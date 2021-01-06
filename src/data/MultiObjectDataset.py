import os
import os.path as osp
import glob
import json
import imageio
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

from util import get_image_to_tensor_balanced, get_mask_to_tensor


class MultiObjectDataset(torch.utils.data.Dataset):
    """Synthetic dataset of scenes with multiple Shapenet objects"""

    def __init__(self, path, stage="train", z_near=4, z_far=9, n_views=None, compose=False,
                 split_seed=1234, val_frac=0.2, test_frac=0.2):
        """
        :param path data directory
        :para stage train | val | test
        :param z_near near bound for ray sampling
        :param z_far far bound for ray sampling
        :param n_views optional: expected number of views per object in dataset
        for validity checking only
        :param compose if true, adds background to images.
        Dataset must have background '*_env.png' in addition to foreground
        '*_obj.png' for each view.
        """
        super().__init__()
        self.base_path = path
        self.stage = stage

        print("Loading NeRF synthetic dataset", self.base_path, "stage", self.stage)
        trans_files = []
        TRANS_FILE = "transforms.json"
        for root, directories, filenames in os.walk(path):
            if TRANS_FILE in filenames:
                trans_files.append(osp.join(root, TRANS_FILE))
        self.trans_files = trans_files
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False

        self.compose = compose
        self.n_views = n_views

        # Load data split
        self._load_split(val_frac, test_frac, split_seed)

    def __len__(self):
        return len(self.trans_files)

    def _check_valid(self, index):
        if self.n_views is None:
            return True
        trans_file = self.trans_files[index]
        dir_path = osp.dirname(trans_file)
        try:
            with open(trans_file, "r") as f:
                transform = json.load(f)
        except Exception as e:
            print("Problematic transforms.json file", trans_file)
            print("JSON loading exception", e)
            return False
        if len(transform["frames"]) != self.n_views:
            return False
        if len(glob.glob(osp.join(dir_path, "*.png"))) != self.n_views:
            return False
        return True

    def _load_split(self, val_frac, test_frac, seed):
        permute_file = osp.join(self.base_path, "split_{}.pth".format(seed))
        num_objs = len(self)
        if osp.isfile(permute_file):
            print("Loading dataset split from {}".format(permute_file))
            permute = torch.load(permute_file)
        else:
            if val_frac == 0 and test_frac == 0:
                warn("creating empty validation and test sets")
            state = np.random.get_state()
            np.random.seed(seed)
            print("Created dataset split in {}".format(permute_file))

            permute = np.random.permutation(num_objs)
            torch.save(permute, permute_file)
            np.random.set_state(state)

        val_size = int(val_frac * num_objs)
        test_size = int(test_frac * num_objs)
        train_end = num_objs - (val_size + test_size)
        val_end = num_objs - test_size

        if self.stage == 'train':
            subset = permute[:train_end]
        elif self.stage == 'val':
            subset = permute[train_end:val_end]
        elif self.stage == 'test':
            subset = permute[val_end:]
        self.trans_files = [self.trans_files[i] for i in subset]
        assert len(self) == len(subset)

    def __getitem__(self, index):
        if not self._check_valid(index):
            return {}

        trans_file = self.trans_files[index]
        dir_path = osp.dirname(trans_file)
        with open(trans_file, "r") as f:
            transform = json.load(f)

        all_imgs = []
        all_bboxes = []
        all_masks = []
        all_poses = []

        for frame in transform["frames"]:
            fpath = frame["file_path"]
            basename = osp.splitext(osp.basename(fpath))[0]
            obj_path = osp.join(dir_path, "{}_obj.png".format(basename))
            img = imageio.imread(obj_path)
            mask = self.mask_to_tensor(img[..., 3])
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                cmin = rmin = 0
                cmax = mask.shape[-1]
                rmax = mask.shape[-2]
            else:
                rmin, rmax = rnz[[0, -1]]
                cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            img_tensor = self.image_to_tensor(img[..., :3])
            if self.compose:
                env_path = osp.join(dir_path, "{}_env.png".format(basename))
                env_img = self.image_to_tensor(imageio.imread(env_path)[..., :3])
                img = img_tensor * mask + (
                    env_img * (1.0 - mask)
                )  # env image where transparent
            else:
                img = img_tensor * mask + (
                    1.0 - mask
                )  # solid white background where transparent

            all_imgs.append(img)
            all_bboxes.append(bbox)
            all_masks.append(mask)
            all_poses.append(torch.tensor(frame["transform_matrix"]))
        imgs = torch.stack(all_imgs)
        masks = torch.stack(all_masks)
        bboxes = torch.stack(all_bboxes)
        poses = torch.stack(all_poses)

        H, W = imgs.shape[-2:]
        camera_angle_x = transform.get("camera_angle_x")
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

        result = {
            "path": dir_path,
            "img_id": index,
            "focal": focal,
            "images": imgs,
            "masks": masks,
            "bbox": bboxes,
            "poses": poses,
        }

        return result
