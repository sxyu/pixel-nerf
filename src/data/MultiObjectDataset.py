import os
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

    def __init__(self, path, stage="train", z_near=4, z_far=9, n_views=None):
        super().__init__()
        path = os.path.join(path, stage)
        self.base_path = path
        print("Loading NeRF synthetic dataset", self.base_path)
        trans_files = []
        TRANS_FILE = "transforms.json"
        for root, directories, filenames in os.walk(self.base_path):
            if TRANS_FILE in filenames:
                trans_files.append(os.path.join(root, TRANS_FILE))
        self.trans_files = trans_files
        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False
        self.n_views = n_views

        print("{} instances in split {}".format(len(self.trans_files), stage))

    def __len__(self):
        return len(self.trans_files)

    def _check_valid(self, index):
        if self.n_views is None:
            return True
        trans_file = self.trans_files[index]
        dir_path = os.path.dirname(trans_file)
        try:
            with open(trans_file, "r") as f:
                transform = json.load(f)
        except Exception as e:
            print("Problematic transforms.json file", trans_file)
            print("JSON loading exception", e)
            return False
        if len(transform["frames"]) != self.n_views:
            return False
        if len(glob.glob(os.path.join(dir_path, "*.png"))) != self.n_views:
            return False
        return True

    def __getitem__(self, index):
        if not self._check_valid(index):
            return {}

        trans_file = self.trans_files[index]
        dir_path = os.path.dirname(trans_file)
        with open(trans_file, "r") as f:
            transform = json.load(f)

        all_imgs = []
        all_bboxes = []
        all_masks = []
        all_poses = []
        for frame in transform["frames"]:
            fpath = frame["file_path"]
            basename = os.path.splitext(os.path.basename(fpath))[0]
            obj_path = os.path.join(dir_path, "{}_obj.png".format(basename))
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
