import os
import torch
import torch.nn.functional as F
import glob
import imageio
import numpy as np
from util import get_image_to_tensor_balanced, get_mask_to_tensor, look_at


class DISNDataset(torch.utils.data.Dataset):
    """
    Dataset from DISN (Xu et al. 2019)
    """

    def __init__(
        self,
        path,
        stage="train",
        mode="easy",
        image_size=(224, 224)
    ):
        """
        :param stage train | test
        :param mode easy | hard NOTE: DISN's provided hard set is broken, do not use
        :param image_size result image size (resizes if different)
        :param world_scale amount to scale entire world by
        """
        super().__init__()
        self.base_path = os.path.join(path, "image")
        self.filelists_path = os.path.join(path, "filelists")

        test_lists = glob.glob(os.path.join(self.filelists_path, "*_test.lst"))
        all_test = {}
        for test_list in test_lists:
            cat = os.path.basename(test_list)[:-9]
            with open(test_list, 'r') as f:
                ids = [x.strip() for x in f.readlines()]
            all_test[cat] = set(ids)

        self.stage = stage
        self.mode = mode
        assert os.path.exists(self.base_path)
        assert os.path.exists(self.filelists_path)  # Please copy data/filelists from DISN repo

        cat_roots = sorted(glob.glob(os.path.join(self.base_path, "*")))

        self.all_objs = []
        test_suffix = "/easy/rendering_metadata.txt"
        for cat_root in cat_roots:
            cat = os.path.basename(cat_root)
            cat_objs = set([os.path.basename(
                x[:-len(test_suffix)]) for x in glob.glob(
                    os.path.join(cat_root, "*" + test_suffix))])

            if cat in all_test:
                test_set = all_test[cat]
            else:
                test_set = set()
            if self.stage == 'test':
                cat_objs = cat_objs.intersection(test_set)
            else:
                cat_objs = cat_objs.difference(test_set)
            cat_objs = sorted(list(cat_objs))
            self.all_objs.extend([(cat, os.path.join(cat_root, x)) for x in cat_objs])

        self.image_to_tensor = get_image_to_tensor_balanced()
        self.mask_to_tensor = get_mask_to_tensor()
        print("Loading DISN dataset", self.base_path, 'stage', stage, 'mode', mode,
                len(self.all_objs), 'objs')

        self.image_size = image_size
        self._coord_trans = torch.tensor(
            [[0, 0, -1, 0],
             [1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]],
            dtype=torch.float32
        )

        self.z_near = 0.5
        self.z_far = 2.5
        self.lindisp = False

    def __len__(self):
        return len(self.all_objs)

    def __getitem__(self, index):
        cat, root_dir = self.all_objs[index]
        if self.mode is not None:
            root_dir = os.path.join(root_dir, self.mode)

        rgb_paths = sorted(glob.glob(os.path.join(root_dir, "*.png")))
        meta_path = os.path.join(root_dir , "rendering_metadata.txt")

        def parse_list(x):
            first, last = x.index('[') + 1, x.rindex(']')
            return list(map(float, x[first:last].split(',')))
        with open(meta_path, 'r') as f:
            all_meta = map(parse_list, f.readlines())

        all_imgs = []
        all_poses = []
        all_masks = []
        all_bboxes = []
        focal = None
        for rgb_path, meta in zip(rgb_paths, all_meta):
            yaw, pitch, roll, ratio, focal_mm, sens_mm, dist, xr, yr, zr = meta
            dist *= ratio

            assert roll == 0.0  # Not supported

            img = imageio.imread(rgb_path)
            mask = img[..., 3:]
            img = img[..., :3]

            img_tensor = self.image_to_tensor(img)
            mask_tensor = self.mask_to_tensor(mask)
            img_tensor = img_tensor * mask_tensor + (1.0 - mask_tensor)

            focal_ = (focal_mm / sens_mm) * img_tensor.shape[-1]
            if focal is None:
                focal = focal_
            else:
                assert focal == focal_

            theta = np.deg2rad(yaw)
            phi = np.deg2rad(pitch)

            camY = dist * np.sin(phi) - yr
            temp = dist * np.cos(phi)
            camX = temp * np.cos(theta) - xr
            camZ = temp * np.sin(theta) - zr
            cam_pos = np.array([camX, camY, camZ])

            pose = torch.from_numpy(look_at(cam_pos, np.zeros(3)))
            pose = self._coord_trans @ pose

            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            rnz = np.where(rows)[0]
            cnz = np.where(cols)[0]
            if len(rnz) == 0:
                raise RuntimeError('ERROR: Bad image at', rgb_path, 'please investigate!')
            rmin, rmax = rnz[[0, -1]]
            cmin, cmax = cnz[[0, -1]]
            bbox = torch.tensor([cmin, rmin, cmax, rmax], dtype=torch.float32)

            all_imgs.append(img_tensor)
            all_masks.append(mask_tensor)
            all_poses.append(pose)
            all_bboxes.append(bbox)

        all_imgs = torch.stack(all_imgs)
        all_poses = torch.stack(all_poses)
        all_masks = torch.stack(all_masks)
        all_bboxes = torch.stack(all_bboxes)

        if all_imgs.shape[-2:] != self.image_size:
            scale = self.image_size[0] / all_imgs.shape[-2]
            focal *= scale
            all_bboxes *= scale

            all_imgs = F.interpolate(all_imgs, size=self.image_size, mode="area")
            all_masks = F.interpolate(all_masks, size=self.image_size, mode="area")

        result = {
            "path": root_dir,
            "img_id": index,
            "focal": focal,
            "images": all_imgs,
            "masks": all_masks,
            "bbox": all_bboxes,
            "poses": all_poses,
        }
        return result
