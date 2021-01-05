import os
import torch
import torch.nn.functional as F
import torchvision.transforms.functional_tensor as F_t
import torchvision.transforms.functional as TF
import numpy as np
import imageio

#  from util import GaussianBlur


class ColorJitterDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dset,
        hue_range=0.1,
        saturation_range=0.1,
        brightness_range=0.1,
        contrast_range=0.1,
        extra_inherit_attrs=[],
    ):
        self.hue_range = [-hue_range, hue_range]
        self.saturation_range = [1 - saturation_range, 1 + saturation_range]
        self.brightness_range = [1 - brightness_range, 1 + brightness_range]
        self.contrast_range = [1 - contrast_range, 1 + contrast_range]
        inherit_attrs = ["z_near", "z_far", "lindisp", "base_path", "image_to_tensor"]
        inherit_attrs.extend(extra_inherit_attrs)

        self.base_dset = base_dset
        for inherit_attr in inherit_attrs:
            setattr(self, inherit_attr, getattr(self.base_dset, inherit_attr))

    def apply_color_jitter(self, images):
        # apply the same color jitter over batch of images
        hue_factor = np.random.uniform(*self.hue_range)
        saturation_factor = np.random.uniform(*self.saturation_range)
        brightness_factor = np.random.uniform(*self.brightness_range)
        contrast_factor = np.random.uniform(*self.contrast_range)
        for i in range(len(images)):
            tmp = (images[i] + 1.0) * 0.5
            tmp = F_t.adjust_saturation(tmp, saturation_factor)
            tmp = F_t.adjust_hue(tmp, hue_factor)
            tmp = F_t.adjust_contrast(tmp, contrast_factor)
            tmp = F_t.adjust_brightness(tmp, brightness_factor)
            images[i] = tmp * 2.0 - 1.0
        return images

    def __len__(self):
        return len(self.base_dset)

    def __getitem__(self, idx):
        data = self.base_dset[idx]
        data["images"] = self.apply_color_jitter(data["images"])
        return data
