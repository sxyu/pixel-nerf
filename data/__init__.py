import os

from .MultiObjectDataset import MultiObjectDataset
from .DISNDataset import DISNDataset
from .DVRDataset import DVRDataset
from .SRNDataset import SRNDataset

from .data_util import split_dataset


def get_split_dataset(dataset_type, datadir, want_split="all", **kwargs):
    if dataset_type == "srn":
        train_set = SRNDataset(datadir, stage="train", **kwargs)
        val_set = SRNDataset(datadir, stage="val", **kwargs)
        test_set = SRNDataset(datadir, stage="test", **kwargs)
    elif dataset_type == "multi_obj":
        train_set = MultiObjectDataset(os.path.join(datadir, "train"))
        test_set = MultiObjectDataset(os.path.join(datadir, "test"))
        val_set = MultiObjectDataset(os.path.join(datadir, "val"))
    elif dataset_type == "disn":
        train_set = DISNDataset(datadir, stage="train", **kwargs)
        test_set = DISNDataset(datadir, stage="test", **kwargs)
        val_set = test_set  # Not yet created
    elif dataset_type == "dvr":
        # For category agnostic training
        train_set = DVRDataset(datadir, stage="train", **kwargs)
        val_set = DVRDataset(datadir, stage="val", **kwargs)
        test_set = DVRDataset(datadir, stage="test", **kwargs)
    elif dataset_type == "dvr_gen":
        # For generalization training (train some categories, eval on others)
        train_set = DVRDataset(datadir, stage="train", list_prefix="gen_", **kwargs)
        val_set = DVRDataset(datadir, stage="val", list_prefix="gen_", **kwargs)
        test_set = DVRDataset(datadir, stage="test", list_prefix="gen_", **kwargs)
    elif dataset_type == "dvr_choy":
        # For Choy et al experiments
        train_set = DVRDataset(datadir, stage="train", list_prefix="gen_",
                image_size=(137, 137), scale_focal=False, z_near=0.4, z_far=1.5, **kwargs)
        val_set = DVRDataset(datadir, stage="val", list_prefix="gen_",
                image_size=(137, 137), scale_focal=False, z_near=0.4, z_far=1.5, **kwargs)
        test_set = val_set  # Not separated
    elif dataset_type == "dvr_dtu" or dataset_type == "dvr_tnt" or dataset_type == "dvr_dtumvs":
        list_prefix = "mvsnet_" if dataset_type == "dvr_dtumvs" else "new_"
        sub = 'dtu' if dataset_type.startswith("dvr_dtu") else 'tnt'

        train_set = DVRDataset(datadir, stage="train",
                list_prefix=list_prefix, max_imgs=49,
                sub_format=sub, z_near=0.1, z_far=5.0, **kwargs)
        val_set = DVRDataset(datadir, stage="val",
                list_prefix=list_prefix, max_imgs=49,
                sub_format=sub, z_near=0.1, z_far=5.0, **kwargs)
        test_set = DVRDataset(datadir, stage="test",
                list_prefix=list_prefix, max_imgs=49,
                sub_format=sub, z_near=0.1, z_far=5.0, **kwargs)
    else:
        raise NotImplementedError("Unsupported dataset type", dataset_type)

    if want_split == "train":
        return train_set
    elif want_split == "val":
        return val_set
    elif want_split == "test":
        return test_set
    return train_set, val_set, test_set
