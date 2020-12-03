import os

from .MultiObjectDataset import MultiObjectDataset
from .DVRDataset import DVRDataset
from .SRNDataset import SRNDataset

from .data_util import ColorJitterDataset


def get_split_dataset(dataset_type, datadir, want_split="all", **kwargs):
    if dataset_type == "srn":
        # For ShapeNet single-category (from SRN)
        train_set = SRNDataset(datadir, stage="train", **kwargs)
        val_set = SRNDataset(datadir, stage="val", **kwargs)
        test_set = SRNDataset(datadir, stage="test", **kwargs)
    elif dataset_type == "multi_obj":
        # For multiple-object
        train_set = MultiObjectDataset(os.path.join(datadir, "train"))
        test_set = MultiObjectDataset(os.path.join(datadir, "test"))
        val_set = MultiObjectDataset(os.path.join(datadir, "val"))
    elif dataset_type == "dvr":
        # For ShapeNet category agnostic training
        train_set = DVRDataset(datadir, stage="train", **kwargs)
        val_set = DVRDataset(datadir, stage="val", **kwargs)
        test_set = DVRDataset(datadir, stage="test", **kwargs)
    elif dataset_type == "dvr_gen":
        # For generalization training (train some categories, eval on others)
        train_set = DVRDataset(datadir, stage="train", list_prefix="gen_", **kwargs)
        val_set = DVRDataset(datadir, stage="val", list_prefix="gen_", **kwargs)
        test_set = DVRDataset(datadir, stage="test", list_prefix="gen_", **kwargs)
    elif dataset_type == "dvr_dtu":
        # DTU dataset
        list_prefix = "new_"
        train_set = DVRDataset(
            datadir,
            stage="train",
            list_prefix=list_prefix,
            max_imgs=49,
            sub_format="dtu",
            z_near=0.1,
            z_far=5.0,
            **kwargs
        )
        val_set = DVRDataset(
            datadir,
            stage="val",
            list_prefix=list_prefix,
            max_imgs=49,
            sub_format="dtu",
            z_near=0.1,
            z_far=5.0,
            **kwargs
        )
        test_set = DVRDataset(
            datadir,
            stage="test",
            list_prefix=list_prefix,
            max_imgs=49,
            sub_format="dtu",
            z_near=0.1,
            z_far=5.0,
            **kwargs
        )
        train_set = ColorJitterDataset(train_set)
    else:
        raise NotImplementedError("Unsupported dataset type", dataset_type)

    if want_split == "train":
        return train_set
    elif want_split == "val":
        return val_set
    elif want_split == "test":
        return test_set
    return train_set, val_set, test_set
