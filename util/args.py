import os
import argparse
from pyhocon import ConfigFactory


def parse_args(
    callback=None,
    default_conf="conf/resnet_fine.conf",
    default_expname="example",
    default_num_epochs=10000,
    default_lr=1e-4,
    default_gamma=1.00,
    default_datadir="/home/group/data/chairs",
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", type=str, default=default_conf)
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU to use")
    parser.add_argument(
        "--extra_gpus",
        type=str,
        default="",
        help="Extra GPUs for data parallel, space delim",
    )
    parser.add_argument(
        "--name", "-n", type=str, default=default_expname, help="experiment name"
    )
    parser.add_argument(
        "--dataset_format",
        "-F",
        type=str,
        default="srn",
        help="Dataset format, nerf | srn | realestate",
    )
    parser.add_argument(
        "--exp_group_name",
        "-G",
        type=str,
        default=None,
        help="if we want to group some experiments together",
    )
    parser.add_argument(
        "--logs_path",
        type=str,
        default="logs",
        help="logs output directory",
    )
    parser.add_argument(
        "--checkpoints_path",
        type=str,
        default="checkpoints",
        help="checkpoints output directory",
    )
    parser.add_argument(
        "--visual_path",
        type=str,
        default="visuals",
        help="visualization output directory",
    )
    parser.add_argument(
        "--skip_epochs",
        type=int,
        default=0,
        help="DEPRECATED: number of epochs to skip",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_num_epochs,
        help="number of epochs to train for",
    )
    parser.add_argument("--lr", type=float, default=default_lr, help="learning rate")
    parser.add_argument(
        "--gamma", type=float, default=default_gamma, help="learning rate decay factor"
    )
    parser.add_argument(
        "--datadir", "-D", type=str, default=default_datadir, help="Dataset directory"
    )
    parser.add_argument(
        "--amp", action="store_true", help="use auto mixed precision, if applicable"
    )
    if callback is not None:
        parser = callback(parser)
    args = parser.parse_args()

    if args.exp_group_name is not None:
        args.logs_path = os.path.join(args.logs_path, args.exp_group_name)
        args.checkpoints_path = os.path.join(args.checkpoints_path, args.exp_group_name)
        args.visual_path = os.path.join(args.visual_path, args.exp_group_name)

    os.makedirs(os.path.join(args.checkpoints_path, args.name), exist_ok=True)
    os.makedirs(os.path.join(args.visual_path, args.name), exist_ok=True)
    conf = ConfigFactory.parse_file(args.conf)
    print("EXPERIMENT NAME:", args.name, "CONTINUE?", "yes" if args.resume else "no")
    return args, conf
