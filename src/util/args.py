import sys
import os


import argparse
from pyhocon import ConfigFactory


def parse_args(
    callback=None,
    training=False,
    default_conf="conf/default_mv.conf",
    default_expname="example",
    default_data_format="dvr",
    default_num_epochs=10000000,
    default_lr=1e-4,
    default_gamma=1.00,
    default_datadir="data",
    default_ray_batch_size=50000,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", "-c", type=str, default=None)
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="GPU(s) to use, space delimited"
    )
    parser.add_argument(
        "--name", "-n", type=str, default=default_expname, help="experiment name"
    )
    parser.add_argument(
        "--dataset_format",
        "-F",
        type=str,
        default=None,
        help="Dataset format, multi_obj | dvr | dvr_gen | dvr_dtu | srn",
    )
    parser.add_argument(
        "--exp_group_name",
        "-G",
        type=str,
        default=None,
        help="if we want to group some experiments together",
    )
    parser.add_argument(
        "--logs_path", type=str, default="logs", help="logs output directory",
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
        "--datadir", "-D", type=str, default=None, help="Dataset directory"
    )
    parser.add_argument(
        "--ray_batch_size", "-R", type=int, default=default_ray_batch_size, help="Ray batch size"
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

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    EXPCONF_PATH = os.path.join(PROJECT_ROOT, "expconf.conf")
    expconf = ConfigFactory.parse_file(EXPCONF_PATH)

    if args.conf is None:
        args.conf = expconf.get_string("config." + args.name, default_conf)

    if args.conf is None:
        args.conf = expconf.get_string("config." + args.name, default_conf)
    if args.datadir is None:
        args.datadir = expconf.get_string("datadir." + args.name, default_datadir)

    conf = ConfigFactory.parse_file(args.conf)

    if args.dataset_format is None:
        args.dataset_format = conf.get_string("data.format", default_data_format)

    args.gpu_id = list(map(int, args.gpu_id.split()))

    print("EXPERIMENT NAME:", args.name)
    if training:
        print("CONTINUE?", "yes" if args.resume else "no")
    print("* Config file:", args.conf)
    print("* Dataset format:", args.dataset_format)
    print("* Dataset location:", args.datadir)
    return args, conf
