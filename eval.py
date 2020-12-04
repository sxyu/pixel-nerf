"""
Full evaluation script, including PSNR+SSIM evaluation with multi-GPU support.

python eval.py --gpu_id=<gpu> -n <expname> -c <conf> -D /home/group/data/chairs -F srn
Parallized, add --extra_gpus='space separated list of additional gpu ids' to use more GPUs
"""
import os
import torch
import numpy as np
import imageio
import skimage.measure
import util
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
import cv2
import tqdm
import ipdb
import warnings

#  from pytorch_memlab import set_target_gpu
#  set_target_gpu(9)


def extra_args(parser):
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Split of data to use train | val | test",
    )
    parser.add_argument(
        "--primary",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, or path to viewlist file",
    )
    parser.add_argument(
        "--eval_view_list", type=str, default=None, help="Path to eval view list"
    )
    parser.add_argument(
        "--ray_batch_size", type=int, default=100000, help="Ray batch size"
    )
    parser.add_argument("--coarse", action="store_true", help="Coarse network as fine")
    parser.add_argument("--no_compare_gt", action="store_true")
    parser.add_argument(
        "--multicat",
        action="store_true",
        help="Prepend category id to object id. Specify if model fits multiple categories.",
    )
    parser.add_argument(
        "--viewlist",
        "-L",
        type=str,
        default="",
        help="Path to source view list e.g. src_dvr.txt; if specified, overrides primary/P",
    )

    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default="eval",
        help="If specified, saves generated images to directory",
    )
    parser.add_argument(
        "--include_src", action="store_true", help="Include source views in calculation"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0, help="Video scale relative to input size"
    )
    parser.add_argument(
        "--black",
        action="store_true",
        help="Force renderer to use a black background.",
    )
    parser.add_argument(
        "--write_depth", action="store_true", help="Write depth image"
    )
    parser.add_argument(
        "--write_compare", action="store_true", help="Write GT comparison image"
    )
    parser.add_argument(
        "--free_pose",
        action="store_true",
        help="Set to indicate poses may change between objects. In most of our datasets, the test set has fixed poses.",
    )
    return parser


args, conf = util.args.parse_args(
    extra_args,
    default_conf="conf/resnet_fine_mv.conf",
    default_expname="shapenet",
)
args.resume = True

device = util.get_cuda(args.gpu_id)

extra_gpus = list(map(int, args.extra_gpus.split()))

dset = get_split_dataset(args.dataset_format, args.datadir, want_split=args.split)
data_loader = torch.utils.data.DataLoader(
    dset, batch_size=1, shuffle=False, num_workers=8, pin_memory=False
)

output_dir = args.output.strip()
has_output = len(output_dir) > 0

total_psnr = 0.0
total_ssim = 0.0
cnt = 0

if has_output:
    finish_path = os.path.join(output_dir, "finish.txt")
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(finish_path):
        with open(finish_path, "r") as f:
            lines = [x.strip().split() for x in f.readlines()]
        lines = [x for x in lines if len(x) == 4]
        finished = set([x[0] for x in lines])
        total_psnr = sum((float(x[1]) for x in lines))
        total_ssim = sum((float(x[2]) for x in lines))
        cnt = sum((int(x[3]) for x in lines))
        if cnt > 0:
            print("resume psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
        else:
            total_psnr = 0.0
            total_ssim = 0.0
    else:
        finished = set()

    finish_file = open(finish_path, "a", buffering=1)
    print("Writing images to", output_dir)


class RenderWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = make_model(conf["model"]).to(device=device)
        self.net.load_weights(args)
        self.renderer = NeRFRenderer.from_conf(conf["renderer"], white_bkgd=not args.black,
                eval_batch_size=args.ray_batch_size).to(device=device)

    def forward(self, x):
        outputs = self.renderer(self.net, x)
        if self.renderer.using_fine:
            rgb = outputs.fine.rgb
            depth = outputs.fine.depth
        else:
            rgb = outputs.coarse.rgb
            depth = outputs.coarse.depth
        return rgb, depth


renderer = RenderWrapper()
renderer.eval()

if args.coarse:
    renderer.net.mlp_fine = None

if renderer.renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.renderer.n_coarse = 64
if args.coarse:
    renderer.renderer.n_coarse = 64
    renderer.renderer.n_fine = 128
    renderer.renderer.using_fine = True

z_near = dset.z_near
z_far = dset.z_far

use_primary_lut = len(args.viewlist) > 0
if use_primary_lut:
    print("Using views from list", args.viewlist)
    with open(args.viewlist, "r") as f:
        tmp = [x.strip().split() for x in f.readlines()]
    primary_lut = {
        x[0] + "/" + x[1]: torch.tensor(list(map(int, x[2:])), dtype=torch.long)
        for x in tmp
    }
else:
    primary = torch.tensor(list(map(int, args.primary.split())), dtype=torch.long)

NV = dset[0]["images"].shape[0]

if args.eval_view_list is not None:
    with open(args.eval_view_list, "r") as f:
        eval_views = torch.tensor(list(map(int, f.readline().split())))
    target_view_mask = torch.zeros(NV, dtype=torch.bool)
    target_view_mask[eval_views] = 1
else:
    target_view_mask = torch.ones(NV, dtype=torch.bool)

all_rays = None
pri_poses = None
focal = None

rays_spl = []

src_view_mask = None
total_objs = len(data_loader)

if len(extra_gpus):
    print("Using extra GPUs", extra_gpus)
    renderer = torch.nn.DataParallel(renderer, [args.gpu_id] + extra_gpus)

with torch.no_grad():
    for obj_idx, data in enumerate(data_loader):
        print(
            "OBJECT",
            obj_idx,
            "OF",
            total_objs,
            "PROGRESS",
            obj_idx / total_objs * 100.0,
            "%",
            data["path"][0],
        )
        dpath = data["path"][0]
        obj_basename = os.path.basename(dpath)
        cat_name = os.path.basename(os.path.dirname(dpath))
        obj_name = cat_name + "_" + obj_basename if args.multicat else obj_basename
        if has_output and obj_name in finished:
            print("(skip)")
            continue
        images = data["images"][0]  # (NV, 3, H, W)

        NV, _, H, W = images.shape

        if args.scale != 1.0:
            Ht = int(H * args.scale)
            Wt = int(W * args.scale)
            if abs(Ht / args.scale - H) > 1e-10 or abs(Wt / args.scale - W) > 1e-10:
                warnings.warn("Inexact scaling, please check {} times ({}, {}) is integral".format(
                    args.scale, H, W))
            H, W = Ht, Wt

        if all_rays is None or use_primary_lut or args.free_pose:
            if use_primary_lut:
                obj_id = cat_name + "/" + obj_basename
                primary = primary_lut[obj_id]

            NS = len(primary)
            src_view_mask = torch.zeros(NV, dtype=torch.bool)
            src_view_mask[primary] = 1

            focal = data["focal"][0]
            if isinstance(focal, float):
                focal = torch.tensor(focal, dtype=torch.float32)
            focal = focal[None]

            c = data.get("c")[0]
            if c is not None:
                c = c.to(device=device).unsqueeze(0)

            poses = data["poses"][0]  # (NV, 4, 4)
            pri_poses = poses[src_view_mask]  # (NS, 4, 4)
            pri_poses = pri_poses.to(device=device)
            if not args.include_src:
                target_view_mask *= ~src_view_mask

            novel_view_idxs = target_view_mask.nonzero(as_tuple=False).reshape(-1)
            poses = poses[target_view_mask]  # (NV-1, 4, 4)

            all_rays = (
                util.gen_rays(poses.reshape(-1, 4, 4), W, H, focal * args.scale, z_near, z_far, c=c * args.scale if c is not None else None)
                .reshape(-1, 8)
                .to(device=device)
            )  # ((NV-1)*H*W, 8)
            rays_spl = torch.split(
                all_rays, args.ray_batch_size, dim=0
            )  # Creates views
            poses = _poses_a = _poses_b = None
            focal = focal.to(device=device)

        n_gen_views = len(novel_view_idxs)

        util.get_module(renderer).net.encode(
            images[src_view_mask].to(device=device).unsqueeze(0),
            pri_poses.unsqueeze(0),
            focal,
            (z_near, z_far),
            c=c
        )

        all_rgb, all_depth = [], []
        for rays in tqdm.tqdm(rays_spl):
            rgb, depth = renderer(rays)
            rgb = rgb.cpu()
            depth = depth.cpu()
            all_rgb.append(rgb)
            all_depth.append(depth)

        all_rgb = torch.cat(all_rgb, dim=0)
        all_depth = torch.cat(all_depth, dim=0)
        all_depth = (all_depth - z_near) / (z_far - z_near)
        all_depth = all_depth.reshape(n_gen_views, H, W).numpy()

        all_rgb = torch.clamp(
            all_rgb.reshape(n_gen_views, H, W, 3), 0.0, 1.0
        ).numpy()  # (NV-NS, H, W, 3)
        if has_output:
            obj_out_dir = os.path.join(output_dir, obj_name)
            os.makedirs(obj_out_dir, exist_ok=True)
            for i in range(n_gen_views):
                out_file = os.path.join(
                    obj_out_dir, "{:06}.png".format(novel_view_idxs[i].item())
                )
                imageio.imwrite(out_file, (all_rgb[i] * 255).astype(np.uint8))

                if args.write_depth:
                    out_depth_file = os.path.join(
                        obj_out_dir, "{:06}_depth.exr".format(novel_view_idxs[i].item())
                    )
                    out_depth_norm_file = os.path.join(
                        obj_out_dir, "{:06}_depth_norm.png".format(novel_view_idxs[i].item())
                    )
                    depth_cmap_norm = util.cmap(all_depth[i])
                    cv2.imwrite(out_depth_file, all_depth[i])
                    imageio.imwrite(out_depth_norm_file, depth_cmap_norm)

        curr_ssim = 0.0
        curr_psnr = 0.0
        if not args.no_compare_gt:
            images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)
            images_gt = images_0to1[target_view_mask]
            rgb_gt_all = (
                images_gt.permute(0, 2, 3, 1).contiguous().numpy()
            )  # (NV-NS, H, W, 3)
            for view_idx in range(n_gen_views):
                ssim = skimage.measure.compare_ssim(
                    all_rgb[view_idx], rgb_gt_all[view_idx], multichannel=True, data_range=1
                )
                psnr = skimage.measure.compare_psnr(
                    all_rgb[view_idx], rgb_gt_all[view_idx], data_range=1
                )
                curr_ssim += ssim
                curr_psnr += psnr

                if args.write_compare:
                    out_file = os.path.join(obj_out_dir, '{:06}_compare.png'.format(
                        novel_view_idxs[view_idx].item()))
                    out_im = np.hstack((all_rgb[view_idx], rgb_gt_all[view_idx]))
                    imageio.imwrite(out_file, (out_im * 255).astype(np.uint8))
        curr_psnr /= n_gen_views
        curr_ssim /= n_gen_views
        curr_cnt = 1
        total_psnr += curr_psnr
        total_ssim += curr_ssim
        cnt += curr_cnt
        if not args.no_compare_gt:
            print(
                "curr psnr",
                curr_psnr,
                "ssim",
                curr_ssim,
                "running psnr",
                total_psnr / cnt,
                "ssim",
                total_ssim / cnt,
            )
        finish_file.write(
            "{} {} {} {}\n".format(obj_name, curr_psnr, curr_ssim, curr_cnt)
        )
print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
