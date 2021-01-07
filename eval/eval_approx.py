"""
Approximate PSNR+SSIM evaluation for use during development, since eval.py is too slow.
Evaluates using only 1 random target view per object. You can try different --seed.

python eval_approx.py --gpu_id=<gpu> -n <expname> -c <conf> -D <datadir> -F <format>
Add --seed <num> to set random seed

May not work for DTU.
"""
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import torch
import numpy as np
import imageio
import skimage.measure
import util
from data import get_split_dataset
from render import NeRFRenderer
from model import make_model
import tqdm


def extra_args(parser):
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Split of data to use train | val | test",
    )

    parser.add_argument(
        "--source",
        "-P",
        type=str,
        default="64",
        help="Source view(s) in image, in increasing order. -1 to use random 1 view.",
    )

    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for selecting target views of each object",
    )
    parser.add_argument("--coarse", action="store_true", help="Coarse network as fine")
    return parser


args, conf = util.args.parse_args(extra_args)
args.resume = True

device = util.get_cuda(args.gpu_id[0])
net = make_model(conf["model"]).to(device=device)
net.load_weights(args)

if args.coarse:
    net.mlp_fine = None

dset = get_split_dataset(
    args.dataset_format, args.datadir, want_split=args.split, training=False
)
data_loader = torch.utils.data.DataLoader(
    dset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False
)

renderer = NeRFRenderer.from_conf(
    conf["renderer"], eval_batch_size=args.ray_batch_size
).to(device=device)

if renderer.n_coarse < 64:
    # Ensure decent sampling resolution
    renderer.n_coarse = 64
if args.coarse:
    renderer.n_coarse = 64
    renderer.n_fine = 128
    renderer.using_fine = True

render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

z_near = dset.z_near
z_far = dset.z_far

torch.random.manual_seed(args.seed)

total_psnr = 0.0
total_ssim = 0.0
cnt = 0


source = torch.tensor(list(map(int, args.source.split())), dtype=torch.long)
NS = len(source)
random_source = NS == 1 and source[0] == -1

with torch.no_grad():
    for data in tqdm.tqdm(data_loader, total=len(data_loader)):
        images = data["images"]  # (SB, NV, 3, H, W)
        masks = data["masks"]  # (SB, NV, 1, H, W)
        poses = data["poses"]  # (SB, NV, 4, 4)
        focal = data["focal"][0]

        images_0to1 = images * 0.5 + 0.5  # (B, 3, H, W)

        SB, NV, _, H, W = images.shape

        if random_source:
            src_view = torch.randint(0, NV, (SB, 1))
        else:
            src_view = source.unsqueeze(0).expand(SB, -1)

        dest_view = torch.randint(0, NV - NS, (SB, 1))
        for i in range(NS):
            dest_view += dest_view >= src_view[:, i : i + 1]

        dest_poses = util.batched_index_select_nd(poses, dest_view)
        all_rays = util.gen_rays(
            dest_poses.reshape(-1, 4, 4), W, H, focal, z_near, z_far
        ).reshape(SB, -1, 8)

        pri_images = util.batched_index_select_nd(images, src_view)  # (SB, NS, 3, H, W)
        pri_poses = util.batched_index_select_nd(poses, src_view)  # (SB, NS, 4, 4)

        net.encode(
            pri_images.to(device=device),
            pri_poses.to(device=device),
            focal.to(device=device),
        )

        rgb_fine, _depth = render_par(all_rays.to(device=device))
        _depth = None
        rgb_fine = rgb_fine.reshape(SB, H, W, 3).cpu().numpy()
        images_gt = util.batched_index_select_nd(images_0to1, dest_view).reshape(
            SB, 3, H, W
        )
        rgb_gt_all = images_gt.permute(0, 2, 3, 1).contiguous().numpy()

        for sb in range(SB):
            ssim = skimage.measure.compare_ssim(
                rgb_fine[sb], rgb_gt_all[sb], multichannel=True, data_range=1
            )
            psnr = skimage.measure.compare_psnr(
                rgb_fine[sb], rgb_gt_all[sb], data_range=1
            )
            total_ssim += ssim
            total_psnr += psnr
        cnt += SB
        print("curr psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
print("final psnr", total_psnr / cnt, "ssim", total_ssim / cnt)
