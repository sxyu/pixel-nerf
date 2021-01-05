"""
Eval on real images from input/*_normalize.png, output to output/
"""
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

import util
import torch
import numpy as np
from model import make_model
from render import NeRFRenderer
import torchvision.transforms as T
import tqdm
import imageio
from PIL import Image


def extra_args(parser):
    parser.add_argument(
        "--input",
        "-I",
        type=str,
        default=os.path.join(ROOT_DIR, "input"),
        help="Image directory",
    )
    parser.add_argument(
        "--output",
        "-O",
        type=str,
        default=os.path.join(ROOT_DIR, "output"),
        help="Output directory",
    )
    parser.add_argument("--size", type=int, default=128, help="Input image maxdim")
    parser.add_argument(
        "--out_size",
        type=str,
        default="128",
        help="Output image size, either 1 or 2 number (w h)",
    )

    parser.add_argument("--focal", type=float, default=131.25, help="Focal length")

    parser.add_argument("--radius", type=float, default=1.3, help="Camera distance")
    parser.add_argument("--z_near", type=float, default=0.8)
    parser.add_argument("--z_far", type=float, default=1.8)

    parser.add_argument(
        "--elevation",
        "-e",
        type=float,
        default=0.0,
        help="Elevation angle (negative is above)",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=24,
        help="Number of video frames (rotated views)",
    )
    parser.add_argument("--fps", type=int, default=15, help="FPS of video")
    parser.add_argument("--gif", action="store_true", help="Store gif instead of mp4")
    parser.add_argument(
        "--no_vid",
        action="store_true",
        help="Do not store video (only image frames will be written)",
    )
    return parser


args, conf = util.args.parse_args(
    extra_args, default_expname="srn_car", default_data_format="srn",
)
args.resume = True

device = util.get_cuda(args.gpu_id[0])
net = make_model(conf["model"]).to(device=device).load_weights(args)
renderer = NeRFRenderer.from_conf(
    conf["renderer"], eval_batch_size=args.ray_batch_size
).to(device=device)
render_par = renderer.bind_parallel(net, args.gpu_id, simple_output=True).eval()

z_near, z_far = args.z_near, args.z_far
focal = torch.tensor(args.focal, dtype=torch.float32, device=device)

in_sz = args.size
sz = list(map(int, args.out_size.split()))
if len(sz) == 1:
    H = W = sz[0]
else:
    assert len(sz) == 2
    W, H = sz

_coord_to_blender = util.coord_to_blender()
_coord_from_blender = util.coord_from_blender()

print("Generating rays")
render_poses = torch.stack(
    [
        _coord_from_blender @ util.pose_spherical(angle, args.elevation, args.radius)
        #  util.pose_spherical(angle, args.elevation, args.radius)
        for angle in np.linspace(-180, 180, args.num_views + 1)[:-1]
    ],
    0,
)  # (NV, 4, 4)

render_rays = util.gen_rays(render_poses, W, H, focal, z_near, z_far).to(device=device)


inputs_all = os.listdir(args.input)
inputs = [
    os.path.join(args.input, x) for x in inputs_all if x.endswith("_normalize.png")
]
os.makedirs(args.output, exist_ok=True)

if len(inputs) == 0:
    if len(inputs_all) == 0:
        print("No input images found, please place an image into ./input")
    else:
        print("No processed input images found, did you run 'scripts/preproc.py'?")
    import sys

    sys.exit(1)

cam_pose = torch.eye(4, device=device)
cam_pose[2, -1] = args.radius
print("SET DUMMY CAMERA")
print(cam_pose)

image_to_tensor = util.get_image_to_tensor_balanced()

with torch.no_grad():
    for i, image_path in enumerate(inputs):
        print("IMAGE", i + 1, "of", len(inputs), "@", image_path)
        image = Image.open(image_path).convert("RGB")
        image = T.Resize(in_sz)(image)
        image = image_to_tensor(image).to(device=device)

        net.encode(
            image.unsqueeze(0), cam_pose.unsqueeze(0), focal,
        )
        print("Rendering", args.num_views * H * W, "rays")
        all_rgb_fine = []
        for rays in tqdm.tqdm(torch.split(render_rays.view(-1, 8), 80000, dim=0)):
            rgb, _depth = render_par(rays[None])
            all_rgb_fine.append(rgb[0])
        _depth = None
        rgb_fine = torch.cat(all_rgb_fine)
        frames = (rgb_fine.view(args.num_views, H, W, 3).cpu().numpy() * 255).astype(
            np.uint8
        )

        im_name = os.path.basename(os.path.splitext(image_path)[0])

        frames_dir_name = os.path.join(args.output, im_name + "_frames")
        os.makedirs(frames_dir_name, exist_ok=True)

        for i in range(args.num_views):
            frm_path = os.path.join(frames_dir_name, "{:04}.png".format(i))
            imageio.imwrite(frm_path, frames[i])

        if not args.no_vid:
            if args.gif:
                vid_path = os.path.join(args.output, im_name + "_vid.gif")
                imageio.mimwrite(vid_path, frames, fps=args.fps)
            else:
                vid_path = os.path.join(args.output, im_name + "_vid.mp4")
                imageio.mimwrite(vid_path, frames, fps=args.fps, quality=8)
        print("Wrote to", vid_path)
