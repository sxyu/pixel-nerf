# Training to all of ShapeNet, with single-view encoding
# tensorboard logs available in logs/<expname>!

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import trainlib
from model import make_model, loss
from render import NeRFRenderer
from data import get_split_dataset
import util
import numpy as np
import torch.nn.functional as F
import torch

# Set if getting NaNs in loss/gradient
# torch.autograd.set_detect_anomaly(True)


def extra_args(parser):
    # Framework for adding file-specific arguments
    parser.add_argument(
        "--num_points_per_iter",
        type=int,
        default=128,
        help="Number of points to sample for each iteration",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")

    parser.add_argument(
        "--nviews",
        "-V",
        type=str,
        default="1",
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch",
    )

    parser.add_argument("--freeze_enc", action="store_true", default=None,
            help="Freeze encoder weights")

    parser.add_argument(
        "--hard",
        type=int,
        default=0,
        help="Use adaptive hard ray sampling with given number of preview rays."
        + "set to 0 to disable. Otherwise, maybe 2048 is a reasonable value.",
    )

    parser.add_argument("--z_near", type=float, default=None)
    parser.add_argument("--z_far", type=float, default=None)
    parser.add_argument("--lindisp", action="store_true", default=None)

    parser.add_argument("--save_tb_images", action="store_true", default=False)
    parser.add_argument(
        "--fixed_test",
        action="store_true",
        help="evaluate and visualize for a fixed batch in the test set",
    )
    parser.add_argument("--eval", action="store_true", help="evaluate the model")
    parser.add_argument("--out_dir", type=str, help="output dir")
    parser.add_argument("--test_idx", type=int, default=None, help="size of test batch")
    parser.add_argument(
        "--use_mask",
        action="store_true",
        help="Allow use of masks. Currently used to remove the background for NeRF supervision.",
    )
    parser.add_argument(
        "--black",
        action="store_true",
        help="Force renderer to use a black background.",
    )
    return parser


args, conf = util.args.parse_args(
    extra_args,
    default_conf="conf/resnet_fine_mv_view.conf",
    default_expname="srn",
    default_num_epochs=50000,
    default_lr=1e-4,
    default_datadir="/home/group/chairs",
)
device = util.get_cuda(args.gpu_id)
assert len(args.extra_gpus) == 0  # Multi-gpu not supported

dset, val_set, _ = get_split_dataset(args.dataset_format, args.datadir,
        ban_views=conf.get_list('data.ban_views', []))
if args.z_near is not None:
    dset.z_near = args.z_near
if args.z_far is not None:
    dset.z_far = args.z_far
if args.lindisp is not None:
    dset.lindisp = args.lindisp

print(
    "dset z_near {}, z_far {}, lindisp {}".format(dset.z_near, dset.z_far, dset.lindisp)
)

data_loader = torch.utils.data.DataLoader(
    dset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False
)

shuffle_test = not args.eval
test_data_loader = torch.utils.data.DataLoader(
    val_set,
    batch_size=min(args.batch_size, 16),
    shuffle=shuffle_test,
    num_workers=4,
    pin_memory=False,
)

net = make_model(conf["model"]).to(device=device)
net.stop_encoder_grad = args.freeze_enc
if args.freeze_enc:
    print("Encoder frozen")
    net.encoder.eval()

# Init
enc_init_path = "inits/%s/enc_init" % args.name
global_enc_init_path = "inits/%s/globalenc_init" % args.name
if net.use_global_encoder and os.path.exists(global_enc_init_path):
    net.global_encoder.load_state_dict(torch.load(global_enc_init_path))
    print("Init global encoder to ", global_enc_init_path)
if net.use_encoder and os.path.exists(enc_init_path):
    net.encoder.load_state_dict(torch.load(enc_init_path))
    print("Init encoder to ", enc_init_path)


nviews = list(map(int, args.nviews.split()))


class ShapenetTrainer(trainlib.Trainer):
    def __init__(self):
        super().__init__(net, data_loader, test_data_loader, args, conf["train"],
                device=device)
        self.renderer = NeRFRenderer.from_conf(
            conf["renderer"],
            white_bkgd=not args.black,
            lindisp=dset.lindisp,
        ).to(device=device)
        self.renderer_state_path = "%s/%s/_renderer" % (
            self.args.checkpoints_path,
            self.args.name,
        )
        self.alpha_prior_state_path = "%s/%s/_alphaprior" % (
            self.args.checkpoints_path,
            self.args.name,
        )

        self.lambda_coarse = conf.get_float("loss.lambda_coarse")
        self.lambda_fine = conf.get_float("loss.lambda_fine", 1.0)
        print("lambda coarse {} and fine {}".format(self.lambda_coarse, self.lambda_fine))
        self.rgb_coarse_crit = loss.get_rgb_loss(conf["loss.rgb"], True)
        fine_loss_conf = conf["loss.rgb"]
        if "rgb_fine" in conf["loss"]:
            print("using fine loss")
            fine_loss_conf = conf["loss.rgb_fine"]
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)
        self.crit_alpha_prior = loss.get_alpha_loss(conf["loss.alpha"]).to(
            device=device
        )
        self.crit_alpha_prior.epoch.fill_(-1)

        self.using_uncertainty = (
            conf["loss.rgb"].get_bool("use_uncertainty", False) and net.pred_betas
        )
        print("Using loss with uncertainty?", self.using_uncertainty)
        self.lambda_sigma_tv = conf.get_float("loss.sigma.lambda_tv", 0.0)
        self.using_sigma_reg = self.lambda_sigma_tv > 0
        self.using_alpha_reg = conf.get_float("loss.alpha.lambda_alpha", 0.0) > 0.0

        if args.resume:
            if os.path.exists(self.renderer_state_path):
                self.renderer.load_state_dict(torch.load(self.renderer_state_path,
                    map_location=device))
            if os.path.exists(self.alpha_prior_state_path):
                self.crit_alpha_prior.load_state_dict(
                    torch.load(self.alpha_prior_state_path, map_location=device)
                )

        self.z_near = dset.z_near
        self.z_far = dset.z_far
        #  _mlp_dummy = torch.zeros(1, net.d_in + net.d_latent, device=device)
        #  self.writer.add_graph(net.mlp_coarse, _mlp_dummy)
        #  self.writer.add_graph(net.mlp_fine, _mlp_dummy)
        #  _img_dummy = torch.zeros(1, 3, 128, 128, device=device)
        #  if net.use_encoder:
        #      self.writer.add_graph(net.encoder, _img_dummy)
        #  if net.use_global_encoder:
        #      self.writer.add_graph(net.global_encoder, _img_dummy)

    def on_epoch(self, epoch):
        self.crit_alpha_prior.sched_step()

    def post_batch(self, epoch, batch):
        self.renderer.sched_step(args.batch_size)

    def extra_save_state(self):
        torch.save(self.renderer.state_dict(), self.renderer_state_path)
        torch.save(self.crit_alpha_prior.state_dict(), self.alpha_prior_state_path)

    def calc_losses(self, data, is_train=True):
        if "images" not in data:
            return {}
        all_images = data["images"].to(device=device)  # (SB, NV, 3, H, W)
        if args.use_mask:
            all_masks = data["masks"]  # (SB, NV, 1, H, W)

        SB, NV, _, H, W = all_images.shape
        all_poses = data["poses"].to(device=device)  # (SB, NV, 4, 4)
        all_bboxes = data.get("bbox")  # (SB, NV, 4)  cmin rmin cmax rmax
        all_focals = data["focal"]  # (SB)
        all_c = data.get("c")  # (SB)

        # TEMP
        #  all_bboxes = None
        # END TEMP

        all_rgb_gt = []
        all_rays = []

        curr_nviews = nviews[torch.randint(0, len(nviews), ()).item()]
        if curr_nviews == 1:
            image_ord = torch.randint(0, NV, (SB, 1))
        else:
            image_ord = torch.empty((SB, curr_nviews), dtype=torch.long)
        for obj_idx in range(SB):
            if all_bboxes is not None:
                bboxes = all_bboxes[obj_idx]
            images = all_images[obj_idx]  # (NV, 3, H, W)
            poses = all_poses[obj_idx]  # (NV, 4, 4)
            focal = all_focals[obj_idx]
            c = None
            if "c" in data:
                c = data["c"][obj_idx]
            if curr_nviews > 1:
                # Somewhat inefficient, don't know better way
                image_ord[obj_idx] = torch.from_numpy(
                        np.random.choice(NV, curr_nviews, replace=False))
            images_0to1 = images * 0.5 + 0.5

            cam_rays = util.gen_rays(
                poses, W, H, focal, self.z_near, self.z_far, c=c
            )  # (NV, H, W, 8)
            if args.use_mask:
                mask = all_masks[obj_idx].to(device)
                rgb_gt_all = images_0to1 * mask + (1.0 - mask)
            else:
                rgb_gt_all = images_0to1
            rgb_gt_all = (
                rgb_gt_all.permute(0, 2, 3, 1).contiguous().reshape(-1, 3)
            )  # (NV, H, W, 3)

            if is_train and args.hard > 0:
                pix_inds = util.adaptive_hard_sample(
                    cam_rays,
                    rgb_gt_all,
                    net,
                    self.renderer,
                    images[image_ord],
                    poses[image_ord],
                    focal,
                    self.z_near,
                    self.z_far,
                    args.num_points_per_iter,
                    args.hard,
                )
            else:
                if all_bboxes is not None:
                    pix = util.bbox_sample(bboxes, args.num_points_per_iter)
                    pix_inds = pix[..., 0] * H * W + pix[..., 1] * W + pix[..., 2]
                else:
                    pix_inds = torch.randint(0, NV * H * W, (args.num_points_per_iter,))

            rgb_gt = rgb_gt_all[pix_inds]  # (num_points_per_iter, 3)
            rays = cam_rays.view(-1, cam_rays.shape[-1])[pix_inds].to(
                device=device
            )  # (num_points_per_iter, 8)

            all_rgb_gt.append(rgb_gt)
            all_rays.append(rays)

        all_rgb_gt = torch.stack(all_rgb_gt)  # (SB, num_points_per_iter, 3)
        all_rays = torch.stack(all_rays)  # (SB, num_points_per_iter, 8)

        image_ord = image_ord.to(device)
        pri_images = util.batched_index_select_nd(
            all_images, image_ord
        )  # (SB, NS, 3, H, W)
        pri_poses = util.batched_index_select_nd(all_poses, image_ord)  # (SB, NS, 4, 4)

        all_bboxes = None
        all_poses = None
        all_images = None

        util.get_module(net).encode(
            pri_images,
            pri_poses,
            all_focals.to(device=device),
            (self.z_near, self.z_far),
            c=all_c.to(device=device) if all_c is not None else None,
        )

        render_dict = self.renderer(
            net,
            all_rays,
            want_weights=True,
            want_betas=self.using_uncertainty,
            want_sigma_grad=self.using_sigma_reg,
        )
        coarse = render_dict.coarse
        fine = render_dict.fine
        using_fine = len(fine) > 0

        loss_dict = {}

        rgb_loss = self.rgb_coarse_crit(coarse.rgb, all_rgb_gt)
        loss_dict["rc"] = rgb_loss.item() * self.lambda_coarse
        if using_fine:
            if self.using_uncertainty:
                fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt, fine.betas)
            else:
                fine_loss = self.rgb_fine_crit(fine.rgb, all_rgb_gt)
            rgb_loss = rgb_loss * self.lambda_coarse + fine_loss * self.lambda_fine
            loss_dict["rf"] = fine_loss.item() * self.lambda_fine


        alpha_loss = 0.0
        if self.using_alpha_reg:
            alpha_coarse = torch.clamp(coarse.weights.sum(dim=-1), 0.0, 1.0)
            alpha_loss = self.crit_alpha_prior(alpha_coarse)
            if using_fine:
                alpha_fine = torch.clamp(fine.weights.sum(dim=-1), 0.0, 1.0)
                alpha_loss = alpha_loss * self.lambda_coarse + self.crit_alpha_prior(
                    alpha_fine
                )
            loss_dict["a"] = alpha_loss.item()
        else:
            alpha_loss = 0.0

        sigma_loss = 0.0
        if self.using_sigma_reg:
            sigma_loss = self.lambda_sigma_tv * coarse.sigma_grad.mean()
            loss_dict["s"] = sigma_loss.item()

        loss = alpha_loss + rgb_loss + sigma_loss
        if is_train:
            self.loss_backward(loss)
        loss_dict["t"] = loss.item()

        return loss_dict

    def train_step(self, data):
        return self.calc_losses(data, True)

    def eval_step(self, data):
        self.renderer.eval()
        losses = self.calc_losses(data, False)
        self.renderer.train()
        return losses

    def vis_step(self, data, idx=None):
        if "images" not in data:
            return {}
        if idx is None:
            batch_idx = np.random.randint(0, data["images"].shape[0])
        else:
            print(idx)
            batch_idx = idx
        images = data["images"][batch_idx].to(device=device)  # (NV, 3, H, W)
        poses = data["poses"][batch_idx].to(device=device)  # (NV, 4, 4)
        focal = data["focal"][batch_idx : batch_idx + 1]  # (1)
        c = data.get("c")
        if c is not None:
            c = c[batch_idx : batch_idx + 1]  # (1)
        NV, _, H, W = images.shape
        cam_rays = util.gen_rays(
            poses, W, H, focal, self.z_near, self.z_far, c=c
        )  # (NV, H, W, 8)
        images_0to1 = images * 0.5 + 0.5  # (NV, 3, H, W)

        curr_nviews = nviews[torch.randint(0, len(nviews), (1,)).item()]
        views_src = np.sort(np.random.choice(NV, curr_nviews, replace=False))
        view_dest = np.random.randint(0, NV - curr_nviews)
        for vs in range(curr_nviews):
            view_dest += view_dest >= views_src[vs]
        views_src = torch.from_numpy(views_src)

        # set renderer net to eval mode
        self.renderer.eval()
        source_views = (
            images_0to1[views_src]
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
            .reshape(-1, H, W, 3)
        )

        if args.use_mask:
            mask = data["masks"][batch_idx]
            images_0to1 = images_0to1 * mask.to(device) + (1.0 - mask.to(device))

        gt = images_0to1[view_dest].permute(1, 2, 0).cpu().numpy().reshape(H, W, 3)
        with torch.no_grad():
            test_rays = cam_rays[view_dest]  # (H, W, 8)
            test_images = images[views_src]  # (NS, 3, H, W)
            util.get_module(net).encode(
                test_images.unsqueeze(0),
                poses[views_src].unsqueeze(0),
                focal.to(device=device),
                (self.z_near, self.z_far),
                c=c.to(device=device) if c is not None else None,
            )
            test_rays = test_rays.reshape(H * W, -1)
            render_dict = self.renderer(
                net, test_rays, want_weights=True, want_betas=self.using_uncertainty
            )
            coarse = render_dict.coarse
            fine = render_dict.fine

            using_fine = len(fine) > 0

            alpha_coarse_np = coarse.weights.sum(dim=-1).cpu().numpy().reshape(H, W)
            rgb_coarse_np = coarse.rgb.cpu().numpy().reshape(H, W, 3)
            depth_coarse_np = coarse.depth.cpu().numpy().reshape(H, W)

            if self.using_uncertainty:
                beta_coarse_cmap = np.zeros((H, W, 3))
                if using_fine:
                    beta_fine_np = np.clip(fine.betas.cpu().numpy().reshape(H, W), 0, 1)
                    beta_fine_cmap = util.cmap(beta_fine_np) / 255

            if using_fine:
                alpha_fine_np = fine.weights.sum(dim=1).cpu().numpy().reshape(H, W)
                depth_fine_np = fine.depth.cpu().numpy().reshape(H, W)
                rgb_fine_np = fine.rgb.cpu().numpy().reshape(H, W, 3)

        print("c rgb min {} max {}".format(rgb_coarse_np.min(), rgb_coarse_np.max()))
        print(
            "c alpha min {}, max {}".format(
                alpha_coarse_np.min(), alpha_coarse_np.max()
            )
        )
        alpha_coarse_cmap = util.cmap(alpha_coarse_np) / 255
        depth_coarse_cmap = util.cmap(depth_coarse_np) / 255
        vis_list = [
            *source_views,
            gt,
            depth_coarse_cmap,
            rgb_coarse_np,
            alpha_coarse_cmap,
        ]

        if self.using_uncertainty:
            vis_list.append(beta_coarse_cmap)

        vis_coarse = np.hstack(vis_list)
        vis = vis_coarse

        if using_fine:
            print("f rgb min {} max {}".format(rgb_fine_np.min(), rgb_fine_np.max()))
            print(
                "f alpha min {}, max {}".format(
                    alpha_fine_np.min(), alpha_fine_np.max()
                )
            )
            depth_fine_cmap = util.cmap(depth_fine_np) / 255
            alpha_fine_cmap = util.cmap(alpha_fine_np) / 255
            vis_list = [
                *source_views,
                gt,
                depth_fine_cmap,
                rgb_fine_np,
                alpha_fine_cmap,
            ]
            if self.using_uncertainty:
                print(
                    "f beta min {}, max {}".format(
                        beta_fine_np.min(), beta_fine_np.max()
                    )
                )
                vis_list.append(beta_fine_cmap)
            vis_fine = np.hstack(vis_list)
            vis = np.vstack((vis_coarse, vis_fine))
            rgb_psnr = rgb_fine_np
        else:
            rgb_psnr = rgb_coarse_np

        psnr = util.psnr(rgb_psnr, gt)
        vals = {"psnr": psnr}
        print("psnr", psnr)

        # set the renderer network back to train mode
        self.renderer.train()
        return vis, vals


trainer = ShapenetTrainer()
if args.eval:
    print("EVAL MODE ON")
    os.makedirs(args.out_dir, exist_ok=True)
    import imageio

    for i, test_data in enumerate(test_data_loader):
        if i > 50:
            break
        trainer.net.eval()
        with torch.no_grad():
            vis, vis_vals = trainer.vis_step(test_data, idx=0)
            print("EVAL BATCH", i, vis_vals)

            vis_u8 = (vis * 255).astype(np.uint8)
            imageio.imwrite(
                os.path.join(args.out_dir, "{:04}_vis.png".format(i)), vis_u8,
            )
else:
    trainer.start()
