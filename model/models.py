"""
Main model implementation
"""
import torch
from .encoder import ImageEncoder
from .code import PositionalEncoding
from .model_util import make_encoder, make_mlp
import torch.autograd.profiler as profiler
from util import repeat_interleave
import os


class PIFuNeRFNet(torch.nn.Module):
    def __init__(self, conf, stop_encoder_grad=False):
        """
        :param conf PyHocon config subtree 'model'
        """
        super().__init__()
        print("Using PIFuNeRF model")

        self.encoder = make_encoder(conf["encoder"])
        self.use_encoder = conf.get_bool("use_encoder", True)  # Image features?

        self.use_xyz = conf.get_bool("use_xyz", False) 

        assert self.use_encoder or self.use_xyz  # Must use some feature..

        # Whether to shift z to align in canonical frame.
        # So that all objects, regardless of camera distance to center, will
        # be centered at z=0.
        # Only makes sense in ShapeNet-type setting.
        self.normalize_z = conf.get_bool("normalize_z", True)

        self.stop_encoder_grad = (
            stop_encoder_grad  # Stop ConvNet gradient (freeze weights)
        )
        self.use_code = conf.get_bool("use_code", False)  # Positional encoding
        self.use_code_viewdirs = conf.get_bool(
            "use_code_viewdirs", True
        )  # Positional encoding applies to viewdirs

        # Enable view directions
        self.use_viewdirs = conf.get_bool("use_viewdirs", False)

        self.use_global_encoder = conf.get_bool(
            "use_global_encoder", False
        )  # Global image features?

        d_latent = self.encoder.latent_size if self.use_encoder else 0
        d_in = 3 if self.use_xyz else 1

        if self.use_viewdirs and self.use_code_viewdirs:
            # Apply positional encoding to viewdirs
            d_in += 3
        if self.use_code and d_in > 0:
            # Positional encoding for x,y,z OR view z
            self.code = PositionalEncoding.from_conf(conf["code"], d_in=d_in)
            d_in = self.code.d_out
        if self.use_viewdirs and not self.use_code_viewdirs:
            # Don't apply positional encoding to viewdirs (concat after encoded)
            d_in += 3

        if self.use_global_encoder:
            # Global image feature
            self.global_encoder = ImageEncoder.from_conf(conf["global_encoder"])
            self.global_latent_size = self.global_encoder.latent_size
            d_latent += self.global_latent_size

        d_out = 4

        self.latent_size = self.encoder.latent_size
        self.mlp_coarse = make_mlp(conf["mlp_coarse"], d_in, d_latent, d_out=d_out)
        self.mlp_fine = make_mlp(
            conf["mlp_fine"], d_in, d_latent, d_out=d_out, allow_empty=True
        )
        self.mlp_far = make_mlp(
            conf["mlp_far"], d_in, d_latent, d_out=d_out, allow_empty=True
        )
        # Note: this is now world -> camera, and bottom row is omitted
        self.register_buffer("poses", torch.empty(1, 3, 4), persistent=False)
        self.register_buffer("image_shape", torch.empty(2), persistent=False)

        self.d_in = d_in
        self.d_out = d_out
        self.d_latent = d_latent
        self.register_buffer("focal", torch.empty(1, 2), persistent=False)
        # Principal point
        self.register_buffer("c", torch.empty(1, 2), persistent=False)
        self.z_bounds = (None, None)

        self.num_objs = 0
        self.num_views_per_obj = 1

    def encode(self, images, poses, focal, z_bounds=(2.5, 6.5), c=None):
        """
        :param images (B, 3, H, W)
        :param poses (B, 4, 4)
        :param focal focal length (B) or (B, 2) [fx, fy]
        :param z_bounds tuple (z_near, z_far) camera distance to center, float.
        Only actually needed if voxel encoder is used.
        :param c principal point None or (B, 2) [cx, cy],
        default is center of image
        """
        self.num_objs = images.size(0)
        if len(images.shape) == 5:
            assert len(poses.shape) == 4
            assert poses.size(1) == images.size(1)  # Be consistent with NS
            self.num_views_per_obj = images.size(1)
            images = images.reshape(-1, *images.shape[2:])
            poses = poses.reshape(-1, 4, 4)
        else:
            self.num_views_per_obj = 1

        self.encoder(images)
        rot = poses[:, :3, :3].transpose(1, 2)  # (B, 3, 3)
        trans = -torch.bmm(rot, poses[:, :3, 3:])  # (B, 3, 1)
        self.poses = torch.cat((rot, trans), dim=-1)  # (B, 3, 4)

        self.z_bounds = z_bounds
        self.image_shape[0] = images.shape[-1]
        self.image_shape[1] = images.shape[-2]

        if len(focal.shape) == 0:
            focal = focal[None, None].repeat((1, 2))
        elif len(focal.shape) == 1:
            focal = focal.unsqueeze(-1).repeat((1, 2))
        focal[..., 1] *= -1.0
        self.focal = focal.float()

        if c is None:
            c = (self.image_shape * 0.5).unsqueeze(0)
        elif len(c.shape) == 0:
            c = c[None, None].repeat((1, 2))
        elif len(c.shape) == 1:
            c = c.unsqueeze(-1).repeat((1, 2))
        self.c = c

        if self.use_global_encoder:
            self.global_encoder(images)

    def forward(self, xyz, coarse=True, viewdirs=None, far=False):
        """
        Predict (r, g, b, sigma) at canonical space points xyz.
        Please call encode first!
        :param xyz (SB, B, 3) or (B, 3);
        SB is batch of objects
        B is batch of points (in rays)
        NS is number of input views
        :return (SB, B, 4) r g b sigma
        """
        #  with profiler.profile(record_shapes=True) as prof:
        with profiler.record_function("model_inference"):
            single = len(xyz.shape) == 2
            if single:
                # Single batch (either object or view), i.e. SB = 1
                xyz = xyz[None]
                if self.num_objs > 1:
                    xyz = xyz.expand(self.num_objs, -1, -1)
            SB, B, _ = xyz.shape
            NS = self.num_views_per_obj

            # Transform query points into the camera spaces of the input views
            xyz = repeat_interleave(xyz, NS, dim=0)  # (SB*NS, B, 3)
            xyz_rot = torch.matmul(self.poses[:, None, :3, :3], xyz.unsqueeze(-1))[..., 0]
            xyz = xyz_rot + self.poses[:, None, :3, 3]

            if self.d_in > 0:
                if self.use_xyz:
                    if self.normalize_z:
                        z_feature = xyz_rot.reshape(-1, 3)  # (SB*B, 3)
                    else:
                        z_feature = xyz.reshape(-1, 3)  # (SB*B, 3)
                else:
                    if self.normalize_z:
                        z_feature = -xyz_rot[..., 2].reshape(-1, 1)  # (SB*B, 1)
                    else:
                        z_feature = -xyz[..., 2].reshape(-1, 1)  # (SB*B, 1)

                if self.use_code and not self.use_code_viewdirs:
                    # Positional encoding (no viewdirs)
                    z_feature = self.code(z_feature)

                if self.use_viewdirs:
                    assert viewdirs is not None
                    # Viewdirs to input view space
                    viewdirs = viewdirs.reshape(SB, B, 3, 1)
                    viewdirs = repeat_interleave(
                        viewdirs, NS, dim=0
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = torch.matmul(
                        self.poses[:, None, :3, :3], viewdirs
                    )  # (SB*NS, B, 3, 1)
                    viewdirs = viewdirs.reshape(-1, 3)  # (SB*B, 3)
                    z_feature = torch.cat((z_feature, viewdirs), dim=1)  # (SB*B, 4 or 6)

                if self.use_code and self.use_code_viewdirs:
                    # Positional encoding (with viewdirs)
                    z_feature = self.code(z_feature)

                mlp_input = z_feature

            if self.use_encoder:
                # Grab encoder's latent code.
                uv = -xyz[:, :, :2] / xyz[:, :, 2:]  # (SB, B, 2)
                uv *= repeat_interleave(
                    self.focal.unsqueeze(1), NS if self.focal.shape[0] > 1 else 1, dim=0
                )
                uv += repeat_interleave(
                    self.c.unsqueeze(1), NS if self.c.shape[0] > 1 else 1, dim=0
                )  # (SB*NS, B, 2)
                latent = self.encoder.index(
                    uv, None, self.image_shape
                )  # (SB * NS, latent, B)

                if self.stop_encoder_grad:
                    latent = latent.detach()
                latent = latent.transpose(1, 2).reshape(
                    -1, self.latent_size
                )  # (SB * NS * B, latent)

                if self.d_in == 0:
                    # z_feature not needed
                    mlp_input = latent
                else:
                    mlp_input = torch.cat((latent, z_feature), dim=-1)

            if self.use_global_encoder:
                # Concat global latent code if enabled
                global_latent = self.global_encoder.latent
                assert mlp_input.shape[0] % global_latent.shape[0] == 0
                num_repeats = mlp_input.shape[0] // global_latent.shape[0]
                global_latent = repeat_interleave(global_latent, num_repeats, 0)
                mlp_input = torch.cat((global_latent, mlp_input), dim=-1)

            combine_index = None
            dim_size = None
            if coarse or self.mlp_fine is None:
                mlp_output = self.mlp_coarse(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )
            else:
                mlp_output = self.mlp_fine(
                    mlp_input,
                    combine_inner_dims=(self.num_views_per_obj, B),
                    combine_index=combine_index,
                    dim_size=dim_size,
                )

            mlp_output = mlp_output.reshape(-1, B, self.d_out)

            rgb = mlp_output[..., :3]
            sigma = mlp_output[..., 3:4]

            output_list = [torch.sigmoid(rgb), torch.relu(sigma)]
            output = torch.cat(output_list, dim=-1)
            output = output.reshape(SB, B, -1)

            if single:
                output = output[0]

        return output

    def load_weights(self, args, opt_init=False, strict=True, device=None):
        """
        Helper for loading weights according to argparse arguments
        """
        if opt_init and not args.resume:
            return
        ckpt_name = (
            "pifu_nerf_init_latest"
            if opt_init or not args.resume
            else "pifu_nerf_latest"
        )
        model_path = "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name)

        if device is None:
            device = self.poses.device

        if os.path.exists(model_path):
            print("Load", model_path)
            self.load_state_dict(
                torch.load(model_path, map_location=device), strict=strict
            )
        elif not opt_init:
            print("WARNING:", model_path, "does not exist, not loaded")

    def save_weights(self, args, opt_init=False):
        """
        Helper for saving weights according to argparse arguments
        """
        ckpt_name = "pifu_nerf_init_latest" if opt_init else "pifu_nerf_latest"
        torch.save(
            self.state_dict(),
            "%s/%s/%s" % (args.checkpoints_path, args.name, ckpt_name),
        )
