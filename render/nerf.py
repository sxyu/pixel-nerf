"""
NeRF differentiable renderer.
References:
https://github.com/bmild/nerf
https://github.com/kwea123/nerf_pl
"""
import torch
import torch.nn.functional as F
import util
import torch.autograd.profiler as profiler
from dotmap import DotMap


class NeRFRenderer(torch.nn.Module):
    def __init__(
        self,
        n_coarse=128,
        n_fine=0,
        n_fine_depth=0,
        n_far=0,
        noise_std=0.0,
        depth_std=0.01,
        eval_batch_size=100000,
        white_bkgd=False,
        lindisp=False,
        sched=None,  # ray sampling schedule for coarse and fine rays
        fused_coarse_fine=False,
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_far = n_far
        self.n_fine = n_fine
        self.n_fine_depth = n_fine_depth

        self.noise_std = noise_std
        self.depth_std = depth_std

        self.eval_batch_size = eval_batch_size
        self.white_bkgd = white_bkgd
        self.lindisp = lindisp
        if lindisp:
            print("Using linear displacement rays")
        self.using_fine = n_fine > 0
        self.using_bg = n_far > 0
        self.sched = sched
        if sched is not None and len(sched) == 0:
            self.sched = None
        self.using_fused_coarse_fine = fused_coarse_fine
        self.register_buffer(
            "iter_idx", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "last_sched", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def sample_coarse(self, rays):
        """
        Stratified sampling. Note this is different from original NeRF slightly.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :return (B, Kc)
        """
        device = rays.device
        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)

        step = 1.0 / self.n_coarse
        B = rays.shape[0]
        z_steps = torch.linspace(0, 1 - step, self.n_coarse, device=device)  # (Kc)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)  # (B, Kc)
        z_steps += torch.rand_like(z_steps) * step
        if not self.lindisp:  # Use linear sampling in depth space
            return near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            return 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)

        # Use linear sampling in depth space
        return near * (1 - z_steps) + far * z_steps  # (B, Kc)

    def sample_fine(self, rays, weights):
        """
        Weighted stratified (importance) sample
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param weights (B, Kc)
        :return (B, Kf-Kfd)
        """
        device = rays.device
        B = rays.shape[0]

        weights = weights.detach() + 1e-5  # Prevent division by zero
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (B, Kc)
        cdf = torch.cumsum(pdf, -1)  # (B, Kc)
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (B, Kc+1)

        u = torch.rand(
            B, self.n_fine - self.n_fine_depth, dtype=torch.float32, device=device
        )  # (B, Kf)
        inds = torch.searchsorted(cdf, u, right=True).float() - 1.0  # (B, Kf)
        inds = torch.clamp_min(inds, 0.0)

        z_steps = (inds + torch.rand_like(inds)) / self.n_coarse  # (B, Kf)

        near, far = rays[:, -2:-1], rays[:, -1:]  # (B, 1)
        if not self.lindisp:  # Use linear sampling in depth space
            z_samp = near * (1 - z_steps) + far * z_steps  # (B, Kf)
        else:  # Use linear sampling in disparity space
            z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps)  # (B, Kf)
        return z_samp

    def sample_fine_depth(self, rays, depth):
        """
        Sample around specified depth
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param depth (B)
        :return (B, Kfd)
        """
        z_samp = depth.unsqueeze(1).repeat((1, self.n_fine_depth))
        z_samp += torch.randn_like(z_samp) * self.depth_std
        # Clamp does not support tensor bounds
        z_samp = torch.max(torch.min(z_samp, rays[:, -1:]), rays[:, -2:-1])
        return z_samp

    def composite(
        self,
        model,
        rays,
        z_samp,
        coarse=True,
        far=False,
        sb=0,
        get_fused_output=False,
        fused_mask=None,
        fused_output=None,
    ):
        """
        Render RGB and depth for each ray using NeRF alpha-compositing formula,
        given sampled positions along each ray (see sample_*)
        :param model should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        should also support 'coarse' boolean argument
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        :param z_samp z positions sampled for each ray (B, K)
        :param coarse whether to evaluate using coarse NeRF
        :param sb super-batch dimension; 0 = disable
        :return weights (B, K), rgb (B, 3), depth (B)
        """
        with profiler.record_function("renderer_composite"):
            B, K = z_samp.shape

            deltas = z_samp[:, 1:] - z_samp[:, :-1]  # (B, K-1)
            if far:
                delta_inf = 1e10 * torch.ones_like(deltas[:, :1])  # infty (B, 1)
            else:
                delta_inf = rays[:, -1:] - z_samp[:, -1:]
            deltas = torch.cat([deltas, delta_inf], -1)  # (B, K)

            # (B, K, 3)
            points = rays[:, None, :3] + z_samp.unsqueeze(2) * rays[:, None, 3:6]
            points = points.reshape(-1, 3)  # (B*K, 3)

            if fused_mask is not None:
                points = points[~fused_mask]

            use_viewdirs = hasattr(model, "use_viewdirs") and model.use_viewdirs

            val_all = []
            if sb > 0:
                points = points.reshape(
                    sb, -1, 3
                )  # (SB, B'*K, 3) B' is real ray batch size
                eval_batch_size = (self.eval_batch_size - 1) // sb + 1
                eval_batch_dim = 1
            else:
                eval_batch_size = self.eval_batch_size
                eval_batch_dim = 0

            split_points = torch.split(points, eval_batch_size, dim=eval_batch_dim)
            if use_viewdirs:
                dim1 = self.n_fine if fused_mask is not None else K
                viewdirs = rays[:, None, 3:6].expand(-1, dim1, -1)  # (B, K, 3)
                if sb > 0:
                    viewdirs = viewdirs.reshape(sb, -1, 3)  # (SB, B'*K, 3)
                else:
                    viewdirs = viewdirs.reshape(-1, 3)  # (B*K, 3)
                split_viewdirs = torch.split(viewdirs, eval_batch_size, dim=eval_batch_dim)
                for pnts, dirs in zip(split_points, split_viewdirs):
                    val_all.append(model(pnts, coarse=coarse, viewdirs=dirs))
            else:
                for pnts in split_points:
                    val_all.append(model(pnts, coarse=coarse))
            points = None
            viewdirs = None
            # (B*K, 4) OR (SB, B'*K, 4)
            out = torch.cat(val_all, dim=eval_batch_dim)
            if fused_mask is not None:
                n_outputs = model.d_out
                tmp = torch.empty(B * K, n_outputs, device=out.device)
                tmp[fused_mask] = fused_output
                tmp[~fused_mask] = out.reshape(-1, n_outputs)
                out = tmp
            out = out.reshape(B, K, -1)  # (B, K, 4 or 5)

            rgbs = out[..., :3]  # (B, K, 3)
            sigmas = out[..., 3]  # (B, K)
            if get_fused_output:
                outputs_for_fine = [rgbs, out[..., 3:4]]
                fused_output = torch.cat(outputs_for_fine, dim=-1)
            else:
                fused_output = None
            if self.training and self.noise_std > 0.0:
                sigmas = sigmas + torch.randn_like(sigmas) * self.noise_std

            # compute the gradients in log space of the alphas, for NV TV occupancy regularizer
            alphas = 1 - torch.exp(-deltas * torch.relu(sigmas))  # (B, K)
            deltas = None
            sigmas = None
            alphas_shifted = torch.cat(
                [torch.ones_like(alphas[:, :1]), 1 - alphas + 1e-10], -1
            )  # (B, K+1) = [1, a1, a2, ...]
            T = torch.cumprod(alphas_shifted, -1)  # (B)
            weights = alphas * T[:, :-1]  # (B, K)
            alphas = None
            alphas_shifted = None

            rgb_final = torch.sum(weights.unsqueeze(-1) * rgbs, -2)  # (B, 3)
            depth_final = torch.sum(weights * z_samp, -1)  # (B)
            if self.white_bkgd:
                # White background
                pix_alpha = weights.sum(dim=1)  # (B), pixel alpha
                rgb_final = rgb_final + 1 - pix_alpha.unsqueeze(-1)  # (B, 3)
            return (
                weights,
                rgb_final,
                depth_final,
                fused_output,
            )

    def forward(
        self,
        model,
        rays,
        want_weights=False,
    ):
        """
        :model nerf model, should return (B, (r, g, b, sigma)) when called with (B, (x, y, z))
        for single-object point batch;
        or (SB, B, (r, g, b, sigma)) when called with (SB, B, (x, y, z)), for multi-object
        NeRF super-batch * per object point batch.
        Should also support 'coarse' boolean argument for coarse NeRF.
        :param rays ray [origins (3), directions (3), near (1), far (1)] (B, 8)
        for single-object OR (SB, B, 8) for super-batch
        :param want_weights if true, returns compositing weights (B, K)
        :return rgb_fine, depth_fine, rgb_coarse, depth_coarse [, weights_fine, weights_coarse]
        """
        with profiler.record_function("renderer_forward"):
            if self.sched is not None and self.last_sched.item() > 0:
                self.n_coarse = self.sched[1][self.last_sched.item() - 1]
                self.n_fine = self.sched[2][self.last_sched.item() - 1]

            use_superbatch = len(rays.shape) == 3
            if use_superbatch:
                superbatch_size = rays.shape[0]
                rays = rays.reshape(-1, 8)
            else:
                superbatch_size = 0

            z_coarse = self.sample_coarse(rays)  # (B, Kc)
            coarse_composite = self.composite(
                model,
                rays,
                z_coarse,
                coarse=None if self.using_fused_coarse_fine else True,
                far=not self.using_bg,
                sb=superbatch_size,
                get_fused_output=self.using_fused_coarse_fine,
            )
            # only output uncertainties for fine model
            outputs = DotMap(
                coarse=self._format_outputs(
                    coarse_composite,
                    superbatch_size,
                    want_weights=want_weights,
                ),
            )

            if self.using_fine:
                all_samps = [z_coarse]
                if self.n_fine - self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine(rays, coarse_composite[0].detach())
                    )  # (B, Kf - Kfd)
                if self.n_fine_depth > 0:
                    all_samps.append(
                        self.sample_fine_depth(rays, coarse_composite[2])
                    )  # (B, Kfd)
                z_combine = torch.cat(all_samps, dim=-1)  # (B, Kc + Kf)
                z_combine_sorted, argsort = torch.sort(z_combine, dim=-1)
                if self.using_fused_coarse_fine:
                    fused_mask = (argsort < self.n_coarse).reshape(-1)  # (B*K)
                    fused_output = coarse_composite[4].reshape(-1, 4)  # (B*Kc, 4)
                else:
                    fused_mask = fused_output = None
                fine_composite = self.composite(
                    model,
                    rays,
                    z_combine_sorted,
                    coarse=False,
                    far=not self.using_bg,
                    sb=superbatch_size,
                    fused_mask=fused_mask,
                    fused_output=fused_output,
                )
                outputs.fine = self._format_outputs(
                    fine_composite,
                    superbatch_size,
                    want_weights=want_weights,
                )

            return outputs

    def _format_outputs(
        self,
        rendered_outputs,
        superbatch_size,
        want_weights=False,
    ):
        weights, rgb, depth, fused = rendered_outputs
        if superbatch_size > 0:
            rgb = rgb.reshape(superbatch_size, -1, 3)
            depth = depth.reshape(superbatch_size, -1)
            weights = weights.reshape(superbatch_size, -1, weights.shape[-1])
        ret_dict = DotMap(rgb=rgb, depth=depth)
        if want_weights:
            ret_dict.weights = weights
        return ret_dict

    def sched_step(self, steps=1):
        if self.sched is None:
            return
        self.iter_idx += steps
        while (
            self.last_sched.item() < len(self.sched[0])
            and self.iter_idx.item() >= self.sched[0][self.last_sched.item()]
        ):
            self.n_coarse = self.sched[1][self.last_sched.item()]
            self.n_fine = self.sched[2][self.last_sched.item()]
            print(
                "INFO: NeRF sampling resolution changed on schedule ==> c",
                self.n_coarse,
                "f",
                self.n_fine,
            )
            self.last_sched += 1

    @classmethod
    def from_conf(cls, conf, white_bkgd=False, lindisp=False):
        return cls(
            conf.get_int("n_coarse", 128),
            conf.get_int("n_fine", 0),
            n_fine_depth=conf.get_int("n_fine_depth", 0),
            n_far=conf.get_int("n_far", 0),
            noise_std=conf.get_float("noise_std", 0.0),
            depth_std=conf.get_float("depth_std", 0.01),
            white_bkgd=white_bkgd,
            lindisp=lindisp,
            eval_batch_size=conf.get_int("eval_batch_size", 200000),
            sched=conf.get_list("sched", None),
            fused_coarse_fine=conf.get_bool("fused_coarse_fine", False),
        )
