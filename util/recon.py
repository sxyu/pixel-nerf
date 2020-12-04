"""
Mesh reconstruction tools
"""
import mcubes
import torch
import numpy as np
import util
import tqdm
import warnings


def marching_cubes(
    occu_net,
    c1=[-1, -1, -1],
    c2=[1, 1, 1],
    reso=[128, 128, 128],
    isosurface=50.0,
    sigma_idx=3,
    eval_batch_size=100000,
    coarse=True,
    device=None,
):
    """
    Run marching cubes on network. Uses PyMCubes.
    WARNING: does not make much sense with viewdirs in current form, since
    sigma depends on viewdirs.
    :param occu_net main NeRF type network
    :param c1 corner 1 of marching cube bounds x,y,z
    :param c2 corner 2 of marching cube bounds x,y,z (all > c1)
    :param reso resolutions of marching cubes x,y,z
    :param isosurface sigma-isosurface of marching cubes
    :param sigma_idx index of 'sigma' value in last dimension of occu_net's output
    :param eval_batch_size batch size for evaluation
    :param coarse whether to use coarse NeRF for evaluation
    :param device optionally, device to put points for evaluation.
    By default uses device of occu_net's first parameter.
    """
    if occu_net.use_viewdirs:
        warnings.warn(
            "Running marching cubes with fake view dirs (pointing to origin), output may be invalid"
        )
    with torch.no_grad():
        grid = util.gen_grid(*zip(c1, c2, reso), ij_indexing=True)
        is_train = occu_net.training

        print("Evaluating sigma @", grid.size(0), "points")
        occu_net.eval()

        all_sigmas = []
        if device is None:
            device = next(occu_net.parameters()).device
        grid_spl = torch.split(grid, eval_batch_size, dim=0)
        if occu_net.use_viewdirs:
            fake_viewdirs = -grid / torch.norm(grid, dim=-1).unsqueeze(-1)
            vd_spl = torch.split(fake_viewdirs, eval_batch_size, dim=0)
            for pnts, vd in tqdm.tqdm(zip(grid_spl, vd_spl), total=len(grid_spl)):
                outputs = occu_net(
                    pnts.to(device=device), coarse=coarse, viewdirs=vd.to(device=device)
                )
                sigmas = outputs[..., sigma_idx]
                all_sigmas.append(sigmas.cpu())
        else:
            for pnts in tqdm.tqdm(grid_spl):
                outputs = occu_net(pnts.to(device=device), coarse=coarse)
                sigmas = outputs[..., sigma_idx]
                all_sigmas.append(sigmas.cpu())
        sigmas = torch.cat(all_sigmas, dim=0)
        sigmas = sigmas.view(*reso).cpu().numpy()

        print("Running marching cubes")
        vertices, triangles = mcubes.marching_cubes(sigmas, isosurface)
        # Scale
        c1, c2 = np.array(c1), np.array(c2)
        vertices *= (c2 - c1) / np.array(reso)

    if is_train:
        occu_net.train()
    return vertices + c1, triangles


def save_obj(vertices, triangles, path, vert_rgb=None):
    """
    Save OBJ file, optionally with vertex colors.
    This version is faster than PyMCubes and supports color.
    Taken from PIFu.
    :param vertices (N, 3)
    :param triangles (N, 3)
    :param vert_rgb (N, 3) rgb
    """
    file = open(path, "w")
    if vert_rgb is None:
        # No color
        for v in vertices:
            file.write("v %.4f %.4f %.4f\n" % (v[0], v[1], v[2]))
    else:
        # Color
        for idx, v in enumerate(vertices):
            c = vert_rgb[idx]
            file.write(
                "v %.4f %.4f %.4f %.4f %.4f %.4f\n"
                % (v[0], v[1], v[2], c[0], c[1], c[2])
            )
    for f in triangles:
        f_plus = f + 1
        file.write("f %d %d %d\n" % (f_plus[0], f_plus[1], f_plus[2]))
    file.close()
