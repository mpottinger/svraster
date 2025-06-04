# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import time
import numpy as np
from tqdm import tqdm
from os import makedirs
import imageio

import torch

from src.config import cfg, update_argparser, update_config

from src.dataloader.data_pack import DataPack
from src.sparse_voxel_model import SparseVoxelModel
from src.cameras import MiniCam
from src.utils.image_utils import im_tensor2np, viz_tensordepth
from src.utils.camera_utils import interpolate_poses


if __name__ == "__main__":
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(
        description="Sparse voxels raster rendering.")
    parser.add_argument('model_path')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--n_frames", default=300, type=int)
    parser.add_argument("--save_scale", default=1.0, type=float)

    # Manually select which frames to interpolate
    parser.add_argument("--ids", default=[], type=int, nargs='*')

    # Use farthest point sampling to select key frame
    parser.add_argument("--starting_id", default=0, type=int)

    # Other tweaking
    parser.add_argument("--step_forward", default=0, type=float)

    args = parser.parse_args()
    print("Rendering " + args.model_path)

    # Load config
    update_config(os.path.join(args.model_path, 'config.yaml'))

    # Load data
    data_pack = DataPack(
        source_path=cfg.data.source_path,
        image_dir_name=cfg.data.image_dir_name,
        res_downscale=cfg.data.res_downscale,
        res_width=cfg.data.res_width,
        skip_blend_alpha=cfg.data.skip_blend_alpha,
        alpha_is_white=cfg.model.white_background,
        data_device=cfg.data.data_device,
        use_test=cfg.data.eval,
        test_every=cfg.data.test_every,
        camera_params_only=True,
    )

    # Interpolate poses
    cams = data_pack.get_train_cameras()
    if len(args.ids):
        key_poses = [cams[i].c2w.cpu().numpy() for i in args.ids]
    else:
        cam_pos = torch.stack([cam.position for cam in cams])
        ids = [args.starting_id]
        for _ in range(3):
            farthest_id = torch.cdist(cam_pos[ids], cam_pos).amin(0).argmax().item()
            ids.append(farthest_id)
        ids[1], ids[2] = ids[2], ids[1]
        key_poses = [cams[i].c2w.cpu().numpy() for i in ids]

    if args.step_forward != 0:
        for i in range(len(key_poses)):
            lookat = key_poses[i][:3, 2]
            key_poses[i][:3, 3] += args.step_forward * lookat

    interp_poses = interpolate_poses(key_poses, n_frame=args.n_frames, periodic=True)

    # Load model
    voxel_model = SparseVoxelModel(
        n_samp_per_vox=cfg.model.n_samp_per_vox,
        sh_degree=cfg.model.sh_degree,
        ss=cfg.model.ss,
        white_background=cfg.model.white_background,
        black_background=cfg.model.black_background,
    )
    loaded_iter = voxel_model.load_iteration(args.model_path, args.iteration)
    voxel_model.freeze_vox_geo()

    # Rendering
    fovx = cams[0].fovx
    fovy = cams[0].fovy
    width = cams[0].image_width
    height = cams[0].image_height

    video = []
    for pose in tqdm(interp_poses, desc="Rendering progress"):

        cam = MiniCam(
            c2w=pose,
            fovx=fovx, fovy=fovy,
            width=width, height=height)

        with torch.no_grad():
            render_pkg = voxel_model.render(cam)
            rendering = render_pkg['color']

        if args.save_scale != 0:
            rendering = torch.nn.functional.interpolate(
                rendering[None],
                scale_factor=args.save_scale,
                mode="bilinear",
                antialias=True)[0]

        video.append(im_tensor2np(rendering))

    outpath = os.path.join(args.model_path, "render_fly_through.mp4")
    imageio.mimwrite(outpath, video, fps=30)
    print("Save to", outpath)
