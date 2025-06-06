# Model functions

We cover the functions to create a model instance in [model_creation_tutor.md](./model_creation_tutor.md). We describe the other functions in the following.

## Properties
- `n_samp_per_vox`: Number of samples per voxel when rendering.
- `ss`: Super-sampling scale.
    - We render higher-resolution image (`[H * ss, W * ss]`) and apply anti-aliasing downsampling.
- `white_background`: Indicate if the background is white.
- `black_background`: Indicate if the background is black.
    - The background will be the averaged color if neither black or white is set.
- `max_sh_degree`: Maximum SH degree. Support `0~3` degrees.
    - This number should be fixed after voxels parameters are allocated.

## Derived properties
- `num_voxels`: Number of voxels.
- `num_grid_pts`: Number of grid points.
    - Recap that a voxel has 8 corner grid points. A grid point can be shared by adjacent voxels. This is the number of unique grid points.
- `scene_min`: Minimum coordinate of entire scene.
- `scene_max`: Maximum coordinate of entire scene.
- `inside_min`: Minimum coordinate of the main foreground region.
    - It's valid when the model is created from `model_init` with `outside_level > 0` which preserves Octree level outside the main foreground bounding box.
- `inside_max`: Maximum coordinate of the main foreground region.
- `inside_mask`: A mask indicating if a voxel is in `inside_min` and `inside_max`.
- `subdivision_priority`: The model automatically tracks and accumulates subdivision priority during rendering backward pass.
    - Larger value means higher priority.
    - Reset by `reset_subdivision_priority()`.

The following is the properties that lazily computed at the first time you access them. It automatically recompute when it detect the voxels allocation is updated (e.g., after pruning or subdivision).
- `vox_center`: Voxel center position in the world space.
- `vox_size`: Voxel size.
- `vox_key`: Index to the unique grid points. It's in shape of `[num_voxels, 8]`.
- `grid_pts_xyz`: The world-space position of the unique grid points.

## Parameters
- `_sh0`: Base color as zero-degree SH component. The shape is `[num_voxels, 3]`.
- `_shs`: Higher-degree SH component for view-dependent color. The shape is `[num_voxels, (max_sh_degree+1)**2 - 1, 3]`.
    - It's the dominant factor of the number of total parameters.
- `_geo_grid_pts`: The density of grid points. The shape is `[num_grid_pts, 1]`.
    - When rendering, it's gathered into `[num_voxels, 8]` as voxel trilinear density field for the CUDA to render.

## Core functions
- `render_pkg = render(camera, track_max_w=False, output_depth=False, output_normal=False, output_T=False)`
    - Rendering a view.
    - `track_max_w` whether to track the maximum blending weigth of each voxel. Access by `render_pkg['max_w']`.
    - `output_depth` whether to render depth. Access by `render_pkg['depth']` or `render_pkg['raw_depth']`.
    - `output_normal` whether to render normal. Access by `render_pkg['normal']` or `render_pkg['raw_normal']`.
    - `output_T` whether to output transmittance. Access by `render_pkg['T']` or `render_pkg['raw_T']`.
    - The outputs with `raw_` prefix are the results without anti-aliasing downsampling.
    - The depth and normal is not normalized by alpha.
    - There output depth is in shape `[3, H, W]` for mean depth, distortion cache, median depth. Only the mean depth support backpropagation.
- `pruning(mask)`
    - Remove voxels indicating by the given mask.
- `subdividing(mask)`
    - Subdivde voxels into their eight octans indicating by the given mask. The source parent voxels are removed after subdivision.

## Useful functions
- `compute_training_stat(camera_lst)`
    - Compute the per-voxel statistic from the given cameras, including `max_w` for maximum blending weight, `min_samp_interval` for the inverse of maximum sampling rates, and `view_cnt` for visibile camera count.
- `reset_sh_from_cameras(camera_lst)`
    - Reset shs to zero.
    - Reset sh0 to yield the colors averaged from the given images.
- `apply_tv_on_density_field(lambda_tv_density)`
    - Add the gradient of total variation loss to the `_geo_grid_pts` parameter.
- `save(path, quantize=False)`
    - Save the model to the given path. You can optionally apply 8-bit quantization to the parameters which save 70% disk space with minor quality difference.
- `load(path)`
    - Load checkpoint from the given path.
- `load_iteration(model_path, iteration=-1)`
    - Load checkpoint from a model output path with the given iteration. The default load the latest iteration.