# Tutorial on model creation

We cover several ways to create a models, including:
1. `model_init` to create dense grid or apply the heuristic with training camera poses for unbounded scenes.
2. `ijkl_init` to create from voxel index coordinate. It's differentiable.
3. `points_init` to create from points. It's differentiable.


## Prepare
To depict the results in later sections, we use the following pipeline to render video circularing around the world origin.
```python
from src.sparse_voxel_model import SparseVoxelModel

from src.utils.camera_utils import gen_circular_poses
from src.cameras import MiniCam

from src.utils.image_utils import im_tensor2np

# Generate camera trajectory circular around the origin.
# The camera coordinate system is y-down z-forward following COLMAP convention.
# The camera path is (0, 0, -z) => (+x, 0, 0) => (0, 0, +z) => (-x, 0, 0) => (0, 0, -z).
traj_radius = 1
n_frame = 100
fov = np.deg2rad(60)
res = 512
cam_traj = [
    MiniCam(c2w=c2w, fovx=fov, fovy=fov, width=res, height=res)
    for c2w in gen_circular_poses(radius=traj_radius, n_frame=n_frame)]

# Determine scene bound for the following testing
center = torch.tensor([0, 0, 0], dtype=torch.float32)
extent = 0.6 * traj_radius
bounding = torch.stack([center - 0.5 * extent, center + 0.5 * extent])

# Create model
voxel_model = SparseVoxelModel(sh_degree=0, black_background=True)
# ... described in the following sections

# Render video
video = []
for cam in cam_traj:
    with torch.no_grad():
        render_pkg = voxel_model.render(cam)
    video.append(im_tensor2np(render_pkg["color"]))

imageio.mimwrite("dev.mp4", video, fps=30, quality=8)
```


## Create a dense grid
To initialize a dense grid, we use `model_init`:
```python
# Create a dense cube
voxel_model = SparseVoxelModel(sh_degree=0, black_background=True)
voxel_model.model_init(
    bounding=bounding,
    outside_level=0, # Number of Octree levels outside the bounding (for background).
    init_n_level=3,  # Starting from (2^init_n_level)^3 voxels.
)

# Fill voxel with high density and random color for visualization
from src.utils.activation_utils import rgb2shzero
voxel_model._geo_grid_pts.data.fill_(100)
voxel_model._sh0.data.copy_(rgb2shzero(torch.rand(voxel_model.num_voxels, 3)))
```

https://github.com/user-attachments/assets/4e34d90c-76bd-4231-9ce1-c0655d6a1662


In [train.py](../train.py), we also use `model_init` to apply the heuristic mentioned in the paper to initialize voxels for unbounded scenes from training camera poses.


## Create from voxels
We can also create model from integer indices of non-empty voxels. For example, we use the following script to generate an ascii art in 3D:
```python
# Generate ascii art
import art
ascii_art = art.text2art("NVIDIA", font="xcourb").strip().split('\n')

# Compute 3D coordinates of non-empty text under a 64^3 cubes
x_max, y_max = 0, 0
for y, s in enumerate(ascii_art):
    for x, c in enumerate(s):
        if c != ' ':
            x_max = max(x_max, x)
            y_max = max(y_max, y)

pad_x = (64 - x_max) // 2
pad_y = (64 - y_max) // 2

ijk = []
rgb = []
for y, s in enumerate(ascii_art):
    for x, c in enumerate(s):
        if c != ' ':
            ijk.append((pad_x + x, pad_y + y, 31))
            ijk.append((pad_x + x, pad_y + y, 32))
            rgb.append((0.462, 0.725, 0))
            rgb.append(np.random.rand(3))
```

We then can create a model using voxel ijk indices:
```python
# Create voxel model by voxel index
voxel_model = SparseVoxelModel(sh_degree=0, black_background=True)
voxel_model.ijkl_init(
    scene_center=center,
    scene_extent=extent,

    # The coodinate of the non-empty voxel
    ijk=torch.tensor(ijk, dtype=torch.int64, device="cuda"),

    # The level of each voxel. It can be a tensor of int8 so voxels are in different levels.
    octlevel=6,

    # Voxel density field. It can also be a tensor that requires gradient.
    # The tensor can be [#grid_pts x 1] or [#voxels x 8] for voxel trilinear density field.
    # In case of [#voxel x 8], you can also specify `reduce_density=True` so it merges
    # shared points by adjacent voxels into [#grid_pts x 1] internally.
    # The given density is pre-activation. It is mapped to non-negative float when rendering.
    density=0,

    # Voxel color. It can also be a tensor that requires gradient.
    # The intensity scale is 0~1. Internally, it converts to zero-deg shperical harmonic.
    rgb=torch.tensor(rgb, dtype=torch.float32, device="cuda"),

    # Voxel higher-deg spherical harmonic. It can also be a tensor that requires gradient.
    # If sh_degree=0, it will be a empty tensor.
    shs=0,
)
```

https://github.com/user-attachments/assets/3fabe2ce-1abe-4e64-9cda-3754a2979272


## Create from points
Creating model from point cloud can be useful. We sample points from a cow mesh for an example:
```python
import trimesh
mesh = trimesh.exchange.load.load_remote('https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj').geometry['cow.obj']
pt_xyz, pt_fid, pt_rgba = trimesh.sample.sample_surface(mesh, count=1_000_000, sample_color=True)

pt_xyz = np.array(pt_xyz)
pt_rgb = np.array(pt_rgba)[:, :3] / 255

pt_xyz = pt_xyz - pt_xyz.mean(0)
pt_xyz = pt_xyz * (0.9 * 0.5 * extent / np.abs(pt_xyz).max(0).max())
pt_xyz[:, 1] *= -1  # flip +y up to +y down
```

You can directly specify the voxel levels or expected voxel sizes, which will be map to a voxel level internally.
Duplicated voxels will be merged and their parameters are averaged.
```python
# Create voxel model by points
voxel_model = SparseVoxelModel(sh_degree=0, black_background=True)
voxel_model.points_init(
    scene_center=center,
    scene_extent=extent,

    # The world-space coodinate of the non-empty voxels.
    # This tensor receive no gradient from rendering losses.
    # If you want point position gradient, you need to transform then into the later density.
    xyz=torch.tensor(pt_xyz, dtype=torch.float32, device="cuda"),

    # The level of each voxel. You can also give a targeted voxel size instead.
    octlevel=None,
    expected_vox_size=extent / 100,

    # Voxel density field. It can also be a tensor that requires gradient.
    # The tensor can be [#grid_pts x 1] or [#voxels x 8] for voxel trilinear density field.
    # In case of [#voxel x 8], you can also specify `reduce_density=True` so it merges
    # shared points by adjacent voxels into [#grid_pts x 1] internally.
    # The given density is pre-activation. It is mapped to non-negative float when rendering.
    density=100,

    # Voxel color. It can also be a tensor that requires gradient.
    # The intensity scale is 0~1. Internally, it converts to zero-deg shperical harmonic.
    rgb=torch.tensor(pt_rgb, dtype=torch.float32, device="cuda"),

    # Voxel higher-deg spherical harmonic. It can also be a tensor that requires gradient.
    # If sh_degree=0, it will be a empty tensor.
    shs=0,
)
```

The following is the result with `expected_vox_size` set to `extent/16`, `extent/25`, `extent/100`, and `extent/1000` from left to right. When voxel sizes are too small (`extent/1000`), it causes holes between voxels and the rendering is more like rendering point cloud.

https://github.com/user-attachments/assets/2df19985-d517-4739-8f68-779d2d5d7b28


## Popping artifact
You are promised to get popping-free rendering if you create model with `model_init` following by a series of `pruning` and `subdividing`. If you use `ijkl_init` or `points_init` with various voxel levels, you may make a small voxel covered by larger voxels which results in popping artifact when rendering.

In the following example, we crate volume overlapping cases with a green and a blue voxel covered by a larger red voxel.

| ijk | level | color |
|:---:|:---:|:---:|
| (0,0,0) | 1 | red |
| (0,0,0) | 2 | green |
| (1,1,1) | 2 | blue |


```python
voxel_model = SparseVoxelModel(sh_degree=0, black_background=True)
voxel_model.ijkl_init(
    scene_center=center + 0.25 * extent,
    scene_extent=extent,

    ijk=torch.tensor([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=torch.int64, device="cuda"),
    octlevel=torch.tensor([1, 2, 2], dtype=torch.int8, device="cuda"),

    density=-2,

    rgb=torch.tensor([[1,0,0], [0,1,0], [0,0,1]], dtype=torch.float32, device="cuda"),
)
```

You can see that the green voxel is always occluded by the red voxel and the order between the blue and the red voxels keep changing. It may or may not be an issue depending on your applications.

https://github.com/user-attachments/assets/4220a76e-f935-4ae5-9f77-2682a3cea1c3
