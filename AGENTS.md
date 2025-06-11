# Agent Instructions

This repository contains CUDA/C++ rendering code for SVRaster. A companion agent should mine code only related to the rendering pipeline and ignore training/evaluation utilities. Important folders and files:

- `cuda/src/` – all CUDA kernels and headers. Contains forward rendering logic (`forward.cu`), preprocess step (`preprocess.cu`), helper headers such as `auxiliary.h`, `config.h`, and state management (`raster_state.cu/h`).
- `cuda/svraster_cuda/renderer.py` – Python bindings that expose the CUDA renderer to PyTorch. Functions here call into `_C` extension (built from CUDA sources) and show argument lists for kernels.
- `render.py` and `render_fly_through.py` at repo root – python front-ends that call the CUDA renderer.

When crawling, focus on kernels/functions showing:

1. **View-dependent Morton ordering** – `order_tables` and `compute_order_rank` in `cuda/src/auxiliary.h` implement a quadrant-based Morton permutation via XOR of octree paths with a precomputed table.
2. **Stable voxel sort** – `forward.cu` uses `cub::DeviceRadixSort::SortPairs` after building keys with `duplicateWithKeys`. Per-tile ranges are extracted by `identifyTileRanges`.
3. **GPU raster pass** – `renderCUDA` in `forward.cu` performs ray–voxel intersection using `ray_aabb` and volume integration with constants from `config.h` (`MAX_ALPHA`, `STEP_SZ_SCALE`, etc.).
4. **Python/CUDA mapping** – `rasterize_voxels` in `cuda/svraster_cuda/renderer.py` packs all arguments and calls `_C.rasterize_voxels`; the preprocess step is `_C.rasterize_preprocess`.

### Extraction Goals
Collect minimal, self-contained code snippets showing:
- `order_tables` table and `compute_order_rank` helper.
- `duplicateWithKeys` kernel creating sort keys/values.
- Radix sort invocation with `cub::DeviceRadixSort::SortPairs`.
- `identifyTileRanges` kernel building per-tile workloads.
- Core of `renderCUDA` including the `ray_aabb` call and volume integration loop.
- Python interface functions calling these kernels.

Preserve line numbers in comments for traceability. Strip license headers for brevity.

### Ongoing Notes
- Sorting uses 64‑bit keys: `[tile ID | order_rank]` where `order_rank` is computed per-quadrant.
- `raster_state.cu/h` allocate GPU buffers and expose helper functions like `resizeFunctional`.
- Preprocess step in `preprocess.cu` computes camera-quadrant bitsets and bounding boxes to decide which voxels are rendered.

Append any new file paths or important functions to this document as they are discovered.
