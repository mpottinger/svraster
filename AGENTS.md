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
Recent additions:
- `docs/rendering_pipeline_snippets.md` stores extracted code snippets with line numbers.
- Core kernels found: `preprocessCUDA`, `duplicateWithKeys`, `identifyTileRanges`, `rasterize_voxels_procedure`, and `renderCUDA`.

The rendering pipeline snippets previously stored in docs/rendering_pipeline_snippets.md are now consolidated below.
# Rendering Pipeline Snippets

```
===== FILE: cuda/src/auxiliary.h =====
```
    16	// Octant ordering tables
    17	template<uint64_t id, int n>
    18	struct repeat_3bits
    19	{
    20	    static constexpr uint64_t value = (repeat_3bits<id, n-1>::value << 3) | id;
    21	};
    22	
    23	template<uint64_t id>
    24	struct repeat_3bits<id, 1>
    25	{
    26	    static constexpr uint64_t value = id;
    27	};
    28	
    29	__constant__ uint64_t order_tables[8] = {
    30	    repeat_3bits<0ULL, MAX_NUM_LEVELS>::value,
    31	    repeat_3bits<1ULL, MAX_NUM_LEVELS>::value,
    32	    repeat_3bits<2ULL, MAX_NUM_LEVELS>::value,
    33	    repeat_3bits<3ULL, MAX_NUM_LEVELS>::value,
    34	    repeat_3bits<4ULL, MAX_NUM_LEVELS>::value,
    35	    repeat_3bits<5ULL, MAX_NUM_LEVELS>::value,
    36	    repeat_3bits<6ULL, MAX_NUM_LEVELS>::value,
    37	    repeat_3bits<7ULL, MAX_NUM_LEVELS>::value
    38	};
    39	
    40	__forceinline__ __device__ uint64_t compute_order_rank(uint64_t octree_path, int quadrant_id)
    41	{
    42	    return octree_path ^ order_tables[quadrant_id];
    43	}
    44	
    45	__forceinline__ __device__ uint64_t encode_order_key(uint64_t tile_id, uint64_t order_rank)
    46	{
    47	    return (tile_id << NUM_BIT_ORDER_RANK) | order_rank;
    48	}
    49	
    50	__forceinline__ __device__ uint32_t encode_order_val(uint32_t vox_id, uint32_t quadrant_id)
    51	{
    52	    return (((uint32_t)quadrant_id) << 29) | vox_id;
    53	}
    54	
    55	__forceinline__ __device__ uint32_t decode_order_val_4_vox_id(uint32_t val)
    56	{
    57	    return (val << 3) >> 3;
    58	}
    59	
    60	__forceinline__ __device__ uint32_t decode_order_val_4_quadrant_id(uint32_t val)
    61	{
    62	    return val >> 29;
    63	}

```
===== FILE: cuda/src/preprocess.cu =====
```
    23	// CUDA implementation of the preprocess step.
    24	__global__ void preprocessCUDA(
    25	    const int P,
    26	    const int W, const int H,
    27	    const float tan_fovx, const float tan_fovy,
    28	    const float focal_x, const float focal_y,
    29	    const float cx, const float cy,
    30	    const float* __restrict__ w2c_matrix,
    31	    const float* __restrict__ c2w_matrix,
    32	    const float near,
    33	
    34	    const float3* __restrict__ vox_centers,
    35	    const float* __restrict__ vox_lengths,
    36	
    37	    int* __restrict__ out_n_duplicates,
    38	    uint32_t* __restrict__ n_duplicates,
    39	    uint2* __restrict__ bboxes,
    40	    uint32_t* __restrict__ cam_quadrant_bitsets,
    41	
    42	    const dim3 tile_grid)
    43	{
    44	    auto idx = cg::this_grid().thread_rank();
    45	    if (idx >= P)
    46	        return;
    47	
    48	    // First things first.
    49	    // Initialize the number of voxel duplication to 0.
    50	    // We later can then skip rendering voxel with 0 duplication.
    51	    out_n_duplicates[idx] = 0;
    52	    n_duplicates[idx] = 0;
    53	
    54	    // Load from global memory.
    55	    const float3 vox_c = vox_centers[idx];
    56	    const float vox_r = 0.5f * vox_lengths[idx];
    57	    const float3 ro = last_col_3x4(c2w_matrix);
    58	    float w2c[12];
    59	    for (int i = 0; i < 12; i++)
    60	        w2c[i] = w2c_matrix[i];
    61	
    62	    // Near plane clipping (it's actually sphere)
    63	    const float3 rel_pos = vox_c - ro;
    64	    if (dot(rel_pos, rel_pos) < near * near)
    65	        return;
    66	
    67	    // Iterate the eight voxel corners and do the following:
    68	    // 1. Compute bbox region of the projected voxel.
    69	    // 2. Check if the voxel touch a camera quadrant.
    70	    uint32_t quadrant_bitset = 0;
    71	    float2 coord_min = {1e9f, 1e9f};
    72	    float2 coord_max = {-1e9f, -1e9f};
    73	    for (int i=0; i<8; ++i)
    74	    {
    75	        float3 shift = make_float3(
    76	            (float)(((i&4)>>2) * 2 - 1),
    77	            (float)(((i&2)>>1) * 2 - 1),
    78	            (float)(((i&1)   ) * 2 - 1)
    79	        );
    80	        float3 world_corner = vox_c + vox_r * shift;
    81	        float3 cam_corner = transform_3x4(w2c, world_corner);
    82	        if (cam_corner.z < near)
    83	            continue;
    84	
    85	        float2 corner_coord;
    86	        int quadrant_id;
    87	        const float inv_z = 1.0f / cam_corner.z;
    88	        corner_coord = make_float2(cam_corner.x * inv_z, cam_corner.y * inv_z);
    89	        quadrant_id = compute_corner_quadrant_id(world_corner, ro);
    90	
    91	        coord_min = min(coord_min, corner_coord);
    92	        coord_max = max(coord_max, corner_coord);
    93	        quadrant_bitset |= (1 << quadrant_id);
    94	    }
    95	
    96	    float cx_h = cx - 0.5f;
    97	    float cy_h = cy - 0.5f;
    98	    float2 bbox_min = {
    99	        max(focal_x * coord_min.x + cx_h, 0.0f),
   100	        max(focal_y * coord_min.y + cy_h, 0.0f)
   101	    };
   102	    float2 bbox_max = {
   103	        min(focal_x * coord_max.x + cx_h, (float)W),
   104	        min(focal_y * coord_max.y + cy_h, (float)H)
   105	    };
   106	    if (bbox_min.x > bbox_max.x || bbox_min.y > bbox_max.y)
   107	        return; // Bbox outside image plane.
   108	
   109	    // Squeeze bbox info into 2 uint.
   110	    const uint2 bbox = {
   111	        (((uint)lrintf(bbox_min.x)) << 16) | ((uint)lrintf(bbox_min.y)),
   112	        (((uint)lrintf(bbox_max.x)) << 16) | ((uint)lrintf(bbox_max.y))
   113	    };
   114	
   115	    // Compute tile range.
   116	    uint2 tile_min, tile_max;
   117	    getBboxTileRect(bbox, tile_min, tile_max, tile_grid);
   118	    int tiles_touched = (1 + tile_max.y - tile_min.y) * (1 + tile_max.x - tile_min.x);
   119	    if (tiles_touched <= 0)
   120	    {
   121	        // TODO: remove sanity check.
   122	        printf("tiles_touched <= 0 !???");
   123	        __trap();
   124	    }
   125	
   126	    // Write back the results.
   127	    const int quadrant_touched = __popc(quadrant_bitset);
   128	    out_n_duplicates[idx] = tiles_touched * quadrant_touched;
   129	    n_duplicates[idx] = tiles_touched * quadrant_touched;
   130	    bboxes[idx] = bbox;
   131	    cam_quadrant_bitsets[idx] = quadrant_bitset;
   132	}
   133	
   134	
   135	// Interface for python to find the voxel to render and compute some init values.
   136	std::tuple<torch::Tensor, torch::Tensor>
   137	rasterize_preprocess(
   138	    const int image_width, const int image_height,
   139	    const float tan_fovx, const float tan_fovy,
   140	    const float cx, const float cy,
   141	    const torch::Tensor& w2c_matrix,
   142	    const torch::Tensor& c2w_matrix,
   143	    const float near,
   144	
   145	    const torch::Tensor& octree_paths,
   146	    const torch::Tensor& vox_centers,
   147	    const torch::Tensor& vox_lengths,
   148	
   149	    const bool debug)
   150	{
   151	    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
   152	        AT_ERROR("vox_centers must have dimensions (num_points, 3)");
   153	
   154	    const int P = vox_centers.size(0);
   155	
   156	    auto t_opt_byte = torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA);
   157	    auto t_opt_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
   158	
   159	    torch::Tensor geomBuffer = torch::empty({0}, t_opt_byte);
   160	    torch::Tensor out_n_duplicates = torch::full({P}, 0, t_opt_int32);
   161	
   162	    if (P == 0)
   163	        return std::make_tuple(out_n_duplicates, geomBuffer);
   164	
   165	    // Allocate GeometryState
   166	    size_t chunk_size = RASTER_STATE::required<RASTER_STATE::GeometryState>(P);
   167	    geomBuffer.resize_({(long long)chunk_size});
   168	    char* chunkptr = reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr());
   169	    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(chunkptr, P);
   170	
   171	    // Derive arguments
   172	    dim3 tile_grid((image_width + BLOCK_X - 1) / BLOCK_X, (image_height + BLOCK_Y - 1) / BLOCK_Y, 1);
   173	    const float focal_x = 0.5f * image_width / tan_fovx;
   174	    const float focal_y = 0.5f * image_height / tan_fovy;
   175	
   176	    // Lanching CUDA
   177	    preprocessCUDA <<<(P + 255) / 256, 256>>> (
   178	        P,
   179	        image_width, image_height,
   180	        tan_fovx, tan_fovy,
   181	        focal_x, focal_y,
   182	        cx, cy,
   183	        w2c_matrix.contiguous().data_ptr<float>(),
   184	        c2w_matrix.contiguous().data_ptr<float>(),
   185	        near,
   186	
   187	        (float3*)(vox_centers.contiguous().data_ptr<float>()),
   188	        vox_lengths.contiguous().data_ptr<float>(),
   189	
   190	        out_n_duplicates.contiguous().data_ptr<int>(),
   191	        geomState.n_duplicates,
   192	        geomState.bboxes,
   193	        geomState.cam_quadrant_bitsets,
   194	
   195	        tile_grid);
   196	    CHECK_CUDA(debug);
   197	
   198	    return std::make_tuple(out_n_duplicates, geomBuffer);
   199	}
   200	
   201	}

```
===== FILE: cuda/src/forward.cu =====
```
    38	// CUDA sparse voxel rendering.
    39	template <bool need_depth, bool need_distortion, bool need_normal, bool track_max_w,
    40	          int n_samp>
    41	__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
    42	renderCUDA(
    43	    const uint2* __restrict__ ranges,
    44	    const uint32_t* __restrict__ vox_list,
    45	    int W, int H,
    46	    const float tan_fovx, const float tan_fovy,
    47	    const float cx, const float cy,
    48	    const float* __restrict__ c2w_matrix,
    49	    const float bg_color,
    50	
    51	    const uint2* __restrict__ bboxes,
    52	    const float3* __restrict__ vox_centers,
    53	    const float* __restrict__ vox_lengths,
    54	    const float* __restrict__ geos,
    55	    const float3* __restrict__ rgbs,
    56	
    57	    uint32_t* __restrict__ tile_last,
    58	    uint32_t* __restrict__ n_contrib,
    59	
    60	    float* __restrict__ out_color,
    61	    float* __restrict__ out_depth,
    62	    float* __restrict__ out_normal,
    63	    float* __restrict__ out_T,
    64	    float* __restrict__ max_w)
    65	{
    66	    // Identify current tile and associated min/max pixel range.
    67	    auto block = cg::this_thread_block();
    68	    uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
    69	    int thread_id = block.thread_rank();
    70	    int tile_id = block.group_index().y * horizontal_blocks + block.group_index().x;
    71	    uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
    72	    uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
    73	
    74	    uint2 pix;
    75	    uint32_t pix_id;
    76	    float2 pixf;
    77	    if (BLOCK_X % 8 == 0 && BLOCK_Y % 4 == 0)
    78	    {
    79	        // Pack the warp threads into a 4x8 macro blocks.
    80	        // It could reduce idle warp threads as the voxels to render
    81	        // are more coherent in 4x8 than 2x16 rectangle.
    82	        int macro_x_num = BLOCK_X / 8;
    83	        int macro_id = thread_id / 32;
    84	        int macro_xid = macro_id % macro_x_num;
    85	        int macro_yid = macro_id / macro_x_num;
    86	        int micro_id = thread_id % 32;
    87	        int micro_xid = micro_id % 8;
    88	        int micro_yid = micro_id / 8;
    89	        pix = { pix_min.x + macro_xid * 8 + micro_xid, pix_min.y + macro_yid * 4 + micro_yid};
    90	        pix_id = W * pix.y + pix.x;
    91	        pixf = { (float)pix.x, (float)pix.y };
    92	    }
    93	    else
    94	    {
    95	        pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
    96	        pix_id = W * pix.y + pix.x;
    97	        pixf = { (float)pix.x, (float)pix.y };
    98	    }
    99	
   100	    // Compute camera info.
   101	    float3 ro, rd, rd_inv;
   102	    float rd_norm_inv;
   103	    const float3 cam_rd = compute_ray_d(pixf, W, H, tan_fovx, tan_fovy, cx, cy);
   104	    const float rd_norm = sqrtf(dot(cam_rd, cam_rd));
   105	    const float3 rd_raw = rotate_3x4(c2w_matrix, cam_rd);
   106	    rd_norm_inv = 1.f / rd_norm;
   107	    ro = last_col_3x4(c2w_matrix);
   108	    rd = rd_raw * rd_norm_inv;
   109	    rd_inv = {1.f/ rd.x, 1.f / rd.y, 1.f / rd.z};
   110	
   111	    const uint32_t pix_quad_id = compute_ray_quadrant_id(rd);
   112	
   113	    // Check if this thread is associated with a valid pixel or outside.
   114	    bool inside = (pix.x < W) && (pix.y < H);
   115	    // Done threads can help with fetching, but don't rasterize
   116	    bool done = !inside;
   117	
   118	    // Load start/end range of IDs to process in BinningState.
   119	    uint2 range = ranges[tile_id];
   120	    const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
   121	    int toDo = range.y - range.x;
   122	
   123	    // Init the last non-occluded range index of the tile.
   124	    if (thread_id == 0)
   125	        tile_last[tile_id] = range.x;
   126	
   127	    // Allocate storage for batches of collectively fetched data.
   128	    // 3090Ti shared memory per-block statistic:
   129	    //   total shared memory      = 49152 bytes
   130	    //   shared memory per-thread = 49152/BLOCK_SIZE = 192 bytes
   131	    //                            = 48 int or float
   132	    __shared__ int collected_vox_id[BLOCK_SIZE];
   133	    __shared__ int collected_quad_id[BLOCK_SIZE];
   134	    __shared__ uint2 collected_bbox[BLOCK_SIZE];
   135	    __shared__ float3 collected_vox_c[BLOCK_SIZE];
   136	    __shared__ float collected_vox_l[BLOCK_SIZE];
   137	    __shared__ float collected_geo_params[BLOCK_SIZE * 8];
   138	    __shared__ float3 collected_rgb[BLOCK_SIZE];
   139	
   140	    // Initialize helper variables.
   141	    float T = 1.f;
   142	    uint32_t contributor = 0;
   143	    uint32_t last_contributor = 0;
   144	    float3 C = {0.f, 0.f, 0.f};
   145	    float3 N = {0.f, 0.f, 0.f};
   146	    float D = 0.f;
   147	    int D_med_vox_id = -1;
   148	    float D_med_T;
   149	    float D_med = 0.f;
   150	    float Ddist = 0.f;
   151	    int j_lst[BLOCK_SIZE];
   152	
   153	    // Iterate over batches until all done or range is complete.
   154	    for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
   155	    {
   156	        // End if entire block votes that it is done rasterizing.
   157	        int num_done = __syncthreads_count(done);
   158	        if (num_done == BLOCK_SIZE)
   159	            break;
   160	
   161	        // Collectively fetch batch of voxel data from global to shared.
   162	        int progress = i * BLOCK_SIZE + thread_id;
   163	        if (range.x + progress < range.y)
   164	        {
   165	            uint32_t order_val = vox_list[range.x + progress];
   166	            uint32_t vox_id = decode_order_val_4_vox_id(order_val);
   167	            uint32_t quad_id = decode_order_val_4_quadrant_id(order_val);
   168	            collected_vox_id[thread_id] = vox_id;
   169	            collected_quad_id[thread_id] = quad_id;
   170	            collected_bbox[thread_id] = bboxes[vox_id];
   171	            collected_vox_c[thread_id] = vox_centers[vox_id];
   172	            collected_vox_l[thread_id] = vox_lengths[vox_id];
   173	            for (int k=0; k<8; ++k)
   174	                collected_geo_params[thread_id*8 + k] = geos[vox_id*8 + k];
   175	            collected_rgb[thread_id] = rgbs[vox_id];
   176	        }
   177	        block.sync();
   178	
   179	        // Iterate over current batch.
   180	        const int end_j = min(BLOCK_SIZE, toDo);
   181	        int j_lst_top = -1;
   182	        for (int j = 0; !done && j < end_j; j++)
   183	        {
   184	            // Check if the pixel in the projected bbox region.
   185	            // Check if the quadrant id match the pixel.
   186	            if (!pix_in_bbox(pix, collected_bbox[j]) || pix_quad_id != collected_quad_id[j])
   187	                continue;
   188	
   189	            // Compute ray aabb intersection
   190	            const float3 vox_c = collected_vox_c[j];
   191	            const float vox_l = collected_vox_l[j];
   192	            const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
   193	            const float a = ab.x;
   194	            const float b = ab.y;
   195	            if (a > b)
   196	                continue;  // Skip if no intersection.
   197	
   198	            j_lst_top += 1;
   199	            j_lst[j_lst_top] = j;
   200	        }
   201	
   202	        int contributor_inc = 0;
   203	        for (int jj = 0; !done && jj <= j_lst_top; jj++)
   204	        {
   205	            int j = j_lst[jj];
   206	            const int vox_id = collected_vox_id[j];
   207	
   208	            // Keep track of current position in range.
   209	            contributor_inc = j + 1;
   210	
   211	            // Compute ray aabb intersection
   212	            const float3 vox_c = collected_vox_c[j];
   213	            const float vox_l = collected_vox_l[j];
   214	            const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
   215	            const float a = ab.x;
   216	            const float b = ab.y;
   217	
   218	            float geo_params[8];
   219	            for (int k=0; k<8; ++k)
   220	                geo_params[k] = collected_geo_params[j*8 + k];
   221	
   222	            // Compute volume density
   223	            float vol_int = 0.f;
   224	            float interp_w[8];
   225	            float local_alphas[n_samp];
   226	
   227	            // Quadrature integral from trilinear sampling.
   228	            float vox_l_inv = 1.f / vox_l;
   229	            const float step_sz = (b - a) * (1.f / n_samp);
   230	            const float3 step = step_sz * rd;
   231	            float3 pt = ro + (a + 0.5f * step_sz) * rd;
   232	            float3 qt = (pt - (vox_c - 0.5f * vox_l)) * vox_l_inv;
   233	            const float3 qt_step = step * vox_l_inv;
   234	
   235	            #pragma unroll
   236	            for (int k=0; k<n_samp; k++, qt=qt+qt_step)
   237	            {
   238	                tri_interp_weight(qt, interp_w);
   239	                float d = 0.f;
   240	                for (int iii=0; iii<8; ++iii)
   241	                    d += geo_params[iii] * interp_w[iii];
   242	
   243	                const float local_vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);
   244	                vol_int += local_vol_int;
   245	
   246	                if (need_depth && n_samp > 1)
   247	                    local_alphas[k] = min(MAX_ALPHA, 1.f - expf(-local_vol_int));
   248	            }
   249	
   250	            // Compute alpha from volume integral.
   251	            float alpha = min(MAX_ALPHA, 1.f - expf(-vol_int));
   252	            if (alpha < MIN_ALPHA)
   253	                continue;
   254	
   255	            // Accumulate to the pixel.
   256	            float pt_w = T * alpha;
   257	            C = C + pt_w * collected_rgb[j];
   258	
   259	            if (need_depth)
   260	            {
   261	                // Mean depth
   262	                float dval;
   263	                if (n_samp == 3)
   264	                {
   265	                    float step_sz = 0.3333333f * (b - a);
   266	                    float a0 = local_alphas[0], a1 = local_alphas[1], a2 = local_alphas[2];
   267	                    float t0 = a + 0.5f * step_sz;
   268	                    float t1 = a + 1.5f * step_sz;
   269	                    float t2 = a + 2.5f * step_sz;
   270	                    dval = a0*t0 + (1.f-a0)*a1*t1 + (1.f-a0)*(1.f-a1)*a2*t2;
   271	                }
   272	                else if (n_samp == 2)
   273	                {
   274	                    float step_sz = 0.5f * (b - a);
   275	                    float a0 = local_alphas[0], a1 = local_alphas[1];
   276	                    float t0 = a + 0.5f * step_sz;
   277	                    float t1 = a + 1.5f * step_sz;
   278	                    dval = a0*t0 + (1.f-a0)*a1*t1;
   279	                }
   280	                else
   281	                {
   282	                    dval = alpha * 0.5f * (a + b);
   283	                }
   284	                D = D + T * dval;
   285	
   286	                // Median depth
   287	                if (T > 0.5f)
   288	                {
   289	                    D_med_vox_id = vox_id;
   290	                    D_med_T = T;
   291	                }
   292	            }
   293	
   294	            // Distortion depth
   295	            if (need_distortion)
   296	                Ddist = Ddist + pt_w * 0.5f * (depth_contracted(a) + depth_contracted(b));
   297	
   298	            // Normal
   299	            if (need_normal)
   300	            {
   301	                const float lin_nx = (
   302	                    (geo_params[0b100] + geo_params[0b101] + geo_params[0b110] + geo_params[0b111]) -
   303	                    (geo_params[0b000] + geo_params[0b001] + geo_params[0b010] + geo_params[0b011]));
   304	                const float lin_ny = (
   305	                    (geo_params[0b010] + geo_params[0b011] + geo_params[0b110] + geo_params[0b111]) -
   306	                    (geo_params[0b000] + geo_params[0b001] + geo_params[0b100] + geo_params[0b101]));
   307	                const float lin_nz = (
   308	                    (geo_params[0b001] + geo_params[0b011] + geo_params[0b101] + geo_params[0b111]) -
   309	                    (geo_params[0b000] + geo_params[0b010] + geo_params[0b100] + geo_params[0b110]));
   310	                const float3 lin_n = make_float3(lin_nx, lin_ny, lin_nz);
   311	                const float r_lin = safe_rnorm(lin_n);
   312	                N = N + pt_w * r_lin * lin_n;
   313	            }
   314	
   315	            T *= (1.f - alpha);
   316	            done |= (T < EARLY_STOP_T);
   317	
   318	            // Keep track of last range entry to update this pixel.
   319	            last_contributor = contributor + contributor_inc;
   320	
   321	            // Keep track of the maxiumum importance weight of each voxel.
   322	            if (track_max_w)
   323	                atomicMax(((int*)max_w) + vox_id, *((int*)(&pt_w)));
   324	        }
   325	        contributor += done ? contributor_inc : end_j;
   326	    }
   327	
   328	    if (need_depth && inside && D_med_vox_id != -1)
   329	    {
   330	        // Finest sampling of median depth
   331	        const int n_samp_dmed = 16;
   332	
   333	        float3 vox_c = vox_centers[D_med_vox_id];
   334	        float vox_l = vox_lengths[D_med_vox_id];
   335	        float geo_params[8];
   336	        for (int k=0; k<8; ++k)
   337	            geo_params[k] = geos[D_med_vox_id*8 + k];
   338	        const float2 ab = ray_aabb(vox_c, vox_l, ro, rd_inv);
   339	        const float a = ab.x;
   340	        const float b = ab.y;
   341	
   342	        float vox_l_inv = 1.f / vox_l;
   343	        const float step_sz = (b - a) * (1.f / n_samp_dmed);
   344	        const float3 step = step_sz * rd;
   345	        float3 pt = ro + (a + 0.5f * step_sz) * rd;
   346	        float3 qt = (pt - (vox_c - 0.5f * vox_l)) * vox_l_inv;
   347	        const float3 qt_step = step * vox_l_inv;
   348	
   349	        D_med = a - 0.5f * step_sz;
   350	        for (int k=0; k<n_samp_dmed && D_med_T > 0.5f; k++, qt=qt+qt_step)
   351	        {
   352	            D_med += step_sz;
   353	
   354	            float interp_w[8];
   355	            tri_interp_weight(qt, interp_w);
   356	            float d = 0.f;
   357	            for (int iii=0; iii<8; ++iii)
   358	                d += geo_params[iii] * interp_w[iii];
   359	
   360	            const float vol_int = STEP_SZ_SCALE * step_sz * exp_linear_11(d);
   361	
   362	            D_med_T *= expf(-vol_int);
   363	        }
   364	    }
   365	
   366	    // All threads that treat valid pixel write out their final
   367	    // rendering data to the frame and auxiliary buffers.
   368	    if (inside)
   369	    {
   370	        n_contrib[pix_id] = last_contributor;
   371	        out_color[0 * H * W + pix_id] = C.x + T * bg_color;
   372	        out_color[1 * H * W + pix_id] = C.y + T * bg_color;
   373	        out_color[2 * H * W + pix_id] = C.z + T * bg_color;
   374	        out_T[pix_id] = T;  // Equal to (1 - alpha).
   375	        if (need_depth)
   376	        {
   377	            out_depth[pix_id] = D * rd_norm_inv;
   378	            out_depth[H * W * 2 + pix_id] = D_med * rd_norm_inv;
   379	        }
   380	        if (need_distortion)
   381	        {
   382	            out_depth[H * W + pix_id] = Ddist;
   383	        }
   384	        if (need_normal)
   385	        {
   386	            out_normal[0 * H * W + pix_id] = N.x;
   387	            out_normal[1 * H * W + pix_id] = N.y;
   388	            out_normal[2 * H * W + pix_id] = N.z;
   389	        }
   390	        atomicMax(tile_last + tile_id, range.x + last_contributor);
   391	    }
   392	}
   393	
   394	
   500	uint32_t getHigherMsb(uint32_t n)
   501	{
   502	    uint32_t msb = sizeof(n) * 4;
   503	    uint32_t step = msb;
   504	    while (step > 1)
   505	    {
   506	        step /= 2;
   507	        if (n >> msb)
   508	            msb += step;
   509	        else
   510	            msb -= step;
   511	    }
   512	    if (n >> msb)
   513	        msb++;
   514	    return msb;
   515	}
   516	
   517	// Duplicate each voxel by #tiles x #cam_quadrant it touches.
   518	__global__ void duplicateWithKeys(
   519	    int P,
   520	    const int64_t* octree_paths,
   521	    const uint2* bboxes,
   522	    const uint32_t* cam_quadrant_bitsets,
   523	    const uint32_t* n_duplicates,
   524	    const uint32_t* n_duplicates_scan,
   525	    uint64_t* vox_list_keys_unsorted,
   526	    uint32_t* vox_list_unsorted,
   527	    dim3 grid)
   528	{
   529	    auto idx = cg::this_grid().thread_rank();
   530	    if (idx >= P || n_duplicates[idx] == 0)
   531	        return;
   532	
   533	    // Find this voxel's array offset in buffer for writing the key/value.
   534	    uint32_t off = (idx == 0) ? 0 : n_duplicates_scan[idx - 1];
   535	    uint2 tile_min, tile_max;
   536	    getBboxTileRect(bboxes[idx], tile_min, tile_max, grid);
   537	
   538	    // For each tile that the bounding rect overlaps, emit a key/value pair.
   539	    // The key bit structure is [  tile ID  |  order_rank  ],
   540	    // so the voxels are first sorted by tile and then by order_ranks.
   541	    // The value bit structure is [  quadrant ID  |  voxel ID  ].
   542	    const uint64_t octree_path = octree_paths[idx];
   543	    uint32_t quadrant_bitsets = cam_quadrant_bitsets[idx];
   544	    for (int quadrant_id = 0; quadrant_id < 8; quadrant_id++)
   545	    {
   546	        if ((quadrant_bitsets & (1 << quadrant_id)) == 0)
   547	            continue;
   548	
   549	        // Compute order_rank for the voxel in this quadrant.
   550	        uint64_t order_rank = compute_order_rank(octree_path, quadrant_id);
   551	
   552	        // Duplicate result to touched tiles.
   553	        for (int y = tile_min.y; y <= tile_max.y; y++)
   554	        {
   555	            for (int x = tile_min.x; x <= tile_max.x; x++)
   556	            {
   557	                uint64_t tile_id = y * grid.x + x;
   558	                vox_list_keys_unsorted[off] = encode_order_key(tile_id, order_rank);
   559	                vox_list_unsorted[off] = encode_order_val(idx, quadrant_id);
   560	                off++;
   561	            }
   562	        }
   563	    }
   564	
   565	    if (off != n_duplicates_scan[idx])
   566	    {
   567	        // TODO: remove sanity check.
   568	        printf("Number of duplication mismatch !???");
   569	        __trap();
   570	    }
   571	}
   572	
   573	// The sorted vox_list_keys is now as:
   574	//   [--sorted voxels for tile 1--  --sorted voxels for tile 2--  ...]
   575	// We want to identify the start/end index of each tile from this list.
   576	__global__ void identifyTileRanges(int L, uint64_t* vox_list_keys, uint2* ranges)
   577	{
   578	    auto idx = cg::this_grid().thread_rank();
   579	    if (idx >= L)
   580	        return;
   581	
   582	    // Read tile ID from key. Update start/end of tile range if at limit.
   583	    uint64_t key = vox_list_keys[idx];
   584	    uint32_t currtile = key >> NUM_BIT_ORDER_RANK;
   585	    if (idx == 0)
   586	        ranges[currtile].x = 0;
   587	    else
   588	    {
   589	        uint32_t prevtile = vox_list_keys[idx - 1] >> NUM_BIT_ORDER_RANK;
   590	        if (currtile != prevtile)
   591	        {
   592	            ranges[prevtile].y = idx;
   593	            ranges[currtile].x = idx;
   594	        }
   595	    }
   596	    if (idx == L - 1)
   597	        ranges[currtile].y = L;
   598	}
   599	
   600	// Mid-level C interface for the entire rasterization procedure.
   601	int rasterize_voxels_procedure(
   602	    char* geom_buffer,
   603	    std::function<char* (size_t)> binningBuffer,
   604	    std::function<char* (size_t)> imageBuffer,
   605	    const int P,
   606	    const int n_samp_per_vox,
   607	    const int width, const int height,
   608	    const float tan_fovx, const float tan_fovy,
   609	    const float cx, float cy,
   610	    const float* w2c_matrix,
   611	    const float* c2w_matrix,
   612	    const float bg_color,
   613	    const bool need_depth,
   614	    const bool need_distortion,
   615	    const bool need_normal,
   616	
   617	    const int64_t* octree_paths,
   618	    const float* vox_centers,
   619	    const float* vox_lengths,
   620	    const float* geos,
   621	    const float* rgbs,
   622	
   623	    float* out_color,
   624	    float* out_depth,
   625	    float* out_normal,
   626	    float* out_T,
   627	    float* max_w,
   628	
   629	    bool debug)
   630	{
   631	    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
   632	    dim3 block(BLOCK_X, BLOCK_Y, 1);
   633	
   634	    // Recover the preprocessing results.
   635	    RASTER_STATE::GeometryState geomState = RASTER_STATE::GeometryState::fromChunk(geom_buffer, P);
   636	
   637	    // Dynamically resize image-based auxiliary buffers during training.
   638	    size_t img_chunk_size = RASTER_STATE::required<RASTER_STATE::ImageState>(width * height, tile_grid.x * tile_grid.y);
   639	    char* img_chunkptr = imageBuffer(img_chunk_size);
   640	    RASTER_STATE::ImageState imgState = RASTER_STATE::ImageState::fromChunk(img_chunkptr, width * height, tile_grid.x * tile_grid.y);
   641	
   642	    // Compute prefix sum over full list of the number of voxel duplications.
   643	    cub::DeviceScan::InclusiveSum(
   644	        geomState.scanning_temp_space,
   645	        geomState.scan_size,
   646	        geomState.n_duplicates,
   647	        geomState.n_duplicates_scan,
   648	        P);
   649	    CHECK_CUDA(debug);
   650	
   651	    // Retrieve total number of voxels after duplication.
   652	    int num_rendered;
   653	    cudaMemcpy(
   654	        &num_rendered,
   655	        geomState.n_duplicates_scan + P - 1,
   656	        sizeof(int),
   657	        cudaMemcpyDeviceToHost);
   658	    CHECK_CUDA(debug);
   659	
   660	    size_t binning_chunk_size = RASTER_STATE::required<RASTER_STATE::BinningState>(num_rendered);
   661	    char* binning_chunkptr = binningBuffer(binning_chunk_size);
   662	    RASTER_STATE::BinningState binningState = RASTER_STATE::BinningState::fromChunk(binning_chunkptr, num_rendered);
   663	
   664	    // For each voxel to be rendered, produce adequate [ tile ID | rank ] key
   665	    // and the corresponding dublicated voxel [ quadrant ID | voxel ID ] to be sorted.
   666	    duplicateWithKeys <<<(P + 255) / 256, 256>>> (
   667	        P,
   668	        octree_paths,
   669	        geomState.bboxes,
   670	        geomState.cam_quadrant_bitsets,
   671	        geomState.n_duplicates,
   672	        geomState.n_duplicates_scan,
   673	        binningState.vox_list_keys_unsorted,
   674	        binningState.vox_list_unsorted,
   675	        tile_grid);
   676	    CHECK_CUDA(debug);
   677	
   678	    int bit = getHigherMsb(tile_grid.x * tile_grid.y);
   679	
   680	    // Sort complete list of (duplicated) ID by keys.
   681	    cub::DeviceRadixSort::SortPairs(
   682	        binningState.list_sorting_space,
   683	        binningState.sorting_size,
   684	        binningState.vox_list_keys_unsorted, binningState.vox_list_keys,
   685	        binningState.vox_list_unsorted, binningState.vox_list,
   686	        num_rendered, 0, NUM_BIT_ORDER_RANK + bit);
   687	    CHECK_CUDA(debug);
   688	
   689	    cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));
   690	    CHECK_CUDA(debug);
   691	
   692	    // Identify start and end of per-tile workloads in sorted list.
   693	    if (num_rendered > 0)
   694	    {
   695	        identifyTileRanges <<<(num_rendered + 255) / 256, 256>>> (
   696	            num_rendered,
   697	            binningState.vox_list_keys,
   698	            imgState.ranges);
   699	        CHECK_CUDA(debug);
   700	    }
   701	
   702	    // Let each tile blend its range of voxels independently in parallel.
   703	    render(
   704	        tile_grid, block,
   705	        imgState.ranges,
   706	        binningState.vox_list,
   707	        n_samp_per_vox,
   708	        width, height,
   709	        tan_fovx, tan_fovy,
   710	        cx, cy,
   711	        c2w_matrix,
   712	        bg_color,
   713	        need_depth,
   714	        need_distortion,
   715	        need_normal,
   716	
   717	        geomState.bboxes,
   718	        (float3*)vox_centers,
   719	        vox_lengths,
   720	        geos,
   721	        (float3*)rgbs,
   722	
   723	        imgState.tile_last,
   724	        imgState.n_contrib,
   725	
   726	        out_color,
   727	        out_depth,
   728	        out_normal,
   729	        out_T,
   730	        max_w);
   731	    CHECK_CUDA(debug);
   732	
   733	    return num_rendered;
   734	}
   735	
   736	
   737	// Interface for python to run forward rasterization.
   738	std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
   739	rasterize_voxels(
   740	    const int n_samp_per_vox,
   740	    const int n_samp_per_vox,
   741	    const int image_width, const int image_height,
   742	    const float tan_fovx, const float tan_fovy,
   743	    const float cx, const float cy,
   744	    const torch::Tensor& w2c_matrix,
   745	    const torch::Tensor& c2w_matrix,
   746	    const float bg_color,
   747	    const bool need_depth,
   748	    const bool need_distortion,
   749	    const bool need_normal,
   750	    const bool track_max_w,
   751	
   752	    const torch::Tensor& octree_paths,
   753	    const torch::Tensor& vox_centers,
   754	    const torch::Tensor& vox_lengths,
   755	    const torch::Tensor& geos,
   756	    const torch::Tensor& rgbs,
   757	
   758	    const torch::Tensor& geomBuffer,
   759	
   760	    const bool debug)
   761	{
   762	    if (vox_centers.ndimension() != 2 || vox_centers.size(1) != 3)
   763	        AT_ERROR("vox_centers must have dimensions (num_points, 3)");
   764	    if (rgbs.ndimension() != 2 || rgbs.size(1) != 3)
   765	        AT_ERROR("rgbs should be either (num_points, 3)");
   766	    if (vox_centers.size(0) != rgbs.size(0))
   767	        AT_ERROR("size mismatch");
   768	
   769	    const int P = vox_centers.size(0);
   770	    const int H = image_height;
   771	    const int W = image_width;
   772	
   773	    auto float_opts = torch::TensorOptions(torch::kFloat32).device(torch::kCUDA);
   774	    auto byte_opts = torch::TensorOptions(torch::kByte).device(torch::kCUDA);
   775	
   776	    torch::Tensor out_color = torch::full({3, H, W}, 0.f, float_opts);
   777	    torch::Tensor out_depth = need_depth || need_distortion ? torch::full({3, H, W}, 0.f, float_opts) : torch::empty({0});
   778	    torch::Tensor out_normal = need_normal ? torch::full({3, H, W}, 0.f, float_opts) : torch::empty({0});
   779	    torch::Tensor out_T = torch::full({1, H, W}, 0.f, float_opts);
   780	    torch::Tensor max_w = track_max_w ? torch::full({P, 1}, 0.f, float_opts) : torch::empty({0});
   781	
   782	    torch::Tensor binningBuffer = torch::empty({0}, byte_opts);
   783	    torch::Tensor imgBuffer = torch::empty({0}, byte_opts);
   784	    std::function<char*(size_t)> binningFunc = RASTER_STATE::resizeFunctional(binningBuffer);
   785	    std::function<char*(size_t)> imgFunc = RASTER_STATE::resizeFunctional(imgBuffer);
   786	
   787	    float* max_w_pointer = nullptr;
   788	    if (track_max_w)
   789	        max_w_pointer = max_w.contiguous().data_ptr<float>();
   790	
   791	    int rendered = 0;
   792	    if(P != 0)
   793	        rendered = rasterize_voxels_procedure(
   794	            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
   795	            binningFunc,
   796	            imgFunc,
   797	            P,
   798	            n_samp_per_vox,
   799	
   800	            W, H,
   801	            tan_fovx, tan_fovy,
   802	            cx, cy,
   803	            w2c_matrix.contiguous().data_ptr<float>(),
   804	            c2w_matrix.contiguous().data_ptr<float>(),
   805	            bg_color,
   806	            need_depth,
   807	            need_distortion,
   808	            need_normal,
   809	
   810	            octree_paths.contiguous().data_ptr<int64_t>(),
   811	            vox_centers.contiguous().data_ptr<float>(),
   812	            vox_lengths.contiguous().data_ptr<float>(),
   813	            geos.contiguous().data_ptr<float>(),
   814	            rgbs.contiguous().data_ptr<float>(),
   815	
   816	            out_color.contiguous().data_ptr<float>(),
   817	            out_depth.contiguous().data_ptr<float>(),
   818	            out_normal.contiguous().data_ptr<float>(),
   819	            out_T.contiguous().data_ptr<float>(),
   820	            max_w_pointer,
   821	
   822	            debug);
   823	
   824	    return std::make_tuple(rendered, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w);
   825	}
   826	
   827	}

```
===== FILE: cuda/svraster_cuda/renderer.py =====
```
    39	def rasterize_voxels(
    40	        raster_settings: RasterSettings,
    41	        octree_paths: torch.Tensor,
    42	        vox_centers: torch.Tensor,
    43	        vox_lengths: torch.Tensor,
    44	        vox_fn,
    45	    ):
    46	
    47	    # Some input checking
    48	    if not isinstance(raster_settings, RasterSettings):
    49	        raise Exception("Expect RasterSettings as first argument.")
    50	    if raster_settings.n_samp_per_vox > _C.MAX_N_SAMP or raster_settings.n_samp_per_vox < 1:
    51	        raise Exception(f"n_samp_per_vox should be in range [1, {_C.MAX_N_SAMP}].")
    52	
    53	    N = octree_paths.numel()
    54	    device = octree_paths.device
    55	    if vox_centers.shape[0] != N or vox_lengths.numel() != N:
    56	        raise Exception("Size mismatched.")
    57	    if len(vox_centers.shape) != 2 or vox_centers.shape[1] != 3:
    58	        raise Exception("Expect vox_centers in shape [N, 3].")
    59	    if raster_settings.w2c_matrix.device != device or \
    60	            raster_settings.c2w_matrix.device != device or \
    61	            vox_centers.device != device or \
    62	            vox_lengths.device != device:
    63	        raise Exception("Device mismatch.")
    64	
    65	    # Preprocess octree
    66	    n_duplicates, geomBuffer = _C.rasterize_preprocess(
    67	        raster_settings.image_width,
    68	        raster_settings.image_height,
    69	        raster_settings.tanfovx,
    70	        raster_settings.tanfovy,
    71	        raster_settings.cx,
    72	        raster_settings.cy,
    73	        raster_settings.w2c_matrix,
    74	        raster_settings.c2w_matrix,
    75	        raster_settings.near,
    76	
    77	        octree_paths,
    78	        vox_centers,
    79	        vox_lengths,
    80	
    81	        raster_settings.debug,
    82	    )
    83	    in_frusts_idx = torch.where(n_duplicates > 0)[0]
    84	
    85	    # Forward voxel parameters
    86	    cam_pos = raster_settings.c2w_matrix[:3, 3]
    87	    vox_params = vox_fn(in_frusts_idx, cam_pos, raster_settings.color_mode)
    88	    geos = vox_params['geos']
    89	    rgbs = vox_params['rgbs']
    90	    subdiv_p = vox_params['subdiv_p']
    91	
    92	    # Some voxel parameters checking
    93	    if geos.shape != (N, 8):
    94	        raise Exception(f"Expect geos in ({N}, 8) but got", geos.shape)
    95	    if rgbs.shape[0] != N:
    96	        raise Exception(f"Expect rgbs in ({N}, 3) but got", rgbs.shape)
    97	    if subdiv_p.shape[0] != N:
    98	        raise Exception(f"Expect subdiv_p in ({N}, 1) but got", subdiv_p.shape)
    99	
   100	    if geos.device != device:
   101	        raise Exception("Device mismatch: geos.")
   102	    if rgbs.device != device:
   103	        raise Exception("Device mismatch: rgbs.")
   104	    if subdiv_p.device != device:
   105	        raise Exception("Device mismatch: subdiv_p.")
   106	
   107	    # Some checking for regularizations
   108	    if raster_settings.lambda_R_concen > 0:
   109	        if len(raster_settings.gt_color.shape) != 3 or \
   110	                raster_settings.gt_color.shape[0] != 3 or \
   111	                raster_settings.gt_color.shape[1] != raster_settings.image_height or \
   112	                raster_settings.gt_color.shape[2] != raster_settings.image_width:
   113	            raise Exception("Except gt_color in shape of [3, H, W]")
   114	        if raster_settings.gt_color.device != device:
   115	            raise Exception("Device mismatch.")
   116	
   117	    # Involk differentiable voxels rasterization.
   118	    return _RasterizeVoxels.apply(
   119	        raster_settings,
   120	        geomBuffer,
   121	        octree_paths,
   122	        vox_centers,
   123	        vox_lengths,
   130	class _RasterizeVoxels(torch.autograd.Function):
   131	    @staticmethod
   132	    def forward(
   133	        ctx,
   134	        raster_settings,
   135	        geomBuffer,
   136	        octree_paths,
   137	        vox_centers,
   138	        vox_lengths,
   139	        geos,
   140	        rgbs,
   141	        subdiv_p,
   142	    ):
   143	
   144	        need_distortion = raster_settings.lambda_dist > 0
   145	
   146	        args = (
   147	            raster_settings.n_samp_per_vox,
   148	            raster_settings.image_width,
   149	            raster_settings.image_height,
   150	            raster_settings.tanfovx,
   151	            raster_settings.tanfovy,
   152	            raster_settings.cx,
   153	            raster_settings.cy,
   154	            raster_settings.w2c_matrix,
   155	            raster_settings.c2w_matrix,
   156	            raster_settings.bg_color,
   157	            raster_settings.need_depth,
   158	            need_distortion,
   159	            raster_settings.need_normal,
   160	            raster_settings.track_max_w,
   161	
   162	            octree_paths,
   163	            vox_centers,
   164	            vox_lengths,
   165	            geos,
   166	            rgbs,
   167	
   168	            geomBuffer,
   169	
   170	            raster_settings.debug,
   171	        )
   172	
   173	        num_rendered, binningBuffer, imgBuffer, out_color, out_depth, out_normal, out_T, max_w = _C.rasterize_voxels(*args)
   174	
   175	        # Keep relevant tensors for backward
   176	        ctx.raster_settings = raster_settings
   177	        ctx.num_rendered = num_rendered
   178	        ctx.save_for_backward(
   179	            octree_paths, vox_centers, vox_lengths,
   180	            geos, rgbs,
   186	    def backward(ctx, dL_dout_color, dL_dout_depth, dL_dout_normal, dL_dout_T, dL_dmax_w):
   187	        # Restore necessary values from context
   188	        raster_settings = ctx.raster_settings
   189	        num_rendered = ctx.num_rendered
   190	        octree_paths, vox_centers, vox_lengths, \
   191	            geos, rgbs, \
   192	            geomBuffer, binningBuffer, imgBuffer, out_T, out_depth, out_normal = ctx.saved_tensors
   193	
   194	        args = (
   195	            num_rendered,
   196	            raster_settings.n_samp_per_vox,
   197	            raster_settings.image_width,
   198	            raster_settings.image_height,
   199	            raster_settings.tanfovx,
   200	            raster_settings.tanfovy,
   201	            raster_settings.cx,
   202	            raster_settings.cy,
   203	            raster_settings.w2c_matrix,
   204	            raster_settings.c2w_matrix,
   205	            raster_settings.bg_color,
   206	
   207	            octree_paths,
   208	            vox_centers,
   209	            vox_lengths,
   210	            geos,
   211	            rgbs,
   212	
   213	            geomBuffer,
   214	            binningBuffer,
   215	            imgBuffer,
   216	            out_T,
   217	
   218	            dL_dout_color,
   219	            dL_dout_depth,
   220	            dL_dout_normal,
   221	            dL_dout_T,
   222	
   223	            raster_settings.lambda_R_concen,
   224	            raster_settings.gt_color,
   225	            raster_settings.lambda_ascending,
   226	            raster_settings.lambda_dist,
   227	            raster_settings.need_depth,
   228	            raster_settings.need_normal,
   229	            out_depth,
   230	            out_normal,
   231	
   232	            raster_settings.debug,
   233	        )
   234	
   235	        dL_dgeos, dL_drgbs, subdiv_p_bw = _C.rasterize_voxels_backward(*args)
   236	
