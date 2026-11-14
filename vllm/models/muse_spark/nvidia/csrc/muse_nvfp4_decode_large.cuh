// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Preparation launch of the large-batch engine of the NVFP4 decode MoE
// (muse_nvfp4_decode_moe.cu includes this file inside its anonymous namespace,
// after muse_nvfp4_decode_persist.cuh). With more than 8 tokens the quantized
// activations no longer fit next to the weight ring in shared memory and the
// plan is too large to build in every CTA, so one small launch precedes the
// persistent kernel (k_persistent<kLargeMaxTokens, true>):
//
//   * CTAs 1..: one warp per token row, quantizing the bf16 activations to
//     NVFP4 into global scratch (codes [T, 2048], linear scales [T, 256]) with
//     the arithmetic of the in-kernel quantization, i.e. of the stock op;
//   * CTA 0: the plan of the layer in global memory (PlanLarge) and the route ->
//     row map, deterministically and without atomics. The routes of local expert
//     e are numbered in route order (rank), an entry is a block of up to
//     kEntryCols of them, entries are ordered by expert and block. (The order
//     inside an entry does not matter to the results: a route's row does not
//     depend on its column.)
//
// It is a programmatic dependent of the kernel before it (the routing kernel, or
// the pre-expert norm) and lets the persistent kernel launch right away; that
// kernel waits for this one to finish before it reads the plan or the scratch.

constexpr int kPrepThreads = 1024;
constexpr int kPrepRowsPerCta = kPrepThreads / 32;

// Inclusive sum of `value` over the threads of the block; all of the
// kPrepThreads threads must call it. `warp_sums` holds 32 ints.
__device__ __forceinline__ int prep_block_scan(int value, int* warp_sums) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int other = __shfl_up_sync(0xffffffffu, value, offset);
    if (lane >= offset) value += other;
  }
  if (lane == 31) warp_sums[warp] = value;
  __syncthreads();
  int prefix = 0;
  for (int w = 0; w < warp; ++w) prefix += warp_sums[w];
  __syncthreads();
  return value + prefix;
}

__global__ void __launch_bounds__(kPrepThreads, 1) k_prep(const DecodeParams p) {
  __shared__ __align__(16) uint8_t wc[kLargeMaxLocal][32];  // routes of expert, warp
  __shared__ int count_of[kLargeMaxLocal];
  __shared__ int entry_end[kLargeMaxLocal];   // inclusive scan of the entry counts
  __shared__ int entry_base[kLargeMaxLocal];  // first entry of the expert
  __shared__ int col_base[kLargeMaxLocal];    // first hq row of the expert
  __shared__ int warp_sums[32];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  // The persistent kernel may launch now: it waits for this kernel to complete
  // before it reads anything this kernel writes.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  // The ids and the activations come from the kernels before this one.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  if (blockIdx.x > 0) {
    const int row = (blockIdx.x - 1) * kPrepRowsPerCta + warp;
    if (row >= p.T) return;
    // The row is 512 vectors of 8 elements, a warp takes 32 of them at a time;
    // 8 loads in flight per lane keep the registers inside the 1024-thread limit.
    constexpr int kBatch = 8;
    const float gs0 = p.gs[0];
    uint8_t* codes = p.xq_buf + static_cast<size_t>(row) * kRowBytes;
    uint8_t* sf = p.xq_sf_buf + static_cast<size_t>(row) * kSfCols;
#pragma unroll
    for (int b = 0; b < 512 / 32 / kBatch; ++b) {
      PackedVec<__nv_bfloat16> vec[kBatch];
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        vec[i] = *reinterpret_cast<const PackedVec<__nv_bfloat16>*>(
            p.hidden + static_cast<size_t>(row) * kHidden +
            ((b * kBatch + i) * 32 + lane) * 8);
      }
#pragma unroll
      for (int i = 0; i < kBatch; ++i) {
        const int idx = (b * kBatch + i) * 32 + lane;  // 8-element vector of the row
        uint8_t* sf_out = (lane & 1) ? nullptr : sf + (idx >> 1);
        const uint32_t c = cvt_warp_fp16_to_fp4<__nv_bfloat16, 2>(vec[i], gs0, sf_out);
        *reinterpret_cast<uint32_t*>(codes + idx * 4) = c;
      }
    }
    return;
  }

  // ---- CTA 0: the plan ----
  PlanLarge* const plan = reinterpret_cast<PlanLarge*>(p.plan);
  const int num_routes = p.T * kTopK;
  int l = -1;  // local expert of route `tid`, or -1
  if (tid < num_routes) {
    const int id = p.topk_ids[tid];
    if (p.expert_map != nullptr) {
      l = (id >= 0 && id < p.num_global) ? p.expert_map[id] : -1;
    } else {
      l = id;
    }
    if (l < 0 || l >= p.num_local) l = -1;
  }
  uint32_t* const wc_words = reinterpret_cast<uint32_t*>(&wc[0][0]);
  for (int i = tid; i < kLargeMaxLocal * 8; i += kPrepThreads) wc_words[i] = 0u;
  __syncthreads();
  // Rank of the route among the routes of its expert in the warp, and the
  // number of those per warp; a route's rank is its in-warp rank plus the
  // counts of the earlier warps.
  const unsigned same = __match_any_sync(0xffffffffu, l);
  const int warp_rank = __popc(same & ((1u << lane) - 1u));
  if (l >= 0 && warp_rank == 0) wc[l][warp] = static_cast<uint8_t>(__popc(same));
  __syncthreads();
  int rank = warp_rank;
  if (l >= 0) {
    for (int w = 0; w < warp; ++w) rank += wc[l][w];
  }
  int count = 0;
  if (tid < kLargeMaxLocal) {
    for (int w = 0; w < 32; ++w) count += wc[tid][w];
  }
  const int n_blocks = (count + kEntryCols - 1) / kEntryCols;
  const int entries_end = prep_block_scan(n_blocks, warp_sums);
  const int cols_end = prep_block_scan(count, warp_sums);
  if (tid < kLargeMaxLocal) {
    count_of[tid] = count;
    entry_end[tid] = entries_end;
    entry_base[tid] = entries_end - n_blocks;
    col_base[tid] = cols_end - count;
  }
  __syncthreads();
  const int n_entries = entry_end[kLargeMaxLocal - 1];
  if (tid == 0) {
    plan->n_entries = n_entries;
    plan->n_cols = col_base[kLargeMaxLocal - 1] + count_of[kLargeMaxLocal - 1];
  }
  if (tid < n_entries) {
    // The expert of entry `tid`: the first one whose entries end past it.
    int lo = 0, hi = kLargeMaxLocal - 1;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (entry_end[mid] > tid) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    const int block = tid - entry_base[lo];
    plan->expert[tid] = lo;
    plan->cnt[tid] = min(kEntryCols, count_of[lo] - block * kEntryCols);
    plan->col0[tid] = col_base[lo] + block * kEntryCols;
  }
  if (tid < num_routes) {
    if (l >= 0) {
      plan->route[entry_base[l] + rank / kEntryCols][rank % kEntryCols] = tid;
    }
    p.perm_idx[tid] = l >= 0 ? tid : -1;
  }
}
