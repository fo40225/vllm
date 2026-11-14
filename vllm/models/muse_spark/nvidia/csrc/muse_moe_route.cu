// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Routing data of the CUTLASS grouped GEMMs of one expert-parallel rank, from
// the global top-k ids. It replaces the remap of the ids to local ones and the
// stock get_cutlass_moe_mm_data (vLLM csrc/quantization/w8a8/cutlass/moe/
// moe_data.cu) for the NVFP4 path, with the same outputs, but:
//  * the remote routes are never visited more than once, and by no atomic: the
//    stock sort lets one block place all of them (seven in eight under EP8)
//    one atomic at a time, which is most of its time;
//  * three short kernels spread over the whole GPU (count, scan, place) replace
//    a block per expert that scans every route;
//  * the map of a remote route is -1 (the stock one is the number of routes),
//    which is what the consumers of the map test for.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/ops/zeros.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "muse_c_ops.h"

namespace muse_spark {

namespace {

constexpr int kCountThreads = 256;
constexpr int kMaxRouteExperts = 1024;
// The grouped GEMM pads the block scales of every expert to whole 128-row tiles.
constexpr int kScaleRowTile = 128;

// The local expert of a route, or -1 for a remote or invalid one.
__device__ __forceinline__ int local_expert(int32_t const* __restrict__ topk_ids,
                                            int32_t const* __restrict__ expert_map,
                                            int map_size, int num_experts,
                                            int64_t route) {
  int const id = topk_ids[route];
  if (expert_map == nullptr) {
    return (id >= 0 && id < num_experts) ? id : -1;
  }
  if (id < 0 || id >= map_size) {
    return -1;
  }
  int const local = expert_map[id];
  return (local >= 0 && local < num_experts) ? local : -1;
}

// counts[e] += number of routes to local expert e. `counts` is zero on entry.
__global__ void __launch_bounds__(kCountThreads)
    count_routes(int32_t const* __restrict__ topk_ids,
                 int32_t const* __restrict__ expert_map, int map_size,
                 int64_t num_routes, int num_experts, int32_t* counts) {
  extern __shared__ int32_t histogram[];
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    histogram[e] = 0;
  }
  __syncthreads();
  for (int64_t route = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       route < num_routes; route += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int const e = local_expert(topk_ids, expert_map, map_size, num_experts, route);
    if (e >= 0) {
      atomicAdd(&histogram[e], 1);
    }
  }
  __syncthreads();
  for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
    if (histogram[e] != 0) {
      atomicAdd(&counts[e], histogram[e]);
    }
  }
}

// Inclusive sum of `value` over the threads of the block (blockDim <= 1024).
__device__ __forceinline__ int block_inclusive_scan(int value, int* warp_sums) {
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    int const other = __shfl_up_sync(0xffffffffu, value, offset);
    if (lane >= offset) {
      value += other;
    }
  }
  if (lane == 31) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  int prefix = 0;
  for (int w = 0; w < warp; ++w) {
    prefix += warp_sums[w];
  }
  __syncthreads();
  return value + prefix;
}

// One block: the offsets of the experts and the problem sizes of the two GEMMs.
__global__ void scan_experts(int32_t const* __restrict__ counts, int num_experts,
                             int n, int k, int32_t* expert_offsets,
                             int32_t* blockscale_offsets,
                             int32_t* problem_sizes1, int32_t* problem_sizes2) {
  __shared__ int warp_sums[32];
  int const e = threadIdx.x;
  int const count = e < num_experts ? counts[e] : 0;
  int const padded = (count + kScaleRowTile - 1) / kScaleRowTile * kScaleRowTile;
  int const end = block_inclusive_scan(count, warp_sums);
  int const padded_end = block_inclusive_scan(padded, warp_sums);
  if (e < num_experts) {
    expert_offsets[e] = end - count;
    blockscale_offsets[e] = padded_end - padded;
    // GEMM1 is [m, 2n, k] (gate and up), GEMM2 [m, k, n].
    problem_sizes1[e * 3] = count;
    problem_sizes1[e * 3 + 1] = 2 * n;
    problem_sizes1[e * 3 + 2] = k;
    problem_sizes2[e * 3] = count;
    problem_sizes2[e * 3 + 1] = k;
    problem_sizes2[e * 3 + 2] = n;
  }
  if (e == num_experts - 1) {
    expert_offsets[num_experts] = end;
    blockscale_offsets[num_experts] = padded_end;
  }
}

// Places every local route in its expert's rows, and clears the rows past them.
// `cursor` is zero on entry. The order of the routes inside an expert follows
// the atomics, as in the stock sort.
__global__ void __launch_bounds__(kCountThreads)
    place_routes(int32_t const* __restrict__ topk_ids,
                 int32_t const* __restrict__ expert_map, int map_size,
                 int64_t num_routes, int top_k, int num_experts,
                 int32_t const* __restrict__ expert_offsets, int32_t* cursor,
                 int32_t* a_map, int32_t* c_map) {
  int const local_rows = expert_offsets[num_experts];
  for (int64_t route = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       route < num_routes; route += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int const e = local_expert(topk_ids, expert_map, map_size, num_experts, route);
    if (e >= 0) {
      int const row = expert_offsets[e] + atomicAdd(&cursor[e], 1);
      a_map[row] = static_cast<int32_t>(route / top_k);
      c_map[route] = row;
    } else {
      c_map[route] = -1;
    }
    // Disjoint from the rows above, which are all below `local_rows`.
    if (route >= local_rows) {
      a_map[route] = -1;
    }
  }
}

}  // namespace

void moe_route_local(at::Tensor const& topk_ids,
                     std::optional<at::Tensor> const& expert_map,
                     int64_t num_local_experts, int64_t n, int64_t k,
                     at::Tensor& expert_offsets, at::Tensor& blockscale_offsets,
                     at::Tensor& problem_sizes1, at::Tensor& problem_sizes2,
                     at::Tensor& a_map, at::Tensor& c_map) {
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() &&
                  topk_ids.dim() == 2 && topk_ids.scalar_type() == at::kInt,
              "topk_ids must be a contiguous 2D int32 CUDA tensor");
  TORCH_CHECK(num_local_experts >= 1 && num_local_experts < kMaxRouteExperts,
              "num_local_experts must be in [1, ", kMaxRouteExperts, ")");
  int64_t const num_routes = topk_ids.numel();
  int64_t const top_k = topk_ids.size(1);
  TORCH_CHECK(num_routes < (int64_t{1} << 31), "too many routes");
  int const E = static_cast<int>(num_local_experts);

  auto check_out = [&](at::Tensor const& t, std::vector<int64_t> shape,
                       char const* name) {
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kInt &&
                    t.device() == topk_ids.device() && t.sizes() == shape,
                name, " must be a contiguous int32 tensor of shape ",
                at::IntArrayRef(shape), " on the device of topk_ids");
  };
  check_out(expert_offsets, {E + 1}, "expert_offsets");
  check_out(blockscale_offsets, {E + 1}, "blockscale_offsets");
  check_out(problem_sizes1, {E, 3}, "problem_sizes1");
  check_out(problem_sizes2, {E, 3}, "problem_sizes2");
  check_out(a_map, {num_routes}, "a_map");
  check_out(c_map, {num_routes}, "c_map");
  int32_t const* map_ptr = nullptr;
  int map_size = 0;
  if (expert_map.has_value()) {
    at::Tensor const& map = expert_map.value();
    TORCH_CHECK(map.is_cuda() && map.is_contiguous() && map.dim() == 1 &&
                    map.scalar_type() == at::kInt &&
                    map.device() == topk_ids.device(),
                "expert_map must be a contiguous 1D int32 tensor on the device "
                "of topk_ids");
    map_ptr = map.data_ptr<int32_t>();
    map_size = static_cast<int>(map.size(0));
  }

  const c10::cuda::CUDAGuard device_guard(topk_ids.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(topk_ids.get_device()).stream();
  // counts, then the cursors of the placement.
  at::Tensor scratch =
      at::zeros({2 * E}, topk_ids.options().dtype(at::kInt));
  int32_t* counts = scratch.data_ptr<int32_t>();
  int32_t* cursor = counts + E;

  if (num_routes > 0) {
    int const blocks = static_cast<int>(std::min<int64_t>(
        (num_routes + kCountThreads - 1) / kCountThreads, 1024));
    count_routes<<<blocks, kCountThreads, E * sizeof(int32_t), stream>>>(
        topk_ids.data_ptr<int32_t>(), map_ptr, map_size, num_routes, E, counts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  int const scan_threads = (E + 31) / 32 * 32;
  scan_experts<<<1, scan_threads, 0, stream>>>(
      counts, E, static_cast<int>(n), static_cast<int>(k),
      expert_offsets.data_ptr<int32_t>(), blockscale_offsets.data_ptr<int32_t>(),
      problem_sizes1.data_ptr<int32_t>(), problem_sizes2.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (num_routes > 0) {
    int const blocks = static_cast<int>(std::min<int64_t>(
        (num_routes + kCountThreads - 1) / kCountThreads, 1024));
    place_routes<<<blocks, kCountThreads, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), map_ptr, map_size, num_routes,
        static_cast<int>(top_k), E, expert_offsets.data_ptr<int32_t>(), cursor,
        a_map.data_ptr<int32_t>(), c_map.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

}  // namespace muse_spark
