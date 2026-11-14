// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Routing-aware L2 prefetch of the experts that the NVFP4 decode kernel
// (muse_nvfp4_decode_moe.cu) is about to read. It runs on a side stream behind
// the routing kernel: it maps this rank's routes to its local experts exactly
// like the kernel's plan does (distinct local experts in order of first
// appearance) and issues cp.async.bulk.prefetch.L2 for the weights and block
// scales of those experts in the order the persistent kernel consumes them
// (GEMM1 of every entry, then GEMM2 of every entry), up to a byte budget. The
// kernel finds the first of them in L2 instead of waiting for HBM; the prefetch
// only moves cache lines and writes nothing.
//
// Consumption order of one expert (see muse_nvfp4_decode_persist.cuh): the 256
// GEMM1 units of 16 rows in groups of 8 (one group = 128 consecutive rows of
// the up half, of the gate half, and the 32 KiB of block scales of each), then
// the 256 GEMM2 units in groups of 8 (128 consecutive rows of w2 and their
// 32 KiB of block scales). A group is cut into requests of 64 KiB (the rows)
// and 32 KiB (a block of scales), the unit of the prefetch: the CTAs issue the
// requests one thread each, so fewer, larger requests reach the memory system
// sooner.

// Only c10 headers: <ATen/cuda/CUDAContext.h> needs cusparse.h and
// cusolverDn.h, which the stock runtime images do not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

#include "muse_c_ops.h"

namespace muse_spark {
namespace expert_pf {

constexpr int kHidden = 4096;
constexpr int kInter = 4096;
constexpr int kTopK = 8;
constexpr int kRowBytes = kHidden / 2;          // packed fp4 bytes of a row
constexpr int kW13Rows = 2 * kInter;
constexpr int kSfCols = kHidden / 16;           // block scale bytes of a row
constexpr int kMaxRoutes = 64;                  // 8 tokens x top-8
constexpr int kBlockThreads = 64;               // one thread per route

constexpr int kRowReqBytes = 64 * 1024;
constexpr int kScaleReqBytes = 32 * 1024;       // the scales of 128 rows
constexpr int kRowsPerGroup = 128;              // 8 units of 16 rows
constexpr int kGroups = 32;                     // 4096 / 128 (and 8192 / 256)
constexpr int kRowReqs = kRowsPerGroup * kRowBytes / kRowReqBytes;  // 4
// Requests of one group: GEMM1 = up rows, gate rows, up scales, gate scales;
// GEMM2 = rows, scales.
constexpr int kReqs1 = 2 * kRowReqs + 2;        // 10
constexpr int kReqs2 = kRowReqs + 1;            // 5
constexpr int kExpertReqs1 = kGroups * kReqs1;  // 320 (18 MiB)
constexpr int kExpertReqs2 = kGroups * kReqs2;  // 160 (9 MiB)
constexpr int kGroupBytes1 = 2 * kRowReqs * kRowReqBytes + 2 * kScaleReqBytes;
constexpr int kGroupBytes2 = kRowReqs * kRowReqBytes + kScaleReqBytes;
static_assert(kSfCols * kRowsPerGroup == kScaleReqBytes,
              "the scales of a 128-row group are one swizzled-layout block");

struct Params {
  const int* topk_ids;      // [T, 8] global expert ids
  const int* expert_map;    // global -> local (-1), or null (ids are local)
  const uint8_t* w13;       // [E, 8192, 2048]
  const uint8_t* sf13;      // [E, 8192, 256] swizzled
  const uint8_t* w2;        // [E, 4096, 2048]
  const uint8_t* sf2;       // [E, 4096, 256] swizzled
  int num_routes;           // T * 8
  int num_global;
  int num_local;
  int up_first;
  int64_t budget_bytes;
  int wait_for_primary;
  int64_t* dump;            // tests: (address, size) of the requests, unused
  int64_t dump_size;
};

// Bytes of the requests before request `k` in the consumption order of `n`
// entries: GEMM1 requests of every entry first, then GEMM2.
__device__ __forceinline__ int64_t bytes_before(int n, int k) {
  const int n1 = n * kExpertReqs1;
  if (k <= n1) {
    const int group = k / kReqs1;
    const int j = k - group * kReqs1;
    const int rows = j < 2 * kRowReqs ? j : 2 * kRowReqs;
    return static_cast<int64_t>(group) * kGroupBytes1 +
           static_cast<int64_t>(rows) * kRowReqBytes +
           static_cast<int64_t>(j - rows) * kScaleReqBytes;
  }
  const int k2 = k - n1;
  const int group = k2 / kReqs2;
  const int j = k2 - group * kReqs2;
  const int rows = j < kRowReqs ? j : kRowReqs;
  return static_cast<int64_t>(n) * kGroups * kGroupBytes1 +
         static_cast<int64_t>(group) * kGroupBytes2 +
         static_cast<int64_t>(rows) * kRowReqBytes +
         static_cast<int64_t>(j - rows) * kScaleReqBytes;
}

// The address and size of request `k` of the consumption order of `n` entries
// whose local experts are `expert[0..n)`.
__device__ __forceinline__ const uint8_t* request(const Params& p,
                                                  const int* expert, int n,
                                                  int k, uint32_t* len) {
  const int n1 = n * kExpertReqs1;
  if (k < n1) {
    const int g = k / kExpertReqs1;
    const int r = k - g * kExpertReqs1;
    const int group = r / kReqs1;
    const int j = r - group * kReqs1;
    const int64_t e = expert[g];
    const int up_base = p.up_first ? 0 : kInter;
    const int gate_base = p.up_first ? kInter : 0;
    const uint8_t* w = p.w13 + e * (static_cast<int64_t>(kW13Rows) * kRowBytes);
    const uint8_t* sf = p.sf13 + e * (static_cast<int64_t>(kW13Rows) * kSfCols);
    if (j < 2 * kRowReqs) {
      const int base = j < kRowReqs ? up_base : gate_base;
      *len = kRowReqBytes;
      return w +
             (static_cast<int64_t>(base) + kRowsPerGroup * group) * kRowBytes +
             static_cast<int64_t>(j % kRowReqs) * kRowReqBytes;
    }
    // Swizzled scales: the 128-row block b of a half is 32 KiB at b * 32 KiB.
    const int base = (j == 2 * kRowReqs) ? up_base : gate_base;
    *len = kScaleReqBytes;
    return sf +
           static_cast<int64_t>(base / kRowsPerGroup + group) * kScaleReqBytes;
  }
  const int k2 = k - n1;
  const int g = k2 / kExpertReqs2;
  const int r = k2 - g * kExpertReqs2;
  const int group = r / kReqs2;
  const int j = r - group * kReqs2;
  const int64_t e = expert[g];
  if (j < kRowReqs) {
    *len = kRowReqBytes;
    return p.w2 +
           (e * kHidden + static_cast<int64_t>(kRowsPerGroup) * group) *
               kRowBytes +
           static_cast<int64_t>(j) * kRowReqBytes;
  }
  *len = kScaleReqBytes;
  return p.sf2 + e * (static_cast<int64_t>(kHidden) * kSfCols) +
         static_cast<int64_t>(group) * kScaleReqBytes;
}

__global__ void __launch_bounds__(kBlockThreads)
    expert_prefetch_kernel(const Params p) {
  __shared__ int lid[kMaxRoutes];
  __shared__ int expert[kMaxRoutes];
  __shared__ uint32_t ballot[2];
  __shared__ int n_entries;

  const int tid = threadIdx.x;
  // The ids are written by the routing kernel in front of this one in the
  // stream. Nothing above touches global memory.
  if (p.wait_for_primary) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
  }
  int l = -1;
  if (tid < p.num_routes) {
    const int id = p.topk_ids[tid];
    if (p.expert_map != nullptr) {
      l = (id >= 0 && id < p.num_global) ? p.expert_map[id] : -1;
    } else {
      l = id;
    }
    if (l < 0 || l >= p.num_local) l = -1;
  }
  lid[tid] = l;
  __syncthreads();
  // One entry per distinct local expert, in order of first appearance (the
  // order of the kernel's plan; an expert with more routes than the kernel's
  // column limit only adds entries of the same expert, which are the same
  // bytes).
  bool creator = l >= 0;
  for (int j = 0; creator && j < tid; ++j) {
    creator = lid[j] != l;
  }
  const uint32_t b = __ballot_sync(0xffffffffu, creator);
  if ((tid & 31) == 0) ballot[tid >> 5] = b;
  __syncthreads();
  if (creator) {
    const uint32_t below = (1u << (tid & 31)) - 1u;
    const int entry = __popc(ballot[tid >> 5] & below) +
                      ((tid >> 5) ? __popc(ballot[0]) : 0);
    expert[entry] = l;
  }
  if (tid == 0) n_entries = __popc(ballot[0]) + __popc(ballot[1]);
  __syncthreads();
  const int n = n_entries;
  if (n == 0 || tid != 0) return;

  // The requests that fit the byte budget: the longest prefix of the order.
  int limit = 0;
  {
    int lo = 0;
    int hi = n * (kExpertReqs1 + kExpertReqs2);
    while (lo < hi) {
      const int mid = (lo + hi + 1) >> 1;
      if (bytes_before(n, mid) <= p.budget_bytes) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    limit = lo;
  }
  if (p.dump != nullptr && 2 * static_cast<int64_t>(limit) > p.dump_size) {
    limit = static_cast<int>(p.dump_size / 2);
  }
  // CTA b issues the requests b, b + G, b + 2 G, ...: they reach the memory
  // system in consumption order across the CTAs.
  for (int k = blockIdx.x; k < limit; k += gridDim.x) {
    uint32_t len = 0;
    const uint8_t* src = request(p, expert, n, k, &len);
    if (p.dump != nullptr) {
      p.dump[2 * k] = reinterpret_cast<int64_t>(src);
      p.dump[2 * k + 1] = len;
      continue;
    }
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src),
                 "r"(len)
                 : "memory");
  }
}

void check_u8(at::Tensor const& t, char const* name, int64_t rows,
              int64_t cols) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name,
              " must be a contiguous CUDA tensor");
  TORCH_CHECK(t.element_size() == 1 && t.dim() == 3 && t.size(1) == rows &&
                  t.size(2) == cols,
              name, " has the wrong layout");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name,
              " must be 16 B aligned");
}

}  // namespace expert_pf

void expert_prefetch(at::Tensor const& topk_ids,
                     std::optional<at::Tensor> const& expert_map,
                     at::Tensor const& w13, at::Tensor const& sf13,
                     at::Tensor const& w2, at::Tensor const& sf2, bool up_first,
                     int64_t num_ctas, int64_t budget_bytes, bool pdl,
                     std::optional<at::Tensor> const& dump) {
  using namespace expert_pf;
  TORCH_CHECK(topk_ids.is_cuda() && topk_ids.is_contiguous() &&
                  topk_ids.scalar_type() == at::kInt && topk_ids.dim() == 2 &&
                  topk_ids.size(1) == kTopK && topk_ids.size(0) >= 1 &&
                  topk_ids.size(0) * kTopK <= kMaxRoutes,
              "topk_ids must be a contiguous int32 [T, 8] CUDA tensor, T <= 8");
  const int64_t num_local = w13.size(0);
  TORCH_CHECK(num_local >= 1, "w13 holds no experts");
  check_u8(w13, "w13", kW13Rows, kRowBytes);
  check_u8(sf13, "sf13", kW13Rows, kSfCols);
  check_u8(w2, "w2", kHidden, kRowBytes);
  check_u8(sf2, "sf2", kHidden, kSfCols);
  TORCH_CHECK(sf13.size(0) == num_local && w2.size(0) == num_local &&
                  sf2.size(0) == num_local,
              "the weights disagree on the number of experts");
  TORCH_CHECK(num_ctas >= 1 && num_ctas <= 256, "bad num_ctas ", num_ctas);
  TORCH_CHECK(budget_bytes >= 0, "bad budget_bytes ", budget_bytes);

  Params p{};
  p.topk_ids = reinterpret_cast<const int*>(topk_ids.data_ptr());
  p.expert_map = nullptr;
  p.num_global = 0;
  if (expert_map.has_value()) {
    const at::Tensor& m = *expert_map;
    TORCH_CHECK(m.is_cuda() && m.is_contiguous() &&
                    m.scalar_type() == at::kInt && m.dim() == 1,
                "expert_map must be a contiguous int32 CUDA vector");
    p.expert_map = reinterpret_cast<const int*>(m.data_ptr());
    p.num_global = static_cast<int>(m.size(0));
  }
  p.w13 = reinterpret_cast<const uint8_t*>(w13.data_ptr());
  p.sf13 = reinterpret_cast<const uint8_t*>(sf13.data_ptr());
  p.w2 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
  p.sf2 = reinterpret_cast<const uint8_t*>(sf2.data_ptr());
  p.num_routes = static_cast<int>(topk_ids.size(0) * kTopK);
  p.num_local = static_cast<int>(num_local);
  p.up_first = up_first ? 1 : 0;
  p.budget_bytes = budget_bytes;
  p.wait_for_primary = pdl ? 1 : 0;
  p.dump = nullptr;
  p.dump_size = 0;
  if (dump.has_value()) {
    TORCH_CHECK(dump->is_cuda() && dump->is_contiguous() &&
                    dump->scalar_type() == at::kLong && dump->dim() == 1,
                "dump must be a contiguous int64 CUDA vector");
    p.dump = reinterpret_cast<int64_t*>(dump->data_ptr());
    p.dump_size = dump->size(0);
  }

  const c10::cuda::OptionalCUDAGuard guard(topk_ids.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(num_ctas));
  cfg.blockDim = dim3(kBlockThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  if (pdl) {
    // A programmatic dependent of the routing kernel: it is launched while that
    // kernel finishes, and waits for it (griddepcontrol.wait) before it reads
    // the ids.
    cfg.attrs = attr;
    cfg.numAttrs = 1;
  }
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, expert_prefetch_kernel, p));
}

}  // namespace muse_spark
