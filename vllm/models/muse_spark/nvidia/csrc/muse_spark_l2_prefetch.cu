// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Asynchronous L2 prefetch of weight tensors for the decode path. One thread per
// CTA issues cp.async.bulk.prefetch.L2 for its share of the chunks (chunk_kib) and
// exits. The requests queue up behind the memory system, so the kernel lasts about
// as long as the transfer (about 100 us for the 55 MB of the QKV and O weights with
// 4 CTAs); it holds a few SMs with one thread each and writes nothing.

// Only c10 headers: <ATen/cuda/CUDAContext.h> needs cusparse.h and
// cusolverDn.h, which the stock runtime images do not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "muse_c_ops.h"

namespace muse_spark {
namespace l2_pf {

static constexpr int kMaxRegions = 4;

struct Regions {
  const char* ptr[kMaxRegions];
  int64_t bytes[kMaxRegions];
  int num;
};

__global__ void l2_prefetch_kernel(Regions regions, int chunk_bytes,
                                   bool evict_last) {
  if (threadIdx.x != 0) {
    return;
  }
  uint64_t policy = 0;
  if (evict_last) {
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;"
                 : "=l"(policy));
  }
  // Each region is chunked on its own, so no byte between two tensors is
  // skipped and none outside a tensor is touched (up to the 16 B granule).
  for (int i = 0; i < regions.num; ++i) {
    for (int64_t off = static_cast<int64_t>(blockIdx.x) * chunk_bytes;
         off < regions.bytes[i];
         off += static_cast<int64_t>(gridDim.x) * chunk_bytes) {
      const char* src = regions.ptr[i] + off;
      const int64_t left = regions.bytes[i] - off;
      const int64_t span = left < chunk_bytes ? left : chunk_bytes;
      const uint32_t len = static_cast<uint32_t>(span & ~int64_t{15});
      if (len == 0) {
        continue;
      }
      if (evict_last) {
        asm volatile(
            "cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::
                "l"(src),
            "r"(len), "l"(policy)
            : "memory");
      } else {
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(src),
                     "r"(len)
                     : "memory");
      }
    }
  }
}

}  // namespace l2_pf

void l2_prefetch(std::vector<at::Tensor> const& tensors, int64_t num_ctas,
                 int64_t chunk_kib, bool evict_last) {
  TORCH_CHECK(!tensors.empty() &&
                  tensors.size() <= l2_pf::kMaxRegions,
              "l2_prefetch takes 1 to ", l2_pf::kMaxRegions, " tensors");
  TORCH_CHECK(num_ctas >= 1 && num_ctas <= 1024, "bad num_ctas ", num_ctas);
  TORCH_CHECK(chunk_kib >= 1 && chunk_kib <= 4096, "bad chunk_kib ",
              chunk_kib);
  l2_pf::Regions regions{};
  for (at::Tensor const& t : tensors) {
    TORCH_CHECK(t.is_cuda(), "l2_prefetch needs CUDA tensors");
    TORCH_CHECK(t.is_contiguous(), "l2_prefetch needs contiguous tensors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "l2_prefetch needs 16 B aligned tensors");
    regions.ptr[regions.num] = static_cast<const char*>(t.data_ptr());
    regions.bytes[regions.num] = static_cast<int64_t>(t.nbytes());
    ++regions.num;
  }
  const c10::cuda::CUDAGuard device_guard(tensors[0].get_device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(tensors[0].get_device()).stream();
  l2_pf::l2_prefetch_kernel<<<static_cast<unsigned>(num_ctas), 32, 0,
                                    stream>>>(
      regions, static_cast<int>(chunk_kib * 1024), evict_last);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace muse_spark
