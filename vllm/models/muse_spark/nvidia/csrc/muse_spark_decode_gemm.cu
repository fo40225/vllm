// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Skinny bf16 GEMM of the Muse Spark decode layers (the QKV projection and the
// sharded pooler / unpooler of a TP8 rank; 1 .. 128 rows, the model uses it up
// to 64):
//   out[t][n] = bf16(sum_k x[t][k] * w[n][k]),  x [T, K], w [N, K], out [T, N]
// fp32 accumulation, one rounding to bf16. cuBLAS runs these as a cutlass
// tensor-op GEMM that splits K over the CTAs and a second kernel
// (cublasLt::splitKreduce_kernel, 2 .. 3 us, with a launch in between) that
// adds the partial tiles; a variant without the split has too few CTAs to read
// the weights at the memory rate (N = 512 is 8 tiles of 64 columns).
//
// Here K is split over a thread block cluster of SPLIT CTAs (<= 8) and the
// partial tiles are added in the same kernel: every CTA stores its fp32 partial
// tile into a global scratch buffer (L2 resident), one cluster barrier orders
// the stores before the loads, and CTA o of the cluster adds the SPLIT pieces
// of the rows [o * BM / SPLIT, (o + 1) * BM / SPLIT) and stores them. No second
// launch, no atomics, and the cluster launch guarantees that the CTAs that wait
// for each other are resident together. (Distributed shared memory was the
// first choice, but a store of 8 KB into another CTA takes 2.3 us on this GPU,
// about 3.5 GB/s per CTA; the round trip through the scratch in L2 is about
// 1 us.)
//
// Decomposition: CTA (nt, s) owns the output columns [nt * BN, (nt + 1) * BN)
// and the K slice [s * K / split, (s + 1) * K / split) for ALL rows (MT tiles
// of 16 rows, padding rows are zero), so every weight byte is read by exactly
// one CTA. K is streamed through a STAGES deep cp.async ring of 64-wide tiles
// (128-byte rows, 16-byte chunks XOR-swizzled by the row for ldmatrix);
// mma.sync m16n8k16 bf16 with fp32 accumulators.
//
// Order of the additions of one output element, independent of the tile
// configuration, of the grid and of the run, and of T as long as SPLIT is the same
// (SPLIT changes it; the QKV projection takes SPLIT 4 up to 32 rows and 8 above, so
// its bits step at 33 rows, as the algorithm cuBLAS picks depends on the row count):
//   slice:   the k16 steps of the slice in ascending order (the accumulation
//            of one mma of the tensor core)
//   cluster: p_0 + p_1, then + p_2, ... in ascending cluster rank, fp32
//   then one round-to-nearest-even to bf16.
// This is not the order of cuBLAS, so the result differs from the cuBLAS one in
// the last bits; the partial sums stay fp32 (the cuBLAS split-K path keeps them
// in a bf16 workspace for some shapes).
//
// PDL: with `pdl` the kernel is a programmatic dependent of the kernel in front
// of it and calls griddepcontrol.wait before it reads x (and before it writes
// out); with `early_w` the first weight tiles are requested before that wait
// (the weights depend on nothing). The kernel never triggers its own dependents
// early: they start when it has finished, as after a cuBLAS kernel.

// Only c10 headers: <ATen/cuda/CUDAContext.h> needs cusparse.h and
// cusolverDn.h, which the stock runtime images do not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <vector>

#include "muse_c_ops.h"

namespace muse_spark {
namespace dgemm {

constexpr int kBK = 64;  // bf16 elements per tile row (128 bytes)
constexpr int kMaxSplit = 8;
constexpr int kMaxDevices = 64;  // devices of one process

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src,
                                           int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst),
               "l"(src), "r"(src_bytes)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4],
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cluster_arrive_release() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void cluster_wait_acquire() {
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void st_global_cg_v2(float* p, float a, float b) {
  asm volatile("st.global.cg.v2.f32 [%0], {%1, %2};\n" ::"l"(p), "f"(a), "f"(b)
               : "memory");
}
__device__ __forceinline__ float4 ld_global_cg_v4(const float* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// MT: tiles of 16 rows; BN: output columns; the CTA has WM x WN warps, a warp
// owns MT / WM row tiles and BN / WN columns.
template <int MT, int BN, int WM, int WN, int STAGES, int SPLIT>
__global__ void __launch_bounds__(WM* WN * 32, 1)
    decode_gemm_kernel(const __nv_bfloat16* __restrict__ x,
                       const __nv_bfloat16* __restrict__ w,
                       __nv_bfloat16* __restrict__ out, int T, int N, int K,
                       float* __restrict__ scratch, int ldx, int ldo,
                       bool early_w) {
  constexpr int BM = MT * 16;
  constexpr int NTHREADS = WM * WN * 32;
  constexpr int MT_W = MT / WM;
  constexpr int NCOL_W = BN / WN;
  constexpr int NT_W = NCOL_W / 8;
  constexpr int X_BYTES = BM * 128;
  constexpr int W_BYTES = BN * 128;
  constexpr int STAGE_BYTES = X_BYTES + W_BYTES;
  constexpr int RB = BM / SPLIT;  // rows of the tile that one CTA adds up
  static_assert(MT % WM == 0 && BN % (WN * 8) == 0, "warp tiling");
  static_assert(SPLIT == 1 || (BM % SPLIT == 0 && SPLIT <= kMaxSplit),
                "each CTA of the cluster owns whole rows");

  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t smem_base = smem_u32(smem);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp / WN;
  const int wn = warp % WN;
  const int s = blockIdx.x % SPLIT;  // rank in the cluster of SPLIT CTAs
  const int n0 = (blockIdx.x / SPLIT) * BN;
  const int k_per = K / SPLIT;
  const int iters = k_per / kBK;

  const __nv_bfloat16* wbase = w + static_cast<size_t>(n0) * K + s * k_per;
  const __nv_bfloat16* xbase = x + s * k_per;

  // One 16-byte chunk per (row, chunk) pair, NTHREADS pairs per round.
  auto load_w = [&](int stage, int it) {
    const uint32_t dst = smem_base + stage * STAGE_BYTES + X_BYTES;
#pragma unroll
    for (int i = 0; i < (BN * 8 + NTHREADS - 1) / NTHREADS; ++i) {
      const int c = tid + i * NTHREADS;
      if (BN * 8 % NTHREADS == 0 || c < BN * 8) {
        const int row = c >> 3;
        const int ch = c & 7;
        cp_async16(dst + row * 128 + ((ch ^ (row & 7)) << 4),
                   wbase + static_cast<size_t>(row) * K + it * kBK + ch * 8,
                   16);
      }
    }
  };
  auto load_x = [&](int stage, int it) {
    const uint32_t dst = smem_base + stage * STAGE_BYTES;
#pragma unroll
    for (int i = 0; i < (BM * 8 + NTHREADS - 1) / NTHREADS; ++i) {
      const int c = tid + i * NTHREADS;
      if (BM * 8 % NTHREADS == 0 || c < BM * 8) {
        const int row = c >> 3;
        const int ch = c & 7;
        const bool ok = row < T;
        cp_async16(dst + row * 128 + ((ch ^ (row & 7)) << 4),
                   xbase + static_cast<size_t>(ok ? row : 0) * ldx + it * kBK +
                       ch * 8,
                   ok ? 16 : 0);
      }
    }
  };

  // Prologue. The weights do not depend on the kernel in front: with early_w
  // they are requested before the wait.
  if (early_w) {
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
      if (st < iters) load_w(st, st);
    }
  }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif
#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    if (st < iters) {
      if (!early_w) load_w(st, st);
      load_x(st, st);
    }
    cp_async_commit();
  }

  float acc[MT_W][NT_W][4];
#pragma unroll
  for (int i = 0; i < MT_W; ++i)
#pragma unroll
    for (int j = 0; j < NT_W; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  for (int it = 0; it < iters; ++it) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int nxt = it + STAGES - 1;
      if (nxt < iters) {
        load_w(nxt % STAGES, nxt);
        load_x(nxt % STAGES, nxt);
      }
      cp_async_commit();
    }
    const uint32_t xs = smem_base + (it % STAGES) * STAGE_BYTES;
    const uint32_t ws = xs + X_BYTES;
#pragma unroll
    for (int kk = 0; kk < kBK / 16; ++kk) {
      uint32_t a[MT_W][4];
#pragma unroll
      for (int mt = 0; mt < MT_W; ++mt) {
        const int row =
            (wm * MT_W + mt) * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ch = kk * 2 + (lane >> 4);
        ldmatrix_x4(a[mt], xs + row * 128 + ((ch ^ (row & 7)) << 4));
      }
      uint32_t b[NT_W][2];
      if constexpr (NT_W % 2 == 0) {
#pragma unroll
        for (int p = 0; p < NT_W / 2; ++p) {
          const int row = wn * NCOL_W + (p * 2 + (lane >> 4)) * 8 + (lane & 7);
          const int ch = kk * 2 + ((lane >> 3) & 1);
          uint32_t r[4];
          ldmatrix_x4(r, ws + row * 128 + ((ch ^ (row & 7)) << 4));
          b[2 * p][0] = r[0];
          b[2 * p][1] = r[1];
          b[2 * p + 1][0] = r[2];
          b[2 * p + 1][1] = r[3];
        }
      } else {
#pragma unroll
        for (int j = 0; j < NT_W; ++j) {
          const int row = wn * NCOL_W + j * 8 + (lane & 7);
          const int ch = kk * 2 + ((lane >> 3) & 1);
          ldmatrix_x2(b[j], ws + row * 128 + ((ch ^ (row & 7)) << 4));
        }
      }
#pragma unroll
      for (int mt = 0; mt < MT_W; ++mt)
#pragma unroll
        for (int j = 0; j < NT_W; ++j)
          mma_bf16(acc[mt][j], a[mt], b[j][0], b[j][1]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();

  if constexpr (SPLIT == 1) {
#pragma unroll
    for (int mt = 0; mt < MT_W; ++mt) {
#pragma unroll
      for (int j = 0; j < NT_W; ++j) {
        const int r0 = (wm * MT_W + mt) * 16 + (lane >> 2);
        const int c = n0 + wn * NCOL_W + j * 8 + (lane & 3) * 2;
        if (r0 < T)
          *reinterpret_cast<uint32_t*>(out + static_cast<size_t>(r0) * ldo +
                                       c) =
              pack_bf16(acc[mt][j][0], acc[mt][j][1]);
        if (r0 + 8 < T)
          *reinterpret_cast<uint32_t*>(
              out + static_cast<size_t>(r0 + 8) * ldo + c) =
              pack_bf16(acc[mt][j][2], acc[mt][j][3]);
      }
    }
    return;
  }

  // The partial tile (rows < T) goes to the scratch buffer, [tile][s][row][BN]
  // fp32: a quad of lanes writes 32 contiguous bytes of a row.
  const size_t tile_id = blockIdx.x / SPLIT;
  float* mine = scratch + (tile_id * SPLIT + s) * BM * BN;
#pragma unroll
  for (int mt = 0; mt < MT_W; ++mt) {
#pragma unroll
    for (int j = 0; j < NT_W; ++j) {
      const int r0 = (wm * MT_W + mt) * 16 + (lane >> 2);
      const int c = wn * NCOL_W + j * 8 + (lane & 3) * 2;
      if (r0 < T)
        st_global_cg_v2(mine + r0 * BN + c, acc[mt][j][0], acc[mt][j][1]);
      if (r0 + 8 < T)
        st_global_cg_v2(mine + (r0 + 8) * BN + c, acc[mt][j][2],
                        acc[mt][j][3]);
    }
  }
  cluster_arrive_release();
  cluster_wait_acquire();

  // CTA s adds the SPLIT pieces of its rows in rank order and stores them.
  const float* tile = scratch + tile_id * SPLIT * BM * BN;
  constexpr int QPR = BN / 4;  // 4-column chunks per row
  for (int q = tid; q < RB * QPR; q += NTHREADS) {
    const int rr = q / QPR;
    const int q4 = q % QPR;
    const int row = s * RB + rr;
    if (row < T) {
      float4 v[SPLIT];
#pragma unroll
      for (int r = 0; r < SPLIT; ++r)
        v[r] = ld_global_cg_v4(tile + (r * BM + row) * BN + q4 * 4);
      float4 sum = v[0];
#pragma unroll
      for (int r = 1; r < SPLIT; ++r) {
        sum.x = __fadd_rn(sum.x, v[r].x);
        sum.y = __fadd_rn(sum.y, v[r].y);
        sum.z = __fadd_rn(sum.z, v[r].z);
        sum.w = __fadd_rn(sum.w, v[r].w);
      }
      uint2 packed;
      packed.x = pack_bf16(sum.x, sum.y);
      packed.y = pack_bf16(sum.z, sum.w);
      *reinterpret_cast<uint2*>(out + static_cast<size_t>(row) * ldo + n0 +
                                q4 * 4) = packed;
    }
  }
}

struct Config {
  int mt, bn, wm, wn, stages, split;
};

// (MT, BN, WM, WN, STAGES, SPLIT). pick_config chooses by the shape.
#define MUSE_DGEMM_CONFIGS(X) \
  X(1, 32, 1, 2, 5, 4) X(1, 32, 1, 2, 5, 8) X(1, 32, 1, 4, 4, 1)  \
  X(1, 32, 1, 4, 4, 2) X(1, 32, 1, 4, 4, 4) X(1, 32, 1, 4, 4, 8)  \
  X(1, 64, 1, 4, 4, 4) X(2, 32, 1, 4, 4, 8) X(2, 32, 2, 2, 4, 1)  \
  X(2, 32, 2, 2, 4, 4) X(2, 32, 2, 2, 4, 8) X(2, 32, 2, 2, 5, 4)  \
  X(2, 64, 2, 4, 4, 4) X(2, 128, 2, 4, 3, 8) X(3, 32, 1, 4, 4, 4)  \
  X(3, 32, 1, 4, 4, 8) X(3, 32, 3, 2, 4, 8) X(3, 128, 3, 4, 3, 8)  \
  X(4, 32, 2, 2, 4, 4) X(4, 32, 2, 2, 5, 8) X(4, 32, 4, 1, 4, 2)  \
  X(4, 32, 4, 1, 4, 4) X(4, 32, 4, 1, 4, 8) X(4, 64, 2, 4, 4, 4)  \
  X(4, 64, 4, 2, 4, 4) X(4, 128, 2, 4, 3, 8) X(4, 128, 4, 2, 3, 8)  \
  X(6, 32, 2, 2, 4, 8) X(6, 32, 3, 2, 4, 4) X(8, 32, 2, 2, 4, 4)  \
  X(8, 32, 4, 1, 4, 4) X(8, 32, 4, 1, 4, 8)

constexpr Config kConfigs[] = {
#define X(MT, BN, WM, WN, ST, SP) {MT, BN, WM, WN, ST, SP},
    MUSE_DGEMM_CONFIGS(X)
#undef X
};
constexpr int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

template <int MT, int BN, int WM, int WN, int STAGES, int SPLIT>
void launch(const __nv_bfloat16* x, const __nv_bfloat16* w,
            __nv_bfloat16* out, float* scratch, int T, int N, int K, int ldx,
            int ldo, bool early_w, bool pdl, cudaStream_t stream) {
  constexpr int smem = STAGES * (MT * 16 + BN) * 128;
  auto kernel = decode_gemm_kernel<MT, BN, WM, WN, STAGES, SPLIT>;
  // The attribute belongs to the function of a device: once per instantiation and device.
  static std::atomic<bool> configured[kMaxDevices];
  int device = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "decode_gemm: device ", device);
  if (!configured[device].load(std::memory_order_acquire)) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    configured[device].store(true, std::memory_order_release);
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / BN) * SPLIT);
  cfg.blockDim = dim3(WM * WN * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attrs[2];
  int na = 0;
  if (SPLIT > 1) {
    attrs[na].id = cudaLaunchAttributeClusterDimension;
    attrs[na].val.clusterDim.x = SPLIT;
    attrs[na].val.clusterDim.y = 1;
    attrs[na].val.clusterDim.z = 1;
    ++na;
  }
  if (pdl) {
    attrs[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, x, w, out, T, N, K, scratch,
                                    ldx, ldo, early_w));
}

inline int find_config(int mt, int bn, int wm, int wn, int stages, int split) {
  for (int i = 0; i < kNumConfigs; ++i) {
    const Config& c = kConfigs[i];
    if (c.mt == mt && c.bn == bn && c.wm == wm && c.wn == wn &&
        c.stages == stages && c.split == split)
      return i;
  }
  return -1;
}

// The tuned choice per (rows, N, K) (benchmarks/kernels/
// benchmark_muse_spark_decode_gemm.py, RTX PRO 6000 Blackwell). The K split is
// a property of the shape (and for the QKV projection of the row count: 4 up to
// 32 rows, 8 above), the tile shape, which does not change a bit, follows the
// row count. Shapes that are not the TP8 ones of the model get a generic
// choice.
inline bool is_tuned_shape(int N, int K) {
  return (N == 512 && K == 8192) || (N == 1024 && K == 4096) ||
         (N == 2304 && K == 8192);
}

inline int pick_config(int T, int N, int K) {
  const int mt = (T + 15) / 16;
  int cfg = -1;
  if (N == 512 && K == 8192) {  // pooler
    cfg = mt == 1   ? find_config(1, 32, 1, 4, 4, 8)
          : mt == 2 ? find_config(2, 32, 2, 2, 4, 8)
          : mt == 3 ? find_config(3, 32, 1, 4, 4, 8)
                    : find_config(4, 32, 4, 1, 4, 8);
  } else if (N == 1024 && K == 4096) {  // unpooler
    cfg = mt == 1   ? find_config(1, 32, 1, 4, 4, 4)
          : mt == 2 ? find_config(2, 32, 2, 2, 4, 4)
          : mt == 3 ? find_config(3, 32, 1, 4, 4, 4)
                    : find_config(4, 32, 4, 1, 4, 4);
  } else if (N == 2304 && K == 8192) {  // QKV
    cfg = mt == 1   ? find_config(1, 64, 1, 4, 4, 4)
          : mt == 2 ? find_config(2, 64, 2, 4, 4, 4)
          : mt == 3 ? find_config(3, 128, 3, 4, 3, 8)
                    : find_config(4, 128, 2, 4, 3, 8);
  }
  if (mt > 4) cfg = -1;
  if (cfg >= 0) return cfg;
  // Generic: the smallest tile that holds the rows, 32 columns, a split of K.
  for (int i = 0; i < kNumConfigs; ++i) {
    const Config& c = kConfigs[i];
    if (c.mt >= mt && c.bn == 32 && c.split >= 2 && N % c.bn == 0 &&
        K % (c.split * kBK) == 0 && (cfg < 0 || c.mt < kConfigs[cfg].mt))
      cfg = i;
  }
  return cfg;
}

}  // namespace dgemm

int64_t decode_gemm_num_configs() { return dgemm::kNumConfigs; }

bool decode_gemm_is_tuned(int64_t N, int64_t K) {
  return dgemm::is_tuned_shape(static_cast<int>(N), static_cast<int>(K));
}

int64_t decode_gemm_scratch_floats(int64_t N, int64_t rows, int64_t bn,
                                   int64_t split) {
  return (N / bn) * split * rows * bn;  // = N * split * rows
}

std::vector<int64_t> decode_gemm_config(int64_t i) {
  TORCH_CHECK(i >= 0 && i < dgemm::kNumConfigs, "decode_gemm: no config ", i);
  const dgemm::Config& c = dgemm::kConfigs[i];
  return {c.mt, c.bn, c.wm, c.wn, c.stages, c.split};
}

void decode_gemm(at::Tensor const& x, at::Tensor const& w, at::Tensor& out,
                 at::Tensor& scratch, int64_t cfg, bool early_w, bool pdl) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(),
              "decode_gemm: CUDA tensors expected");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                  w.scalar_type() == at::kBFloat16 &&
                  out.scalar_type() == at::kBFloat16,
              "decode_gemm: bf16 tensors expected");
  TORCH_CHECK(x.dim() == 2 && w.dim() == 2 && out.dim() == 2 &&
                  x.stride(1) == 1 && w.stride(1) == 1 && out.stride(1) == 1,
              "decode_gemm: 2-D tensors with a contiguous row expected");
  const int T = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(w.size(0));
  TORCH_CHECK(w.size(1) == K && w.stride(0) == K,
              "decode_gemm: w must be a dense [N, K] matrix");
  TORCH_CHECK(out.size(0) == T && out.size(1) == N,
              "decode_gemm: out must be [T, N]");
  TORCH_CHECK(T >= 1 && T <= 128, "decode_gemm: 1 <= T <= 128, got ", T);
  TORCH_CHECK(x.stride(0) % 8 == 0 && out.stride(0) % 4 == 0,
              "decode_gemm: row strides must be multiples of 8 (x), 4 (out)");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0,
              "decode_gemm: operands must be 16-byte aligned (out 8)");
  if (cfg < 0) cfg = dgemm::pick_config(T, N, K);
  TORCH_CHECK(cfg >= 0 && cfg < dgemm::kNumConfigs, "decode_gemm: config ",
              cfg);
  const dgemm::Config& c = dgemm::kConfigs[cfg];
  TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == at::kFloat &&
                  scratch.is_contiguous() &&
                  reinterpret_cast<uintptr_t>(scratch.data_ptr()) % 16 == 0,
              "decode_gemm: scratch must be a contiguous fp32 CUDA tensor");
  const int64_t need =
      c.split == 1
          ? 0
          : decode_gemm_scratch_floats(N, c.mt * 16, c.bn, c.split);
  TORCH_CHECK(scratch.numel() >= need, "decode_gemm: scratch is too small (",
              scratch.numel(), " floats, need ", need, ")");
  TORCH_CHECK(c.mt * 16 >= T, "decode_gemm: config ", cfg, " has ", c.mt * 16,
              " rows, T = ", T);
  TORCH_CHECK(K % (c.split * dgemm::kBK) == 0 && N % c.bn == 0,
              "decode_gemm: N ", N, " / K ", K, " do not fit config ", cfg);
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  const auto* wp = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
  auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  auto* sp = reinterpret_cast<float*>(scratch.data_ptr());
  const int ldx = static_cast<int>(x.stride(0));
  const int ldo = static_cast<int>(out.stride(0));
  int i = 0;
#define X(MT, BN, WM, WN, ST, SP)                                          \
  if (i++ == cfg) {                                                        \
    dgemm::launch<MT, BN, WM, WN, ST, SP>(xp, wp, op, sp, T, N, K, ldx,    \
                                          ldo, early_w, pdl, stream);      \
    return;                                                                \
  }
  MUSE_DGEMM_CONFIGS(X)
#undef X
}

}  // namespace muse_spark
