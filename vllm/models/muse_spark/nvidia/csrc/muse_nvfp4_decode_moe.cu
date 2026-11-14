// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// NVFP4 (W4A4) MoE experts for a handful of tokens (decode) on SM120, EP or TP.
// One torch op, `nvfp4_decode_moe`, runs the whole expert computation of a
// layer with the weights streamed straight from HBM:
//
//   K1: routing prologue + NVFP4 quantization of the activations + GEMM1 (gate
//       and up) + SiLU * up + NVFP4 quantization of the intermediate -> hq
//   K2: GEMM2 on hq -> route-major, unweighted bf16 rows
//
// The GEMMs are "swap-AB": the 16 weight rows of a tile are the M dimension of
// an m16n8k64 block-scaled FP4 MMA and the (at most 8) routes of one expert
// are its N dimension, so a tile of weights is read once for all of them.
// Weights and their swizzled block scales are consumed in place. A producer
// warp streams 16 rows x 1 KB of weights plus their scale words into a 3-slot
// shared memory ring with bulk copies, four consumer warps split the K64
// slices of every stage between them. Work is split statically: the units of
// all experts of the layer are cut into one contiguous range per CTA, so there
// is no queue, no inter-CTA waiting and no floating point atomic anywhere, and
// the result is deterministic.
//
// Two engines share the tile math:
//   * k_persistent (default): one launch, grid = SM count. A warp-specialized CTA
//     claims work from a global atomic ticket queue: tickets [0, N1) are GEMM1
//     units, tickets [N1, 2 N1) GEMM2 units (N1 = 256 per expert entry), so all
//     GEMM1 units are claimed before any GEMM2 unit. A GEMM2 unit of an entry
//     starts reading its intermediate once the per-entry done counter says that
//     every GEMM1 unit of the entry has stored it (release / acquire). Every
//     claimed unit belongs to a running CTA, so nothing deadlocks when fewer
//     CTAs than the grid are resident. The counters reset themselves.
//   * k1_gemm1 + k2_gemm2 (kTwoLaunch flag, "plan B"): the two-launch version
//     with a static unit partition described above.
//
// Numerics follow the CUTLASS composition (stock activation quantization, bf16
// GEMM1 output, silu(gate) * up in fp32 rounded to bf16, NVFP4 intermediate),
// with the gate and up global scales applied exactly instead of through the
// s3/s1 fold.

// Only c10 headers: <ATen/cuda/CUDAContext.h> needs cusparse.h and cusolverDn.h,
// which the stock runtime images do not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>

#include "muse_c_ops.h"
#include "nvfp4_mma.cuh"
#include "nvfp4_utils.cuh"

namespace muse_spark {
namespace {

using namespace nvfp4;

// ---- problem constants ------------------------------------------------------
constexpr int kHidden = 4096;          // GEMM1 K, GEMM2 N
constexpr int kInter = 4096;           // GEMM1 N per half, GEMM2 K
constexpr int kTopK = 8;
constexpr int kRowBytes = kHidden / 2;  // packed fp4 bytes of one weight row
constexpr int kHalfBytes = kRowBytes / 2;  // one K half of a row
constexpr int kW13Rows = 2 * kInter;
constexpr int kSfCols = kHidden / 16;  // scale columns of a weight row
constexpr int kKTiles = kHidden / 64;  // K64 tiles of a row (64 scale words)
constexpr int kHalfSfBytes = kSfCols / 2;  // the scales of one K half of a row
constexpr int kMaxEntries = 64;
constexpr int kMaxCols = 64;
// Large-batch engine (k_prep + k_persistent<..., true>): tokens and entries it
// serves. An entry is one expert and at most kEntryCols routes, so a rank with
// up to kLargeMaxLocal experts has at most kLargeMaxLocal + kTopK * kLargeMaxTokens /
// kEntryCols entries. An entry is computed as kEntryCols / 8 column groups of one
// m16n8k64 MMA each, on kLargeStages ring stages (the B operand slots grow with the
// columns and share the shared memory with the ring).
#ifndef MUSE_LARGE_COLS
#define MUSE_LARGE_COLS 16
#endif
#ifndef MUSE_LARGE_STAGES
#define MUSE_LARGE_STAGES 2
#endif
constexpr int kLargeMaxTokens = 128;
constexpr int kLargeMaxLocal = 256;
constexpr int kEntryCols = MUSE_LARGE_COLS;  // routes of one expert in one entry
constexpr int kLargeStages = MUSE_LARGE_STAGES;
constexpr int kLargeMaxEntries = kLargeMaxLocal + kLargeMaxTokens * kTopK / kEntryCols;
static_assert(kEntryCols == 8 || kEntryCols == 16, "one or two MMA column groups");
static_assert(kLargeStages >= 2, "the ring needs two stages");
constexpr int kUnitsPerEntry = 256;    // 4096 / 16 tiles of 16 rows

// Op flags (the upper 16 bits override the grid size, 0 = one CTA per SM).
constexpr int kFlagTwoLaunch = 1;       // plan B: K1 + K2 instead of k_persistent
constexpr int kFlagCtlMemset = 2;       // zero the counters with a memset node
constexpr int kFlagZeroNonLocal = 4;    // zero the rows of non-local routes
constexpr int kFlagNoPdl = 8;           // no PDL: K2 / k_persistent are plain launches
// Large-batch kernel: the loader warp fetches the B operand with one bulk copy per
// column and K half instead of a cp.async per 16 bytes (same bytes in shared memory).
constexpr int kFlagLargeBulkLoader = 16;

// ---- shared memory layout ---------------------------------------------------
#ifndef MUSE_DECODE_STAGES
#define MUSE_DECODE_STAGES 3
#endif
#ifndef MUSE_DECODE_CMAX
#define MUSE_DECODE_CMAX 8
#endif
#ifndef MUSE_DECODE_GRID
#define MUSE_DECODE_GRID 0
#endif
constexpr int kStages = MUSE_DECODE_STAGES;
constexpr int kWPitch = 1040;          // 1 KB row + 16 B: conflict-free ldmatrix
constexpr int kWBytes = 16 * kWPitch;  // 16640
constexpr int kSfBytes = 16 * 32 * 4;  // 2048: 16 rows x 32 K64 tiles x 4 B
constexpr int kSlot1 = kWBytes + kSfBytes;
constexpr int kXqSfOffset = 2064;
constexpr int kXqBytes = 2336;         // per token: codes, then 256 scales (+16)
constexpr int kHPitch = 1040;          // GEMM2 B operand: codes of one K half
constexpr int kHsfPitch = 144;         // and its 128 scale bytes (+16)
constexpr int kPartBytes1 = 2 * 2 * 4 * 32 * 16;  // [parity][tile][warp][lane]
constexpr int kPartBytes2 = 2 * 4 * 32 * 16;      // [parity][warp][lane]

constexpr int kBlockThreads = 192;     // warps 0-3 consumers, 4 producer, 5 H loader
constexpr int kProducerWarp = 4;
constexpr int kLoaderWarp = 5;
constexpr int kConsumerBarrier = 1;    // named barrier of the 4 consumer warps

template <int T_PAD>
constexpr int smem_bytes_k1() {
  return kStages * kSlot1 + T_PAD * kXqBytes + kPartBytes1;
}
template <int T_PAD>
constexpr int slot2_bytes() {
  return kWBytes + kSfBytes + T_PAD * (kHPitch + kHsfPitch);
}
template <int T_PAD>
constexpr int smem_bytes_k2() {
  return kStages * slot2_bytes<T_PAD>() + kPartBytes2;
}

// The routes of the layer grouped by expert. Built redundantly by every CTA of
// K1 (a pure function of the ids and the expert map), published by CTA 0 for
// K2 through global memory. An entry is one expert and at most `cmax` routes
// (a real top-k gives an expert at most T routes, so normally an entry is a
// distinct expert; more routes of one expert make several entries).
template <int N, int C>
struct PlanT {
  int n_entries;
  int n_cols;                          // total routes served
  int pad[2];
  int expert[N];                       // local expert id
  int cnt[N];                          // routes of the entry (1..cmax)
  int col0[N];                         // first hq row of the entry
  int route[N][C];                     // route index (token * 8 + k) of column n
};
// The plan of the small-batch kernels lives in shared memory, that of the
// large-batch kernels (up to kLargeMaxTokens tokens, built by k_prep) in global
// memory.
using Plan = PlanT<kMaxEntries, 8>;
using PlanLarge = PlanT<kLargeMaxEntries, kEntryCols>;
static_assert(sizeof(Plan) % 16 == 0, "plan is copied in 16-byte words");
static_assert(sizeof(PlanLarge) % 16 == 0, "plan is read in 16-byte words");

struct DecodeParams {
  const __nv_bfloat16* hidden;   // [T, 4096]
  const int* topk_ids;           // [T, 8] global expert ids
  const int* expert_map;         // [num_global] global -> local (-1), or null
  int num_global;                // size of expert_map
  int num_local;                 // experts of this rank (any count >= 1)
  const uint8_t* w13;            // [num_local, 8192, 2048]
  const uint8_t* sf13;           // [num_local, 8192, 256] swizzled e4m3
  const uint8_t* w2;             // [num_local, 4096, 2048]
  const uint8_t* sf2;            // [num_local, 4096, 256] swizzled e4m3
  const float* prm;              // [num_local, 4]: alpha_g, alpha_u, alpha2, 0
  const float* gs;               // {1 / a13, 1 / a2}
  __nv_bfloat16* rows;           // [T_PAD * 8, 4096]
  int* perm_idx;                 // [T_PAD * 8]
  uint8_t* hq_data;              // [64, 2048]
  uint8_t* hq_sf;                // [64, 256]
  int* plan;                     // Plan (published) / PlanLarge (built by k_prep), as ints
  uint8_t* xq_buf;               // large batches: quantized activations [T, 2048]
  uint8_t* xq_sf_buf;            // and their linear scales [T, 256], written by k_prep
  float* dbg_acc;                // optional [units, 2, 16, 8] raw GEMM1 sums
  unsigned* ctl;                 // persistent kernel counters (self-resetting)
  const uint8_t* xq_data;        // optional pre-quantized activations [T, 2048]
  const uint8_t* xq_sf;          // and their linear scales [T, 256]
  int flags;
  int T;
  int num_perm;                  // T_PAD * 8
  int up_first;
};

// ---- routing prologue ---------------------------------------------------------

struct PlanScratch {
  int lid[64];
  int rank[64];
  int entry_of[64];
  uint32_t ballot[2];
};

// Called by all threads of a block (it synchronizes). `cmax` is the largest
// number of routes of one expert in one entry.
__device__ void build_plan(const DecodeParams& p, int cmax, Plan& plan,
                           PlanScratch& s) {
  const int tid = threadIdx.x;
  const int num_routes = p.T * kTopK;
  if (tid < 64) {
    int l = -1;
    if (tid < num_routes) {
      int id = p.topk_ids[tid];
      if (p.expert_map != nullptr) {
        l = (id >= 0 && id < p.num_global) ? p.expert_map[id] : -1;
      } else {
        l = id;
      }
      if (l < 0 || l >= p.num_local) l = -1;
    }
    s.lid[tid] = l;
    plan.cnt[tid] = 0;
  }
  __syncthreads();
  int rank = 0;
  bool creator = false;
  if (tid < 64) {
    const int l = s.lid[tid];
    for (int j = 0; j < tid; ++j) rank += (l >= 0 && s.lid[j] == l);
    creator = l >= 0 && (rank % cmax == 0);
    const uint32_t b = __ballot_sync(0xffffffffu, creator);
    if ((tid & 31) == 0) s.ballot[tid >> 5] = b;
    s.rank[tid] = rank;
  }
  __syncthreads();
  int entry = -1;
  if (tid < 64) {
    if (creator) {
      const uint32_t below = (1u << (tid & 31)) - 1u;
      entry = __popc(s.ballot[tid >> 5] & below) +
              ((tid >> 5) ? __popc(s.ballot[0]) : 0);
      s.entry_of[tid] = entry;
    }
  }
  __syncthreads();
  if (tid < 64) {
    const int l = s.lid[tid];
    if (l >= 0) {
      if (!creator) {
        // The creator is the route of the same expert whose rank is the
        // largest multiple of cmax not above ours.
        const int want = rank - rank % cmax;
        for (int j = 0; j < tid; ++j) {
          if (s.lid[j] == l && s.rank[j] == want) entry = s.entry_of[j];
        }
      }
      plan.route[entry][rank % cmax] = tid;
      atomicAdd(&plan.cnt[entry], 1);  // integer: the sum is order independent
      if (creator) plan.expert[entry] = l;
    }
  }
  __syncthreads();
  if (tid == 0) {
    int n = __popc(s.ballot[0]) + __popc(s.ballot[1]);
    int col = 0;
    for (int g = 0; g < n; ++g) {
      plan.col0[g] = col;
      col += plan.cnt[g];
    }
    plan.n_entries = n;
    plan.n_cols = col;
  }
  __syncthreads();
}

// CTA 0: publish the plan for K2 and write the route -> row map.
__device__ void publish_plan(const DecodeParams& p, const Plan& plan,
                             const PlanScratch& s) {
  const int tid = threadIdx.x;
  const int* src = reinterpret_cast<const int*>(&plan);
  for (int i = tid; i < static_cast<int>(sizeof(Plan) / 4); i += blockDim.x) {
    p.plan[i] = src[i];
  }
  const int num_routes = p.T * kTopK;
  for (int i = tid; i < p.num_perm; i += blockDim.x) {
    p.perm_idx[i] = (i < num_routes && s.lid[i] >= 0) ? i : -1;
  }
}

// Debug aid (kFlagZeroNonLocal): CTA b zeroes the rows of the non-local routes
// r = b, b + grid, ..., so that a consumer of `rows` never sees stale data.
__device__ void zero_nonlocal_row(const DecodeParams& p, const PlanScratch& s) {
  for (int r = blockIdx.x; r < p.num_perm; r += gridDim.x) {
    if (r < p.T * kTopK && s.lid[r] >= 0) continue;
    uint4* dst = reinterpret_cast<uint4*>(p.rows + static_cast<size_t>(r) * kHidden);
    for (int i = threadIdx.x; i < kHidden / 8; i += blockDim.x) {
      dst[i] = make_uint4(0, 0, 0, 0);
    }
  }
}

// kFlagZeroNonLocal of the large-batch kernel: CTA b zeroes the rows of the
// non-local routes r = b, b + grid, ..., which it knows from the route -> row map
// (perm_idx) that k_prep wrote.
__device__ void zero_nonlocal_row_large(const DecodeParams& p) {
  for (int r = blockIdx.x; r < p.num_perm; r += gridDim.x) {
    if (p.perm_idx[r] >= 0) continue;
    uint4* dst = reinterpret_cast<uint4*>(p.rows + static_cast<size_t>(r) * kHidden);
    for (int i = threadIdx.x; i < kHidden / 8; i += blockDim.x) {
      dst[i] = make_uint4(0, 0, 0, 0);
    }
  }
}

// ---- activation quantization ---------------------------------------------------

// Loads this thread's share of `T` rows of bf16 [T, 4096]: row t, vector
// it * NTHREADS + ltid (8 elements). All loads are independent, so a thread
// issues every one before it needs the first (a row-by-row loop costs one
// memory latency per row).
template <int NTHREADS, int T_PAD>
__device__ __forceinline__ void load_rows(
    const __nv_bfloat16* hidden, int T, int ltid,
    PackedVec<__nv_bfloat16> (&vec)[T_PAD * (512 / NTHREADS)]) {
  constexpr int kIters = 512 / NTHREADS;
#pragma unroll
  for (int t = 0; t < T_PAD; ++t) {
#pragma unroll
    for (int it = 0; it < kIters; ++it) {
      if (t < T) {
        vec[t * kIters + it] = *reinterpret_cast<const PackedVec<__nv_bfloat16>*>(
            hidden + static_cast<size_t>(t) * kHidden + (it * NTHREADS + ltid) * 8);
      }
    }
  }
}

// Quantizes rows loaded by load_rows with the arithmetic of the stock NVFP4
// activation quantization (cvt_warp_fp16_to_fp4: two lanes per 16-element
// block, 8 elements each) into `xq`: per token 2048 bytes of codes (low nibble
// = even K) and, at kXqSfOffset, 256 e4m3 scales in K order. `ltid` is the
// calling thread's index among `NTHREADS` cooperating threads (whole warps,
// ltid = 0 starting a warp, so that lanes 2i, 2i + 1 hold one block).
template <int NTHREADS, int T_PAD>
__device__ __forceinline__ void quantize_rows(
    PackedVec<__nv_bfloat16> (&vec)[T_PAD * (512 / NTHREADS)], int T, float gs0,
    uint8_t* xq, int ltid) {
  static_assert(512 % NTHREADS == 0, "a row is 512 thread-vectors");
  constexpr int kIters = 512 / NTHREADS;
#pragma unroll
  for (int t = 0; t < T_PAD; ++t) {
    if (t < T) {
      uint8_t* row = xq + t * kXqBytes;
#pragma unroll
      for (int it = 0; it < kIters; ++it) {
        const int idx = it * NTHREADS + ltid;  // 8-element vector of the row
        uint8_t* sf_out = (ltid & 1) ? nullptr : row + kXqSfOffset + (idx >> 1);
        uint32_t codes = cvt_warp_fp16_to_fp4<__nv_bfloat16, 2>(
            vec[t * kIters + it], gs0, sf_out);
        *reinterpret_cast<uint32_t*>(row + idx * 4) = codes;
      }
    }
  }
}

// Fallback of the in-kernel quantization (DecodeParams::xq_data): the T rows
// quantized by the stock op (codes [T, 2048], linear scales [T, 256]) are copied
// into `xq` in the same layout quantize_rows writes. `ltid` of `NTHREADS` (128).
template <int NTHREADS>
__device__ __forceinline__ void load_xq_from_global(const DecodeParams& p,
                                                    uint8_t* xq, int ltid) {
  for (int t = 0; t < p.T; ++t) {
    uint8_t* row = xq + t * kXqBytes;
    for (int i = ltid; i < kRowBytes / 16; i += NTHREADS) {
      reinterpret_cast<uint4*>(row)[i] =
          reinterpret_cast<const uint4*>(p.xq_data + t * kRowBytes)[i];
    }
    for (int i = ltid; i < kSfCols / 16; i += NTHREADS) {
      reinterpret_cast<uint4*>(row + kXqSfOffset)[i] =
          reinterpret_cast<const uint4*>(p.xq_sf + t * kSfCols)[i];
    }
  }
}

// Test entry: the in-kernel quantization of T rows, written out as codes
// [T, 2048] and linear scales [T, 256].
template <int NTHREADS>
__global__ void __launch_bounds__(NTHREADS) act_quant_kernel(
    const __nv_bfloat16* hidden, const float* gs, int T, uint8_t* data,
    uint8_t* sf) {
  extern __shared__ __align__(16) uint8_t xq[];
  PackedVec<__nv_bfloat16> vec[8 * (512 / NTHREADS)];
  load_rows<NTHREADS, 8>(hidden, T, threadIdx.x, vec);
  quantize_rows<NTHREADS, 8>(vec, T, gs[0], xq, threadIdx.x);
  __syncthreads();
  for (int t = 0; t < T; ++t) {
    for (int i = threadIdx.x; i < kHidden / 2; i += NTHREADS) {
      data[t * (kHidden / 2) + i] = xq[t * kXqBytes + i];
    }
    for (int i = threadIdx.x; i < kSfCols; i += NTHREADS) {
      sf[t * kSfCols + i] = xq[t * kXqBytes + kXqSfOffset + i];
    }
  }
}

// ---- helpers shared by K1 / K2 -----------------------------------------------

__device__ __forceinline__ float bf16_round(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

// Global byte offset of the 4-byte scale word of weight row `m`, K64 tile `kt`
// inside an expert's swizzled scale plane (128 x 4 swizzle, 64 K tiles).
__device__ __forceinline__ size_t sf_word_offset(int m, int kt) {
  return (static_cast<size_t>((m >> 7) * kKTiles + kt) << 9) + ((m & 31) << 4) +
         (((m >> 5) & 3) << 2);
}

// Consumer-side fragment loads and MMA of one K64 slice. `a_addr` is the
// lane's ldmatrix address of the A tile slice, `sfa_ptr` its scale word,
// `b_addr` / `sfb_ptr` those of the B operand.
__device__ __forceinline__ void mma_slice(float (&acc)[4], uint32_t a_addr,
                                          const uint32_t* sfa_ptr,
                                          uint32_t b_addr,
                                          const uint32_t* sfb_ptr) {
  uint32_t a0, a1, a2, a3, b0, b1;
  ldmatrix_x4(a_addr, a0, a1, a2, a3);
  ldmatrix_x2(b_addr, b0, b1);
  mma_nvfp4_m16n8k64(acc, a0, a1, a2, a3, b0, b1, *sfa_ptr, *sfb_ptr);
}

// mma_slice with the A fragment loaded by the caller (and its scale word read), for
// the column groups of an entry that share it.
__device__ __forceinline__ void mma_slice_b(float (&acc)[4], const uint32_t (&a)[4],
                                            uint32_t sfa, uint32_t b_addr,
                                            const uint32_t* sfb_ptr) {
  uint32_t b0, b1;
  ldmatrix_x2(b_addr, b0, b1);
  mma_nvfp4_m16n8k64(acc, a[0], a[1], a[2], a[3], b0, b1, sfa, *sfb_ptr);
}

// ---- K1: routing + activation quant + GEMM1 + SiLU*up + requant ----------------

template <int T_PAD>
__global__ void __launch_bounds__(kBlockThreads, 1)
    k1_gemm1(const DecodeParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Plan plan;
  __shared__ PlanScratch scratch;
  __shared__ __align__(8) uint64_t full[kStages];
  __shared__ __align__(8) uint64_t empty[kStages];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  uint8_t* const slots = smem;
  uint8_t* const xq = smem + kStages * kSlot1;
  float4* const part = reinterpret_cast<float4*>(xq + T_PAD * kXqBytes);

  // K2 only reads the weights (streamed into its ring) before it waits for K1
  // to finish, so it may start on any SM that K1 has vacated.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // The activations do not depend on the routing: start loading them now.
  PackedVec<__nv_bfloat16> hv[T_PAD * 4];
  if (warp < 4 && p.xq_data == nullptr) load_rows<128, T_PAD>(p.hidden, p.T, tid, hv);

  if (tid == 0) {
    for (int i = 0; i < kStages; ++i) {
      mbar_init(smem_u32(&full[i]), 1 + 32);
      mbar_init(smem_u32(&empty[i]), 4);
    }
    mbar_fence_init();
  }
  build_plan(p, T_PAD, plan, scratch);  // also the barrier-init fence
  if (blockIdx.x == 0) publish_plan(p, plan, scratch);
  if (p.flags & kFlagZeroNonLocal) zero_nonlocal_row(p, scratch);
  if (plan.n_entries == 0) return;  // no local route: nothing to compute

  const int n_units = kUnitsPerEntry * plan.n_entries;
  const int G = gridDim.x;
  const int u0 = static_cast<int>(static_cast<long long>(blockIdx.x) * n_units / G);
  const int u1 =
      static_cast<int>(static_cast<long long>(blockIdx.x + 1) * n_units / G);
  const int n_st = (u1 - u0) * 4;  // 2 tiles (up, gate) x 2 K halves
  const int up_base = p.up_first ? 0 : kInter;
  const int gate_base = p.up_first ? kInter : 0;

  if (warp == kProducerWarp) {
    for (int st = 0; st < n_st; ++st) {
      const int slot = st % kStages;
      mbar_wait(smem_u32(&empty[slot]), ((st / kStages) & 1) ^ 1);
      const int unit = u0 + (st >> 2);
      const int tile = (st >> 1) & 1;  // 0: up, 1: gate
      const int kh = st & 1;
      const int g = unit >> 8;
      const int w = unit & 255;
      const int e = plan.expert[g];
      // w = (a << 3) | (xh << 2) | d, tile index jb = 8 a + 2 d + xh: the four
      // d siblings share 32-byte scale sectors.
      const int jb = 8 * (w >> 3) + 2 * (w & 3) + ((w >> 2) & 1);
      const int row0 = (tile == 0 ? up_base : gate_base) + 16 * jb;
      uint8_t* sl = slots + slot * kSlot1;
      const uint32_t fb = smem_u32(&full[slot]);
      if (lane == 0) mbar_arrive_expect_tx(fb, 16 * kHalfBytes);
      __syncwarp();
      if (lane < 16) {
        bulk_copy_g2s(smem_u32(sl + lane * kWPitch),
                      p.w13 + (static_cast<size_t>(e) * kW13Rows + row0 + lane) *
                                  kRowBytes + kh * kHalfBytes,
                      kHalfBytes, fb);
      }
      const uint8_t* sfp = p.sf13 + static_cast<size_t>(e) * kW13Rows * kSfCols;
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        const int idx = lane + 32 * t;  // = ktl * 16 + row
        const int row = idx & 15;
        const int kt = kh * 32 + (idx >> 4);
        cp_async_4(smem_u32(sl + kWBytes + idx * 4),
                   sfp + sf_word_offset(row0 + row, kt));
      }
      cp_async_arrive_noinc(fb);
    }
  } else if (warp < 4) {
    if (u1 > u0) {
      if (p.xq_data == nullptr) {
        quantize_rows<128, T_PAD>(hv, p.T, p.gs[0], xq, tid);
      } else {
        load_xq_from_global<128>(p, xq, tid);
      }
    }
    named_barrier(kConsumerBarrier, 128);

    const int q = lane >> 2;
    const int c = lane & 3;
    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int sfa_row = q + 8 * (c & 1);
    const float gs1 = p.gs[1];
    for (int ui = 0; ui < u1 - u0; ++ui) {
      const int unit = u0 + ui;
      const int g = unit >> 8;
      const int w = unit & 255;
      const int e = plan.expert[g];
      const int cnt = plan.cnt[g];
      const int jb = 8 * (w >> 3) + 2 * (w & 3) + ((w >> 2) & 1);
      // B operand: column n of the MMA is route n of the entry.
      const int tok_b = plan.route[g][min(lane & 7, cnt - 1)] >> 3;
      const int tok_s = plan.route[g][min(q, cnt - 1)] >> 3;
      const uint32_t b_base = smem_u32(xq + tok_b * kXqBytes) + ((lane >> 3) & 1) * 16;
      const uint32_t* sfb_base = reinterpret_cast<const uint32_t*>(
          xq + tok_s * kXqBytes + kXqSfOffset);
      float acc[2][4];
#pragma unroll
      for (int t = 0; t < 2; ++t)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[t][i] = 0.f;

      for (int s = 0; s < 4; ++s) {
        const int st = ui * 4 + s;
        const int slot = st % kStages;
        mbar_wait(smem_u32(&full[slot]), (st / kStages) & 1);
        const int tile = s >> 1;
        const int kh = s & 1;
        const uint8_t* sl = slots + slot * kSlot1;
        const uint32_t a_base = smem_u32(sl) + a_row * kWPitch + (lane >> 4) * 16;
        const uint32_t* sf_w = reinterpret_cast<const uint32_t*>(sl + kWBytes);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int kk = warp + 4 * i;  // K64 slice inside this half
          const int kt = kh * 32 + kk;
          if (tile == 0) {
            mma_slice(acc[0], a_base + kk * 32, sf_w + kk * 16 + sfa_row,
                      b_base + kt * 32, sfb_base + kt);
          } else {
            mma_slice(acc[1], a_base + kk * 32, sf_w + kk * 16 + sfa_row,
                      b_base + kt * 32, sfb_base + kt);
          }
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&empty[slot]));
      }

      // Partial sums of the four warps: stored by all, summed in warp order by
      // the epilogue warp (rotating, so none is always the late one).
      float4* pbuf = part + (ui & 1) * (2 * 4 * 32);
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        pbuf[(t * 4 + warp) * 32 + lane] =
            make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]);
      }
      named_barrier(kConsumerBarrier, 128);
      if (warp != (ui & 3)) continue;

      float4 sum[2];
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const float4 p0 = pbuf[(t * 4 + 0) * 32 + lane];
        const float4 p1 = pbuf[(t * 4 + 1) * 32 + lane];
        const float4 p2 = pbuf[(t * 4 + 2) * 32 + lane];
        const float4 p3 = pbuf[(t * 4 + 3) * 32 + lane];
        sum[t].x = ((p0.x + p1.x) + p2.x) + p3.x;
        sum[t].y = ((p0.y + p1.y) + p2.y) + p3.y;
        sum[t].z = ((p0.z + p1.z) + p2.z) + p3.z;
        sum[t].w = ((p0.w + p1.w) + p2.w) + p3.w;
      }
      const float us[4] = {sum[0].x, sum[0].y, sum[0].z, sum[0].w};  // up
      const float gt[4] = {sum[1].x, sum[1].y, sum[1].z, sum[1].w};  // gate
      if (p.dbg_acc != nullptr) {
        // Raw sums: [unit][tile][row][col], rows q, q + 8, columns 2c, 2c + 1.
        float* d = p.dbg_acc + static_cast<size_t>(unit) * 256;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
          const float* v = t == 0 ? us : gt;
          d[t * 128 + q * 8 + 2 * c] = v[0];
          d[t * 128 + q * 8 + 2 * c + 1] = v[1];
          d[t * 128 + (q + 8) * 8 + 2 * c] = v[2];
          d[t * 128 + (q + 8) * 8 + 2 * c + 1] = v[3];
        }
      }

      // h = bf16(silu(bf16(gate * alpha_g)) * bf16(up * alpha_u)), as the stock
      // silu_and_mul quantization sees the bf16 GEMM1 output.
      const float alpha_g = p.prm[e * 4 + 0];
      const float alpha_u = p.prm[e * 4 + 1];
      float h[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float gv = bf16_round(gt[i] * alpha_g);
        const float uv = bf16_round(us[i] * alpha_u);
        h[i] = bf16_round(silu(gv) * uv);
      }
      // The 16 rows of the tile are one NVFP4 block per route column. A column
      // is held by the 8 lanes with equal c; its rows are q and q + 8.
      float amax0 = fmaxf(fabsf(h[0]), fabsf(h[2]));  // column 2c
      float amax1 = fmaxf(fabsf(h[1]), fabsf(h[3]));  // column 2c + 1
#pragma unroll
      for (int m = 4; m <= 16; m <<= 1) {
        amax0 = fmaxf(amax0, __shfl_xor_sync(0xffffffffu, amax0, m));
        amax1 = fmaxf(amax1, __shfl_xor_sync(0xffffffffu, amax1, m));
      }
      // Scale arithmetic of cvt_warp_fp16_to_fp4, for one column.
      float out_scale[2];
      uint8_t sf8[2];
      const float vmax[2] = {amax0, amax1};
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        float sf = gs1 * (vmax[k] * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(sf);
        reinterpret_cast<__nv_fp8_e4m3&>(sf8[k]) = tmp;
        sf = float(tmp);
        out_scale[k] = sf != 0.0f ? reciprocal_approximate_ftz(
                                        sf * reciprocal_approximate_ftz(gs1))
                                  : 0.0f;
      }
      float v[4], pv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[i] = h[i] * out_scale[i & 1];
        pv[i] = __shfl_xor_sync(0xffffffffu, v[i], 4);  // row q + 1 (q even)
      }
      const int col0 = plan.col0[g];
      const int n0 = 2 * c;
      if ((q & 1) == 0) {
        // Rows (q, q + 1) and (q + 8, q + 9) pair up into bytes q / 2 and
        // 4 + q / 2 of the 8 bytes of this tile and column.
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int n = n0 + k;
          if (n < cnt) {
            uint8_t* dst =
                p.hq_data + static_cast<size_t>(col0 + n) * kRowBytes + jb * 8;
            dst[q >> 1] = static_cast<uint8_t>(cvt_e2m1x2(v[k], pv[k]));
            dst[4 + (q >> 1)] = static_cast<uint8_t>(cvt_e2m1x2(v[2 + k], pv[2 + k]));
          }
        }
      }
      if (q == 0) {
#pragma unroll
        for (int k = 0; k < 2; ++k) {
          const int n = n0 + k;
          if (n < cnt) p.hq_sf[static_cast<size_t>(col0 + n) * kSfCols + jb] = sf8[k];
        }
      }
    }
  }
}

// ---- K2: GEMM2 ----------------------------------------------------------------

template <int T_PAD>
__global__ void __launch_bounds__(kBlockThreads, 1)
    k2_gemm2(const DecodeParams p) {
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Plan plan;
  __shared__ PlanScratch scratch;
  __shared__ __align__(8) uint64_t full[kStages];
  __shared__ __align__(8) uint64_t empty[kStages];

  constexpr int kSlot = slot2_bytes<T_PAD>();
  constexpr int kHOff = kWBytes + kSfBytes;
  constexpr int kHsfOff = kHOff + T_PAD * kHPitch;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  uint8_t* const slots = smem;
  float4* const part = reinterpret_cast<float4*>(smem + kStages * kSlot);

  if (tid == 0) {
    for (int i = 0; i < kStages; ++i) {
      mbar_init(smem_u32(&full[i]), 1 + 32 + 32);
      mbar_init(smem_u32(&empty[i]), 4);
    }
    mbar_fence_init();
  }
  // The plan is a pure function of the ids: recomputed here instead of read
  // from K1, so that nothing before the weight stream depends on K1.
  build_plan(p, T_PAD, plan, scratch);
  if (plan.n_entries == 0) return;  // no local route: nothing to compute
  // Columns of an entry beyond its route count are never read as results;
  // keep their (unwritten) B operand finite.
  for (int i = tid; i < kStages * kSlot / 16; i += blockDim.x) {
    if (i % (kSlot / 16) >= kHOff / 16) {
      reinterpret_cast<uint4*>(slots)[i] = make_uint4(0, 0, 0, 0);
    }
  }
  __syncthreads();

  const int n_units = kUnitsPerEntry * plan.n_entries;
  const int G = gridDim.x;
  const int u0 = static_cast<int>(static_cast<long long>(blockIdx.x) * n_units / G);
  const int u1 =
      static_cast<int>(static_cast<long long>(blockIdx.x + 1) * n_units / G);
  const int n_st = (u1 - u0) * 2;  // two K halves

  if (warp == kProducerWarp) {
    for (int st = 0; st < n_st; ++st) {
      const int slot = st % kStages;
      mbar_wait(smem_u32(&empty[slot]), ((st / kStages) & 1) ^ 1);
      const int unit = u0 + (st >> 1);
      const int kh = st & 1;
      const int g = unit >> 8;
      const int o = unit & 255;
      const int e = plan.expert[g];
      uint8_t* sl = slots + slot * kSlot;
      const uint32_t fb = smem_u32(&full[slot]);
      if (lane == 0) mbar_arrive_expect_tx(fb, 16 * kHalfBytes);
      __syncwarp();
      if (lane < 16) {
        // Output row of tile row i: 128 (o >> 3) + 32 (i >> 2) + 4 (o & 7) + (i & 3);
        // it puts the scale words of a K64 tile into 64 contiguous bytes.
        const int m = 128 * (o >> 3) + 32 * (lane >> 2) + 4 * (o & 7) + (lane & 3);
        bulk_copy_g2s(smem_u32(sl + lane * kWPitch),
                      p.w2 + (static_cast<size_t>(e) * kHidden + m) * kRowBytes +
                          kh * kHalfBytes,
                      kHalfBytes, fb);
      }
      const uint8_t* sfp = p.sf2 + static_cast<size_t>(e) * kHidden * kSfCols +
                           static_cast<size_t>((o >> 3) * kKTiles + kh * 32) * 512 +
                           (o & 7) * 64;
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        const int ci = lane + 32 * t;  // 16-byte chunk: kt_local * 4 + part
        cp_async_16(smem_u32(sl + kWBytes + ci * 16),
                    sfp + static_cast<size_t>(ci >> 2) * 512 + (ci & 3) * 16);
      }
      cp_async_arrive_noinc(fb);
    }
  } else if (warp == kLoaderWarp) {
    // The entry's intermediate for this K half: per route column 1 KB of codes
    // and 128 scale bytes of hq. hq is written by K1: wait for it to finish
    // (a no-op when K2 was not launched as a programmatic dependent).
    asm volatile("griddepcontrol.wait;" ::: "memory");
    for (int st = 0; st < n_st; ++st) {
      const int slot = st % kStages;
      mbar_wait(smem_u32(&empty[slot]), ((st / kStages) & 1) ^ 1);
      const int unit = u0 + (st >> 1);
      const int kh = st & 1;
      const int g = unit >> 8;
      const int cnt = plan.cnt[g];
      const int col0 = plan.col0[g];
      uint8_t* sl = slots + slot * kSlot;
      const uint32_t fb = smem_u32(&full[slot]);
      for (int idx = lane; idx < cnt * 72; idx += 32) {
        const int n = idx / 72;
        const int ch = idx - n * 72;
        if (ch < 64) {
          cp_async_16(smem_u32(sl + kHOff + n * kHPitch + ch * 16),
                      p.hq_data + static_cast<size_t>(col0 + n) * kRowBytes +
                          kh * kHalfBytes + ch * 16);
        } else {
          cp_async_16(smem_u32(sl + kHsfOff + n * kHsfPitch + (ch - 64) * 16),
                      p.hq_sf + static_cast<size_t>(col0 + n) * kSfCols + kh * 128 +
                          (ch - 64) * 16);
        }
      }
      cp_async_arrive_noinc(fb);
    }
  } else if (warp < 4) {
    const int q = lane >> 2;
    const int c = lane & 3;
    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int sfa_row = q + 8 * (c & 1);
    // Scale word of tile row r at K tile kt: kt * 16 words + (r & 3) * 4 + (r >> 2).
    const int sfa_word = (sfa_row & 3) * 4 + (sfa_row >> 2);
    for (int ui = 0; ui < u1 - u0; ++ui) {
      const int unit = u0 + ui;
      const int g = unit >> 8;
      const int o = unit & 255;
      const int e = plan.expert[g];
      const int cnt = plan.cnt[g];
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      for (int kh = 0; kh < 2; ++kh) {
        const int st = ui * 2 + kh;
        const int slot = st % kStages;
        mbar_wait(smem_u32(&full[slot]), (st / kStages) & 1);
        const uint8_t* sl = slots + slot * kSlot;
        const uint32_t a_base = smem_u32(sl) + a_row * kWPitch + (lane >> 4) * 16;
        // Columns past the padded token count repeat the last one (never a
        // result): the B operand stays inside the slot's H region.
        const uint32_t b_base = smem_u32(sl + kHOff) +
                                min(lane & 7, T_PAD - 1) * kHPitch +
                                ((lane >> 3) & 1) * 16;
        const uint32_t* sf_w = reinterpret_cast<const uint32_t*>(sl + kWBytes);
        const uint32_t* sfb_w = reinterpret_cast<const uint32_t*>(
            sl + kHsfOff + min(q, T_PAD - 1) * kHsfPitch);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int kk = warp + 4 * i;
          mma_slice(acc, a_base + kk * 32, sf_w + kk * 16 + sfa_word,
                    b_base + kk * 32, sfb_w + kk);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&empty[slot]));
      }
      float4* pbuf = part + (ui & 1) * (4 * 32);
      pbuf[warp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
      named_barrier(kConsumerBarrier, 128);
      if (warp != (ui & 3)) continue;

      const float4 p0 = pbuf[0 * 32 + lane];
      const float4 p1 = pbuf[1 * 32 + lane];
      const float4 p2 = pbuf[2 * 32 + lane];
      const float4 p3 = pbuf[3 * 32 + lane];
      const float s[4] = {((p0.x + p1.x) + p2.x) + p3.x, ((p0.y + p1.y) + p2.y) + p3.y,
                          ((p0.z + p1.z) + p2.z) + p3.z, ((p0.w + p1.w) + p2.w) + p3.w};
      const float alpha2 = p.prm[e * 4 + 2];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int r = q + 8 * (i >> 1);  // tile row
        const int n = 2 * c + (i & 1);   // route column
        if (n < cnt) {
          const int m = 128 * (o >> 3) + 32 * (r >> 2) + 4 * (o & 7) + (r & 3);
          const int route = plan.route[g][n];
          p.rows[static_cast<size_t>(route) * kHidden + m] =
              __float2bfloat16_rn(s[i] * alpha2);
        }
      }
    }
  }
}

#include "muse_nvfp4_decode_persist.cuh"
#include "muse_nvfp4_decode_large.cuh"

// ---- host side -------------------------------------------------------------------

int sm_count(int device) {
  constexpr int kMaxCachedDevices = 64;
  static std::atomic<int> cache[kMaxCachedDevices];
  const bool cacheable = device >= 0 && device < kMaxCachedDevices;
  if (cacheable) {
    const int cached = cache[device].load(std::memory_order_relaxed);
    if (cached > 0) return cached;
  }
  int value = 0;
  C10_CUDA_CHECK(
      cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device));
  if (cacheable) cache[device].store(value, std::memory_order_relaxed);
  return value;
}

template <typename K>
void ensure_smem(K kernel, int bytes, std::atomic<int>& done) {
  // Not safe to call while a stream is being captured the first time: the
  // first call of every instantiation must come from a warm-up.
  if (done.load(std::memory_order_acquire) >= bytes) return;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  done.store(bytes, std::memory_order_release);
}

// Shared memory the persistent kernel may use: static + dynamic must fit in the
// opt-in limit of the device.
template <typename K>
void ensure_smem_checked(K kernel, int dynamic_bytes, std::atomic<int>& done) {
  if (done.load(std::memory_order_acquire) >= dynamic_bytes) return;
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, kernel));
  int device = 0, limit = 0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  TORCH_CHECK(static_cast<int>(attrs.sharedSizeBytes) + dynamic_bytes <= limit,
              "decode kernel shared memory ", attrs.sharedSizeBytes, " + ",
              dynamic_bytes, " exceeds ", limit);
  ensure_smem(kernel, dynamic_bytes, done);
}

// Launches `kernel` as a programmatic dependent of the previous kernel of the
// stream: it may start before the predecessor has finished, and must execute
// griddepcontrol.wait before it reads anything the predecessor writes.
void launch_dependent(void (*kernel)(const DecodeParams), const DecodeParams& p,
                      int grid, int smem, cudaStream_t stream,
                      int block = kBlockThreads) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(block);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kernel, p));
}

template <int T_PAD>
void launch_persistent(const DecodeParams& p, int grid, cudaStream_t stream) {
  static std::atomic<int> attr{0};
  constexpr int kSmem = smem_bytes_persist<T_PAD, false>();
  ensure_smem_checked(k_persistent<T_PAD, false>, kSmem, attr);
  if (p.flags & kFlagCtlMemset) {
    C10_CUDA_CHECK(cudaMemsetAsync(p.ctl, 0, kCtlWords * sizeof(unsigned), stream));
  }
  if (!(p.flags & kFlagNoPdl)) {
    launch_dependent(k_persistent<T_PAD, false>, p, grid, kSmem, stream);
  } else {
    k_persistent<T_PAD, false><<<grid, kBlockThreads, kSmem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

// Large batches: k_prep, then the persistent kernel as its programmatic dependent.
template <bool BULK>
void launch_large_kernels(const DecodeParams& p, int grid, cudaStream_t stream) {
  static std::atomic<int> attr{0};
  constexpr int kSmem = smem_bytes_persist<kLargeMaxTokens, true>();
  static_assert(kSmem <= 99 * 1024,
                "the ring and the B operand slots of the large-batch kernel do not "
                "fit the shared memory of an SM120 block (MUSE_LARGE_COLS, "
                "MUSE_LARGE_STAGES)");
  ensure_smem_checked(k_persistent<kLargeMaxTokens, true, BULK>, kSmem, attr);
  const int prep_grid = 1 + (p.T + kPrepRowsPerCta - 1) / kPrepRowsPerCta;
  if (!(p.flags & kFlagNoPdl)) {
    launch_dependent(k_prep, p, prep_grid, 0, stream, kPrepThreads);
    launch_dependent(k_persistent<kLargeMaxTokens, true, BULK>, p, grid, kSmem, stream);
  } else {
    k_prep<<<prep_grid, kPrepThreads, 0, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    k_persistent<kLargeMaxTokens, true, BULK><<<grid, kBlockThreads, kSmem, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void launch_large(const DecodeParams& p, int grid, cudaStream_t stream) {
  if (p.flags & kFlagLargeBulkLoader) {
    launch_large_kernels<true>(p, grid, stream);
  } else {
    launch_large_kernels<false>(p, grid, stream);
  }
}

template <int T_PAD>
void launch_decode(const DecodeParams& p, int grid, cudaStream_t stream) {
  if (!(p.flags & kFlagTwoLaunch)) {
    launch_persistent<T_PAD>(p, grid, stream);
    return;
  }
  static std::atomic<int> attr1{0}, attr2{0};
  constexpr int kSmem1 = smem_bytes_k1<T_PAD>();
  constexpr int kSmem2 = smem_bytes_k2<T_PAD>();
  ensure_smem(k1_gemm1<T_PAD>, kSmem1, attr1);
  ensure_smem(k2_gemm2<T_PAD>, kSmem2, attr2);
  k1_gemm1<T_PAD><<<grid, kBlockThreads, kSmem1, stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (!(p.flags & kFlagNoPdl)) {
    launch_dependent(k2_gemm2<T_PAD>, p, grid, kSmem2, stream);
  } else {
    k2_gemm2<T_PAD><<<grid, kBlockThreads, kSmem2, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void check_tensor(at::Tensor const& t, at::ScalarType type, char const* name,
                  std::initializer_list<int64_t> shape) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name,
              " must be a contiguous CUDA tensor");
  TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
  TORCH_CHECK(t.dim() == static_cast<int64_t>(shape.size()), name,
              " has the wrong rank");
  int d = 0;
  for (int64_t s : shape) {
    TORCH_CHECK(s < 0 || t.size(d) == s, name, " has the wrong shape in dim ", d);
    ++d;
  }
}

}  // namespace

void nvfp4_decode_moe(at::Tensor const& hidden, at::Tensor const& topk_ids,
                      std::optional<at::Tensor> const& expert_map,
                      at::Tensor const& w13, at::Tensor const& sf13,
                      at::Tensor const& w2, at::Tensor const& sf2,
                      at::Tensor const& prm, at::Tensor const& gs, bool up_first,
                      at::Tensor& rows, at::Tensor& perm_idx,
                      at::Tensor& hq_data, at::Tensor& hq_sf, at::Tensor& plan,
                      at::Tensor& ctl, std::optional<at::Tensor> const& dbg_acc,
                      int64_t flags, std::optional<at::Tensor> const& xq_data,
                      std::optional<at::Tensor> const& xq_sf) {
  const int64_t T = hidden.size(0);
  const int64_t num_perm = perm_idx.size(0);
  check_tensor(hidden, at::kBFloat16, "hidden", {-1, kHidden});
  check_tensor(topk_ids, at::kInt, "topk_ids", {T, kTopK});
  const int64_t num_local = w13.size(0);
  TORCH_CHECK(num_local >= 1, "w13 holds no experts");
  check_tensor(w13, at::kByte, "w13", {num_local, kW13Rows, kRowBytes});
  check_tensor(sf13, at::kFloat8_e4m3fn, "sf13", {num_local, kW13Rows, kSfCols});
  check_tensor(w2, at::kByte, "w2", {num_local, kHidden, kRowBytes});
  check_tensor(sf2, at::kFloat8_e4m3fn, "sf2", {num_local, kHidden, kSfCols});
  check_tensor(prm, at::kFloat, "prm", {num_local, 4});
  check_tensor(gs, at::kFloat, "gs", {2});
  check_tensor(perm_idx, at::kInt, "perm_idx", {-1});
  check_tensor(rows, at::kBFloat16, "rows", {num_perm, kHidden});
  check_tensor(hq_data, at::kByte, "hq_data", {-1, kRowBytes});
  check_tensor(hq_sf, at::kByte, "hq_sf", {-1, kSfCols});
  TORCH_CHECK(hq_data.size(0) >= kMaxCols && hq_sf.size(0) >= kMaxCols,
              "hq scratch must hold ", kMaxCols, " rows");
  check_tensor(plan, at::kInt, "plan", {-1});
  TORCH_CHECK(plan.numel() * 4 >= static_cast<int64_t>(sizeof(Plan)),
              "plan scratch is too small");
  check_tensor(ctl, at::kInt, "ctl", {-1});
  TORCH_CHECK(ctl.numel() >= kCtlWords, "ctl must hold ", kCtlWords, " words");
  TORCH_CHECK(xq_data.has_value() == xq_sf.has_value(),
              "xq_data and xq_sf go together");
  TORCH_CHECK(num_perm % kTopK == 0, "perm_idx must hold T_PAD * 8 entries");
  const int64_t t_pad = num_perm / kTopK;
  TORCH_CHECK(T >= 1 && T <= t_pad, "token count ", T, " exceeds the padded ", t_pad);

  DecodeParams p{};
  p.hidden = reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr());
  p.topk_ids = reinterpret_cast<const int*>(topk_ids.data_ptr());
  p.expert_map = nullptr;
  p.num_global = 0;
  if (expert_map.has_value()) {
    check_tensor(*expert_map, at::kInt, "expert_map", {-1});
    p.expert_map = reinterpret_cast<const int*>(expert_map->data_ptr());
    p.num_global = static_cast<int>(expert_map->size(0));
  }
  p.num_local = static_cast<int>(num_local);
  p.w13 = reinterpret_cast<const uint8_t*>(w13.data_ptr());
  p.sf13 = reinterpret_cast<const uint8_t*>(sf13.data_ptr());
  p.w2 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
  p.sf2 = reinterpret_cast<const uint8_t*>(sf2.data_ptr());
  p.prm = reinterpret_cast<const float*>(prm.data_ptr());
  p.gs = reinterpret_cast<const float*>(gs.data_ptr());
  p.rows = reinterpret_cast<__nv_bfloat16*>(rows.data_ptr());
  p.perm_idx = reinterpret_cast<int*>(perm_idx.data_ptr());
  p.hq_data = reinterpret_cast<uint8_t*>(hq_data.data_ptr());
  p.hq_sf = reinterpret_cast<uint8_t*>(hq_sf.data_ptr());
  p.plan = reinterpret_cast<int*>(plan.data_ptr());
  p.dbg_acc = dbg_acc.has_value() ? reinterpret_cast<float*>(dbg_acc->data_ptr())
                                  : nullptr;
  p.ctl = reinterpret_cast<unsigned*>(ctl.data_ptr());
  p.xq_data = nullptr;
  p.xq_sf = nullptr;
  if (xq_data.has_value()) {
    check_tensor(*xq_data, at::kByte, "xq_data", {T, kRowBytes});
    check_tensor(*xq_sf, at::kByte, "xq_sf", {T, kSfCols});
    p.xq_data = reinterpret_cast<const uint8_t*>(xq_data->data_ptr());
    p.xq_sf = reinterpret_cast<const uint8_t*>(xq_sf->data_ptr());
  }
  p.flags = static_cast<int>(flags & 0xffff);
  p.T = static_cast<int>(T);
  p.num_perm = static_cast<int>(num_perm);
  p.up_first = up_first ? 1 : 0;

  const c10::cuda::OptionalCUDAGuard guard(hidden.device());
  int grid = static_cast<int>(flags >> 16);
  if (grid <= 0) grid = MUSE_DECODE_GRID > 0 ? MUSE_DECODE_GRID : sm_count(hidden.get_device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  switch (t_pad) {
    case 1:
      launch_decode<1>(p, grid, stream);
      break;
    case 2:
      launch_decode<2>(p, grid, stream);
      break;
    case 4:
      launch_decode<4>(p, grid, stream);
      break;
    case 8:
      launch_decode<8>(p, grid, stream);
      break;
    default:
      TORCH_CHECK(false, "the padded token count must be 1, 2, 4 or 8, got ", t_pad);
  }
}

void nvfp4_decode_moe_large(
    at::Tensor const& hidden, at::Tensor const& topk_ids,
    std::optional<at::Tensor> const& expert_map, at::Tensor const& w13,
    at::Tensor const& sf13, at::Tensor const& w2, at::Tensor const& sf2,
    at::Tensor const& prm, at::Tensor const& gs, bool up_first, at::Tensor& rows,
    at::Tensor& perm_idx, at::Tensor& hq_data, at::Tensor& hq_sf,
    at::Tensor& xq_data, at::Tensor& xq_sf, at::Tensor& plan, at::Tensor& ctl,
    int64_t flags) {
  const int64_t T = hidden.size(0);
  const int64_t num_routes = T * kTopK;
  check_tensor(hidden, at::kBFloat16, "hidden", {-1, kHidden});
  check_tensor(topk_ids, at::kInt, "topk_ids", {T, kTopK});
  TORCH_CHECK(T >= 1 && T <= kLargeMaxTokens, "token count ", T,
              " is outside 1..", kLargeMaxTokens);
  const int64_t num_local = w13.size(0);
  TORCH_CHECK(num_local >= 1 && num_local <= kLargeMaxLocal, "w13 holds ", num_local,
              " experts, the large-batch kernel takes 1..", kLargeMaxLocal);
  check_tensor(w13, at::kByte, "w13", {num_local, kW13Rows, kRowBytes});
  check_tensor(sf13, at::kFloat8_e4m3fn, "sf13", {num_local, kW13Rows, kSfCols});
  check_tensor(w2, at::kByte, "w2", {num_local, kHidden, kRowBytes});
  check_tensor(sf2, at::kFloat8_e4m3fn, "sf2", {num_local, kHidden, kSfCols});
  check_tensor(prm, at::kFloat, "prm", {num_local, 4});
  check_tensor(gs, at::kFloat, "gs", {2});
  // The outputs and the scratch hold the largest batch; a smaller one uses
  // their leading rows.
  check_tensor(rows, at::kBFloat16, "rows", {num_routes, kHidden});
  check_tensor(perm_idx, at::kInt, "perm_idx", {num_routes});
  check_tensor(hq_data, at::kByte, "hq_data", {-1, kRowBytes});
  check_tensor(hq_sf, at::kByte, "hq_sf", {-1, kSfCols});
  TORCH_CHECK(hq_data.size(0) >= num_routes && hq_sf.size(0) >= num_routes,
              "hq scratch must hold one row per route");
  check_tensor(xq_data, at::kByte, "xq_data", {-1, kRowBytes});
  check_tensor(xq_sf, at::kByte, "xq_sf", {-1, kSfCols});
  TORCH_CHECK(xq_data.size(0) >= T && xq_sf.size(0) >= T,
              "activation scratch must hold one row per token");
  // The loader warp copies whole rows of these in 16-byte pieces or bulk copies; the
  // pitch of a row is a multiple of 16 bytes, so the base must be aligned.
  for (const at::Tensor* t : {&hq_data, &hq_sf, &xq_data, &xq_sf}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "the intermediate and activation scratch must be 16-byte aligned");
  }
  check_tensor(plan, at::kInt, "plan", {-1});
  TORCH_CHECK(plan.numel() * 4 >= static_cast<int64_t>(sizeof(PlanLarge)),
              "plan scratch is too small");
  check_tensor(ctl, at::kInt, "ctl", {-1});
  TORCH_CHECK(ctl.numel() >= kCtlWordsLarge, "ctl must hold ", kCtlWordsLarge,
              " words");

  DecodeParams p{};
  p.hidden = reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr());
  p.topk_ids = reinterpret_cast<const int*>(topk_ids.data_ptr());
  p.expert_map = nullptr;
  p.num_global = 0;
  if (expert_map.has_value()) {
    check_tensor(*expert_map, at::kInt, "expert_map", {-1});
    p.expert_map = reinterpret_cast<const int*>(expert_map->data_ptr());
    p.num_global = static_cast<int>(expert_map->size(0));
  }
  p.num_local = static_cast<int>(num_local);
  p.w13 = reinterpret_cast<const uint8_t*>(w13.data_ptr());
  p.sf13 = reinterpret_cast<const uint8_t*>(sf13.data_ptr());
  p.w2 = reinterpret_cast<const uint8_t*>(w2.data_ptr());
  p.sf2 = reinterpret_cast<const uint8_t*>(sf2.data_ptr());
  p.prm = reinterpret_cast<const float*>(prm.data_ptr());
  p.gs = reinterpret_cast<const float*>(gs.data_ptr());
  p.rows = reinterpret_cast<__nv_bfloat16*>(rows.data_ptr());
  p.perm_idx = reinterpret_cast<int*>(perm_idx.data_ptr());
  p.hq_data = reinterpret_cast<uint8_t*>(hq_data.data_ptr());
  p.hq_sf = reinterpret_cast<uint8_t*>(hq_sf.data_ptr());
  p.plan = reinterpret_cast<int*>(plan.data_ptr());
  p.xq_buf = reinterpret_cast<uint8_t*>(xq_data.data_ptr());
  p.xq_sf_buf = reinterpret_cast<uint8_t*>(xq_sf.data_ptr());
  p.dbg_acc = nullptr;
  p.ctl = reinterpret_cast<unsigned*>(ctl.data_ptr());
  p.xq_data = nullptr;
  p.xq_sf = nullptr;
  p.flags = static_cast<int>(flags & 0xffff);
  p.T = static_cast<int>(T);
  p.num_perm = static_cast<int>(num_routes);
  p.up_first = up_first ? 1 : 0;

  const c10::cuda::OptionalCUDAGuard guard(hidden.device());
  int grid = static_cast<int>(flags >> 16);
  if (grid <= 0) {
    grid = MUSE_DECODE_GRID > 0 ? MUSE_DECODE_GRID : sm_count(hidden.get_device());
  }
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  launch_large(p, grid, stream);
}

void nvfp4_decode_act_quant(at::Tensor const& hidden, at::Tensor const& gs,
                            at::Tensor& data, at::Tensor& sf) {
  const int64_t T = hidden.size(0);
  check_tensor(hidden, at::kBFloat16, "hidden", {-1, kHidden});
  check_tensor(gs, at::kFloat, "gs", {-1});
  check_tensor(data, at::kByte, "data", {T, kHidden / 2});
  check_tensor(sf, at::kByte, "sf", {T, kSfCols});
  TORCH_CHECK(T >= 1 && T <= 8, "at most 8 tokens");
  const c10::cuda::OptionalCUDAGuard guard(hidden.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  const int smem = static_cast<int>(T) * kXqBytes;
  act_quant_kernel<128><<<1, 128, smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr()),
      reinterpret_cast<const float*>(gs.data_ptr()), static_cast<int>(T),
      reinterpret_cast<uint8_t*>(data.data_ptr()),
      reinterpret_cast<uint8_t*>(sf.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace muse_spark
