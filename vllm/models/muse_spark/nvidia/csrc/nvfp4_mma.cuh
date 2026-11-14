// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// PTX helpers of the NVFP4 decode MoE kernel (muse_nvfp4_decode_moe.cu) for
// SM120: the block-scaled FP4 tensor core MMA, ldmatrix, mbarriers, bulk and
// asynchronous copies. Everything here is a thin wrapper around one
// instruction; the fragment maps are documented where they are used.

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace muse_spark {
namespace nvfp4 {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- tensor core ------------------------------------------------------------

// D += A * B for one m16n8k64 tile of e2m1 values with e4m3 block scales (one
// scale per 16 K elements, four per operand row / column). With lane L,
// q = L >> 2 and c = L & 3:
//   a0..a3: A[q + 8 * (r & 1)][32 * (r >> 1) + 8 * c .. +7]   (r = 0..3)
//   b0, b1: B[k = 32 * j + 8 * c .. +7][column q]               (j = 0, 1)
//   sfa:    the 4 scales of A row q + 8 * (c & 1) (lanes c = 2, 3 are ignored)
//   sfb:    the 4 scales of B column q
//   d0..d3: D[q][2c], D[q][2c + 1], D[q + 8][2c], D[q + 8][2c + 1]
__device__ __forceinline__ void mma_nvfp4_m16n8k64(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1, uint32_t sfa, uint32_t sfb) {
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, "
      "{%13}, {%14,%15};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
        "h"(static_cast<uint16_t>(0)), "h"(static_cast<uint16_t>(0)),
        "r"(sfb), "h"(static_cast<uint16_t>(0)),
        "h"(static_cast<uint16_t>(0)));
}

// Lane L supplies the address of row (L & 7) + 8 * ((L >> 3) & 1), byte
// (L >> 4) * 16 of a 16-row tile: the result is the A fragment above.
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0,
                                            uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

// Lanes 0-15 supply the address of row (L & 7), byte ((L >> 3) & 1) * 16 of an
// 8-row tile: the result is the B fragment above.
__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t& r0,
                                            uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(addr));
}

// Two e2m1 codes of two floats: `lo` in the low nibble, `hi` in the high one.
__device__ __forceinline__ uint32_t cvt_e2m1x2(float lo, float hi) {
  uint32_t r;
  asm("{\n"
      ".reg .b8 b;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
      "cvt.u32.u8 %0, b;\n"
      "}\n"
      : "=r"(r)
      : "f"(lo), "f"(hi));
  return r;
}

// ---- synchronization --------------------------------------------------------

// Barrier among the `count` threads that name it (id 1..15; 0 is __syncthreads).
__device__ __forceinline__ void named_barrier(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count));
}

__device__ __forceinline__ void mbar_fence_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// One arrival plus `bytes` of expected asynchronous (bulk copy) traffic.
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar,
                                                      uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar),
               "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

// Blocks until the phase with the given parity of `bar` has completed.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
  }
}

// ---- copies -----------------------------------------------------------------

// Global -> shared bulk copy (16-byte aligned, multiple of 16 bytes) that
// completes on `bar` through its transaction count.
__device__ __forceinline__ void bulk_copy_g2s(uint32_t dst, const void* src,
                                              uint32_t bytes, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void cp_async_4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src)
               : "memory");
}

// 16-byte copy through L2 only (the source may have been written by an
// earlier kernel, never use the non-coherent path for it).
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src)
               : "memory");
}

// Arrive on `bar` once every cp.async of this thread so far has landed. The
// arrival is not pre-counted: the barrier's expected count includes it.
__device__ __forceinline__ void cp_async_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar)
               : "memory");
}


// ---- global memory counters --------------------------------------------------

// Load with acquire semantics at gpu scope (spin on a counter that other CTAs
// release with red_release_add_gpu).
__device__ __forceinline__ unsigned ld_acquire_gpu(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// Add with release semantics at gpu scope: every write of the calling thread
// (and, through a preceding bar / syncwarp, of the threads that synchronized
// with it) is visible to a reader that acquires the new value.
__device__ __forceinline__ void red_release_add_gpu(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(p), "r"(v)
               : "memory");
}

__device__ __forceinline__ void fence_acq_rel_gpu() {
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
}

}  // namespace nvfp4
}  // namespace muse_spark
