// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// The persistent single-launch engine of the NVFP4 decode MoE
// (muse_nvfp4_decode_moe.cu includes this file inside its anonymous namespace,
// after the constants, the plan builder and the helpers it uses).
//
// Work queue. N1 = 256 * n_entries. Ticket t < N1 is the GEMM1 unit (g, w) of
// k1_gemm1 (g = t >> 8, w = t & 255): 4 stages (up / gate tile x 2 K halves).
// Ticket N1 <= t < 2 N1 is the GEMM2 unit (g, o) of k2_gemm2 (g = (t - N1) >> 8,
// o = (t - N1) & 255): 2 stages. A CTA claims tickets with an atomicAdd on
// ctl[0], one unit ahead of the stage it is loading, so every GEMM1 ticket is
// claimed (by a CTA that is running) before any GEMM2 ticket. A GEMM2 unit waits
// for ctl done[g] == 256, which the epilogues of the 256 GEMM1 units of the
// entry increment with release semantics after storing their slice of hq.
//
// Roles (192 threads): warps 0-3 consumers (MMA and epilogue), warp 4 weight
// producer and ticket claimer, warp 5 loader of the GEMM2 B operand (the
// intermediate). Every pipeline stage has a header (ticket * 4 + stage of the
// unit) written by the producer behind hdr_ready[slot]; a stop ticket (>= 2 N1)
// is the last stage of every CTA, so all three roles leave their loops together.
//
// Shared memory: [kStages weight slots][union][partial sums]. The union holds
// the quantized activations (GEMM1) or, after the consumers of the CTA are done
// with them (xq_free), kStages slots of the intermediate (GEMM2).
//
// Counters (ctl, uint32): [0] ticket, [1] exit counter, [2 + g] done of entry g.
// The CTA that exits last resets all of them, so a launch leaves them zero;
// kFlagCtlMemset additionally zeroes them with a memset node before the launch.

constexpr int kCtlTicket = 0;
constexpr int kCtlExit = 1;
constexpr int kCtlDone = 2;
constexpr int kCtlWords = kCtlDone + kMaxEntries;
constexpr unsigned kDoneTarget = kUnitsPerEntry;

constexpr int kHcolBytes = kHPitch + kHsfPitch;  // codes + scales of one column

template <int T_PAD>
constexpr int cmax_of() {
  return T_PAD < MUSE_DECODE_CMAX ? T_PAD : MUSE_DECODE_CMAX;
}
template <int T_PAD>
constexpr int persist_union_bytes() {
  constexpr int x = T_PAD * kXqBytes;
  constexpr int h = kStages * cmax_of<T_PAD>() * kHcolBytes;
  return x > h ? x : h;
}
template <int T_PAD>
constexpr int smem_bytes_persist() {
  return kStages * kSlot1 + persist_union_bytes<T_PAD>() + kPartBytes1;
}

template <int T_PAD>
__global__ void __launch_bounds__(kBlockThreads, 1)
    k_persistent(const DecodeParams p) {
  constexpr int CMAX = cmax_of<T_PAD>();
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ Plan plan;
  __shared__ PlanScratch scratch;
  __shared__ __align__(8) uint64_t full[kStages];
  __shared__ __align__(8) uint64_t empty[kStages];
  __shared__ __align__(8) uint64_t hdr_ready[kStages];
  __shared__ __align__(8) uint64_t xq_free;
  __shared__ int hdr[kStages];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  uint8_t* const slots = smem;
  uint8_t* const uni = smem + kStages * kSlot1;  // x_q, then the H slots
  float4* const part = reinterpret_cast<float4*>(uni + persist_union_bytes<T_PAD>());
  unsigned* const ctl = p.ctl;
#ifdef MUSE_DECODE_TRACE
  // Per-CTA timestamps (globaltimer, ns) into dbg_acc viewed as uint64 [grid][8]:
  // 0 entry, 1 plan built, 2 first full barrier seen (consumer 0), 3 first unit
  // done (consumer 0), 4 exit.
  unsigned long long* const trace =
      reinterpret_cast<unsigned long long*>(p.dbg_acc) + blockIdx.x * 8;
  auto stamp = [&](int i) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    trace[i] = t;
  };
  if (tid == 0) stamp(0);
#define MUSE_STAMP(i, cond) if (cond) stamp(i)
#else
#define MUSE_STAMP(i, cond)
#endif


  if (tid == 0) {
    for (int i = 0; i < kStages; ++i) {
      mbar_init(smem_u32(&full[i]), 1 + 32 + 32);
      mbar_init(smem_u32(&empty[i]), 4);
      mbar_init(smem_u32(&hdr_ready[i]), 1);
    }
    mbar_init(smem_u32(&xq_free), 4);
    mbar_fence_init();
  }
  // PDL: everything above only touches this CTA's shared memory. The plan reads
  // the top-k ids and the weights read the activations, so wait for the
  // predecessor first (the routing kernel, or the pre-expert norm when the routing
  // runs on the aux stream; a no-op when this launch is not a programmatic
  // dependent). Then let the successor (the finalize kernel, which waits for
  // this kernel to finish before it reads anything) launch.
  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  build_plan(p, CMAX, plan, scratch);  // also the barrier-init fence
  MUSE_STAMP(1, tid == 0);
  if (blockIdx.x == 0) publish_plan(p, plan, scratch);
  if (p.flags & kFlagZeroNonLocal) zero_nonlocal_row(p, scratch);
  if (plan.n_entries == 0) {
    // No local route: no ticket was claimed and no counter was touched, so the
    // counters stay zero without the exit protocol.
    return;
  }

  // The activations are loaded only now: with an early load, the hidden rows of
  // all CTAs compete with the first weight stages and, at n = 0, delay the exit.
  PackedVec<__nv_bfloat16> hv[T_PAD * 4];
  if (warp < 4 && p.xq_data == nullptr) load_rows<128, T_PAD>(p.hidden, p.T, tid, hv);
  const unsigned n1 = kUnitsPerEntry * plan.n_entries;
  const unsigned total = 2 * n1;
  const int up_base = p.up_first ? 0 : kInter;
  const int gate_base = p.up_first ? kInter : 0;

  if (warp == kProducerWarp) {
    // ---- weight producer + ticket claimer ----
    unsigned next = 0;
    if (lane == 0) next = atomicAdd(&ctl[kCtlTicket], 1u);
    unsigned ticket = __shfl_sync(0xffffffffu, next, 0);
    int st = 0;
    for (;;) {
      const bool stop = ticket >= total;
      const bool ph1 = ticket < n1;
      const int n_stages = stop ? 1 : (ph1 ? 4 : 2);
      const unsigned t = ph1 ? ticket : ticket - n1;
      const int g = static_cast<int>(t >> 8);
      const int w = static_cast<int>(t & 255);
      const int e = stop ? 0 : plan.expert[g];
      for (int s = 0; s < n_stages; ++s, ++st) {
        const int slot = st % kStages;
        mbar_wait(smem_u32(&empty[slot]), ((st / kStages) & 1) ^ 1);
        const uint32_t fb = smem_u32(&full[slot]);
        if (lane == 0) {
          reinterpret_cast<volatile int*>(hdr)[slot] =
              static_cast<int>(ticket * 4 + s);
          mbar_arrive(smem_u32(&hdr_ready[slot]));  // release: publishes hdr
          mbar_arrive_expect_tx(fb, stop ? 0 : 16 * kHalfBytes);
        }
        __syncwarp();
        uint8_t* sl = slots + slot * kSlot1;
        if (!stop && ph1) {
          // w = (a << 3) | (xh << 2) | d, tile index jb = 8 a + 2 d + xh: the
          // four d siblings share 32-byte scale sectors.
          const int jb = 8 * (w >> 3) + 2 * (w & 3) + ((w >> 2) & 1);
          const int tile = s >> 1;  // 0: up, 1: gate
          const int kh = s & 1;
          const int row0 = (tile == 0 ? up_base : gate_base) + 16 * jb;
          if (lane < 16) {
            bulk_copy_g2s(smem_u32(sl + lane * kWPitch),
                          p.w13 + (static_cast<size_t>(e) * kW13Rows + row0 + lane) *
                                      kRowBytes + kh * kHalfBytes,
                          kHalfBytes, fb);
          }
          const uint8_t* sfp = p.sf13 + static_cast<size_t>(e) * kW13Rows * kSfCols;
#pragma unroll
          for (int i = 0; i < 16; ++i) {
            const int idx = lane + 32 * i;  // = ktl * 16 + row
            const int row = idx & 15;
            const int kt = kh * 32 + (idx >> 4);
            cp_async_4(smem_u32(sl + kWBytes + idx * 4),
                       sfp + sf_word_offset(row0 + row, kt));
          }
        } else if (!stop) {
          const int kh = s;
          if (lane < 16) {
            // Output row of tile row i: 128 (o >> 3) + 32 (i >> 2) + 4 (o & 7) +
            // (i & 3); it puts the scale words of a K64 tile into 64 contiguous
            // bytes.
            const int m = 128 * (w >> 3) + 32 * (lane >> 2) + 4 * (w & 7) + (lane & 3);
            bulk_copy_g2s(smem_u32(sl + lane * kWPitch),
                          p.w2 + (static_cast<size_t>(e) * kHidden + m) * kRowBytes +
                              kh * kHalfBytes,
                          kHalfBytes, fb);
          }
          const uint8_t* sfp = p.sf2 + static_cast<size_t>(e) * kHidden * kSfCols +
                               static_cast<size_t>((w >> 3) * kKTiles + kh * 32) * 512 +
                               (w & 7) * 64;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int ci = lane + 32 * i;  // 16-byte chunk: kt_local * 4 + part
            cp_async_16(smem_u32(sl + kWBytes + ci * 16),
                        sfp + static_cast<size_t>(ci >> 2) * 512 + (ci & 3) * 16);
          }
        }
        cp_async_arrive_noinc(fb);
        // Claim the next ticket once this unit's first stage is in flight; the
        // result is only needed after its last stage was issued.
        if (s == 0 && !stop && lane == 0) next = atomicAdd(&ctl[kCtlTicket], 1u);
      }
      if (stop) break;
      ticket = __shfl_sync(0xffffffffu, next, 0);
    }
  } else if (warp == kLoaderWarp) {
    // ---- loader of the GEMM2 B operand (the intermediate hq) ----
    bool xq_released = false;
    for (int st = 0;; ++st) {
      const int slot = st % kStages;
      mbar_wait(smem_u32(&hdr_ready[slot]), (st / kStages) & 1);
      const int h = reinterpret_cast<volatile int*>(hdr)[slot];
      const unsigned ticket = static_cast<unsigned>(h) >> 2;
      const int s = h & 3;
      const bool stop = ticket >= total;
      const uint32_t fb = smem_u32(&full[slot]);
      if (!stop && ticket >= n1) {
        const unsigned t = ticket - n1;
        const int g = static_cast<int>(t >> 8);
        // The H slots overlay the quantized activations.
        if (!xq_released) {
          mbar_wait(smem_u32(&xq_free), 0);
          xq_released = true;
        }
        if (s == 0) {
          // hq of the entry is complete once all its GEMM1 units have released.
          if (lane == 0) {
            while (ld_acquire_gpu(&ctl[kCtlDone + g]) < kDoneTarget) __nanosleep(32);
          }
          __syncwarp();
          fence_acq_rel_gpu();
        }
        const int cnt = plan.cnt[g];
        const int col0 = plan.col0[g];
        uint8_t* hs = uni + slot * (CMAX * kHcolBytes);
        for (int idx = lane; idx < cnt * 72; idx += 32) {
          const int n = idx / 72;
          const int ch = idx - n * 72;
          if (ch < 64) {
            cp_async_16(smem_u32(hs + n * kHPitch + ch * 16),
                        p.hq_data + static_cast<size_t>(col0 + n) * kRowBytes +
                            s * kHalfBytes + ch * 16);
          } else {
            cp_async_16(smem_u32(hs + CMAX * kHPitch + n * kHsfPitch + (ch - 64) * 16),
                        p.hq_sf + static_cast<size_t>(col0 + n) * kSfCols + s * 128 +
                            (ch - 64) * 16);
          }
        }
      }
      cp_async_arrive_noinc(fb);
      if (stop) break;
    }
  } else {
    // ---- consumers ----
    uint8_t* const xq = uni;
    if (p.xq_data == nullptr) {
      quantize_rows<128, T_PAD>(hv, p.T, p.gs[0], xq, tid);
    } else {
      load_xq_from_global<128>(p, xq, tid);
    }
    named_barrier(kConsumerBarrier, 128);

    const int q = lane >> 2;
    const int c = lane & 3;
    const int a_row = (lane & 7) + 8 * ((lane >> 3) & 1);
    const int sfa_row = q + 8 * (c & 1);
    // GEMM2: scale word of tile row r at K tile kt: kt * 16 words + (r & 3) * 4 + (r >> 2).
    const int sfa_word2 = (sfa_row & 3) * 4 + (sfa_row >> 2);
    const float gs1 = p.gs[1];
    bool xq_released = false;
    int st = 0;
    for (int ui = 0;; ++ui) {
      {
        const int slot = st % kStages;
        mbar_wait(smem_u32(&hdr_ready[slot]), (st / kStages) & 1);
      }
      const int h = reinterpret_cast<volatile int*>(hdr)[st % kStages];
      const unsigned ticket = static_cast<unsigned>(h) >> 2;
      const bool stop = ticket >= total;
      if (!xq_released && (stop || ticket >= n1)) {
        // Done with the quantized activations: the loader may overwrite them.
        xq_released = true;
        __syncwarp();
        if (lane == 0) mbar_arrive(smem_u32(&xq_free));
      }
      if (stop) break;
      float4* pbuf = part + (ui & 1) * (2 * 4 * 32);

      if (ticket < n1) {
        // ---- GEMM1 unit ----
        const int g = static_cast<int>(ticket >> 8);
        const int w = static_cast<int>(ticket & 255);
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
          const int slot = (st + s) % kStages;
          mbar_wait(smem_u32(&full[slot]), ((st + s) / kStages) & 1);
          const int tile = s >> 1;
          MUSE_STAMP(2, tid == 0 && ui == 0 && s == 0);
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
        st += 4;

        MUSE_STAMP(3, tid == 0 && ui == 0);
        // Partial sums of the four warps: stored by all, summed in warp order by
        // the epilogue warp (rotating, so none is always the late one).
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
#ifndef MUSE_DECODE_TRACE
        if (p.dbg_acc != nullptr) {
          // Raw sums: [unit][tile][row][col], rows q, q + 8, columns 2c, 2c + 1.
          float* d = p.dbg_acc + static_cast<size_t>(ticket) * 256;
#pragma unroll
          for (int t = 0; t < 2; ++t) {
            const float* v = t == 0 ? us : gt;
            d[t * 128 + q * 8 + 2 * c] = v[0];
            d[t * 128 + q * 8 + 2 * c + 1] = v[1];
            d[t * 128 + (q + 8) * 8 + 2 * c] = v[2];
            d[t * 128 + (q + 8) * 8 + 2 * c + 1] = v[3];
          }
        }
#endif

        // h = bf16(silu(bf16(gate * alpha_g)) * bf16(up * alpha_u)), as the stock
        // silu_and_mul quantization sees the bf16 GEMM1 output.
        const float alpha_g = p.prm[e * 4 + 0];
        const float alpha_u = p.prm[e * 4 + 1];
        float hh[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float gv = bf16_round(gt[i] * alpha_g);
          const float uv = bf16_round(us[i] * alpha_u);
          hh[i] = bf16_round(silu(gv) * uv);
        }
        // The 16 rows of the tile are one NVFP4 block per route column. A column
        // is held by the 8 lanes with equal c; its rows are q and q + 8.
        float amax0 = fmaxf(fabsf(hh[0]), fabsf(hh[2]));  // column 2c
        float amax1 = fmaxf(fabsf(hh[1]), fabsf(hh[3]));  // column 2c + 1
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
          v[i] = hh[i] * out_scale[i & 1];
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
        // Publish: every writing lane fences its stores, then one lane releases.
        __threadfence();
        __syncwarp();
        if (lane == 0) red_release_add_gpu(&ctl[kCtlDone + g], 1u);
      } else {
        // ---- GEMM2 unit ----
        const unsigned t2 = ticket - n1;
        const int g = static_cast<int>(t2 >> 8);
        const int o = static_cast<int>(t2 & 255);
        const int e = plan.expert[g];
        const int cnt = plan.cnt[g];
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int kh = 0; kh < 2; ++kh) {
          const int slot = (st + kh) % kStages;
          mbar_wait(smem_u32(&full[slot]), ((st + kh) / kStages) & 1);
          const uint8_t* sl = slots + slot * kSlot1;
          const uint8_t* hs = uni + slot * (CMAX * kHcolBytes);
          const uint32_t a_base = smem_u32(sl) + a_row * kWPitch + (lane >> 4) * 16;
          // Columns past the route count of the entry repeat the last one (never
          // a result): the B operand never reads unwritten shared memory.
          const uint32_t b_base = smem_u32(hs) + min(lane & 7, cnt - 1) * kHPitch +
                                  ((lane >> 3) & 1) * 16;
          const uint32_t* sf_w = reinterpret_cast<const uint32_t*>(sl + kWBytes);
          const uint32_t* sfb_w = reinterpret_cast<const uint32_t*>(
              hs + CMAX * kHPitch + min(q, cnt - 1) * kHsfPitch);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int kk = warp + 4 * i;
            mma_slice(acc, a_base + kk * 32, sf_w + kk * 16 + sfa_word2,
                      b_base + kk * 32, sfb_w + kk);
          }
          __syncwarp();
          if (lane == 0) mbar_arrive(smem_u32(&empty[slot]));
        }
        st += 2;
        pbuf[warp * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        named_barrier(kConsumerBarrier, 128);
        if (warp != (ui & 3)) continue;

        const float4 p0 = pbuf[0 * 32 + lane];
        const float4 p1 = pbuf[1 * 32 + lane];
        const float4 p2 = pbuf[2 * 32 + lane];
        const float4 p3 = pbuf[3 * 32 + lane];
        const float s4[4] = {((p0.x + p1.x) + p2.x) + p3.x, ((p0.y + p1.y) + p2.y) + p3.y,
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
                __float2bfloat16_rn(s4[i] * alpha2);
          }
        }
      }
    }
  }

  // ---- exit: the last CTA to finish resets the counters ----
  __syncthreads();
  MUSE_STAMP(4, tid == 0);
  if (tid == 0) {
    __threadfence();
    const unsigned old = atomicAdd(&ctl[kCtlExit], 1u);
    if (old == gridDim.x - 1) {
      __threadfence();
      volatile unsigned* v = ctl;
      for (int i = 0; i < kCtlWords; ++i) v[i] = 0u;
    }
  }
}
