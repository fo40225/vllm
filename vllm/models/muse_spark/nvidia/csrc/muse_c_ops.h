// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Declarations of the kernels exposed through the `_muse_C` torch library.
// Register each one in muse_c_bindings.cpp.

#pragma once

#include <ATen/core/Tensor.h>

#include <optional>
#include <string>
#include <vector>

namespace muse_spark {

// Fused mul-and-silu + per-expert NVFP4 quantization. The input halves are
// [up | gate], so the result is quant(up * silu(gate)). This is the stock
// silu_and_mul_scaled_fp4_experts_quant with the two halves exchanged.
void mul_and_silu_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts);

// SiLU * up + per-expert NVFP4 quantization of the exact GEMM1's output, which
// holds the two halves as stacked matrices: input [2 * m_topk, k], the first
// half in rows [0, m_topk) and the second in rows [m_topk, 2 * m_topk). The
// first half is the gate if gate_first, else the up half. Quantizes
// up * silu(gate) to output [m_topk, k / 2], bit-identical to the stock
// silu_and_mul_scaled_fp4_experts_quant on the same values laid out
// [gate | up].
void silu_and_mul_split_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts, bool gate_first);

// Tables of the exact GEMM1 (muse_nvfp4_split_gemm1.cu): the grouped GEMM
// reads w13 [E, 2N, K / 2] as [2E, N, K / 2], group 2e + h being half h of
// expert e, on two copies of the activation rows. From the tables of the E
// experts (expert_offsets and blockscale_offsets [E + 1], problem_sizes [E, 3]
// = (M, 2N, K), as get_cutlass_moe_mm_data makes them) and the activations
// a_fp4 [R, K / 2], it writes expert_offsets_x2 and blockscale_offsets_x2
// [2E], problem_sizes_x2 [2E, 3] = (M, N, K), and the rows that experts own,
// twice, to a_fp4_x2 [2R, K / 2] (copy h at rows [h * R, (h + 1) * R)); the
// other rows are untouched.
void split_gemm1_prepare(at::Tensor const& a_fp4,
                         at::Tensor const& expert_offsets,
                         at::Tensor const& blockscale_offsets,
                         at::Tensor const& problem_sizes, at::Tensor& a_fp4_x2,
                         at::Tensor& expert_offsets_x2,
                         at::Tensor& blockscale_offsets_x2,
                         at::Tensor& problem_sizes_x2);

// The stock scaled_fp4_experts_quant of an activation matrix permuted by row_map,
// without the permuted copy: row r of the result is the quantization of row
// input[row_map[r]], with the global scale and scale-factor layout of the expert
// that owns row r. input [T, K] bf16/fp16 (K a multiple of 256), row_map [M]
// int32, output [M, K / 2] u8; only the first input_offset_by_experts[E] rows
// are written. It gives the bytes the stock op gives on the shuffled copy.
void gather_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& row_map, at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts);

// Dense NVFP4 quantization into packed rows (muse_nvfp4_experts_quant.cu), the
// producer half of gather_packed_fp4_experts. input [M, K] bf16/fp16 (K a
// multiple of 256), input_global_scale one float, packed [M, W] u8 with W a
// multiple of 16 and W >= K / 2 + K / 16: bytes [0, K / 2) of a row are the e2m1
// codes, [K / 2, K / 2 + K / 16) the linear e4m3 scale factors (the bytes the stock
// scaled_fp4_quant gives with is_sf_swizzled_layout=False); the tail of the row is
// left alone.
void pack_nvfp4_rows(at::Tensor& packed, at::Tensor const& input,
                     at::Tensor const& input_global_scale);

// gather_scaled_fp4_experts_quant on rows quantized by pack_nvfp4_rows (which may
// have travelled through a collective): row r of output [M, K / 2] is the code
// bytes of packed[row_map[r]], and its scale factors are written to the
// swizzled layout of the expert that owns row r in output_scale (int32,
// [rows, K / 64]). packed [T, W] as above with a 16 B aligned base address. Only
// the first input_offset_by_experts[E] rows are written; the bytes are those
// gather_scaled_fp4_experts_quant gives on the unpacked bf16 rows.
void gather_packed_fp4_experts(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& packed,
    at::Tensor const& row_map, at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts, int64_t k);

// Routing data of the CUTLASS grouped GEMMs of one expert-parallel rank
// (muse_moe_route.cu): what remap_topk_to_local and the stock
// get_cutlass_moe_mm_data give for the NVFP4 path, from the global ids.
//   topk_ids [T, top_k] int32 (global ids; -1 and ids outside the map are
//   remote), expert_map [E_global] int32 (local id or -1; null: the ids are
//   local already). Outputs, all int32: expert_offsets and blockscale_offsets
//   [E + 1] (rows of every expert, and the same padded to whole 128-row tiles),
//   problem_sizes1 [E, 3] = (m, 2n, k), problem_sizes2 [E, 3] = (m, k, n),
//   a_map [T * top_k] (token of every local row, -1 past them) and c_map
//   [T * top_k] (row of every route, -1 for a remote one).
void moe_route_local(at::Tensor const& topk_ids,
                     std::optional<at::Tensor> const& expert_map,
                     int64_t num_local_experts, int64_t n, int64_t k,
                     at::Tensor& expert_offsets, at::Tensor& blockscale_offsets,
                     at::Tensor& problem_sizes1, at::Tensor& problem_sizes2,
                     at::Tensor& a_map, at::Tensor& c_map);

// NVFP4 (W4A4) MoE experts for a few tokens on SM120 (muse_nvfp4_decode_moe.cu).
// One persistent launch: routing + activation quantization + GEMM1 + SiLU * up +
// NVFP4 requantization into hq, then GEMM2 into route-major unweighted bf16
// `rows` [T_PAD * 8, 4096] (rows of non-local routes are left untouched) and
// `perm_idx` [T_PAD * 8] (route index, or -1 for a non-local route).
//   hidden [T, 4096] bf16, topk_ids [T, 8] int32 (global ids), expert_map
//   [E_global] int32 (null: identity), w13 [E_local, 8192, 2048] u8 and w2
//   [E_local, 4096, 2048] u8 (packed e2m1), sf13 [E_local, 8192, 256] / sf2
//   [E_local, 4096, 256] e4m3 swizzled, prm [E_local, 4] fp32 = (alpha_g, alpha_u,
//   alpha2, 0), gs [2] fp32 = (1 / a13, 1 / a2). w13 rows are [up; gate] if
//   up_first, else [gate; up]. hq_data [64, 2048] u8, hq_sf [64, 256] u8 and
//   plan (int32, >= 708 entries) are scratch; ctl (int32, >= 66 entries) holds
//   the persistent kernel's counters: zero it once, every launch leaves it zero.
//   dbg_acc optionally receives the raw GEMM1 sums. flags: 1 = two launches
//   (plan B), 2 = zero the counters with a memset node first, 4 = zero the rows
//   of non-local routes, 8 = no programmatic dependent launch (of K2 with 1, of
//   the persistent kernel otherwise); bits 16.. override the grid size (0 = one
//   CTA per SM).
//   xq_data [T, 2048] u8 / xq_sf [T, 256] u8 (linear scales), when given, are the
//   activations quantized by the stock op (ops.scaled_fp4_quant with
//   is_sf_swizzled_layout=False and gs[0]) and replace the in-kernel quantization.
void nvfp4_decode_moe(at::Tensor const& hidden, at::Tensor const& topk_ids,
                      std::optional<at::Tensor> const& expert_map,
                      at::Tensor const& w13, at::Tensor const& sf13,
                      at::Tensor const& w2, at::Tensor const& sf2,
                      at::Tensor const& prm, at::Tensor const& gs, bool up_first,
                      at::Tensor& rows, at::Tensor& perm_idx,
                      at::Tensor& hq_data, at::Tensor& hq_sf, at::Tensor& plan,
                      at::Tensor& ctl, std::optional<at::Tensor> const& dbg_acc,
                      int64_t flags, std::optional<at::Tensor> const& xq_data,
                      std::optional<at::Tensor> const& xq_sf);

// Large-batch variant of nvfp4_decode_moe for 1..128 tokens and up to 256 local
// experts (muse_nvfp4_decode_large.cuh, the persistent kernel with LARGE): the
// same expert computation, bit for bit the rows of nvfp4_decode_moe, with the
// activations quantized and the plan built by a first launch into global scratch
// instead of in every CTA.
//   hidden [T, 4096] bf16, topk_ids [T, 8] int32, expert_map, w13 .. gs, up_first
//   as above. rows [T * 8, 4096] bf16 and perm_idx [T * 8] int32 are the outputs
//   (leading rows of the caller's static buffers), hq_data [>= T * 8, 2048] u8 and
//   hq_sf [>= T * 8, 256] u8 the intermediate, xq_data [>= T, 2048] u8 and xq_sf
//   [>= T, 256] u8 the quantized activations (scratch, written by the first
//   launch), plan (int32, >= 6084 entries) and ctl (int32, >= 322 entries, zero it
//   once, every launch leaves it zero) as above (16 columns per entry; the 8-column
//   build, MUSE_LARGE_COLS=8, needs 4228 and 386). flags: 4 = zero the rows
//   of non-local routes, 8 = no programmatic dependent launches, 16 = the loader
//   warp fetches the B operand with bulk copies (the scratch above must be 16-byte
//   aligned, as it is for every launch); the upper 16 bits (flags >> 16) override
//   the grid size of the persistent kernel.
void nvfp4_decode_moe_large(
    at::Tensor const& hidden, at::Tensor const& topk_ids,
    std::optional<at::Tensor> const& expert_map, at::Tensor const& w13,
    at::Tensor const& sf13, at::Tensor const& w2, at::Tensor const& sf2,
    at::Tensor const& prm, at::Tensor const& gs, bool up_first, at::Tensor& rows,
    at::Tensor& perm_idx, at::Tensor& hq_data, at::Tensor& hq_sf,
    at::Tensor& xq_data, at::Tensor& xq_sf, at::Tensor& plan, at::Tensor& ctl,
    int64_t flags);

// The in-kernel activation quantization of nvfp4_decode_moe on its own, for
// tests: hidden [T, 4096] bf16 -> codes [T, 2048] u8 and linear e4m3 scales
// [T, 256] u8, gs[0] = 1 / a13.
void nvfp4_decode_act_quant(at::Tensor const& hidden, at::Tensor const& gs,
                            at::Tensor& data, at::Tensor& sf);

// Muse Spark attention-side ops, ported from vllm-project/vllm#62132 (the
// stock `torch.ops._C` versions of the same ops). Schemas are identical.

// out = soft_cap_temp * tanh(input * multiplier / soft_cap_temp), in fp32.
// input is float32, float16 or bfloat16; out is float32 with the same shape.
void muse_spark_mul_tanh_soft_cap(at::Tensor& out, at::Tensor const& input,
                                  double multiplier, double soft_cap_temp);

// In-place weightless QK RMSNorm + GPT-J RoPE over [q | k | v | o_gate] rows,
// optionally writing k/v into a paged [nb, nkv, bs, 2*head_dim] KV cache
// (bf16/fp16, or fp8 with k_scale/v_scale) and Q as scaled E4M3 into
// q_fp8_out. With q_out (a dense [N, num_heads * head_dim] tensor of the qkv
// dtype) Q goes there, bit for bit what the in-place store would have written,
// and the Q columns of qkv stay untouched: the attention kernel reads a dense
// Q, which a strided view of qkv is not (a copy in front of it otherwise).
void fused_muse_spark_qknorm_rope_kv_insert(
    at::Tensor& qkv, at::Tensor const& positions, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_dim, double eps,
    std::optional<at::Tensor> const& cos_sin_cache, int64_t rotary_dim,
    std::optional<at::Tensor> const& slot_mapping,
    std::optional<at::Tensor> const& kv_cache, int64_t block_size,
    std::string const& kv_cache_dtype,
    std::optional<at::Tensor> const& k_scale,
    std::optional<at::Tensor> const& v_scale,
    std::optional<at::Tensor> const& q_fp8_out,
    std::optional<at::Tensor> const& q_scale,
    std::optional<at::Tensor> const& q_out);

// Asynchronous L2 prefetch of up to 4 contiguous 16 B aligned tensors
// (muse_spark_l2_prefetch.cu): each CTA issues cp.async.bulk.prefetch.L2 for its
// share of `chunk_kib` KiB chunks and exits. With evict_last the lines get the
// L2 evict-last priority. Reads and writes no tensor data.
void l2_prefetch(std::vector<at::Tensor> const& tensors, int64_t num_ctas,
                 int64_t chunk_kib, bool evict_last);

// Routing-aware L2 prefetch of the experts nvfp4_decode_moe is about to read
// (muse_spark_expert_prefetch.cu): maps topk_ids [T, 8] (T <= 8, global expert
// ids; expert_map is the global-to-local map, null if the ids are local) to the
// distinct local experts and prefetches their w13 / sf13 / w2 / sf2 slices (the
// layout of nvfp4_decode_moe) in the order that kernel consumes them, GEMM1 of
// all entries first, up to budget_bytes. With pdl it is launched as a
// programmatic dependent of the kernel in front of it in the stream (the
// routing kernel) and waits for it before it reads the ids. Reads the ids,
// writes nothing. For tests, `dump` (int64) receives the (address, size) pairs
// of the requests, 2 words each, in place of the prefetches.
void expert_prefetch(at::Tensor const& topk_ids,
                     std::optional<at::Tensor> const& expert_map,
                     at::Tensor const& w13, at::Tensor const& sf13,
                     at::Tensor const& w2, at::Tensor const& sf2, bool up_first,
                     int64_t num_ctas, int64_t budget_bytes, bool pdl,
                     std::optional<at::Tensor> const& dump);

// Router GEMV of the decode layer (muse_spark_router_gemv.cu): out [T, N] =
// x [T, 8192] . w [N, 8192]^T in float32 for 1 <= T <= 8, in a fixed order that
// does not depend on T. With pdl it is launched as a programmatic dependent
// and loads the weights before it waits for the kernel in front of it.
void router_gemv(at::Tensor const& x, at::Tensor const& w, at::Tensor& out,
                 bool pdl);

}  // namespace muse_spark
