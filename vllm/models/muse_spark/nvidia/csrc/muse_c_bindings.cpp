// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// torch library registration for the `_muse_C` extension. To add a kernel:
// declare it in muse_c_ops.h, add its .cu file to _SOURCES in
// vllm/models/muse_spark/nvidia/ops/muse_c.py and register it here.

#include <torch/library.h>

#include "muse_c_ops.h"

TORCH_LIBRARY(_muse_C, m) {
  // [up | gate] variant of silu_and_mul_scaled_fp4_experts_quant.
  m.def(
      "mul_and_silu_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale, Tensor input, Tensor input_global_scale, "
      "Tensor input_offset_by_experts, "
      "Tensor output_scale_offset_by_experts) -> ()");

  // Exact GEMM1 of the CUTLASS grouped-GEMM path (separate gate / up alphas).
  m.def(
      "silu_and_mul_split_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale, Tensor input, Tensor input_global_scale, "
      "Tensor input_offset_by_experts, "
      "Tensor output_scale_offset_by_experts, bool gate_first) -> ()");
  m.def(
      "split_gemm1_prepare(Tensor a_fp4, Tensor expert_offsets, "
      "Tensor blockscale_offsets, Tensor problem_sizes, Tensor! a_fp4_x2, "
      "Tensor! expert_offsets_x2, Tensor! blockscale_offsets_x2, "
      "Tensor! problem_sizes_x2) -> ()");

  // Stock scaled_fp4_experts_quant of the rows input[row_map[r]].
  m.def(
      "gather_scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale, "
      "Tensor input, Tensor row_map, Tensor input_global_scale, "
      "Tensor input_offset_by_experts, "
      "Tensor output_scale_offset_by_experts) -> ()");

  // Split of the gather quantization around a collective: pack the rows as fp4
  // codes + linear scales, then gather them into the experts' permuted rows.
  m.def(
      "pack_nvfp4_rows(Tensor! packed, Tensor input, Tensor "
      "input_global_scale) -> ()");
  m.def(
      "gather_packed_fp4_experts(Tensor! output, Tensor! output_scale, "
      "Tensor packed, Tensor row_map, Tensor input_offset_by_experts, "
      "Tensor output_scale_offset_by_experts, int k) -> ()");

  // Routing data of the CUTLASS grouped GEMMs of an EP rank.
  m.def(
      "moe_route_local(Tensor topk_ids, Tensor? expert_map, "
      "int num_local_experts, int n, int k, Tensor! expert_offsets, "
      "Tensor! blockscale_offsets, Tensor! problem_sizes1, "
      "Tensor! problem_sizes2, Tensor! a_map, Tensor! c_map) -> ()");

  // NVFP4 (W4A4) MoE experts for a few tokens (decode), see muse_c_ops.h.
  m.def(
      "nvfp4_decode_moe(Tensor hidden, Tensor topk_ids, Tensor? expert_map, "
      "Tensor w13, Tensor sf13, Tensor w2, Tensor sf2, Tensor prm, "
      "Tensor gs, bool up_first, Tensor! rows, Tensor! perm_idx, "
      "Tensor! hq_data, Tensor! hq_sf, Tensor! plan, Tensor! ctl, "
      "Tensor? dbg_acc=None, int flags=0, Tensor? xq_data=None, "
      "Tensor? xq_sf=None) -> ()");
  m.def(
      "nvfp4_decode_moe_large(Tensor hidden, Tensor topk_ids, Tensor? "
      "expert_map, Tensor w13, Tensor sf13, Tensor w2, Tensor sf2, Tensor prm, "
      "Tensor gs, bool up_first, Tensor! rows, Tensor! perm_idx, "
      "Tensor! hq_data, Tensor! hq_sf, Tensor! xq_data, Tensor! xq_sf, "
      "Tensor! plan, Tensor! ctl, int flags=0) -> ()");
  m.def(
      "nvfp4_decode_act_quant(Tensor hidden, Tensor gs, Tensor! data, "
      "Tensor! sf) -> ()");

  // Decode weight prefetch into L2, see muse_c_ops.h.
  m.def(
      "l2_prefetch(Tensor[] tensors, int num_ctas, int chunk_kib, "
      "bool evict_last) -> ()");
  m.def(
      "expert_prefetch(Tensor topk_ids, Tensor? expert_map, Tensor w13, "
      "Tensor sf13, Tensor w2, Tensor sf2, bool up_first, int num_ctas, "
      "int budget_bytes, bool pdl, Tensor!? dump=None) -> ()");

  // Skinny split-K GEMM of the decode layers, see muse_c_ops.h.
  m.def(
      "decode_gemm(Tensor x, Tensor w, Tensor! out, Tensor! scratch, "
      "int cfg=-1, bool early_w=False, bool pdl=False) -> ()");
  m.def("decode_gemm_scratch_floats(int N, int rows, int bn, int split) -> int");
  m.def("decode_gemm_num_configs() -> int");
  m.def("decode_gemm_is_tuned(int N, int K) -> bool");
  m.def("decode_gemm_config(int i) -> int[]");

  // Router GEMV of the decode layer, see muse_c_ops.h.
  m.def("router_gemv(Tensor x, Tensor w, Tensor! out, bool pdl) -> ()");

  // Muse Spark attention ops of vllm-project/vllm#62132; the schemas are those
  // of the stock torch.ops._C ops, except for the trailing q_out of the fused
  // insert (callers check the schema before they pass it).
  m.def(
      "muse_spark_mul_tanh_soft_cap(Tensor! out, Tensor input, "
      "float multiplier, float soft_cap_temp) -> ()");
  m.def(
      "fused_muse_spark_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor positions, int num_heads, int num_kv_heads, "
      "int head_dim, float eps, Tensor? cos_sin_cache, int rotary_dim, "
      "Tensor? slot_mapping, Tensor!? kv_cache, int block_size, "
      "str kv_cache_dtype, Tensor? k_scale=None, Tensor? v_scale=None, "
      "Tensor!? q_fp8_out=None, Tensor? q_scale=None, Tensor!? q_out=None) "
      "-> ()");
}

// The tile table of decode_gemm: no tensor argument, so no backend key to dispatch on.
TORCH_LIBRARY_IMPL(_muse_C, CompositeExplicitAutograd, m) {
  m.impl("decode_gemm_num_configs", &muse_spark::decode_gemm_num_configs);
  m.impl("decode_gemm_is_tuned", &muse_spark::decode_gemm_is_tuned);
  m.impl("decode_gemm_config", &muse_spark::decode_gemm_config);
  m.impl("decode_gemm_scratch_floats", &muse_spark::decode_gemm_scratch_floats);
}

TORCH_LIBRARY_IMPL(_muse_C, CUDA, m) {
  m.impl("mul_and_silu_scaled_fp4_experts_quant",
         &muse_spark::mul_and_silu_scaled_fp4_experts_quant);
  m.impl("silu_and_mul_split_scaled_fp4_experts_quant",
         &muse_spark::silu_and_mul_split_scaled_fp4_experts_quant);
  m.impl("split_gemm1_prepare", &muse_spark::split_gemm1_prepare);
  m.impl("gather_scaled_fp4_experts_quant",
         &muse_spark::gather_scaled_fp4_experts_quant);
  m.impl("pack_nvfp4_rows", &muse_spark::pack_nvfp4_rows);
  m.impl("gather_packed_fp4_experts", &muse_spark::gather_packed_fp4_experts);
  m.impl("moe_route_local", &muse_spark::moe_route_local);
  m.impl("nvfp4_decode_moe", &muse_spark::nvfp4_decode_moe);
  m.impl("nvfp4_decode_moe_large", &muse_spark::nvfp4_decode_moe_large);
  m.impl("nvfp4_decode_act_quant", &muse_spark::nvfp4_decode_act_quant);
  m.impl("l2_prefetch", &muse_spark::l2_prefetch);
  m.impl("expert_prefetch", &muse_spark::expert_prefetch);
  m.impl("router_gemv", &muse_spark::router_gemv);
  m.impl("decode_gemm", &muse_spark::decode_gemm);
  m.impl("muse_spark_mul_tanh_soft_cap",
         &muse_spark::muse_spark_mul_tanh_soft_cap);
  m.impl("fused_muse_spark_qknorm_rope_kv_insert",
         &muse_spark::fused_muse_spark_qknorm_rope_kv_insert);
}
