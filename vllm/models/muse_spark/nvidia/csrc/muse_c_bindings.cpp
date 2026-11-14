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

  // Muse Spark attention ops of vllm-project/vllm#62132; the schemas are those
  // of the stock torch.ops._C ops.
  m.def(
      "muse_spark_mul_tanh_soft_cap(Tensor! out, Tensor input, "
      "float multiplier, float soft_cap_temp) -> ()");
  m.def(
      "fused_muse_spark_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor positions, int num_heads, int num_kv_heads, "
      "int head_dim, float eps, Tensor? cos_sin_cache, int rotary_dim, "
      "Tensor? slot_mapping, Tensor!? kv_cache, int block_size, "
      "str kv_cache_dtype, Tensor? k_scale=None, Tensor? v_scale=None, "
      "Tensor!? q_fp8_out=None, Tensor? q_scale=None) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(_muse_C, CUDA, m) {
  m.impl("mul_and_silu_scaled_fp4_experts_quant",
         &muse_spark::mul_and_silu_scaled_fp4_experts_quant);
  m.impl("muse_spark_mul_tanh_soft_cap",
         &muse_spark::muse_spark_mul_tanh_soft_cap);
  m.impl("fused_muse_spark_qknorm_rope_kv_insert",
         &muse_spark::fused_muse_spark_qknorm_rope_kv_insert);
}
