# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse-Spark QK-norm + RoPE + KV-cache insert as one launch.

Not a custom op since ``MuseSpark1xModel`` is ``@ignore_torch_compile``;
re-enabling compile requires hiding the forward-context lookup from dynamo.
"""

import torch

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.models.muse_spark.common.ops import resolve_compiled_op

logger = init_logger(__name__)

_FUSED_KV_INSERT_OP = "fused_muse_spark_qknorm_rope_kv_insert"

# nvfp4 pages a different layout and keeps the unfused path.
SUPPORTED_KV_CACHE_DTYPES = frozenset(
    {"auto", "bfloat16", "float16", "fp8", "fp8_e4m3", "fp8_e5m2"}
)


def supports_fused_kv_insert(attn_layer: torch.nn.Module) -> bool:
    """Whether this attention layer's backend and cache can take the insert.

    The backend must keep its KV write out of ``forward``. Off under DCP, whose
    FlashInfer prefill attends over the raw new-token k/v. Also off when no
    fused kernel is available (see ``resolve_compiled_op``). Call it from layer
    construction: the first call may build the JIT extension.
    """
    parallel_config = get_current_vllm_config().parallel_config
    if parallel_config.decode_context_parallel_size > 1:
        return False
    if attn_layer.attn_backend.forward_includes_kv_cache_update:
        return False
    if not attn_layer.attn_backend.supports_kv_cache_update_bypass:
        return False
    if attn_layer.kv_sharing_target_layer_name is not None:
        return False
    if attn_layer.kv_cache_dtype not in SUPPORTED_KV_CACHE_DTYPES:
        return False
    # Last, so that a layer that cannot use the kernel never triggers the build.
    return resolve_compiled_op(_FUSED_KV_INSERT_OP) is not None


def _op_has_dense_q_output(op: object) -> bool:
    schema = op.default._schema  # type: ignore[attr-defined]
    return any(arg.name == "q_out" for arg in schema.arguments)


def supports_dense_q(attn_layer: torch.nn.Module) -> bool:
    """Whether the insert can write Q densely for this layer's attention backend.

    Only the ``_muse_C`` op has the output (the stock ``torch.ops._C`` one does not),
    and only FlashInfer gains from it: its decode and prefill paths make Q dense with
    a copy when it is a strided view of the QKV row. Call it from layer construction,
    after ``supports_fused_kv_insert``.
    """
    if attn_layer.attn_backend.get_name() != "FLASHINFER":
        return False
    op = resolve_compiled_op(_FUSED_KV_INSERT_OP)
    if op is None or not _op_has_dense_q_output(op):
        logger.warning_once(
            "VLLM_MUSE_SPARK_SM120_DENSE_Q is not applied: the fused QK-norm/RoPE op "
            "in use has no dense Q output."
        )
        return False
    return True


def fused_qk_norm_rope_kv_insert(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    layer_name: str,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    dense_q: bool = False,
) -> torch.Tensor | None:
    """Norm/rotate and insert KV; return the Q that replaces the view of ``qkv``.

    That is the FP8 Q when all attention tokens use it, or with ``dense_q`` (see
    ``supports_dense_q``) the dense Q buffer the kernel wrote in place of the Q
    columns of ``qkv``; None when Q stays in ``qkv``.
    """
    metadata, attn_layer, kv_cache, slot_mapping = get_attention_context(layer_name)

    q_fp8 = None
    if attn_layer.attn_backend.get_name() == "FLASHINFER" and metadata is not None:
        # SM90 may mix FP8 prefill Q with BF16 decode Q; emit FP8 Q only when
        # every token uses it. The builder's Q dtypes are fixed for its
        # lifetime, so full CUDA graph replays stay consistent.
        from vllm.v1.attention.backends.flashinfer import FlashInferMetadata

        if (
            isinstance(metadata, FlashInferMetadata)
            and metadata.num_actual_tokens > 0
            and attn_layer.kv_cache_dtype in ("fp8", "fp8_e4m3")
            and attn_layer._q_scale.numel() == 1
            and (
                metadata.num_prefill_tokens == 0
                or metadata.q_data_type_prefill == torch.float8_e4m3fn
            )
            and (
                metadata.num_decode_tokens == 0
                or metadata.q_data_type_decode == torch.float8_e4m3fn
            )
        ):
            q_fp8 = torch.empty(
                (qkv.shape[0], num_heads * head_dim),
                dtype=torch.float8_e4m3fn,
                device=qkv.device,
            )

    # Not tied to the metadata, so that the graph of a capture size has the same
    # allocation whatever the batch it is replayed with.
    q_out = None
    if dense_q and q_fp8 is None:
        q_out = torch.empty(
            (qkv.shape[0], num_heads * head_dim), dtype=qkv.dtype, device=qkv.device
        )

    # No cache bound (profiling) or no slots: skip the insert, still norm+rotate.
    insert = slot_mapping is not None and kv_cache.numel() > 0
    slot_mapping_arg = None
    kv_cache_arg = None
    block_size = 0
    kv_cache_dtype = "auto"
    k_scale = None
    v_scale = None
    if insert:
        slot_mapping_arg = slot_mapping
        kv_cache_arg = kv_cache
        block_size = kv_cache.shape[2]
        kv_cache_dtype = attn_layer.kv_cache_dtype
        if kv_cache_dtype not in ("auto", "bfloat16", "float16"):
            k_scale = attn_layer._k_scale
            v_scale = attn_layer._v_scale

    fused_op = resolve_compiled_op(_FUSED_KV_INSERT_OP)
    assert fused_op is not None
    # The op schema has no defaults for the cache arguments: pass them all. The
    # stock op has no q_out: it is passed only when there is one.
    extra = () if q_out is None else (q_out,)
    fused_op(
        qkv,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        cos_sin_cache,
        rotary_dim,
        slot_mapping_arg,
        kv_cache_arg,
        block_size,
        kv_cache_dtype,
        k_scale,
        v_scale,
        q_fp8,
        attn_layer._q_scale if q_fp8 is not None else None,
        *extra,
    )
    return q_fp8 if q_out is None else q_out
