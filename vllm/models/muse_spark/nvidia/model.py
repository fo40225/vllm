# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark HF text decoder with gated FP32 residuals and compressed MoE."""

import inspect
import math
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from typing import Any, cast

import torch
from torch import nn

try:
    from flashinfer.norm import (
        gated_residual_next_rmsnorm,
        supports_gated_residual_next_rmsnorm,
    )
except (ImportError, AttributeError):  # pragma: no cover - optional FlashInfer API
    gated_residual_next_rmsnorm = None

    def supports_gated_residual_next_rmsnorm(*_args: object) -> bool:
        return False


try:
    from flashinfer.norm.kernels.gated_residual_next_rmsnorm import (
        run as dual_norm_router,
    )

    HAS_ROUTER_OUTPUT = "router_input" in inspect.signature(dual_norm_router).parameters
except ImportError:
    HAS_ROUTER_OUTPUT = False


try:
    from flashinfer.norm import rmsnorm_sigmoid_gate as o_norm_sigmoid_gate
    from flashinfer.norm import (
        supports_rmsnorm_sigmoid_gate as supports_o_norm_sigmoid_gate,
    )
except (ImportError, AttributeError):  # pragma: no cover - optional FlashInfer API
    o_norm_sigmoid_gate = None

    def supports_o_norm_sigmoid_gate(*_args: object) -> bool:
        return False


import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.compilation.decorators import ignore_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.utils import dispatch_unquantized_gemm
from vllm.model_executor.models.interfaces import MixtureOfExperts, SupportsLoRA
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaMLP, LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    sequence_parallel_chunk,
)
from vllm.models.common.ops.sequence_parallel import (
    is_sequence_parallel_active,
    sp_all_gather,
    sp_all_gather_columns,
    sp_reduce_scatter,
    sp_shard,
    tp_reduce_scatter_matches_all_reduce,
)
from vllm.models.muse_spark.common.layers import (  # noqa: F401 re-exported
    MulTanhSoftCap,
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xQKVOGateParallelLinear,
    MuseSpark1xResidualConnection,
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    calculate_gain_center,
    materialize_residual_path_constants,
    norm_then_gated_residual,
)
from vllm.models.muse_spark.common.ops.fused_qk_norm_rope_kv import (
    fused_qk_norm_rope_kv_insert,
    supports_fused_kv_insert,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
    supports_norm_residual_norm,
)
from vllm.models.muse_spark.common.ops.fused_routing import fused_musespark1x_routing
from vllm.models.muse_spark.nvidia.mega_moe import MuseSparkMegaMoEExperts
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseMoERunner,
    MuseSpark1xRouter,
    MuseSparkMoEConfig,
    MuseSparkRoutedExperts,
)
from vllm.models.muse_spark.nvidia.ops import fused_boundary, muse_c
from vllm.models.muse_spark.nvidia.ops.cute_dsl.qk_norm import (
    qk_norm,
    supports_qk_norm,
)
from vllm.models.muse_spark.nvidia.ops.expert_prefetch import ExpertPrefetch
from vllm.models.muse_spark.nvidia.ops.l2_prefetch import (
    L2_PREFETCH_MAX_CTAS,
    L2_PREFETCH_MAX_TOKENS,
    L2Prefetcher,
)
from vllm.models.muse_spark.nvidia.ops.mnnvl_fusions import (
    maybe_create_all_gather_dual_norm,
    maybe_create_all_gather_norm_quant,
    tp_group_spans_nodes,
)
from vllm.models.muse_spark.nvidia.ops.muse_decode_moe import (
    EXPERT_PREFETCH_MAX_TOKENS,
    MUSE_MAX_TOKENS,
)
from vllm.models.muse_spark.nvidia.ops.prepare_megamoe_nvfp4 import (
    prepare_megamoe_nvfp4_with_norm,
)
from vllm.models.muse_spark.nvidia.ops.prepare_trtllm_nvfp4 import (
    norm_quant_trtllm_nvfp4,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.third_party.flash_linear_attention.ops.fused_norm_gate import (
    rms_norm_gated,
)
from vllm.transformers_utils.configs.muse_spark import (
    MuseSparkDSparkConfig,
    MuseSparkTextConfig,
)
from vllm.utils.math_utils import cdiv
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel
from vllm.utils.torch_utils import aux_stream

logger = init_logger(__name__)

_FLASHINFER_DUAL_NORM_HAS_POST_WEIGHT = (
    "post_weight" in inspect.signature(supports_gated_residual_next_rmsnorm).parameters
)
_ROUTER_MULTI_STREAM_TOKEN_THRESHOLD = 256
# Decode-sized batches only; prefill keeps the replicated projections.
_SHARDED_PROJECTION_MAX_TOKENS = 256
# Measured on GB300 TP8 (two nodes): below this the cross-node gather costs
# more than the smaller unpooler GEMM saves.
_MULTI_NODE_SHARDED_UNPOOLER_MIN_TOKENS = 32
_PREPARED_NVFP4_MAX_TOKENS = 256
# The most tokens the router GEMV of _muse_C serves (_muse_C::router_gemv), and the
# multiple of experts that its CTAs cover (2 rows for 1-2 tokens, 4 for more).
_ROUTER_GEMV_MAX_TOKENS = 8
_ROUTER_GEMV_ROW_MULTIPLE = 4


def gather_columns(shard: torch.Tensor) -> torch.Tensor:
    """Concatenate every TP rank's column block, in rank order."""
    return sp_all_gather_columns(shard)


@dataclass(slots=True)
class MuseSpark1xColumnShardedStream(MuseSpark1xResidualStream):
    """A deferred boundary whose branch is still this rank's column block."""

    all_gather: Any

    def materialize(self) -> MuseSpark1xResidualStream:
        return MuseSpark1xResidualStream(
            stream=self.stream,
            branch=gather_columns(self.branch),
            post_norm=self.post_norm,
            connection=self.connection,
        )


@dataclass(slots=True)
class MuseSpark1xUnpoolerInput:
    """The input of this rank's unpooler columns, before the GEMV: the layer leaves the
    GEMV to the next boundary, so that a TP communicator which fuses it with the column
    gather (``ops/fused_boundary.py``) can run both as one call."""

    x: torch.Tensor
    # This rank's rows of the unpooler weight.
    weight: torch.Tensor
    # x -> this rank's columns, the layer's own (cuBLAS) GEMV.
    project: Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class MuseSpark1xProjectedStream(MuseSpark1xColumnShardedStream):
    """A column-sharded boundary whose columns are not computed yet.

    ``branch`` is the input of this rank's unpooler GEMV, ``weight`` its rows of the
    weight and ``project`` the layer's own GEMV, which the boundary runs when the
    communicator does not fuse the two.
    """

    weight: torch.Tensor
    project: Callable[[torch.Tensor], torch.Tensor]

    def local_columns(self) -> MuseSpark1xColumnShardedStream:
        return MuseSpark1xColumnShardedStream(
            stream=self.stream,
            branch=self.project(self.branch),
            post_norm=self.post_norm,
            connection=self.connection,
            all_gather=self.all_gather,
        )

    def materialize(self) -> MuseSpark1xResidualStream:
        return self.local_columns().materialize()


@dataclass(slots=True)
class MuseSpark1xPartialSumStream(MuseSpark1xResidualStream):
    """A deferred boundary whose branch is still this rank's partial sum.

    The all-reduce of the branch is left to the boundary, so that a TP communicator
    which fuses the two (``ops/fused_boundary.py``) can run it as one kernel.
    """

    def materialize(self) -> MuseSpark1xResidualStream:
        return MuseSpark1xResidualStream(
            stream=self.stream,
            branch=tensor_model_parallel_all_reduce(self.branch),
            post_norm=self.post_norm,
            connection=self.connection,
        )


def _triton_boundary_kwargs(
    stream: torch.Tensor,
    post_norm: MuseSpark1xRMSNorm,
    connection: MuseSpark1xResidualConnection,
    pre_norm: MuseSpark1xRMSNorm,
    branch_shape: tuple[int, ...],
    out_dtype: torch.dtype,
    router_input: torch.Tensor | None,
) -> dict[str, Any] | None:
    """Arguments of a communicator-fused boundary, None if it cannot stand in.

    The communicator reproduces the Triton boundary kernel
    (``fused_norm_residual_norm_fwd``) bit for bit, so it may only stand in where
    ``norm_residual_then_norm`` would run exactly that kernel: bf16 output, the
    gated residual of this model with its per-channel fp32 coefficients, an
    unweighted or weight-first post-norm, a weight-last pre-norm with a weight,
    and no unrounded fp32 router input (another kernel computes that). The branch
    is a contiguous bf16 tensor of ``branch_shape``, which need not exist yet.
    """
    alpha = getattr(connection, "alpha", None)
    beta = getattr(connection, "beta", None)
    if not (
        router_input is None
        and out_dtype == torch.bfloat16
        and isinstance(connection, MuseSpark1xNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and len(branch_shape) == 2
        and stream.is_cuda
        and stream.dtype == torch.float32
        and stream.is_contiguous()
        and tuple(stream.shape) == tuple(branch_shape)
        and post_norm.variance_size_override is None
        and pre_norm.variance_size_override is None
        and post_norm.hidden_size == pre_norm.hidden_size == branch_shape[-1]
        # Where FlashInfer's dual-norm kernel could run the boundary, the Triton
        # one does not: leave that to the two-step path.
        and (
            gated_residual_next_rmsnorm is None
            or not _uses_flashinfer_dual_norm(
                post_norm, connection, pre_norm, out_dtype
            )
        )
        and pre_norm.has_weight
        and not pre_norm.applies_weight_before_norm
        and (not post_norm.has_weight or post_norm.applies_weight_before_norm)
        and isinstance(alpha, torch.Tensor)
        and isinstance(beta, torch.Tensor)
        and alpha.dtype == beta.dtype == torch.float32
        and alpha.shape == beta.shape == (branch_shape[-1],)
    ):
        return None
    return {
        "alpha": alpha,
        "beta": beta,
        "pre_weight": pre_norm.weight,
        "post_weight": post_norm.weight if post_norm.has_weight else None,
        "post_eps": post_norm.variance_epsilon,
        "pre_eps": pre_norm.variance_epsilon,
        "post_gain_center": post_norm.fused_gain_center,
        "pre_gain_center": pre_norm.fused_gain_center,
    }


def _all_reduce_norm_residual_then_norm(
    x: MuseSpark1xPartialSumStream,
    pre_norm: MuseSpark1xRMSNorm,
    out_dtype: torch.dtype,
    router_input: torch.Tensor | None,
    out_fp32: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The TP all-reduce of the branch and the boundary in one communicator kernel."""
    if not fused_boundary.supports_fused_boundary(
        fused_boundary.KIND_ALL_REDUCE_BOUNDARY, x.branch.shape[0]
    ):
        return None
    kwargs = _triton_boundary_kwargs(
        x.stream,
        x.post_norm,
        x.connection,
        pre_norm,
        tuple(x.branch.shape),
        out_dtype,
        router_input,
    )
    if (
        kwargs is None
        or x.branch.dtype != torch.bfloat16
        or not x.branch.is_contiguous()
    ):
        return None
    return fused_boundary.all_reduce_residual_boundary(
        x.branch, x.stream, out_fp32=out_fp32, **kwargs
    )


def _communicator_all_gather_norm_residual_then_norm(
    x: MuseSpark1xColumnShardedStream,
    pre_norm: MuseSpark1xRMSNorm,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The column gather of the unpooler shards and the boundary in one kernel.

    ``x.branch`` is this rank's column block; the branch of the boundary would be
    ``gather_columns`` of it.
    """
    shard = x.branch
    num_tokens, width = shard.shape
    if not fused_boundary.supports_fused_boundary(
        fused_boundary.KIND_ALL_GATHER_BOUNDARY, num_tokens
    ):
        return None
    kwargs = _triton_boundary_kwargs(
        x.stream,
        x.post_norm,
        x.connection,
        pre_norm,
        (num_tokens, width * get_tensor_model_parallel_world_size()),
        out_dtype,
        None,
    )
    if kwargs is None or shard.dtype != torch.bfloat16 or not shard.is_contiguous():
        return None
    return fused_boundary.all_gather_residual_boundary(
        shard, x.stream, out_fp32=None, **kwargs
    )


def _communicator_gemv_all_gather_norm_residual_then_norm(
    x: MuseSpark1xProjectedStream,
    pre_norm: MuseSpark1xRMSNorm,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The unpooler GEMV, the column gather and the boundary as one call."""
    num_tokens = x.branch.shape[0]
    if not fused_boundary.supports_fused_boundary(
        fused_boundary.KIND_GEMV_ALL_GATHER_BOUNDARY, num_tokens
    ):
        return None
    kwargs = _triton_boundary_kwargs(
        x.stream,
        x.post_norm,
        x.connection,
        pre_norm,
        (num_tokens, x.weight.shape[0] * get_tensor_model_parallel_world_size()),
        out_dtype,
        None,
    )
    if (
        kwargs is None
        or x.branch.dtype != torch.bfloat16
        or not x.branch.is_contiguous()
        or x.weight.dtype != torch.bfloat16
    ):
        return None
    return fused_boundary.gemv_all_gather_residual_boundary(
        x.branch, x.weight, x.stream, out_fp32=None, **kwargs
    )


def _uses_flashinfer_dual_norm(
    post_norm: MuseSpark1xRMSNorm,
    connection: MuseSpark1xResidualConnection,
    pre_norm: MuseSpark1xRMSNorm,
    out_dtype: torch.dtype,
) -> bool:
    """Whether the FlashInfer dual-norm contract covers this boundary."""
    return (
        isinstance(connection, MuseSpark1xNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and (not post_norm.has_weight or post_norm.applies_weight_before_norm)
        and pre_norm.has_weight
        and pre_norm.weight is not None
        and not pre_norm.applies_weight_before_norm
        and out_dtype == torch.bfloat16
    )


def _dual_norm_constants(
    post_norm: MuseSpark1xRMSNorm, pre_norm: MuseSpark1xRMSNorm
) -> dict[str, float]:
    return {
        "post_eps": post_norm.variance_epsilon,
        "next_eps": pre_norm.variance_epsilon,
        "next_gain_center": pre_norm.fused_gain_center,
        "post_gain_center": post_norm.fused_gain_center,
    }


def _all_gather_norm_residual_then_norm(
    x: MuseSpark1xColumnShardedStream,
    pre_norm: MuseSpark1xRMSNorm,
    out_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The fused gather + boundary; same contract as the dual-norm kernel."""
    from flashinfer.comm.mnnvl_cutedsl_ag import (
        all_gather_gated_residual_next_rmsnorm,
        supports_all_gather_gated_residual_next_rmsnorm,
    )

    post_norm, connection = x.post_norm, x.connection
    post_weight = post_norm.weight if post_norm.has_weight else None
    if not (
        _uses_flashinfer_dual_norm(post_norm, connection, pre_norm, out_dtype)
        and post_norm.variance_size_override is None
        and pre_norm.variance_size_override is None
        and supports_all_gather_gated_residual_next_rmsnorm(
            x.all_gather,
            x.branch,
            x.stream,
            connection.alpha,
            connection.beta,
            pre_norm.weight,
            post_weight,
        )
    ):
        return None
    new_stream, out = all_gather_gated_residual_next_rmsnorm(
        x.branch,
        x.stream,
        connection.alpha,
        connection.beta,
        pre_norm.weight,
        x.all_gather,
        post_weight=post_weight,
        **_dual_norm_constants(post_norm, pre_norm),
    )
    return out, new_stream


def norm_residual_then_norm(
    x: MuseSpark1xResidualStream,
    pre_norm: MuseSpark1xRMSNorm,
    *,
    out_dtype: torch.dtype,
    router_input: torch.Tensor | None = None,
    out_fp32: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Post-norm, blend and pre-norm; returns ``(out, new_stream)``.

    If supplied, router_input receives the unrounded FP32 pre-norm result.
    out_fp32 instead receives ``out`` widened to FP32, bit for bit the cast of
    the returned tensor; the Triton boundary writes it without a cast kernel.
    """
    assert router_input is None or out_fp32 is None
    if isinstance(x, MuseSpark1xPartialSumStream):
        if (
            fused := _all_reduce_norm_residual_then_norm(
                x, pre_norm, out_dtype, router_input, out_fp32
            )
        ) is not None:
            return fused
        x = x.materialize()
    if isinstance(x, MuseSpark1xProjectedStream):
        if router_input is None and out_fp32 is None:
            fused = _communicator_gemv_all_gather_norm_residual_then_norm(
                x, pre_norm, out_dtype
            )
            if fused:
                return fused
        x = x.local_columns()
    if isinstance(x, MuseSpark1xColumnShardedStream):
        if router_input is None and out_fp32 is None:
            fused = (
                _all_gather_norm_residual_then_norm(x, pre_norm, out_dtype)
                if x.all_gather is not None
                else _communicator_all_gather_norm_residual_then_norm(
                    x, pre_norm, out_dtype
                )
            )
            if fused:
                return fused
        x = x.materialize()
    post_norm, connection = x.post_norm, x.connection
    branch, stream = x.branch, x.stream
    can_fuse_boundary = (
        isinstance(connection, MuseSpark1xNormPreservingResidualConnection)
        and getattr(connection, "_coefficients_ready", False)
        and post_norm.supports_fused_kernel(branch)
        and pre_norm.supports_fused_kernel(stream)
    )
    pre_weight = pre_norm.weight
    post_weight_args = {"post_weight": post_norm.weight} if post_norm.has_weight else {}
    fp32_copy_written = False
    if (
        can_fuse_boundary
        and gated_residual_next_rmsnorm is not None
        and _uses_flashinfer_dual_norm(post_norm, connection, pre_norm, out_dtype)
        and (not post_norm.has_weight or _FLASHINFER_DUAL_NORM_HAS_POST_WEIGHT)
        and supports_gated_residual_next_rmsnorm(
            branch,
            stream,
            pre_weight,
            connection.alpha,
            connection.beta,
            **post_weight_args,
        )
    ):
        if router_input is not None and HAS_ROUTER_OUTPUT:
            new_stream = torch.empty_like(stream)
            out = torch.empty_like(branch)
            dual_norm_router(
                branch,
                stream,
                connection.alpha,
                connection.beta,
                pre_weight,
                post_norm.variance_epsilon,
                pre_norm.variance_epsilon,
                pre_norm.fused_gain_center,
                new_stream,
                out,
                post_norm.weight if post_norm.has_weight else None,
                post_norm.fused_gain_center if post_norm.has_weight else 0.0,
                router_input=router_input,
                enable_pdl=True,
            )
            return out, new_stream
        new_stream, out = gated_residual_next_rmsnorm(
            branch,
            stream,
            pre_weight,
            connection.alpha,
            connection.beta,
            post_eps=post_norm.variance_epsilon,
            next_eps=pre_norm.variance_epsilon,
            next_gain_center=pre_norm.fused_gain_center,
            enable_pdl=True,
            **post_weight_args,
            **(
                {"post_gain_center": post_norm.fused_gain_center}
                if post_norm.has_weight
                else {}
            ),
        )
    elif can_fuse_boundary and supports_norm_residual_norm(branch, stream):
        out, new_stream = fused_norm_residual_norm_fwd(
            branch,
            stream,
            post_norm.weight if post_norm.has_weight else None,
            pre_weight,
            post_norm.variance_epsilon,
            pre_norm.variance_epsilon,
            alpha=connection.alpha,
            beta=connection.beta,
            post_gain_center=post_norm.fused_gain_center,
            pre_gain_center=pre_norm.fused_gain_center,
            post_input_scale=(
                post_norm.applies_weight_before_norm and post_norm.has_weight
            ),
            pre_input_scale=(
                pre_norm.applies_weight_before_norm and pre_norm.has_weight
            ),
            out_dtype=out_dtype,
            out_fp32=out_fp32,
        )
        fp32_copy_written = out_fp32 is not None
    else:
        new_stream = norm_then_gated_residual(x)
        out = pre_norm(new_stream, out_dtype=out_dtype)
    if router_input is not None:
        router_input.copy_(pre_norm(new_stream, out_dtype=torch.float32))
    if out_fp32 is not None and not fp32_copy_written:
        out_fp32.copy_(out)
    return out, new_stream


def is_sequence_parallel_enabled(vllm_config: VllmConfig) -> bool:
    """Whether the residual stream is TP-sharded along the sequence."""
    pc = vllm_config.parallel_config
    if vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe":
        return False
    return (
        pc.pipeline_parallel_size == 1
        and pc.enable_expert_parallel
        and pc.tensor_parallel_size > 1
    )


# The backends known to give the SM120 stream modular experts.
_AUTO_SP_MOE_BACKENDS = ("auto", "cutlass")


def _on_sm120() -> bool:
    return (
        current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    )


def _sm120_sequence_parallel_blocker(vllm_config: VllmConfig) -> str | None:
    """Why the sharded stream cannot run on this launch, None if it can."""
    pc = vllm_config.parallel_config
    mlp_layer_types = getattr(
        vllm_config.model_config.hf_text_config, "mlp_layer_types", None
    )
    if not _on_sm120():
        return "the device is not SM120"
    if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
        return "--moe-backend deep_gemm_mega_moe has its own sequence-parallel stream"
    if not pc.enable_expert_parallel:
        return "it needs --enable-expert-parallel"
    if pc.tensor_parallel_size <= 1:
        return "it needs tensor parallelism"
    if pc.pipeline_parallel_size != 1 or pc.data_parallel_size != 1:
        return "it needs pipeline and data parallel sizes of 1"
    if pc.use_sequence_parallel_moe:
        return "the MoE already shards the tokens"
    if envs.VLLM_BATCH_INVARIANT:
        return "batch-invariant mode is on"
    # The sharded FFN branch is the MoE one.
    if mlp_layer_types is None or not all(t == "sparse" for t in mlp_layer_types):
        return "it needs every layer to be a sparse (MoE) layer"
    return None


def is_sm120_sequence_parallel_enabled(vllm_config: VllmConfig) -> bool:
    """Whether large prefill batches may shard the stream along the sequence on SM120.

    ``VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL=1`` (the default) shards every batch
    of at least ``sm120_sequence_parallel_min_tokens`` and ``0`` never does. With
    ``auto`` the stream is sharded only where that cannot change a bit: the TP
    communicator must guarantee that reduce-scatter + all-gather equals its
    all-reduce, and ``_SequenceParallelRowGate`` then admits the batch sizes whose
    row-sharded projections equal the replicated ones. The experts still see every
    token, through the same runner as the replicated stream.

    A launch that did not set the flag stays silent, and on the replicated stream,
    where the sharded one cannot apply (another layout, or a backend that may hand
    the routing to monolithic experts); one that set it to 1 is warned.
    """
    setting = envs.VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL
    if setting is False:
        return False
    blocker = _sm120_sequence_parallel_blocker(vllm_config)
    if setting is True and os.environ.get("VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL"):
        if blocker is not None:
            logger.warning_once(
                "VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL is set but ignored: %s.",
                blocker,
            )
        return blocker is None
    # Not a request: stay silent where it cannot apply. Another backend may hand
    # the routing to monolithic experts, which the stream cannot feed.
    if blocker is not None:
        return False
    if vllm_config.kernel_config.moe_backend not in _AUTO_SP_MOE_BACKENDS:
        return False
    return setting is True or tp_reduce_scatter_matches_all_reduce()


def sm120_sequence_parallel_is_gated() -> bool:
    """Whether the sharded stream takes only the batch sizes the row gate verifies."""
    return envs.VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL is None


def _sequence_parallel_active(
    enabled: bool,
    num_tokens: int,
    min_tokens: int | None,
    row_gate: Callable[[int], bool] | None,
) -> bool:
    """``is_sequence_parallel_active`` plus the SM120 batch-size gate, if any."""
    return is_sequence_parallel_active(enabled, num_tokens, min_tokens) and (
        row_gate is None or row_gate(num_tokens)
    )


def sm120_sequence_parallel_min_tokens(vllm_config: VllmConfig) -> int:
    """Smallest batch (in tokens) that shards the stream.

    Never a batch that may run in a CUDA graph (every size up to the largest
    captured one: the dispatcher pads into a graph only up to that size and runs
    larger batches eagerly) or with the fused decode-sized all-reduce and the
    sharded projections (up to ``_SHARDED_PROJECTION_MAX_TOKENS``).
    """
    # 0 without CUDA graphs (enforce-eager or cudagraph mode NONE).
    max_graph_tokens = vllm_config.compilation_config.max_cudagraph_capture_size or 0
    return max(
        envs.VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS,
        _SHARDED_PROJECTION_MAX_TOKENS + 1,
        max_graph_tokens + 1,
    )


@contextmanager
def _fp32_split_k_reduction() -> Iterator[None]:
    """Keep cuBLAS from reducing split-K partial sums in bf16.

    On the small batches of the sharded stream cuBLAS can split K to fill the GPU
    and, by default, reduces the slices in bf16: an extra rounding (rel. L2 error
    2.6e-3 against 1.7e-3 for a single rounding of the fp32 sum).
    """
    matmul = torch.backends.cuda.matmul
    previous = matmul.allow_bf16_reduced_precision_reduction
    matmul.allow_bf16_reduced_precision_reduction = False
    try:
        yield
    finally:
        matmul.allow_bf16_reduced_precision_reduction = previous


def _agreed_by_tp_group(verdict: bool) -> bool:
    """The verdict of the whole TP group, whose ranks must take one branch."""
    group = get_tp_group()
    if group.world_size == 1:
        return verdict
    flag = torch.tensor([int(verdict)], dtype=torch.int32)
    torch.distributed.all_reduce(
        flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    return bool(flag.item())


def _agreed_scale_by_tp_group(scale: float | None) -> bool:
    """Whether every rank of the TP group has a scale and all have the same one.

    Rows that one rank quantized are consumed by the experts of every rank, so a
    scale that differs between ranks (say, an input scale that is a rank-local max
    and not the global one) would silently rescale the output of the layer. One
    collective carries the presence and, as the minimum of ``v`` and of ``-v``, the
    smallest and the largest value.
    """
    usable = scale is not None and math.isfinite(scale) and scale > 0
    group = get_tp_group()
    if group.world_size == 1:
        return usable
    value = scale if usable else 0.0
    stats = torch.tensor([float(usable), value, -value], dtype=torch.float64)
    torch.distributed.all_reduce(
        stats, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
    )
    present, lowest, negated_highest = stats.tolist()
    return present == 1.0 and lowest == -negated_highest


class _SequenceParallelRowGate:
    """Whether a batch of ``num_tokens`` takes the sharded stream without a changed bit.

    The sharded stream runs the pooler, the unpooler and the fp32 router on
    ``ceil(T / tp)`` rows where the replicated stream runs them on T rows. cuBLAS
    picks its kernel per M, and a kernel that splits K sums the slices in another
    order (the default settings even round them to bf16), so the rows of the two
    agree bit for bit only at the sizes where both calls take the same split. Of
    the 1992 sizes from 512 to 16384 tokens in steps of 8, 100 pass on 8 RTX PRO
    6000 (4104-4224, 5312-5472, 6920-7168 and 8008-8224: among them the full
    8192-token prefill chunk and the chunks that decode tokens fill up), and at the
    others up to 95% of the router logits differ in the last bit. The kernel choice
    depends on the shapes and not on the values, so the check runs the three
    projections of one layer on random inputs of the real shapes, once per batch
    size, and compares the rows. Every other op of the sharded stream is a row-wise
    kernel of fixed configuration (identical at any size) or a collective with the
    guarantee ``tp_reduce_scatter_matches_all_reduce`` asks for.
    """

    # Admitted sizes that are logged at info level; the rest at debug level.
    _ANNOUNCED = 4

    def __init__(self, feed_forward: "MuseSpark1xMoE") -> None:
        self._feed_forward = feed_forward
        self._verdicts: dict[int, bool] = {}

    def __call__(self, num_tokens: int) -> bool:
        verdict = self._verdicts.get(num_tokens)
        if verdict is None:
            verdict = self._agreed(self._rows_match(num_tokens))
            announce = verdict and sum(self._verdicts.values()) < self._ANNOUNCED
            self._verdicts[num_tokens] = verdict
            (logger.info if announce else logger.debug)(
                "MuseSpark SM120 sequence parallel: batches of %d tokens %s.",
                num_tokens,
                "shard the stream (the sharded projections are bitwise the "
                "replicated ones)"
                if verdict
                else "stay replicated (the sharded projections are not bitwise "
                "the replicated ones)",
            )
        return verdict

    @staticmethod
    def _agreed(verdict: bool) -> bool:
        return _agreed_by_tp_group(verdict)

    @torch.no_grad()
    def _rows_match(self, num_tokens: int) -> bool:
        feed_forward = self._feed_forward
        tp_size = feed_forward.tp_size
        rows = cdiv(num_tokens, tp_size)
        for layer in (
            feed_forward.pooler,
            feed_forward.unpooler,
            feed_forward.router,
        ):
            assert layer is not None
            weight = layer.weight
            generator = torch.Generator(device=weight.device)
            generator.manual_seed(0)
            x = torch.empty(
                rows * tp_size,
                weight.shape[1],
                dtype=weight.dtype,
                device=weight.device,
            ).normal_(generator=generator)
            full = self._project(layer, x[:num_tokens])
            for rank in range(tp_size):
                valid = min(rows, num_tokens - rank * rows)
                if valid <= 0:
                    break
                # As forward_sequence_parallel runs the pooler and the unpooler.
                with _fp32_split_k_reduction():
                    local = self._project(layer, x[rank * rows : (rank + 1) * rows])
                start = rank * rows
                if not torch.equal(full[start : start + valid], local[:valid]):
                    return False
        return True

    @staticmethod
    def _project(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = layer(x)
        return out[0] if isinstance(out, tuple) else out


class MuseSpark1xMoE(nn.Module):
    post_norm_routed_experts_cls = MuseSparkRoutedExperts
    post_norm_runner_cls = MuseMoERunner

    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_idx: int,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        parallel_config = vllm_config.parallel_config
        mega_backend = vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"
        layer_is_nvfp4 = isinstance(
            quant_config, ModelOptNvFp4Config
        ) and not quant_config.is_layer_excluded(f"{prefix}.experts")
        if (
            mega_backend
            and quant_config is not None
            and not isinstance(quant_config, ModelOptNvFp4Config)
        ):
            raise NotImplementedError("MuseSpark MegaMoE supports BF16 and NVFP4.")
        self.use_mega_moe = mega_backend
        self.is_nvfp4_mega_moe = mega_backend and layer_is_nvfp4

        self.ep_size = get_ep_group().device_group.size()
        self.n_routed_experts = config.num_local_experts
        self.n_logical_experts = config.num_local_experts
        self.n_redundant_experts = 0
        self.n_physical_experts = self.n_logical_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.n_shared_experts = config.num_shared_experts

        self.fp32_router_input = (
            getattr(config, "muse_spark_fp32_router_input", True)
            and torch.backends.cuda.matmul.allow_tf32
        )
        if self.fp32_router_input and not HAS_ROUTER_OUTPUT:
            raise RuntimeError(
                "FP32 router input requires FlashInfer dual-norm router output"
            )
        self.router_dtype = torch.float32
        self.router = GateLinear(
            config.hidden_size,
            config.num_local_experts,
            bias=False,
            out_dtype=self.router_dtype,
            params_dtype=self.router_dtype,
            prefix=f"{prefix}.router",
        )
        self.register_buffer(
            "expert_bias",
            torch.zeros(config.num_local_experts, dtype=torch.float32),
        )
        self.layer_idx = layer_idx

        self.route_scale = config.routed_scaling_factor

        self.router_output_multiplier = config.output_multiplier

        self.moe_router = MuseSpark1xRouter(
            top_k=config.num_experts_per_tok,
            num_experts=config.num_local_experts,
            expert_bias=self.expert_bias,
            route_scale=self.route_scale,
            router_output_multiplier=self.router_output_multiplier,
            router_score_normalization="sigmoid",
            router_score_division_eps=1e-15,
        )

        hidden_size = config.moe_hidden_size
        self.pooler: ReplicatedLinear | None = ReplicatedLinear(
            config.hidden_size,
            hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.pooler",
        )
        self.unpooler: ReplicatedLinear | None = ReplicatedLinear(
            hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.unpooler",
        )

        # Set by the model for the decode expert prefetch.
        self.expert_prefetch: ExpertPrefetch | None = None
        # Aux streams have hung on ROCm.
        self._router_events = (torch.cuda.Event(), torch.cuda.Event())
        self._router_aux_stream = (
            aux_stream()
            if self.pooler is not None and not current_platform.is_rocm()
            else None
        )

        if self.pooler is not None:
            self.pre_expert_norm: MuseSpark1xRMSNorm | None = MuseSpark1xRMSNorm(
                hidden_size=hidden_size,
                eps=config.rms_norm_eps,
                has_weight=True,
                is_post_norm=False,
                zero_centered_gamma=True,
                post_norm_gain_center_type="one",
                residual_id=2 * layer_idx + 2,
                output_norm_gain_center_type="zero",
            )
        else:
            self.pre_expert_norm = None

        self.intermediate_size = config.moe_intermediate_size

        if self.use_mega_moe and not vllm_config.parallel_config.enable_expert_parallel:
            raise NotImplementedError(
                "MuseSpark1x MegaMoE requires expert parallel. Enable it with "
                "--enable-expert-parallel, or drop the moe backend."
            )
        self.is_sequence_parallel = is_sequence_parallel_enabled(vllm_config)
        # Large prefill batches only (see forward_sequence_parallel); this does not
        # change how a replicated batch runs, so it stays out of the flags below.
        self.sm120_sequence_parallel = (
            not self.use_mega_moe and is_sm120_sequence_parallel_enabled(vllm_config)
        )
        # Only the megakernel takes the shard; FusedMoE gets replicated tokens.
        self.use_sequence_parallel_moe = (
            False
            if self.is_sequence_parallel
            else parallel_config.use_sequence_parallel_moe
        )
        # Splitting the pooler/unpooler output columns over TP needs the same
        # tokens, in the same order, on every TP rank. The full weights stay
        # loaded, so larger batches keep the replicated projections.
        replicated_tokens = (
            self.tp_size > 1
            and not self.use_mega_moe
            and not self.is_sequence_parallel
            and not self.use_sequence_parallel_moe
            and parallel_config.data_parallel_size == 1
            and not envs.VLLM_BATCH_INVARIANT
        )
        tep_default = (
            replicated_tokens
            and current_platform.is_cuda()
            and current_platform.has_device_capability(100)
        )
        self.shard_pooler = replicated_tokens and getattr(
            config, "muse_spark_tep_shard_pooler", tep_default
        )
        self.shard_unpooler = replicated_tokens and getattr(
            config, "muse_spark_tep_shard_unpooler", tep_default
        )
        if self.shard_pooler or self.shard_unpooler:
            logger.info_once(
                "Sharding MuseSpark projection outputs over TP: pooler=%s, "
                "unpooler=%s.",
                self.shard_pooler,
                self.shard_unpooler,
            )
        self.tp_rank = get_tensor_model_parallel_rank()
        self._projection_gemm = dispatch_unquantized_gemm(
            vllm_config.kernel_config.linear_backend
        )
        self.min_sharded_unpooler_tokens = (
            _MULTI_NODE_SHARDED_UNPOOLER_MIN_TOKENS
            if self.shard_unpooler and tp_group_spans_nodes()
            else 1
        )
        # NVFP4 TRT-LLM layers can take inputs prepared outside the experts:
        # the pre-expert norm fused with the FP4 quant, and/or top-k routing
        # finished on the router's aux stream.
        self.fused_input_quant = not self.use_mega_moe and getattr(
            config, "muse_spark_tep_fused_moe_input_quant", tep_default
        )
        self.aux_stream_routing = not self.use_mega_moe and getattr(
            config, "muse_spark_tep_aux_stream_routing", tep_default
        )
        self.all_gather_norm_quant: Any | None = None
        # Whether the pooled latent is gathered as NVFP4 (None until the first
        # sequence-parallel batch); see _packed_latent_scale.
        self._latent_fp4: bool | None = None
        # Whether the TP communicator may run the pre-expert norm as the epilogue
        # of the pooler all-gather (ops/fused_boundary.py).
        self.fuses_boundary_collectives = (
            envs.VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY and _on_sm120()
        )
        # Whether it may also run the sharded pooler / unpooler GEMV together with
        # the gather (the GEMV arithmetic is then the communicator's own).
        self.fuses_gemv_gather = (
            self.fuses_boundary_collectives and envs.VLLM_MUSE_SPARK_SM120_FUSED_GEMV
        )
        # Set by the model once the _muse_C extension is loaded
        # (_configure_router_gemv).
        self.router_gemv_ready = False
        if self.use_mega_moe and self.tp_size > 1 and not self.is_sequence_parallel:
            raise NotImplementedError(
                "MuseSpark1x MegaMoE needs the sequence-parallel stream when "
                f"tensor parallel is on (tp_size={self.tp_size}), which rules "
                "out pipeline parallel "
                f"(pipeline_parallel_size={parallel_config.pipeline_parallel_size})."
            )
        if self.use_mega_moe and self.n_physical_experts % self.ep_size != 0:
            raise ValueError(
                f"n_physical_experts={self.n_physical_experts} must be "
                f"divisible by ep_size={self.ep_size}."
            )

        self.post_expert_norm: MuseSpark1xRMSNorm | None = (
            MuseSpark1xRMSNormWithInputScale(
                hidden_size=hidden_size,
                eps=config.post_norm_eps,
                has_weight=True,
                is_post_norm=True,
                zero_centered_gamma=True,
                post_norm_gain_center_type="one",
                residual_id=None,
                output_norm_gain_center_type="zero",
            )
        )
        if self.use_mega_moe:
            max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
            if self.is_sequence_parallel:
                # Unsharded small batches carry up to tp_size - 1 tokens.
                max_tokens = max(cdiv(max_tokens, self.tp_size), self.tp_size - 1)
            self.experts: nn.Module = MuseSparkMegaMoEExperts(
                vllm_config,
                num_experts=self.n_physical_experts,
                num_local_experts=self.n_local_physical_experts,
                experts_start_idx=(
                    get_ep_group().rank_in_group * self.n_local_physical_experts
                ),
                top_k=self.top_k,
                hidden_size=hidden_size,
                intermediate_size=self.intermediate_size,
                post_expert_norm=self.post_expert_norm,
                is_nvfp4=layer_is_nvfp4,
                num_max_tokens_per_rank=max_tokens,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.experts",
            )
            return
        self.experts = FusedMoEFactory(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=self.intermediate_size,
            renormalize=True,
            router=self.moe_router,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            is_sequence_parallel=self.use_sequence_parallel_moe,
            router_logits_dtype=self.router_dtype,
            scoring_func="sigmoid",
            e_score_correction_bias=self.expert_bias,
            routed_scaling_factor=self.route_scale,
            routed_experts_cls=self.post_norm_routed_experts_cls,
            runner_cls=self.post_norm_runner_cls,
            runner_args={
                "post_expert_norm": self.post_expert_norm,
                "fuse_all_reduce": getattr(
                    config, "muse_spark_tep_fused_moe_all_reduce", tep_default
                ),
            },
            moe_config_cls=MuseSparkMoEConfig,
        )
        if (
            self.sm120_sequence_parallel
            and self.experts.routed_experts.quant_method.is_monolithic
        ):
            raise NotImplementedError(
                "VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL needs experts that take "
                "the routing from the caller; the monolithic "
                f"--moe-backend {vllm_config.kernel_config.moe_backend} does not."
            )

    def finalize_mega_moe_weights(self) -> None:
        if self.use_mega_moe:
            self.experts.finalize_weights()

    def fold_router_output_multiplier(self) -> None:
        """Fold the router output multiplier into the gate weight; idempotent."""
        if self.router_output_multiplier is None:
            return
        with torch.no_grad():
            self.router.weight.mul_(self.router_output_multiplier)
        self.router_output_multiplier = None
        self.moe_router.router_output_multiplier = None

    def shards_pooler(self, num_tokens: int) -> bool:
        return self.shard_pooler and num_tokens <= _SHARDED_PROJECTION_MAX_TOKENS

    def shards_unpooler(self, num_tokens: int) -> bool:
        return (
            self.shard_unpooler
            and self.min_sharded_unpooler_tokens
            <= num_tokens
            <= _SHARDED_PROJECTION_MAX_TOKENS
        )

    def _local_weight_rows(self, layer: ReplicatedLinear) -> torch.Tensor:
        """This rank's rows of the projection weight (its block of output columns)."""
        shard_size = layer.output_size // self.tp_size
        return layer.weight.narrow(0, self.tp_rank * shard_size, shard_size)

    def _project_local_columns(
        self, layer: ReplicatedLinear, x: torch.Tensor
    ) -> torch.Tensor:
        """This rank's contiguous block of the projection's output columns."""
        return self._projection_gemm(layer, x, self._local_weight_rows(layer), None)

    def sharded_weights(
        self, num_tokens: int, layer: ReplicatedLinear | None
    ) -> list[torch.Tensor]:
        """The rows of a pooler or unpooler weight this rank reads at this size.

        Empty where the projection runs whole: that is a read of tens of MB,
        too much to keep in L2.
        """
        if layer is None:
            return []
        shard = (
            self.shards_pooler(num_tokens)
            if layer is self.pooler
            else self.shards_unpooler(num_tokens)
        )
        if not shard:
            return []
        size = layer.output_size // self.tp_size
        return [layer.weight.data.narrow(0, self.tp_rank * size, size)]

    def _project(
        self, layer: ReplicatedLinear, x: torch.Tensor, shard: bool
    ) -> torch.Tensor:
        if shard:
            return gather_columns(self._project_local_columns(layer, x))
        return layer(x)[0]

    def _fused_all_gather_rmsnorm(self, shard: torch.Tensor) -> torch.Tensor | None:
        """The pooler gather and the pre-expert norm as one communicator kernel."""
        norm = self.pre_expert_norm
        if not (
            self.fuses_boundary_collectives
            and norm is not None
            and norm.has_weight
            and not norm.applies_weight_before_norm
            and norm.variance_size_override is None
            and shard.dtype == torch.bfloat16
            and shard.is_contiguous()
            and fused_boundary.supports_fused_boundary(
                fused_boundary.KIND_ALL_GATHER_RMSNORM, shard.shape[0]
            )
        ):
            return None
        return fused_boundary.all_gather_rmsnorm(
            shard, norm.weight, norm.variance_epsilon, norm.fused_gain_center
        )

    def _fused_gemv_all_gather_rmsnorm(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor | None:
        """The pooler GEMV, the column gather and the pre-expert norm as one call."""
        pooler, norm = self.pooler, self.pre_expert_norm
        if not (
            self.fuses_gemv_gather
            and pooler is not None
            and norm is not None
            and norm.has_weight
            and not norm.applies_weight_before_norm
            and norm.variance_size_override is None
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and pooler.weight.dtype == torch.bfloat16
            and fused_boundary.supports_fused_boundary(
                fused_boundary.KIND_GEMV_ALL_GATHER_RMSNORM, hidden_states.shape[0]
            )
        ):
            return None
        return fused_boundary.gemv_all_gather_rmsnorm(
            hidden_states,
            self._local_weight_rows(pooler),
            norm.weight,
            norm.variance_epsilon,
            norm.fused_gain_center,
        )

    def _deferred_unpooler_columns(
        self, experts_out: torch.Tensor
    ) -> torch.Tensor | MuseSpark1xUnpoolerInput:
        """This rank's unpooler columns for the boundary that gathers them, or their
        GEMV input where the communicator runs the GEMV with the gather."""
        unpooler = self.unpooler
        assert unpooler is not None
        if (
            self.fuses_gemv_gather
            and experts_out.dtype == torch.bfloat16
            and experts_out.is_contiguous()
            and unpooler.weight.dtype == torch.bfloat16
            and fused_boundary.supports_fused_boundary(
                fused_boundary.KIND_GEMV_ALL_GATHER_BOUNDARY, experts_out.shape[0]
            )
        ):
            return MuseSpark1xUnpoolerInput(
                x=experts_out,
                weight=self._local_weight_rows(unpooler),
                project=lambda x: self._project_local_columns(unpooler, x),
            )
        return self._project_local_columns(unpooler, experts_out)

    def _pool_and_norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.pooler is not None
        if self.shards_pooler(hidden_states.shape[0]):
            fused = self._fused_gemv_all_gather_rmsnorm(hidden_states)
            if fused is not None:
                return fused
            shard = self._project_local_columns(self.pooler, hidden_states)
            if (fused := self._fused_all_gather_rmsnorm(shard)) is not None:
                return fused
            hidden_states = gather_columns(shard)
        else:
            hidden_states = self.pooler(hidden_states)[0]
        if self.pre_expert_norm is not None:
            hidden_states = self.pre_expert_norm(hidden_states)
        return hidden_states

    def wants_fp32_router_input(self, num_tokens: int) -> bool:
        if not self.fp32_router_input or (
            self.is_sequence_parallel and not self.use_mega_moe
        ):
            return False
        if self.use_sequence_parallel_moe:
            num_tokens = cdiv(num_tokens, self.tp_size)
        return 32 < num_tokens <= 128

    def router_overlaps_pooler(self, num_tokens: int) -> bool:
        """Whether the router chain runs on the aux stream beside the pooler."""
        if (
            self.pooler is None
            or self._router_aux_stream is None
            or num_tokens > _ROUTER_MULTI_STREAM_TOKEN_THRESHOLD
        ):
            return False
        # maybe_execute_in_parallel runs both branches in order there.
        return not BreakableCUDAGraphCapture.is_active()

    def wants_cast_router_input(self, num_tokens: int) -> bool:
        """Whether the boundary should also write ``ffn_in`` widened to FP32.

        Without an FP32 router input the router casts its bf16 input to FP32
        before the GEMM. The widening is exact, so the boundary that stores
        ``ffn_in`` can produce the same tensor and save the cast kernel. That
        pays only where the cast is on the critical path: beside the pooler it
        hides, and the extra store would slow the boundary by about 0.5 us.

        With the expert prefetch the routing is on the critical path as well: the
        prefetch starts when the routing ends, and every microsecond the routing
        ends earlier is one more that the experts' first tiles are already in L2.

        SM120 only: that is where the fused store was measured (and shown bitwise
        equal). On the other GPUs the routers keep the cast of their own, even where
        their last tier would accept the widened input (SM90).
        """
        if not (
            _on_sm120()
            and not self.is_sequence_parallel
            and not self.wants_fp32_router_input(num_tokens)
            and (
                not self.router_overlaps_pooler(num_tokens)
                or self.prefetches_experts(num_tokens)
            )
            and self.router.widens_bf16_input()
        ):
            return False
        logger.info_once(
            "MuseSpark MoE: the boundary kernel writes the FP32 router input where "
            "the router would cast it on the critical path."
        )
        return True

    def prefetches_experts(self, num_tokens: int) -> bool:
        """Whether the decode graph of this size queues the expert prefetch.

        It needs a CUDA graph capture (an eager forward queues no prefetch, and the
        fused router cast would only slow its boundary), the routing on the aux
        stream and the muse decode kernel (padded batches of up to
        ``MUSE_MAX_TOKENS``, of which the prefetch maps the first
        ``EXPERT_PREFETCH_MAX_TOKENS``); the layers on another expert path would pay
        the fused router cast for nothing.
        """
        return (
            self.expert_prefetch is not None
            and torch.cuda.is_current_stream_capturing()
            and num_tokens <= min(MUSE_MAX_TOKENS, EXPERT_PREFETCH_MAX_TOKENS)
            and self._routes_on_aux_stream(num_tokens)
            and cast(MuseMoERunner, self.experts).has_muse_decode_engine
        )

    def _router_logits(self, router_input: torch.Tensor) -> torch.Tensor:
        """The router logits: the ``_muse_C`` decode GEMV where it serves the batch."""
        weight = self.router.weight
        if (
            self.router_gemv_ready
            and router_input.dtype == torch.float32
            and router_input.is_contiguous()
            and router_input.dim() == 2
            and 1 <= router_input.shape[0] <= _ROUTER_GEMV_MAX_TOKENS
            and weight.dtype == torch.float32
            and weight.is_contiguous()
            and tuple(weight.shape) == (self.n_logical_experts, router_input.shape[1])
            and router_input.shape[1] == 8192
            # The kernel owns 2 or 4 rows of the weight per CTA.
            and weight.shape[0] % _ROUTER_GEMV_ROW_MULTIPLE == 0
        ):
            logits = torch.empty(
                router_input.shape[0],
                weight.shape[0],
                dtype=torch.float32,
                device=router_input.device,
            )
            torch.ops._muse_C.router_gemv(router_input, weight, logits, False)
            return logits
        self._note_router_gemv_unused(router_input)
        return self.router(router_input)[0]

    def _note_router_gemv_unused(self, router_input: torch.Tensor) -> None:
        """Say once that a decode graph routes on cuBLAS for want of the FP32 input.

        Only a capture of a layer on the muse decode kernel can tell: that is where the
        boundary kernel writes the FP32 router input (``wants_cast_router_input``), and
        it does so beside the pooler only with the expert prefetch. The eager forwards
        and the BF16 layers always route from the BF16 input, which is no news.
        """
        if (
            self.router_gemv_ready
            and router_input.dtype != torch.float32
            and router_input.dim() == 2
            and router_input.shape[0] <= _ROUTER_GEMV_MAX_TOKENS
            and torch.cuda.is_current_stream_capturing()
            and getattr(self.experts, "has_muse_decode_engine", False)
        ):
            logger.warning_once(
                "VLLM_MUSE_SPARK_SM120_ROUTER_GEMV=1, but the decode graphs route on "
                "cuBLAS: the router GEMV needs the FP32 router input, which the "
                "boundary kernel writes only with the expert prefetch "
                "(VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH=1) active in the layer."
            )

    def _route_and_pool(
        self, hidden_states: torch.Tensor, router_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Router logits, plus the pooled expert input."""
        if self.pooler is None:
            router_logits, _ = self.router(router_input)
            return router_logits, hidden_states

        aux_stream_or_none = (
            self._router_aux_stream
            if hidden_states.shape[0] <= _ROUTER_MULTI_STREAM_TOKEN_THRESHOLD
            else None
        )
        pooled, router_logits = maybe_execute_in_parallel(
            lambda: self._pool_and_norm(hidden_states),
            lambda: self._router_logits(router_input),
            self._router_events[0],
            self._router_events[1],
            aux_stream_or_none,
        )
        return router_logits, pooled

    def _route_topk_and_pool(
        self, hidden_states: torch.Tensor, router_input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Overlap the full routing chain with the expert input projection."""

        def route() -> tuple[torch.Tensor, torch.Tensor]:
            logits = self._router_logits(router_input)
            if self.use_mega_moe:
                return self.moe_router._compute_routing(
                    hidden_states, logits, torch.int32
                )
            return cast(MuseMoERunner, self.experts).select_experts(
                hidden_states, logits
            )

        if self.pooler is None:
            weights, ids = route()
            return hidden_states, weights, ids

        parallel = (
            self._router_aux_stream is not None
            and hidden_states.shape[0] <= _ROUTER_MULTI_STREAM_TOKEN_THRESHOLD
        )
        pooled, (weights, ids) = maybe_execute_in_parallel(
            lambda: self._pool_and_norm(hidden_states),
            route,
            self._router_events[0],
            self._router_events[1],
            self._router_aux_stream if parallel else None,
        )
        if parallel and self.prefetches_experts(hidden_states.shape[0]):
            # Behind the routing kernel on the aux stream, after the event that
            # hands the routing to the experts: they do not wait for it.
            self.expert_prefetch.launch(self.experts, ids)

        return pooled, weights, ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_input: torch.Tensor | None = None,
        defer_column_gather: bool = False,
    ) -> torch.Tensor | MuseSpark1xUnpoolerInput:
        """With ``defer_column_gather``, returns only this rank's unpooler columns,
        or their GEMV input where the communicator fuses the GEMV with the gather."""
        return self._routed_forward(hidden_states, router_input, defer_column_gather)

    def _can_fuse_megamoe_inputs(self, hidden_states: torch.Tensor) -> bool:
        norm = self.pre_expert_norm
        return (
            self.is_nvfp4_mega_moe
            and self.pooler is not None
            and norm is not None
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[0] <= 256
            and norm.hidden_size == 4096
            and norm.variance_size_override is None
            and norm.has_weight
            and not norm.applies_weight_before_norm
        )

    def _stage_megamoe_inputs(
        self, hidden_states: torch.Tensor, router_input: torch.Tensor
    ):
        num_tokens = hidden_states.shape[0]
        buffer, padding = self.experts.input_buffer(num_tokens)
        norm = self.pre_expert_norm
        pooler = self.pooler
        assert norm is not None and pooler is not None

        def pool_and_quantize():
            pooled, _ = pooler(hidden_states)
            prepare_megamoe_nvfp4_with_norm(
                pooled,
                norm.weight,
                norm.variance_epsilon,
                norm.fused_gain_center,
                buffer.x[:num_tokens],
                buffer.x_sf[:num_tokens].view(torch.uint8),
            )
            return pooled

        def route():
            logits, _ = self.router(router_input)
            router = self.moe_router
            fused_musespark1x_routing(
                logits,
                router.expert_bias,
                router.top_k,
                router.route_scale,
                router_output_multiplier=router.router_output_multiplier,
                router_score_normalization=router.router_score_normalization,
                router_score_division_eps=router.router_score_division_eps,
                topk_ids=buffer.topk_idx[:num_tokens],
                topk_weights=buffer.topk_weights[:num_tokens],
                is_padding=padding,
            )

        # The fork event also orders both writers after the previous layer's
        # MegaMoE, which may have consumed this same symmetric buffer.
        pooled, _ = maybe_execute_in_parallel(
            pool_and_quantize, route, *self._router_events, self._router_aux_stream
        )
        return self.experts.forward_prepared(pooled, buffer)

    def _routes_on_aux_stream(self, num_tokens: int) -> bool:
        """Whether the experts take routing finished beside the pooler.

        The router GEMM and the top-k then run on the aux stream while the
        pooler, its all-gather and the norm run on the main one, and the experts
        start with both ready (``_prepared_nvfp4_forward`` does the same for
        experts that take prepared inputs).

        SM120 only: that is where the path was shown bitwise equal to the runner's
        own routing; the other GPUs keep the routing inside the experts.
        """
        if not (
            _on_sm120()
            and self.aux_stream_routing
            and not self.use_mega_moe
            and not self.use_sequence_parallel_moe
            and self.router_overlaps_pooler(num_tokens)
            and isinstance(self.experts, MuseMoERunner)
            and self.experts.supports_routed_forward
        ):
            return False
        logger.info_once(
            "MuseSpark MoE: the router GEMM and top-k run on the aux stream beside "
            "the pooler for up to %d tokens.",
            _ROUTER_MULTI_STREAM_TOKEN_THRESHOLD,
        )
        return True

    def _prepares_nvfp4_inputs(self, num_tokens: int) -> bool:
        return (
            (self.fused_input_quant or self.aux_stream_routing)
            and num_tokens <= _PREPARED_NVFP4_MAX_TOKENS
            and not self.use_sequence_parallel_moe
            and self.norm_fuses_into_nvfp4_quant
            and self.experts.nvfp4_input_global_scale is not None
        )

    @property
    def norm_fuses_into_nvfp4_quant(self) -> bool:
        """Whether the pre-expert norm can run inside the expert input quant."""
        norm = self.pre_expert_norm
        return (
            self.pooler is not None
            and norm is not None
            and norm.has_weight
            and norm.variance_size_override is None
            and not norm.applies_weight_before_norm
        )

    def _all_gather_norm_quant(
        self,
        shard: torch.Tensor,
        norm: MuseSpark1xRMSNorm,
        global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Pooler columns gathered straight into the pre-expert norm + NVFP4."""
        all_gather = self.all_gather_norm_quant
        if all_gather is None:
            return None
        from flashinfer.comm.mnnvl_cutedsl_ag import (
            all_gather_rmsnorm_fp4_quantize,
            supports_all_gather_rmsnorm_fp4_quantize,
        )

        if not supports_all_gather_rmsnorm_fp4_quantize(
            all_gather, shard, norm.weight, global_scale
        ):
            return None
        return all_gather_rmsnorm_fp4_quantize(
            shard,
            norm.weight,
            global_scale,
            all_gather,
            eps=norm.variance_epsilon,
            gain_center=norm.fused_gain_center,
        )

    def _prepared_nvfp4_forward(
        self, hidden_states: torch.Tensor, router_input: torch.Tensor
    ) -> torch.Tensor:
        """Experts on NVFP4 input and routing prepared on parallel streams."""
        num_tokens = hidden_states.shape[0]
        norm = self.pre_expert_norm
        pooler = self.pooler
        assert norm is not None and pooler is not None
        global_scale = self.experts.nvfp4_input_global_scale

        def pool_and_quantize() -> tuple[torch.Tensor, torch.Tensor]:
            if not self.fused_input_quant:
                return ops.scaled_fp4_quant(
                    self._pool_and_norm(hidden_states),
                    global_scale,
                    is_sf_swizzled_layout=False,
                )
            if self.shards_pooler(num_tokens):
                pooled = self._project_local_columns(pooler, hidden_states)
                if fused := self._all_gather_norm_quant(pooled, norm, global_scale):
                    return fused
                # The quant reads the gathered shards in place: no reorder copy.
                pooled = sp_all_gather(pooled)
            else:
                pooled = pooler(hidden_states)[0]
            return norm_quant_trtllm_nvfp4(
                pooled,
                num_tokens,
                norm.weight,
                norm.variance_epsilon,
                norm.fused_gain_center,
                global_scale,
            )

        def route() -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            logits, _ = self.router(router_input)
            if not self.aux_stream_routing:
                return logits
            weights, ids = self.moe_router._compute_routing(
                hidden_states, logits, torch.int32
            )
            return ids, weights

        (a1q, a1q_scale), routing = maybe_execute_in_parallel(
            pool_and_quantize,
            route,
            self._router_events[0],
            self._router_events[1],
            self._router_aux_stream
            if num_tokens <= _ROUTER_MULTI_STREAM_TOKEN_THRESHOLD
            else None,
        )
        return self.experts.forward_prepared_nvfp4(a1q, a1q_scale, routing)

    def _routed_forward(
        self,
        hidden_states: torch.Tensor,
        router_input: torch.Tensor | None = None,
        defer_column_gather: bool = False,
    ) -> torch.Tensor | MuseSpark1xUnpoolerInput:
        num_tokens = hidden_states.shape[0]
        if self.use_sequence_parallel_moe:
            hidden_states = sequence_parallel_chunk(hidden_states)
            if router_input is not None:
                router_input = sequence_parallel_chunk(router_input)
        if router_input is None:
            router_input = hidden_states

        if self._can_fuse_megamoe_inputs(hidden_states):
            experts_out = self._stage_megamoe_inputs(hidden_states, router_input)
        elif self._prepares_nvfp4_inputs(num_tokens):
            experts_out = self._prepared_nvfp4_forward(hidden_states, router_input)
        elif self.use_mega_moe:
            hidden_states, topk_weights, topk_ids = self._route_topk_and_pool(
                hidden_states, router_input
            )
            experts_out = self.experts(hidden_states, topk_weights, topk_ids)
        elif self._routes_on_aux_stream(num_tokens):
            hidden_states, topk_weights, topk_ids = self._route_topk_and_pool(
                hidden_states, router_input
            )
            experts_out = cast(MuseMoERunner, self.experts).forward_routed(
                hidden_states, topk_weights, topk_ids
            )
        else:
            router_logits, hidden_states = self._route_and_pool(
                hidden_states, router_input
            )
            experts_out = self.experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )

        if defer_column_gather:
            return self._deferred_unpooler_columns(experts_out)
        if self.unpooler is not None:
            experts_out = self._project(
                self.unpooler, experts_out, self.shards_unpooler(num_tokens)
            )

        if self.use_sequence_parallel_moe:
            experts_out = tensor_model_parallel_all_gather(experts_out, 0)
            experts_out = experts_out[:num_tokens]

        return experts_out

    def _packed_latent_scale(self) -> torch.Tensor | None:
        """The global scale of the latent this layer gathers as NVFP4, or None.

        Whether the layer gathers it packed is decided once, on the first
        sequence-parallel batch, and agreed by the TP group: ranks that gathered rows
        of different widths would hang in the collective, and rows quantized with
        another rank's scale would be rescaled by the experts. Only the decision is
        kept: the scale is read from the experts on every call, because a weight
        reload builds new experts with new scales.
        """
        if self._latent_fp4 is None:
            value = None
            if envs.VLLM_MUSE_SPARK_SM120_PREFILL_LATENT_FP4:
                # Every rank has to reach the collective below, so no exception
                # leaves this block: a rank that fails takes the bf16 gather, and
                # so does the group.
                try:
                    scale = self.experts.packed_latent_scale()
                    value = None if scale is None else float(scale)
                except Exception as e:
                    logger.warning_once(
                        "MuseSpark SM120 sequence parallel: the pooled latent stays "
                        "bf16 on this rank, its NVFP4 scale could not be read "
                        "(%s: %s).",
                        type(e).__name__,
                        e,
                    )
            self._latent_fp4 = _agreed_scale_by_tp_group(value)
            if self._latent_fp4:
                logger.info_once(
                    "MuseSpark SM120 sequence parallel: the pooled latent is "
                    "quantized to NVFP4 before its all-gather "
                    "(VLLM_MUSE_SPARK_SM120_PREFILL_LATENT_FP4=1)."
                )
        if not self._latent_fp4:
            return None
        scale = self.experts.packed_latent_scale()
        if scale is None:
            # The group gathers packed rows; a rank that would gather bf16 ones
            # would hang the others in the collective.
            raise RuntimeError(
                "MuseSpark SM120 sequence parallel: the experts no longer take the "
                "pooled latent as NVFP4 (a weight reload changed them), but the "
                "TP group gathers it packed."
            )
        return scale

    def forward_sequence_parallel(
        self,
        hidden_states: torch.Tensor,
        num_tokens: int,
        router_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's rows of the branch, for a stream sharded along the sequence.

        ``hidden_states`` holds this rank's rows of the pre-FFN norm output (the
        batch padded to a multiple of the TP size) and ``num_tokens`` is the
        unpadded batch size. The router, the pooler and the pre-expert norm run on
        the local rows only. The experts take every row, as in the replicated
        stream, from the gathered latent and routes; their per-rank partial sums
        are reduce-scattered to the rows' owners, ahead of the local unpooler.
        ``router_input``, if given, is ``hidden_states`` widened to fp32 (written by
        the boundary kernel), which the router would otherwise cast itself.
        """
        assert self.sm120_sequence_parallel
        assert self.pooler is not None and self.unpooler is not None
        assert self.pre_expert_norm is not None
        router = self.moe_router
        top_k = router.top_k
        router_logits, _ = self.router(
            hidden_states if router_input is None else router_input
        )
        # Ids and the bits of the fp32 weights share one buffer, so one gather.
        routes = torch.empty(
            (hidden_states.shape[0], 2 * top_k),
            dtype=torch.int32,
            device=hidden_states.device,
        )
        fused_musespark1x_routing(
            router_logits,
            router.expert_bias,
            top_k,
            router.route_scale,
            router_output_multiplier=router.router_output_multiplier,
            router_score_normalization=router.router_score_normalization,
            router_score_division_eps=router.router_score_division_eps,
            topk_ids=routes[:, :top_k],
            topk_weights=routes[:, top_k:].view(torch.float32),
        )
        # Not the sharded projections: those need the same rows on every rank.
        with _fp32_split_k_reduction():
            pooled = self.pooler(hidden_states)[0]
        pooled = self.pre_expert_norm(pooled)
        # The experts quantize the latent to NVFP4 row by row: quantized here, ahead
        # of the gather, it travels as 2304 B per row instead of 8192 B.
        latent_scale = self._packed_latent_scale()
        if latent_scale is not None:
            packed = sp_all_gather(muse_c.pack_nvfp4_rows(pooled, latent_scale))
            packed = packed[:num_tokens]
        else:
            pooled = sp_all_gather(pooled)[:num_tokens]
        routes = sp_all_gather(routes)[:num_tokens]
        topk_ids = routes[:, :top_k].contiguous()
        topk_weights = routes[:, top_k:].contiguous().view(torch.float32)
        if router.capture_fn is not None:
            router.capture_fn(topk_ids)
        if latent_scale is not None:
            partial = self.experts.forward_routed_partial_packed(
                packed, topk_weights, topk_ids
            )
        else:
            partial = self.experts.forward_routed_partial(
                pooled, topk_weights, topk_ids
            )
        rows = sp_reduce_scatter(partial)
        with _fp32_split_k_reduction():
            return self.unpooler(rows)[0]


class MuseSpark1xAttention(nn.Module):
    def __init__(
        self,
        config: MuseSparkTextConfig | MuseSparkDSparkConfig,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        is_draft = isinstance(config, MuseSparkDSparkConfig)
        if is_draft:
            self.use_rope = config.use_rope
            self.count_backward_layer_id = self.layer_idx
            if config.every_n_layers_nope is not None:
                self.count_backward_layer_id += (
                    config.every_n_layers_nope
                    - config.num_hidden_layers % config.every_n_layers_nope
                )
                self.use_rope = self.use_rope and (
                    (config.num_hidden_layers - self.layer_idx - 1)
                    % config.every_n_layers_nope
                    != 0
                )
            rope_theta = config.rope_theta
        else:
            rope_theta = config.layer_rope_theta[self.layer_idx]
            self.use_rope = rope_theta > 0
        self.use_qk_norm = config.use_qk_norm if is_draft else True
        self.use_o_norm = config.use_attn_o_norm if is_draft else True
        self.use_o_gate = config.use_attn_o_gate if is_draft else True
        norm_affine = config.norm_affine if is_draft else True
        zero_centered_gamma = config.norm_zero_centered_gamma if is_draft else True
        gain_center_type = config.post_norm_gain_center_type if is_draft else "one"

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qk_norm = (
            MuseSpark1xRMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,  # matches the training side
                is_post_norm=False,
                zero_centered_gamma=False,  # matches the training side
                post_norm_gain_center_type=gain_center_type,
            )
            if self.use_qk_norm
            else None
        )
        # Decided at init: a per-call dtype check would show up in the graph.
        self.default_dtype = torch.get_default_dtype()
        self.fused_qk_norm = self.use_qk_norm and supports_qk_norm(
            self.default_dtype, self.head_dim
        )
        self.o_norm = (
            MuseSpark1xRMSNorm(
                hidden_size=self.head_dim,
                eps=config.rms_norm_eps,
                has_weight=False,
                is_post_norm=False,
                zero_centered_gamma=False,
                post_norm_gain_center_type=gain_center_type,
            )
            if self.use_o_norm
            else None
        )
        self.fuse_o_norm_gate = self.use_o_norm and self.use_o_gate
        self.rms_norm_eps = config.rms_norm_eps
        self.cute_o_norm_gate = self.fuse_o_norm_gate and supports_o_norm_sigmoid_gate(
            torch.get_default_dtype(), self.head_dim
        )

        self.scale_query_by = (
            (
                3.87
                if config.use_sqrt_attn_scaling
                else 43.7840518911 / math.sqrt(self.head_dim)
            )
            if is_draft
            else config.qk_scale_factor
        )
        if self.use_qk_norm:
            # Exact since RoPE is a rotation; keeps the QK-norm weightless.
            self.scaling *= self.scale_query_by

        self.pre_qkv_proj_norm = MuseSpark1xRMSNorm(
            hidden_size=self.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=norm_affine,
            is_post_norm=False,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 1,  # matches the reference indexing
        )

        if not self.use_o_gate:
            self.qkv_proj = QKVParallelLinear(
                hidden_size=self.hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        else:
            self.qkv_proj = MuseSpark1xQKVOGateParallelLinear(
                hidden_size=self.hidden_size,
                head_size=self.head_dim,
                total_num_heads=self.total_num_heads,
                total_num_kv_heads=self.total_num_kv_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.qkv_proj",
            )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.is_sequence_parallel_output = False
        # Smallest batch that shards the stream; None: one token per rank.
        self.sp_min_tokens: int | None = None
        self.sp_row_gate: Callable[[int], bool] | None = None
        self.gemm_rs = None

        # get_rope shares one instance (and its cos/sin cache, ~256 MiB in
        # fp32 at 1M positions) across all layers with the same parameters.
        self.rotary_emb = (
            get_rope(
                head_size=self.head_dim,
                max_position=max_position_embeddings,
                is_neox_style=False,
                rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
                dtype=(
                    torch.float
                    if not is_draft or config.rope_use_fp32_in_outer_product
                    else self.default_dtype
                ),
            )
            if self.use_rope
            else None
        )

        sliding_window = None
        if is_draft:
            if global_attn_cfg := config.global_attn_cfg:
                sw_entry = global_attn_cfg[
                    self.count_backward_layer_id % len(global_attn_cfg)
                ]
                if isinstance(sw_entry, int) and sw_entry > 0:
                    sliding_window = sw_entry
        elif config.layer_types[self.layer_idx] == "sliding_attention":
            sliding_window = config.sliding_window

        self.use_attn_temperature_tuning = (
            is_draft
            and config.attn_temperature_tuning_floor_scale is not None
            and config.attn_temperature_tuning_q_scale_constant is not None
            and config.attn_temperature_tuning_layers == "global"
            and sliding_window is None
        )
        if self.use_attn_temperature_tuning:
            self.attn_temperature_tuning_floor_scale = (
                config.attn_temperature_tuning_floor_scale
            )
            self.attn_temperature_tuning_q_scale = (
                config.attn_temperature_tuning_q_scale_constant
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

        # Set by the model for the decode L2 prefetch; called with the token count.
        self.after_qkv_hook: Callable[[int], None] | None = None
        self.fused_qk_norm_rope_kv_insert = (
            self.use_qk_norm
            and self.qk_norm is not None
            and not self.qk_norm.has_weight
            and not self.use_attn_temperature_tuning
            and self.head_dim in (64, 128, 256)
            and self.default_dtype in (torch.bfloat16, torch.float16)
            and supports_fused_kv_insert(self.attn)
        )

    @staticmethod
    def compute_attn_temperature_scale(
        positions: torch.Tensor,
        floor_scale: float,
        q_scale: float,
    ) -> torch.Tensor:
        """Query scale: 1.0 up to ``floor_scale``, logarithmic above it."""
        scaling = torch.log((positions.float() + 1.0) / floor_scale) * q_scale + 1.0
        return scaling.clamp_min(1.0).unsqueeze(-1)

    def _get_attn_temperature_scale(self, positions: torch.Tensor) -> torch.Tensor:
        assert self.attn_temperature_tuning_floor_scale is not None
        assert self.attn_temperature_tuning_q_scale is not None
        return self.compute_attn_temperature_scale(
            positions,
            self.attn_temperature_tuning_floor_scale,
            self.attn_temperature_tuning_q_scale,
        )

    def _qk_norm_rope_kv_insert(
        self, qkv: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """QK-norm, temperature scale and RoPE; returns ``(q, k, v, o_gate)``."""
        cos_sin_cache = None
        rotary_dim = 0
        if self.use_rope:
            assert self.rotary_emb is not None
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            rotary_dim = self.rotary_emb.rotary_dim

        if self.fused_qk_norm_rope_kv_insert:
            q_fp8 = fused_qk_norm_rope_kv_insert(
                qkv,
                positions,
                cos_sin_cache,
                self.attn.layer_name,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                rotary_dim,
                self.rms_norm_eps,
            )

        if self.use_o_gate:
            q, k, v, og = qkv.split(
                [self.q_size, self.kv_size, self.kv_size, self.q_size], dim=-1
            )
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            og = None
        if self.fused_qk_norm_rope_kv_insert:
            return q if q_fp8 is None else q_fp8, k, v, og

        if self.qk_norm is not None:
            eps = self.qk_norm.variance_epsilon
            if self.fused_qk_norm:
                q = qk_norm(q, self.head_dim, eps)
                k = qk_norm(k, self.head_dim, eps)
            else:
                q = self.qk_norm(
                    q.reshape(-1, self.head_dim), out_dtype=torch.float32
                ).reshape(-1, self.q_size)
                k = self.qk_norm(
                    k.reshape(-1, self.head_dim), out_dtype=torch.float32
                ).reshape(-1, self.kv_size)
            q = q.to(v.dtype)
            k = k.to(v.dtype)

        if self.use_attn_temperature_tuning:
            q = (q * self._get_attn_temperature_scale(positions)).to(q.dtype)

        if self.rotary_emb is not None:
            # CustomOp dispatch costs more than the kernel does.
            q, k = self.rotary_emb.forward_cuda(positions, q, k)
        return q, k, v, og

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        defer_all_reduce: bool = False,
    ) -> torch.Tensor:
        """Attention over an input the caller has already normalized.

        With ``defer_all_reduce`` the output is this rank's partial sum: the caller
        reduces it (together with the residual boundary, where it can).
        """
        qkv, _ = self.qkv_proj(hidden_states)
        if self.after_qkv_hook is not None:
            self.after_qkv_hook(qkv.shape[0])

        q, k, v, og = self._qk_norm_rope_kv_insert(qkv, positions)

        # Withhold k/v so the attention layer does not rewrite the KV cache.
        if self.fused_qk_norm_rope_kv_insert:
            attn_output = self.attn(q, None, None, output_dtype=qkv.dtype)
        else:
            attn_output = self.attn(q, k, v)

        if self.fuse_o_norm_gate:
            assert og is not None
            attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            og = og.reshape(-1, self.num_heads, self.head_dim)
            if self.cute_o_norm_gate:
                assert o_norm_sigmoid_gate is not None
                o_norm_sigmoid_gate(attn_output, og, self.rms_norm_eps, enable_pdl=True)
            else:
                attn_output = rms_norm_gated(
                    attn_output,
                    og,
                    None,
                    None,
                    "sigmoid",
                    eps=self.rms_norm_eps,
                )
        else:
            if self.use_o_norm:
                assert self.o_norm is not None
                attn_output = self.o_norm(attn_output.reshape(-1, self.head_dim))
                attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
            if self.use_o_gate:
                assert og is not None
                og = og.view(-1, self.num_heads, self.head_dim)
                if attn_output.dim() == 2:
                    attn_output = attn_output.view(-1, self.num_heads, self.head_dim)
                attn_output = og.sigmoid() * attn_output

        attn_output = attn_output.view(-1, self.num_heads * self.head_dim)
        return self._project_output(attn_output, defer_all_reduce)

    def _project_output(
        self, attn_output: torch.Tensor, defer_all_reduce: bool = False
    ) -> torch.Tensor:
        sp_active = _sequence_parallel_active(
            self.is_sequence_parallel_output,
            attn_output.shape[0],
            self.sp_min_tokens,
            self.sp_row_gate,
        )
        if defer_all_reduce and not sp_active:
            reduce_results = self.o_proj.reduce_results
            self.o_proj.reduce_results = False
            try:
                output, _ = self.o_proj(attn_output)
            finally:
                self.o_proj.reduce_results = reduce_results
            return output
        if (
            sp_active
            and self.gemm_rs is not None
            and self.gemm_rs.supports_projection(self.o_proj)
            and self.gemm_rs.should_run(attn_output)
        ):
            return self.gemm_rs(attn_output, self.o_proj.weight)
        output, _ = self.o_proj(attn_output)
        if sp_active:
            output = sp_reduce_scatter(output)
        elif self.is_sequence_parallel_output:
            output = tensor_model_parallel_all_reduce(output)
        return output


class MuseSpark1xDecoderLayer(nn.Module):
    moe_cls = MuseSpark1xMoE

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        is_draft = isinstance(config, MuseSparkDSparkConfig)
        norm_affine = config.norm_affine if is_draft else True
        post_norm_affine = config.post_norm_affine if is_draft else False
        zero_centered_gamma = config.norm_zero_centered_gamma if is_draft else True
        gain_center_type = config.post_norm_gain_center_type if is_draft else "one"
        output_gain_center_type = (
            config.output_norm_gain_center_type if is_draft else "zero"
        )
        learnable_scales = config.use_ffn_learnable_scales if is_draft else True
        post_norm_eps = (
            (config.rms_post_norm_eps or config.rms_norm_eps)
            if is_draft
            else config.post_norm_eps
        )

        self.layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size

        self.self_attn = MuseSpark1xAttention(
            config=config,
            max_position_embeddings=max(
                config.max_position_embeddings,
                vllm_config.model_config.max_model_len,
            ),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.is_moe_layer = (
            (
                config.moe.is_enabled
                and config.moe.interleave_moe_layer_step > 0
                and (self.layer_idx + 1) % config.moe.interleave_moe_layer_step == 0
                and self.layer_idx >= config.moe.n_dense_layers_before_moe
            )
            if is_draft
            else config.mlp_layer_types[self.layer_idx] == "sparse"
        )
        self.pre_ffn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=norm_affine,
            is_post_norm=False,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 2,
            output_norm_gain_center_type=output_gain_center_type,
        )

        if self.is_moe_layer:
            self.feed_forward = self.moe_cls(
                vllm_config=vllm_config,
                layer_idx=self.layer_idx,
                prefix=f"{prefix}.feed_forward",
            )
        else:
            self.feed_forward = LlamaMLP(
                hidden_size=self.hidden_size,
                intermediate_size=(
                    config.intermediate_size_mlp
                    if is_draft
                    else config.intermediate_size
                ),
                hidden_act="silu",
                quant_config=quant_config,
                bias=False,
                prefix=f"{prefix}.feed_forward",
            )
        self.post_attn_norm = MuseSpark1xRMSNorm(
            config.hidden_size,
            eps=post_norm_eps,
            has_weight=post_norm_affine,
            is_post_norm=True,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 1,
            output_norm_gain_center_type=output_gain_center_type,
        )
        post_ffn_cls = (
            MuseSpark1xRMSNormWithInputScale if learnable_scales else MuseSpark1xRMSNorm
        )
        self.post_ffn_norm = post_ffn_cls(
            config.hidden_size,
            eps=post_norm_eps,
            has_weight=(learnable_scales if learnable_scales else post_norm_affine),
            is_post_norm=True,
            zero_centered_gamma=zero_centered_gamma,
            post_norm_gain_center_type=gain_center_type,
            residual_id=2 * self.layer_idx + 2,
            output_norm_gain_center_type=output_gain_center_type,
        )
        self.residual_dtype = config.residual_dtype if is_draft else torch.float32
        self.default_dtype = torch.get_default_dtype()
        # Set by the model when the next boundary may consume unpooler shards.
        self.all_gather_dual_norm: Any | None = None
        # Whether the TP communicator may run the boundaries as the epilogue of its
        # collectives (ops/fused_boundary.py).
        self.fuses_boundary_collectives = (
            envs.VLLM_MUSE_SPARK_SM120_FUSED_BOUNDARY and _on_sm120()
        )
        # The unpooler shard travels to the next boundary, like the one of the
        # FlashInfer gather: only where the layers fuse across (no pipeline
        # parallelism, see MuseSpark1xModel.fuse_across_layers), and not where the
        # checkpoint's config turns that fusion off.
        self.fuses_column_gather = (
            self.fuses_boundary_collectives
            and get_pp_group().world_size == 1
            and getattr(config, "muse_spark_tep_fused_boundary", True)
        )

        self.sm120_sequence_parallel = (
            self.is_moe_layer and is_sm120_sequence_parallel_enabled(vllm_config)
        )
        self.is_sequence_parallel = (
            is_sequence_parallel_enabled(vllm_config) or self.sm120_sequence_parallel
        )
        # Smallest batch that shards the stream; None: one token per rank.
        self.sp_min_tokens = (
            sm120_sequence_parallel_min_tokens(vllm_config)
            if self.sm120_sequence_parallel
            else None
        )
        self.is_sequence_parallel_input = self.is_sequence_parallel
        self.self_attn.is_sequence_parallel_output = self.is_sequence_parallel
        self.self_attn.sp_min_tokens = self.sp_min_tokens
        # Set by the model in the auto mode (see _SequenceParallelRowGate).
        self.sp_row_gate: Callable[[int], bool] | None = None
        if self.is_sequence_parallel:
            self.self_attn.o_proj.reduce_results = False

        if not is_draft or config.use_residual_gate:
            self.add_attn = MuseSpark1xNormPreservingResidualConnection(
                dim=config.hidden_size,
                activation="sigmoid",
                temperature=0.3 if is_draft else config.residual_gate_temperature,
                use_stream_gate=True,
                channel_wise=True,
            )
            self.add_ffn = MuseSpark1xNormPreservingResidualConnection(
                dim=config.hidden_size,
                activation="sigmoid",
                temperature=0.3 if is_draft else config.residual_gate_temperature,
                use_stream_gate=True,
                channel_wise=True,
            )
        else:
            self.add_attn = MuseSpark1xResidualConnection()
            self.add_ffn = MuseSpark1xResidualConnection()

    def forward(
        self,
        positions: torch.Tensor,
        x: MuseSpark1xResidualStream | torch.Tensor,
    ) -> MuseSpark1xResidualStream:
        """Run attention and FFN; the FFN boundary is returned unblended."""
        if isinstance(x, torch.Tensor):
            stream = x
            attn_input = self.self_attn.pre_qkv_proj_norm(
                stream, out_dtype=self.default_dtype
            )
        else:
            attn_input, stream = norm_residual_then_norm(
                x, self.self_attn.pre_qkv_proj_norm, out_dtype=self.default_dtype
            )
        assert stream.dtype == self.residual_dtype

        sp_active = _sequence_parallel_active(
            self.is_sequence_parallel,
            positions.shape[0],
            self.sp_min_tokens,
            self.sp_row_gate,
        )
        sm120_sp = sp_active and self.sm120_sequence_parallel
        if sp_active and self.is_sequence_parallel_input:
            # Positions carries the unpadded length of the gathered sequence.
            attn_input = sp_all_gather(attn_input)[: positions.shape[0]]
        defer_all_reduce = self._defers_all_reduce(positions.shape[0], sp_active)
        attn_out = self.self_attn(
            positions=positions,
            hidden_states=attn_input,
            defer_all_reduce=defer_all_reduce,
        )
        if sp_active and not self.is_sequence_parallel_input:
            stream = sp_shard(stream)
        router_input = None
        cast_router_input = False
        if isinstance(self.feed_forward, MuseSpark1xMoE):
            if not sm120_sp and self.feed_forward.wants_fp32_router_input(
                attn_out.shape[0]
            ):
                router_input = torch.empty_like(attn_out, dtype=torch.float32)
            elif self.feed_forward.wants_cast_router_input(attn_out.shape[0]):
                router_input = torch.empty_like(attn_out, dtype=torch.float32)
                cast_router_input = True
        attn_boundary_cls = (
            MuseSpark1xPartialSumStream
            if defer_all_reduce
            else MuseSpark1xResidualStream
        )
        ffn_in, stream = norm_residual_then_norm(
            attn_boundary_cls(
                stream=stream,
                branch=attn_out,
                post_norm=self.post_attn_norm,
                connection=self.add_attn,
            ),
            self.pre_ffn_norm,
            out_dtype=self.default_dtype,
            router_input=None if cast_router_input else router_input,
            out_fp32=router_input if cast_router_input else None,
        )
        if sm120_sp:
            # The router reads the boundary's fp32 copy of this rank's rows, when the
            # boundary wrote one (it never holds the unrounded mega-MoE input here).
            branch = cast(MuseSpark1xMoE, self.feed_forward).forward_sequence_parallel(
                ffn_in, positions.shape[0], router_input=router_input
            )
        elif sp_active and not self.feed_forward.use_mega_moe:
            # Non-megakernel experts need every token.
            branch = sp_shard(self.feed_forward(sp_all_gather(ffn_in)))
        elif self._defers_column_gather(ffn_in.shape[0]):
            columns = self.feed_forward(
                ffn_in, router_input=router_input, defer_column_gather=True
            )
            if isinstance(columns, MuseSpark1xUnpoolerInput):
                return MuseSpark1xProjectedStream(
                    stream=stream,
                    branch=columns.x,
                    post_norm=self.post_ffn_norm,
                    connection=self.add_ffn,
                    all_gather=self.all_gather_dual_norm,
                    weight=columns.weight,
                    project=columns.project,
                )
            return MuseSpark1xColumnShardedStream(
                stream=stream,
                branch=columns,
                post_norm=self.post_ffn_norm,
                connection=self.add_ffn,
                all_gather=self.all_gather_dual_norm,
            )
        else:
            branch = (
                self.feed_forward(ffn_in, router_input=router_input)
                if router_input is not None
                else self.feed_forward(ffn_in)
            )
        return MuseSpark1xResidualStream(
            stream=stream,
            branch=branch,
            post_norm=self.post_ffn_norm,
            connection=self.add_ffn,
        )

    def _defers_all_reduce(self, num_tokens: int, sp_active: bool) -> bool:
        """Whether the attention-output all-reduce is left to the boundary after it.

        True where the TP communicator fuses the two. Not where the boundary also
        writes the unrounded fp32 router input: another kernel computes that.
        """
        if not self.fuses_boundary_collectives or sp_active:
            return False
        if isinstance(
            self.feed_forward, MuseSpark1xMoE
        ) and self.feed_forward.wants_fp32_router_input(num_tokens):
            return False
        return fused_boundary.supports_fused_boundary(
            fused_boundary.KIND_ALL_REDUCE_BOUNDARY, num_tokens
        )

    def _defers_column_gather(self, num_tokens: int) -> bool:
        if not isinstance(self.feed_forward, MuseSpark1xMoE):
            return False
        if self.all_gather_dual_norm is not None:
            # Only sharded MoE layers get a workspace (see _configure_column_gather).
            return self.feed_forward.shards_unpooler(num_tokens)
        # The TP communicator may gather the columns inside the next boundary.
        return (
            self.fuses_column_gather
            and self.feed_forward.shards_unpooler(num_tokens)
            and fused_boundary.supports_fused_boundary(
                fused_boundary.KIND_ALL_GATHER_BOUNDARY, num_tokens
            )
        )


# Runs uncompiled under breakable cudagraphs; drop LlamaModel's torch.compile.
@ignore_torch_compile
class MuseSpark1xModel(LlamaModel):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] = MuseSpark1xDecoderLayer,
    ):
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

        config = self.config
        if get_pp_group().is_last_rank:
            self.norm = MuseSpark1xRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
                is_post_norm=False,
                has_weight=True,
                zero_centered_gamma=True,
                post_norm_gain_center_type="one",
                residual_id=-1,
            )
        else:
            self.norm = PPMissingLayer()

        self.tok_embeddings_norm = MuseSpark1xRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, has_weight=False
        )

        self.default_dtype = torch.get_default_dtype()
        self.residual_dtype = torch.float32

        self.fuse_across_layers = get_pp_group().world_size == 1
        sm120_sp = is_sm120_sequence_parallel_enabled(vllm_config)
        self.is_sequence_parallel = (
            is_sequence_parallel_enabled(vllm_config) or sm120_sp
        )
        # Smallest batch that shards the stream; None: one token per rank.
        self.sp_min_tokens = (
            sm120_sequence_parallel_min_tokens(vllm_config) if sm120_sp else None
        )
        self.sp_row_gate: Callable[[int], bool] | None = None
        if self.sp_min_tokens is not None:
            logger.info_once(
                "MuseSpark SM120: sequence-parallel residual stream for batches of "
                "at least %d tokens (VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS"
                "=%d, CUDA graphs up to %d tokens)%s.",
                self.sp_min_tokens,
                envs.VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS,
                vllm_config.compilation_config.max_cudagraph_capture_size or 0,
                ", only for the batch sizes whose sharded projections are bitwise "
                "the replicated ones (auto: the TP communicator's reduce-scatter + "
                "all-gather equals its all-reduce)"
                if sm120_sequence_parallel_is_gated()
                else "",
            )
        self.make_empty_intermediate_tensors = self._make_empty_intermediate_tensors
        self.is_sequence_parallel_input = self.is_sequence_parallel
        self._configure_replicated_prefix()
        self._configure_column_gather()
        self._configure_pooler_gather()
        if sm120_sp and sm120_sequence_parallel_is_gated():
            self._configure_sequence_parallel_row_gate()
        self._l2_prefetcher: L2Prefetcher | None = None
        if envs.VLLM_MUSE_SPARK_SM120_L2_PREFETCH and _on_sm120():
            self._configure_l2_prefetch()
        self._expert_prefetch: ExpertPrefetch | None = None
        if envs.VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH and _on_sm120():
            self._configure_expert_prefetch()
        # After the expert prefetch: the router GEMV needs the layers that queue it.
        self._configure_router_gemv()
        if self.is_sequence_parallel:
            from vllm.models.muse_spark.nvidia.ops.gemm_rs import (
                maybe_create_gemm_rs,
            )

            attention_layers = [
                layer.self_attn
                for layer in islice(self.layers, self.start_layer, self.end_layer)
            ]
            gemm_rs = maybe_create_gemm_rs(
                vllm_config, [attn.o_proj for attn in attention_layers]
            )
            if gemm_rs is not None:
                for attn in attention_layers:
                    if gemm_rs.supports_projection(attn.o_proj):
                        attn.gemm_rs = gemm_rs

    def _configure_sequence_parallel_row_gate(self) -> None:
        """Admit the batch sizes whose sharded projections equal the replicated."""
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        feed_forward = next(
            layer.feed_forward
            for layer in layers
            if isinstance(layer.feed_forward, MuseSpark1xMoE)
            and layer.feed_forward.sm120_sequence_parallel
        )
        self.sp_row_gate = _SequenceParallelRowGate(feed_forward)
        for layer in layers:
            layer.sp_row_gate = self.sp_row_gate
            layer.self_attn.sp_row_gate = self.sp_row_gate

    def _configure_replicated_prefix(self) -> None:
        # Layer 0 (BF16 runner) stays replicated until layer 1's attention RS.
        if not self.is_sequence_parallel or self.start_layer != 0 or self.end_layer < 2:
            return
        first, second = self.layers[0], self.layers[1]
        if (
            not isinstance(first.feed_forward, MuseSpark1xMoE)
            or first.feed_forward.use_mega_moe
            or not isinstance(second.feed_forward, MuseSpark1xMoE)
            or not second.feed_forward.use_mega_moe
        ):
            return
        self.is_sequence_parallel_input = False
        first.is_sequence_parallel_input = False
        first.is_sequence_parallel = False
        first.self_attn.is_sequence_parallel_output = False
        first.self_attn.o_proj.reduce_results = True
        first.feed_forward.is_sequence_parallel = False
        second.is_sequence_parallel_input = False

    def _configure_column_gather(self) -> None:
        """Let sharded unpoolers hand their columns to the next boundary."""
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        sharded = [
            idx
            for idx, layer in enumerate(layers)
            if isinstance(layer.feed_forward, MuseSpark1xMoE)
            and layer.feed_forward.shard_unpooler
        ]
        if (
            not sharded
            or not self.fuse_across_layers
            or not getattr(self.config, "muse_spark_tep_fused_boundary", True)
        ):
            return
        all_gather = maybe_create_all_gather_dual_norm(
            self.config.hidden_size,
            _SHARDED_PROJECTION_MAX_TOKENS,
            enable_pdl=getattr(self.config, "muse_spark_tep_fused_boundary_pdl", True),
        )
        if all_gather is None:
            return
        for idx in sharded:
            layer = layers[idx]
            layer.all_gather_dual_norm = all_gather
            next_norm = (
                layers[idx + 1].self_attn.pre_qkv_proj_norm
                if idx + 1 < len(layers)
                else self.norm
            )
            # Compile ahead of CUDA graph capture.
            all_gather.compile(
                has_post_weight=layer.post_ffn_norm.has_weight,
                **_dual_norm_constants(layer.post_ffn_norm, next_norm),
            )

    def _configure_pooler_gather(self) -> None:
        """Let sharded poolers gather straight into the NVFP4 expert input."""
        moes = [
            layer.feed_forward
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if isinstance(layer.feed_forward, MuseSpark1xMoE)
            and layer.feed_forward.shard_pooler
            and layer.feed_forward.fused_input_quant
            and layer.feed_forward.norm_fuses_into_nvfp4_quant
        ]
        if not moes:
            return
        all_gather = maybe_create_all_gather_norm_quant(
            self.config.moe_hidden_size, _SHARDED_PROJECTION_MAX_TOKENS
        )
        if all_gather is None:
            return
        for moe in moes:
            norm = moe.pre_expert_norm
            assert norm is not None
            moe.all_gather_norm_quant = all_gather
            # Compile ahead of CUDA graph capture.
            all_gather.compile(
                eps=norm.variance_epsilon, gain_center=norm.fused_gain_center
            )

    def _configure_router_gemv(self) -> None:
        """Let the decode router of the MoE layers run as ``_muse_C::router_gemv``.

        The kernel reads the FP32 router input, which the boundary kernel writes in
        the decode graphs only for the layers that queue the expert prefetch
        (``wants_cast_router_input``); the other layers keep cuBLAS, so only these
        are marked, and without any of them nothing is built. Must run after
        ``_configure_expert_prefetch``.
        """
        if not (envs.VLLM_MUSE_SPARK_SM120_ROUTER_GEMV and _on_sm120()):
            return
        moes = [
            layer.feed_forward
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if isinstance(layer.feed_forward, MuseSpark1xMoE)
        ]
        if not moes:
            return
        moes = [moe for moe in moes if moe.expert_prefetch is not None]
        if not moes:
            logger.info_once(
                "The router GEMV is off: the decode graphs get its FP32 input only "
                "from the boundary kernel beside the expert prefetch, which no layer "
                "queues (VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH=0)."
            )
            return
        if (reason := muse_c.preflight()) is not None:
            logger.info_once("The router GEMV is off: %s", reason)
            return
        try:
            # Never build inside a forward pass or a capture.
            muse_c.load()
        except Exception as e:
            logger.warning_once(
                "The router GEMV is off: the _muse_C extension could not be built "
                "(%s: %s).",
                type(e).__name__,
                e,
            )
            return
        for moe in moes:
            moe.router_gemv_ready = True
        logger.info_once(
            "MuseSpark: router GEMV of _muse_C for decode batches of up to %d tokens "
            "in %d layers.",
            _ROUTER_GEMV_MAX_TOKENS,
            len(moes),
        )

    def _configure_expert_prefetch(self) -> None:
        """Prefetch the routed experts of a decode layer into L2 (see
        ``ops/expert_prefetch.py``).

        The layers whose experts run the muse decode kernel behind routing on the
        router's aux stream queue it; it is a cache hint that changes no output.
        """
        moes = [
            layer.feed_forward
            for layer in islice(self.layers, self.start_layer, self.end_layer)
            if isinstance(layer.feed_forward, MuseSpark1xMoE)
            and isinstance(layer.feed_forward.experts, MuseMoERunner)
            and layer.feed_forward.experts.supports_after_gemm2_hook
            and layer.feed_forward._router_aux_stream is not None
        ]
        if not moes:
            return
        budget_bytes = envs.VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES
        if budget_bytes <= 0:
            raise ValueError(
                "VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES must be positive, got "
                f"{budget_bytes} (VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH=0 turns the "
                "prefetch off)."
            )
        prefetch = ExpertPrefetch(moes[0]._router_aux_stream, budget_bytes)
        for moe in moes:
            moe.expert_prefetch = prefetch
        self._expert_prefetch = prefetch
        logger.info_once(
            "MuseSpark: expert L2 prefetch of the first %.4g MiB in %d layers.",
            prefetch.budget_bytes / (1 << 20),
            len(moes),
        )

    def _configure_l2_prefetch(self) -> None:
        """Prefetch weights into L2 behind the collectives of the layer before.

        Decode only, inside the CUDA graph, on a side stream (see
        ``ops/l2_prefetch.py``). Behind the experts of layer ``i``, which are
        followed by the MoE all-reduce, it pulls the unpooler of layer ``i`` and
        the QKV and O weights of layer ``i + 1``; behind the QKV projection of
        layer ``i`` (the attention reads little from HBM there), the pooler and
        router weights of layer ``i``. The weights are looked up when the graph
        is captured, after they are final.

        A hint that changes no output, so a host that cannot build the ``_muse_C``
        extension, and a model without the SM120 native NVFP4 experts, run without it.
        """
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        # No extension (and no stream) for a model whose experts have no call site.
        if not any(
            isinstance(layer.feed_forward, MuseSpark1xMoE)
            and isinstance(layer.feed_forward.experts, MuseMoERunner)
            and layer.feed_forward.experts.supports_after_gemm2_hook
            for layer in layers[:-1]
        ):
            return
        if (reason := muse_c.preflight()) is not None:
            logger.info_once("The L2 weight prefetch is off: %s", reason)
            return
        after_experts_ctas = envs.VLLM_MUSE_SPARK_SM120_L2_PREFETCH_AFTER_EXPERTS_CTAS
        if not 1 <= after_experts_ctas <= L2_PREFETCH_MAX_CTAS:
            raise ValueError(
                "VLLM_MUSE_SPARK_SM120_L2_PREFETCH_AFTER_EXPERTS_CTAS must be in "
                f"[1, {L2_PREFETCH_MAX_CTAS}], got {after_experts_ctas}."
            )
        try:
            prefetcher = L2Prefetcher()
        except Exception as e:
            # The preflight does not know whether the build works.
            logger.warning_once(
                "The L2 weight prefetch is off: the _muse_C extension could not be "
                "built (%s: %s).",
                type(e).__name__,
                e,
            )
            return

        def after_experts(
            moe: MuseSpark1xMoE, next_attn: MuseSpark1xAttention
        ) -> Callable[[int], None]:
            def hook(num_tokens: int) -> None:
                if num_tokens <= L2_PREFETCH_MAX_TOKENS:
                    prefetcher.launch(
                        [
                            *moe.sharded_weights(num_tokens, moe.unpooler),
                            next_attn.qkv_proj.weight.data,
                            next_attn.o_proj.weight.data,
                        ],
                        num_ctas=after_experts_ctas,
                    )

            return hook

        def after_qkv(moe: MuseSpark1xMoE) -> Callable[[int], None]:
            def hook(num_tokens: int) -> None:
                if num_tokens <= L2_PREFETCH_MAX_TOKENS:
                    prefetcher.launch(
                        [
                            *moe.sharded_weights(num_tokens, moe.pooler),
                            moe.router.weight.data,
                        ],
                    )

            return hook

        hooked = 0
        for layer, next_layer in zip(layers, layers[1:]):
            moe = layer.feed_forward
            if isinstance(moe, MuseSpark1xMoE) and isinstance(
                moe.experts, MuseMoERunner
            ):
                if moe.experts.set_after_gemm2_hook(
                    after_experts(moe, next_layer.self_attn)
                ):
                    layer.self_attn.after_qkv_hook = after_qkv(moe)
                    hooked += 1
        if hooked:
            self._l2_prefetcher = prefetcher
            logger.info_once(
                "MuseSpark: L2 prefetch of the weights of the next decode ops in %d "
                "layers (%d CTAs behind the experts).",
                hooked,
                after_experts_ctas,
            )

    def _make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        return IntermediateTensors(
            {
                "hidden_states": torch.zeros(
                    (batch_size, self.config.hidden_size),
                    dtype=self.residual_dtype,
                    device=device,
                ),
            }
        )

    def finalize_mega_moe_weights(self) -> None:
        finalized = False
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            if isinstance(layer.feed_forward, MuseSpark1xMoE):
                layer.feed_forward.finalize_mega_moe_weights()
                finalized = finalized or layer.feed_forward.use_mega_moe
        if finalized:
            # Free the pre-repack blocks before the KV cache is sized.
            torch.accelerator.empty_cache()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        return self.tok_embeddings_norm(hidden_states, out_dtype=self.residual_dtype)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        x: MuseSpark1xResidualStream | torch.Tensor = hidden_states.to(
            self.residual_dtype
        )
        sp_active = _sequence_parallel_active(
            self.is_sequence_parallel,
            positions.shape[0],
            self.sp_min_tokens,
            self.sp_row_gate,
        )
        if sp_active and self.is_sequence_parallel_input:
            x = sp_shard(x)
        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)
        aux_hidden_states: list[torch.Tensor] = []
        if (
            get_pp_group().is_first_rank
            and self.start_layer in self.aux_hidden_state_layers
        ):
            initial_state = (
                sp_shard(x) if sp_active and not self.is_sequence_parallel_input else x
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, self.start_layer, initial_state, None
            )
        for idx, layer in enumerate(
            self.layers[self.start_layer : self.end_layer], start=self.start_layer
        ):
            x = layer(positions, x)
            if (
                isinstance(x, MuseSpark1xColumnShardedStream)
                and idx + 1 in self.aux_hidden_state_layers
            ):
                x = x.materialize()
            if not self.fuse_across_layers and isinstance(x, MuseSpark1xResidualStream):
                x = norm_then_gated_residual(x)
            if idx + 1 in self.aux_hidden_state_layers:
                # Don't replace x: the target keeps its fused dual-norm path.
                state = (
                    norm_then_gated_residual(x)
                    if isinstance(x, MuseSpark1xResidualStream)
                    else x
                )
                if sp_active and not layer.is_sequence_parallel:
                    state = sp_shard(state)
                aux_hidden_states.append(state)

        if self._l2_prefetcher is not None:
            self._l2_prefetcher.join()
        if self._expert_prefetch is not None:
            self._expert_prefetch.join()

        if not get_pp_group().is_last_rank:
            assert isinstance(x, torch.Tensor)  # fuse_across_layers is off here
            return IntermediateTensors(
                {
                    "hidden_states": x,
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )

        if isinstance(x, torch.Tensor):
            output = self.norm(x, out_dtype=self.default_dtype)
        else:
            output, _ = norm_residual_then_norm(
                x, self.norm, out_dtype=self.default_dtype
            )
        if sp_active:
            output = sp_all_gather(output)[: positions.shape[0]]
            aux_hidden_states = [
                sp_all_gather(a)[: positions.shape[0]] for a in aux_hidden_states
            ]
        aux_hidden_states = remote_aux + aux_hidden_states
        if aux_hidden_states:
            return output, aux_hidden_states
        return output


class MuseSpark1xForCausalLM(LlamaForCausalLM, SupportsLoRA, MixtureOfExperts):
    model_cls = MuseSpark1xModel
    decoder_layer_cls = MuseSpark1xDecoderLayer
    supported_quantization_configs: tuple[type[QuantizationConfig], ...] = (
        ModelOptNvFp4Config,
    )
    supported_quantization_description = "NVFP4 (modelopt_fp4)"

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper()

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] | None = None,
    ):
        config = vllm_config.model_config.hf_text_config
        gated_delta_net = getattr(config, "gated_delta_net", None)
        if gated_delta_net is not None and getattr(gated_delta_net, "enable", False):
            raise NotImplementedError(
                "MuseSpark1x gated_delta_net layers are not supported by the "
                "native vLLM text stack."
            )
        if vllm_config.quant_config is not None and not isinstance(
            vllm_config.quant_config, self.supported_quantization_configs
        ):
            raise NotImplementedError(
                "MuseSpark1x only supports "
                f"{self.supported_quantization_description} quantized checkpoints; "
                "got quantization "
                f"{vllm_config.quant_config.get_name()!r}."
            )
        if layer_type is None:
            layer_type = self.decoder_layer_cls

        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)

        # qkv_proj fuses Q+K+V+OG, so LoRA must treat it as one module.
        self.packed_modules_mapping = {  # type: ignore[misc]
            k: v for k, v in self.packed_modules_mapping.items() if k != "qkv_proj"
        }

        self.metap_output_multiplier = config.output_multiplier
        self.output_soft_cap_temp = config.final_logit_softcapping

        self.mul_tanh_soft_cap = (
            MulTanhSoftCap(
                multiplier=self.metap_output_multiplier,
                soft_cap_temp=self.output_soft_cap_temp,
            )
            if (
                self.metap_output_multiplier != 1.0
                and self.output_soft_cap_temp is not None
            )
            else None
        )

        if get_pp_group().is_last_rank:
            self.logits_processor.scale = 1.0

        self.set_moe_parameters()

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[MuseSpark1xDecoderLayer] = MuseSpark1xDecoderLayer,
    ):
        return self.model_cls(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    @property
    def min_tokens_per_ubatch(self) -> int:
        """Smallest microbatch (in tokens) that takes the path of a large batch.

        Below ``sp_min_tokens`` the model keeps the residual stream replicated; a
        split batch is only worth running as microbatches when every microbatch
        shards the stream like the whole batch would (see
        ``vllm.v1.worker.gpu.ubatch_utils.fit_prefill_threshold_to_model``).
        """
        return self.model.sp_min_tokens or 0

    def set_moe_parameters(self):
        self.moe_layers: list[nn.Module] = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            assert isinstance(layer, MuseSpark1xDecoderLayer)
            if isinstance(layer.feed_forward, MuseSpark1xMoE):
                example_moe = layer.feed_forward
                self.moe_layers.append(layer.feed_forward.experts)

        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
            logger.warning("No MuseSpark1xMoE layer found in model.layers.")
        else:
            self.num_moe_layers = len(self.moe_layers)
            self.num_expert_groups = 1
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.feed_forward, MuseSpark1xMoE):
                moe = layer.feed_forward
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        hidden_states = hidden_states.to(self.lm_head.weight.dtype)
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is None:
            return None
        if self.mul_tanh_soft_cap is not None:
            return self.mul_tanh_soft_cap(logits)
        if logits.dtype != torch.float32:
            logits = logits.to(torch.float32)
        if self.metap_output_multiplier != 1.0:
            logits = logits * self.metap_output_multiplier
        if self.output_soft_cap_temp is not None:
            logits = self.output_soft_cap_temp * torch.tanh(
                logits / self.output_soft_cap_temp
            )
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def interleaved_weights() -> Iterable[tuple[str, torch.Tensor]]:
            config = self.model.config
            for name, weight in weights:
                if name.startswith(
                    ("model.language_model.layers.", "model.layers.", "layers.")
                ) and name.endswith(
                    (".self_attn.q_proj.weight", ".self_attn.k_proj.weight")
                ):
                    # HF stores Q/K rows as [even | odd] per head (NoPE too);
                    # restore interleaved pairs before TP slicing.
                    shape = weight.shape
                    weight = (
                        weight.reshape(-1, 2, config.head_dim // 2, shape[1])
                        .transpose(1, 2)
                        .reshape(shape)
                    )
                yield name, weight

        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            interleaved_weights(), mapper=self.hf_to_vllm_mapper
        )
        self.model.finalize_mega_moe_weights()
        n_gates = materialize_residual_path_constants(self)
        logger.info(
            "Materialized residual-path constants for %d gated residual connections",
            n_gates,
        )
        return loaded

    def process_weights_after_loading(self) -> None:
        """Model-level post-load hook, run by every loader.

        ``--load-format dummy`` never calls ``load_weights``, so materialize the
        residual-path constants here too (idempotent): dummy-weight runs then
        take the same fused residual kernels and folded router as real weights.
        """
        already = any(
            getattr(m, "_coefficients_ready", False) for m in self.modules()
        )
        n_gates = materialize_residual_path_constants(self)
        if not already:
            logger.info(
                "Materialized residual-path constants for %d gated residual "
                "connections after loading",
                n_gates,
            )
