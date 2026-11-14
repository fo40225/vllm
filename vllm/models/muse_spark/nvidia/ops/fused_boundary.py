# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Residual boundaries that the TP device communicator fuses into its collectives.

A Muse Spark decode layer has three collective -> element-wise kernel pairs on its
critical path: the attention-output all-reduce followed by the residual boundary
(post-norm, gated residual, next pre-norm), the unpooler all-gather followed by the
next boundary, and the pooler all-gather followed by the pre-expert RMSNorm. Each
pair costs a launch and a round trip of the collective's result through memory.

A device communicator that can run the element-wise part as the epilogue of its
collective implements the optional interface below (a host plugin does it for its
machine; nothing here depends on one). The communicator must reproduce, bit for
bit, what the model's own kernels compute from the collective's result, so that
taking this path changes no output: ``fused_norm_residual_norm_fwd`` for the
boundaries and ``fused_rmsnorm_fwd`` for the RMSNorm
(``vllm/models/muse_spark/common/ops/fused_rmsnorm.py``). The collective itself keeps
the arithmetic of the communicator's own ``all_reduce`` / ``all_gather``.

Interface (all optional; the model calls a method only after
``supports_fused_residual_boundary`` approved the batch size, and handles a ``None``
result by running the two steps itself)::

    supports_fused_residual_boundary(kind, num_tokens) -> bool
        kind: "all_reduce_boundary" | "all_gather_boundary" | "all_gather_rmsnorm" |
        "all_gather_rmsnorm_nvfp4".
        Depends on the batch size only, identically on every rank of the group.
    all_reduce_residual_boundary(branch, stream, *, alpha, beta, pre_weight,
            post_weight, post_eps, pre_eps, post_gain_center, pre_gain_center,
            out_fp32=None) -> (out, new_stream) | None
        branch: this rank's partial sum, bf16 [T, D]; stream: fp32 [T, D].
        Returns ``fused_norm_residual_norm_fwd(all_reduce(branch), stream, ...)`` with
        ``out_dtype=bfloat16`` and ``post_input_scale = post_weight is not None``.
    all_gather_residual_boundary(shard, stream, *, <same keywords>)
        -> (out, new_stream) | None
        shard: this rank's column block, bf16 [T, D / tp]; the branch is the column
        concatenation of the shards.
    all_gather_rmsnorm(shard, weight, eps, gain_center) -> Tensor | None
        ``fused_rmsnorm_fwd(all_gather(shard, -1), weight, eps, gain_center=...)``.
    all_gather_rmsnorm_nvfp4(shard, weight, eps, gain_center, global_scale, codes, scales)
            -> Tensor | None
        ``all_gather_rmsnorm`` that also fills ``codes`` (uint8 [T, 2048]) and ``scales``
        (uint8 [T, 256]) with the NVFP4 quantization of the rows it returns that the muse
        large-batch decode kernel makes of its input (``_muse_C::nvfp4_decode_act_quant``
        with ``global_scale[0]``: 16-element blocks, e4m3 linear scales, e2m1 codes), so
        that the kernel can skip its own quantization. Kind ``"all_gather_rmsnorm_nvfp4"``
        of ``supports_fused_residual_boundary``.

The two all-gathers follow a GEMV: the sharded pooler / unpooler of the MoE layer
computes this rank's columns of ``x @ W.T`` (``W`` = the rank's rows of the replicated
weight), and the columns are gathered. A communicator may also run the GEMV itself,
together with the gather, so that every GEMV block sends its columns to the peers as
soon as they are done (a host plugin does it for its machine, and loads the weights
before its predecessor kernel has finished). It then owns the arithmetic of the GEMV:
fp32 accumulation, one rounding to the output dtype, in an order of its own that does
not depend on the batch size, the rank or the run (the model's cuBLAS GEMV has another
order; the results differ in the last bit of some elements, deterministically)::

    supports_fused_residual_boundary(kind, num_tokens) -> bool
        also: kind: "gemv_all_gather_rmsnorm" | "gemv_all_gather_boundary"
    gemv_all_gather_rmsnorm(x, weight, norm_weight, eps, gain_center) -> Tensor | None
        x: bf16 [T, 8192], weight: this rank's bf16 rows [512, 8192]; the pooler.
        Returns the rows ``fused_rmsnorm_fwd(all_gather(gemv(x, weight), -1),
        norm_weight, eps, gain_center=...)`` as ``all_gather_rmsnorm`` does for the
        gathered shard.
    gemv_all_gather_residual_boundary(x, weight, stream, *, <same keywords>)
        -> (out, new_stream) | None
        x: bf16 [T, 4096], weight: this rank's bf16 rows [1024, 4096]; the unpooler.
        As ``all_gather_residual_boundary`` for the shard ``gemv(x, weight)``; the
        keywords are those of ``all_gather_residual_boundary``.

The decode CUDA graphs capture whatever the communicator launches.
"""

from typing import Any

import torch

from vllm.distributed import get_tp_group

KIND_ALL_REDUCE_BOUNDARY = "all_reduce_boundary"
KIND_ALL_GATHER_BOUNDARY = "all_gather_boundary"
KIND_ALL_GATHER_RMSNORM = "all_gather_rmsnorm"
KIND_ALL_GATHER_RMSNORM_NVFP4 = "all_gather_rmsnorm_nvfp4"
KIND_GEMV_ALL_GATHER_BOUNDARY = "gemv_all_gather_boundary"
KIND_GEMV_ALL_GATHER_RMSNORM = "gemv_all_gather_rmsnorm"


def _communicator() -> Any | None:
    communicator = get_tp_group().device_communicator
    if communicator is None or not hasattr(
        communicator, "supports_fused_residual_boundary"
    ):
        return None
    return communicator


def supports_fused_boundary(kind: str, num_tokens: int) -> bool:
    """Whether the TP communicator fuses ``kind`` for a batch of ``num_tokens``."""
    communicator = _communicator()
    return communicator is not None and bool(
        communicator.supports_fused_residual_boundary(kind, num_tokens)
    )


def all_reduce_residual_boundary(
    branch: torch.Tensor, stream: torch.Tensor, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    communicator = _communicator()
    if communicator is None:
        return None
    return communicator.all_reduce_residual_boundary(branch, stream, **kwargs)


def all_gather_residual_boundary(
    shard: torch.Tensor, stream: torch.Tensor, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    communicator = _communicator()
    if communicator is None:
        return None
    return communicator.all_gather_residual_boundary(shard, stream, **kwargs)


def all_gather_rmsnorm(
    shard: torch.Tensor, weight: torch.Tensor, eps: float, gain_center: float
) -> torch.Tensor | None:
    communicator = _communicator()
    if communicator is None:
        return None
    return communicator.all_gather_rmsnorm(shard, weight, eps, gain_center)


def all_gather_rmsnorm_nvfp4(
    shard: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gain_center: float,
    global_scale: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor | None:
    communicator = _communicator()
    if communicator is None or not hasattr(communicator, "all_gather_rmsnorm_nvfp4"):
        return None
    return communicator.all_gather_rmsnorm_nvfp4(
        shard, weight, eps, gain_center, global_scale, codes, scales
    )


def gemv_all_gather_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    gain_center: float,
) -> torch.Tensor | None:
    communicator = _communicator()
    if communicator is None or not hasattr(communicator, "gemv_all_gather_rmsnorm"):
        return None
    return communicator.gemv_all_gather_rmsnorm(
        x, weight, norm_weight, eps, gain_center
    )


def gemv_all_gather_residual_boundary(
    x: torch.Tensor, weight: torch.Tensor, stream: torch.Tensor, **kwargs: Any
) -> tuple[torch.Tensor, torch.Tensor] | None:
    communicator = _communicator()
    if communicator is None or not hasattr(
        communicator, "gemv_all_gather_residual_boundary"
    ):
        return None
    return communicator.gemv_all_gather_residual_boundary(x, weight, stream, **kwargs)
