# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny bf16 GEMM of the SM120 decode layers (``_muse_C::decode_gemm``).

The QKV projection and the sharded pooler / unpooler of a TP8 rank are
``[rows, K] x [N, K]^T`` products with a weight of 8 to 38 MB and few rows. cuBLAS runs
them as a cutlass tensor-op GEMM that splits K over the CTAs plus a second kernel
(``cublasLt::splitKreduce_kernel``) that adds the partial tiles, two launches with a
dependency in between. ``decode_gemm`` is one kernel: K is split over a thread block
cluster, every CTA stores its fp32 partial tile to a scratch buffer, one cluster
barrier, and each CTA of the cluster adds the pieces of its rows in rank order.

The result is not the cuBLAS one: the order of the additions is another one (fixed,
independent of the row count, of the tile configuration, of the run and of the rank;
it depends on the K split of the shape only), and the partial sums stay in fp32 where
cuBLAS keeps them in a bf16 workspace for some shapes. See
``csrc/muse_spark_decode_gemm.cu``.

All calls of a device share one fp32 scratch buffer for the partial tiles. That is
correct because the calls are ordered by the stream they run on (the decode graph runs
the QKV, pooler and unpooler projections one after the other on the main stream, and a
programmatic dependent waits for the whole previous grid before it writes the scratch);
a caller that ran two of them concurrently on different streams would need a buffer per
stream.
"""

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.models.muse_spark.common.ops import resolve_compiled_op

logger = init_logger(__name__)

_OP = "decode_gemm"
# The most rows that the kernel serves where it is not slower than cuBLAS (1 GPU
# microbench, benchmarks/kernels/benchmark_muse_spark_decode_gemm.py): above 64 rows the
# tensor-core main loop of the kernel is the limit and cuBLAS wins.
MAX_ROWS = 64
# The kernel splits K over at most 8 CTAs and takes tiles of at most 64 rows here.
_MAX_SPLIT = 8
# fp32 elements of the scratch buffer of a device: N * split * rows for the widest
# weight (N <= 8192 at 8 splits of 64 rows). Allocated once, while the model is built
# and before any CUDA graph is captured (a capture bakes the address in).
_SCRATCH_FLOATS = 4 << 20
_scratch: dict[int, torch.Tensor] = {}


def _index(device: torch.device) -> int:
    """The index of a CUDA device (``cuda`` is the current one)."""
    return device.index if device.index is not None else torch.cuda.current_device()


def _op() -> Callable | None:
    return resolve_compiled_op(_OP)


def supports_weight(weight: torch.Tensor) -> bool:
    """Whether the kernel can take this weight, and the op is available."""
    if _op() is None:
        return False
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        return False
    n, k = weight.shape
    return (
        weight.stride(1) == 1
        and weight.stride(0) == k
        and weight.data_ptr() % 16 == 0
        and n % 64 == 0
        and n * _MAX_SPLIT * MAX_ROWS <= _SCRATCH_FLOATS
        and k % (_MAX_SPLIT * 64) == 0
    )


def available() -> bool:
    """Whether the ``_muse_C`` extension has the op; a warning when it does not."""
    if _op() is None:
        logger.warning_once(
            "VLLM_MUSE_SPARK_SM120_DECODE_GEMM is not applied: the _muse_C extension "
            "is unavailable or has no decode_gemm. cuBLAS runs the projections."
        )
        return False
    return True


def reserve_scratch(device: torch.device | None = None) -> None:
    """Allocate the scratch buffer of ``device`` (default: the current CUDA device)."""
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda" and _index(device) not in _scratch:
        _scratch[_index(device)] = torch.empty(
            _SCRATCH_FLOATS, dtype=torch.float32, device=device
        )


def decode_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for ``1 <= rows <= MAX_ROWS`` (the caller checks ``usable``)."""
    op = _op()
    assert op is not None
    logger.info_once(
        "MuseSpark SM120: the QKV projection and the sharded pooler / unpooler of up "
        "to %d rows run as _muse_C::decode_gemm (VLLM_MUSE_SPARK_SM120_DECODE_GEMM=1): "
        "one split-K kernel instead of the cuBLAS GEMM and its reduction kernel.",
        MAX_ROWS,
    )
    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    op(x, weight, out, _scratch[_index(x.device)])
    return out


def usable(x: torch.Tensor) -> bool:
    """Whether an activation can go through ``decode_gemm`` (the weight was checked)."""
    return (
        x.dtype == torch.bfloat16
        and x.dim() == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and _index(x.device) in _scratch
    )


def wrap_projection(gemm: Callable) -> Callable:
    """The ``(layer, x, weight, bias)`` GEMM of the sharded projections, with
    ``decode_gemm`` for the batches it serves and ``gemm`` for the others.
    """
    if not available():
        return gemm
    reserve_scratch()

    def projection(
        layer: torch.nn.Module,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is None and usable(x) and supports_weight(weight):
            return decode_gemm(x, weight)
        return gemm(layer, x, weight, bias)

    return projection
