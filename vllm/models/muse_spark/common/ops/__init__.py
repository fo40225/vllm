# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-platform MuseSpark1x operators."""

import functools
from collections.abc import Callable

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def muse_pdl_enabled() -> bool:
    """Whether the Muse Spark kernels launch as programmatic dependents.

    Needs a GPU that supports PDL (SM90+) and ``VLLM_MUSE_SPARK_SM120_PDL``. A
    kernel launched this way may start before its predecessor has finished, so
    it must call ``gdc_wait`` before it reads anything a kernel of the step
    writes (or writes anything such a kernel reads).
    """
    from vllm.platforms import current_platform

    return (
        envs.VLLM_MUSE_SPARK_SM120_PDL
        and current_platform.is_cuda()
        and current_platform.is_arch_support_pdl()
    )


def has_compiled_op(name: str) -> bool:
    """Whether the compiled ``torch.ops._C`` extension registers op ``name``.

    Prebuilt wheels older than the Muse Spark kernels (vllm-project/vllm#62132)
    lack these ops, so callers must keep a pure-PyTorch or unfused fallback.
    """
    return hasattr(torch.ops._C, name)


def _is_sm120() -> bool:
    # Imported here: common/ is shared with ROCm.
    from vllm.platforms import current_platform

    return current_platform.is_cuda() and current_platform.is_device_capability_family(
        120
    )


def _jit_op(name: str) -> Callable | None:
    """The op ``name`` of the JIT-built ``_muse_C`` extension, if it can serve."""
    if not _is_sm120():
        return None
    # Imported here: the loader is NVIDIA-only and common/ is shared with ROCm.
    try:
        from vllm.models.muse_spark.nvidia.ops import muse_c
    except ImportError:
        return None
    if (reason := muse_c.preflight()) is not None:
        logger.info_once(
            "Muse Spark: the _muse_C extension is unavailable (%s).", reason
        )
        return None
    # Builds on the first use of a cache dir, so call this from layer
    # construction (never from a forward pass or a CUDA graph capture).
    try:
        muse_c.load()
    except Exception as e:
        logger.warning_once(
            "Muse Spark: the _muse_C extension could not be built (%s: %s).",
            type(e).__name__,
            e,
        )
        return None
    return getattr(torch.ops._muse_C, name, None)


@functools.cache
def resolve_compiled_op(name: str) -> Callable | None:
    """The fused kernel ``name`` for this process, or None for the fallback.

    Decided once per process: on SM120 the JIT ``torch.ops._muse_C`` variant
    when the extension builds or loads (never ``torch.ops._C``, see below),
    else None; on every other device ``torch.ops._C`` if the build has the op,
    else None. ``VLLM_MUSE_SPARK_FUSED_KERNELS=0`` forces None. The result must
    not change between calls, so that it cannot differ between a CUDA graph
    capture and its replay.
    """
    if not envs.VLLM_MUSE_SPARK_FUSED_KERNELS:
        logger.info(
            "Muse Spark %s: using the unfused fallback "
            "(VLLM_MUSE_SPARK_FUSED_KERNELS=0).",
            name,
        )
        return None
    if _is_sm120():
        # SM120 always runs our JIT variant, even when torch.ops._C ships the
        # upstream kernel of the same name: the two differ in behavior (dense Q,
        # device-derived group width) and only ours is validated on SM120.
        op: Callable | None = _jit_op(name)
        source = "torch.ops._muse_C (JIT-built)"
    elif has_compiled_op(name):
        op = getattr(torch.ops._C, name)
        source = "torch.ops._C"
    else:
        op = None
        source = "none"
    if op is None:
        logger.info("Muse Spark %s: using the unfused fallback.", name)
    else:
        logger.info("Muse Spark %s: using %s.", name, source)
    return op
