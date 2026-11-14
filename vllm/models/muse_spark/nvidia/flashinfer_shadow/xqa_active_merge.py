# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XQA decode (``kernel_mha``) that merges only the sub-sequences with work (SM120).

FlashInfer's ``xqa`` derives the maximum sequence length from the width of the page
table (``page_table.shape[-1] * page_size``, the ``max_seq_len`` argument is ignored),
and vLLM's block table is as wide as ``--max-model-len``. At a batch of 1 the kernel
then launches 94 CTAs per KV head (one per SM, at most one per 256 tokens of that
length) whatever the context is: a 512 token context fills 2 tiles, and the last CTA
that finishes still reads and merges all 94 partial results, 13.4 us per layer
(about 0.4 ms of a 9.2 ms decode step) where the tiles take 7 us. The shadow
``xqa/mha.cu`` computes on the device, from the sequence length, how many
sub-sequences have a tile after the sliding-window skip: the others return at once
and the semaphore and the merge loops run over the active ones only. The grid is not
changed (a CUDA graph cannot change it), the output is bitwise the one of the stock
kernel (the partials that are left out are exact zeros).

``install`` routes ``flashinfer.xqa.gen_xqa_module`` through the shadow for every XQA
module that the process builds afterwards (the model calls it when
``VLLM_MUSE_SPARK_SM120_XQA_ACTIVE_MERGE`` is on). ``shadow_xqa_module`` gives the
shadow module of one configuration next to the stock one (tests and benchmarks, in a
process that did not call ``install``).
"""

import contextlib
import functools
import importlib
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from vllm.logger import init_logger
from vllm.models.muse_spark.nvidia import flashinfer_shadow as shadow

logger = init_logger(__name__)

_TAG = "activemerge"
_SHADOW_DIR = Path(__file__).resolve().parent / "xqa"
_MANIFEST = _SHADOW_DIR / "stock.md5"
_SHADOW_SOURCE = _SHADOW_DIR / "mha.cu"
_PATCHED_ATTR = "_muse_active_merge"
# ``import flashinfer.xqa`` would bind the function that the package re-exports.
fi_xqa = importlib.import_module("flashinfer.xqa")
# What flashinfer.xqa.get_xqa_module builds from, taken before anything swaps
# ``gen_xqa_module``: the stock spec generator and the module factory behind the cache.
_STOCK_GEN = fi_xqa.gen_xqa_module
_STOCK_FACTORY = fi_xqa._get_xqa_module_cached.__wrapped__
_swap_lock = threading.Lock()
_shadow_modules: dict[tuple, Any] = {}


@functools.cache
def unavailable_reason() -> str | None:
    """Why the shadow cannot be used with this FlashInfer and device, None if it can."""
    from flashinfer.compilation_context import CompilationContext

    # The SM90 build adds mha_sm90.cu (its own kernel) to the module.
    if any(major != 12 for major, _ in CompilationContext().TARGET_CUDA_ARCHS):
        return "the JIT targets other architectures than SM12x"
    return shadow.stock_mismatch(_MANIFEST)


def _derive(stock_spec: Any) -> Any | None:
    from flashinfer.jit import env as jit_env

    xqa_dir = Path(jit_env.FLASHINFER_CSRC_DIR) / "xqa"
    return shadow.derive_spec(
        stock_spec,
        tag=_TAG,
        replace={xqa_dir / "mha.cu": _SHADOW_SOURCE},
        include_dirs=[xqa_dir],
    )


def _gen_xqa_module(strict: bool, *args: Any, **kwargs: Any) -> Any:
    stock_spec = _STOCK_GEN(*args, **kwargs)
    reason = unavailable_reason()
    shadow_spec = None if reason else _derive(stock_spec)
    if shadow_spec is None:
        if strict:
            raise RuntimeError(f"no shadow XQA module: {reason or 'no mha.cu source'}")
        return stock_spec
    return shadow.build_or_stock(shadow_spec, stock_spec, strict=strict)


def install() -> bool:
    """Build the XQA modules that follow as shadow modules. True when installed.

    Has to run before the first XQA call of the process: FlashInfer caches the
    module of a configuration, and one that exists already stays stock.
    """
    if getattr(fi_xqa.gen_xqa_module, _PATCHED_ATTR, False):
        return True
    reason = unavailable_reason()
    if reason is not None:
        logger.warning(
            "VLLM_MUSE_SPARK_SM120_XQA_ACTIVE_MERGE is not applied: %s. The stock "
            "XQA kernel runs.",
            reason,
        )
        return False
    if fi_xqa._get_xqa_module_cached.cache_info().currsize:
        logger.warning(
            "XQA modules were built before the active merge was installed; they "
            "stay stock."
        )

    @functools.wraps(_STOCK_GEN)
    def gen_xqa_module(*args: Any, **kwargs: Any) -> Any:
        return _gen_xqa_module(False, *args, **kwargs)

    setattr(gen_xqa_module, _PATCHED_ATTR, True)
    fi_xqa.gen_xqa_module = gen_xqa_module
    logger.info_once(
        "MuseSpark SM120: XQA decode merges only the sub-sequences with work "
        "(VLLM_MUSE_SPARK_SM120_XQA_ACTIVE_MERGE=1, bitwise the stock output); the "
        "shadow modules build at their first use."
    )
    return True


def shadow_xqa_module(*module_args: Any) -> Any:
    """The shadow module of a configuration, built even where ``install`` did not run.

    ``module_args`` are the arguments of ``flashinfer.xqa._get_xqa_module_cached``
    (input dtype, KV dtype, page size, head dim, head group ratio, sliding window,
    output dtype, q_seq_len, ragged Q). The module has its own name and torch op, so
    it works next to the stock module of the same configuration. Raises when it
    cannot be the shadow, and in a process that called ``install`` (the module of a
    configuration that ``install`` built has the same name, and FlashInfer would
    register its torch op a second time).
    """
    if getattr(fi_xqa.gen_xqa_module, _PATCHED_ATTR, False):
        raise RuntimeError(
            "shadow_xqa_module cannot be used after install(): the installed hook "
            "serves the shadow modules through flashinfer.xqa itself"
        )
    with _swap_lock:
        module = _shadow_modules.get(module_args)
        if module is None:
            installed_gen = fi_xqa.gen_xqa_module
            fi_xqa.gen_xqa_module = functools.partial(_gen_xqa_module, True)
            try:
                module = _STOCK_FACTORY(*module_args)
            finally:
                fi_xqa.gen_xqa_module = installed_gen
            _shadow_modules[module_args] = module
        return module


@contextlib.contextmanager
def use_shadow_modules() -> Iterator[None]:
    """Inside, ``flashinfer.xqa`` runs the shadow module of every configuration.

    For tests and benchmarks that compare the shadow with the stock module in one
    process (outside, the stock modules run again).
    """
    stock_factory = fi_xqa._get_xqa_module_cached
    fi_xqa._get_xqa_module_cached = shadow_xqa_module
    try:
        yield
    finally:
        fi_xqa._get_xqa_module_cached = stock_factory
