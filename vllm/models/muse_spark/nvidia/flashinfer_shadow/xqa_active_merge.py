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

Spec-dec modules can also be built with ``one_wave``
(``VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE``, off by default): the verify launch of 5 to
8 query tokens per request (8 query heads per KV head, so 2 blocks of 32 query rows per
KV head) is then sized to one wave of CTAs. The stock launcher divides the SMs by the
batch and the KV heads only, ignores the 2 blocks, and launches twice as many CTAs as
SMs (376 at a batch of 1; the kernel runs 1 CTA per SM, so that is two waves, each with
the CTA start-up and the merge of twice as many partial results). One wave halves the
split count (47 instead of 94 CTAs per block at a batch of 1). It applies where that
grid fills at least 90% of the SMs (batch 1 to 5 on 188 SMs; at larger batches the
rounding down leaves SMs idle and a global layer at long context is slower than with the
stock two waves) and, for a sliding-window layer (9 to 10 tiles per sequence), at every
batch. The split count sets the order in which the partial results are merged: the
output is no longer bitwise the stock one (rounding level, deterministic, the same on
every rank) where a sequence has more tiles than the new split count. Where it has not,
or where the stock split count is kept, the output is still bitwise the stock one.

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
# The spec-dec modules built with the one-wave split count have a name (and a build) of
# their own: the compile flag is not part of the digest of the sources.
_ONE_WAVE_TAG = "activemerge_onewave"
_ONE_WAVE_FLAG = "-DMUSE_XQA_SPEC_ONE_WAVE=1"
_SPEC_DEC_FLAG = "-DSPEC_DEC=1"
_SHADOW_DIR = Path(__file__).resolve().parent / "xqa"
_MANIFEST = _SHADOW_DIR / "stock.md5"
_SHADOW_SOURCE = _SHADOW_DIR / "mha.cu"
_PATCHED_ATTR = "_muse_active_merge"
_ONE_WAVE_ATTR = "_muse_one_wave"
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


def _derive(stock_spec: Any, one_wave: bool = False) -> Any | None:
    """The shadow spec; ``one_wave`` only changes the spec-dec modules (q_len > 1)."""
    from flashinfer.jit import env as jit_env

    xqa_dir = Path(jit_env.FLASHINFER_CSRC_DIR) / "xqa"
    one_wave = one_wave and _SPEC_DEC_FLAG in (stock_spec.extra_cuda_cflags or ())
    return shadow.derive_spec(
        stock_spec,
        tag=_ONE_WAVE_TAG if one_wave else _TAG,
        replace={xqa_dir / "mha.cu": _SHADOW_SOURCE},
        include_dirs=[xqa_dir],
        cuda_cflags=[_ONE_WAVE_FLAG] if one_wave else (),
    )


def _gen_xqa_module(
    strict: bool, *args: Any, one_wave: bool = False, **kwargs: Any
) -> Any:
    stock_spec = _STOCK_GEN(*args, **kwargs)
    reason = unavailable_reason()
    shadow_spec = None if reason else _derive(stock_spec, one_wave)
    if shadow_spec is None:
        if strict:
            raise RuntimeError(f"no shadow XQA module: {reason or 'no mha.cu source'}")
        return stock_spec
    return shadow.build_or_stock(shadow_spec, stock_spec, strict=strict)


def install(one_wave: bool = False) -> bool:
    """Build the XQA modules that follow as shadow modules. True when installed.

    ``one_wave`` builds the spec-dec modules with the one-wave split count (see the
    module docstring). Has to run before the first XQA call of the process: FlashInfer
    caches the module of a configuration, and one that exists already stays stock.
    """
    installed = fi_xqa.gen_xqa_module
    if getattr(installed, _PATCHED_ATTR, False):
        if getattr(installed, _ONE_WAVE_ATTR, False) != one_wave:
            logger.warning(
                "The XQA shadow modules are installed already (one_wave=%s); "
                "one_wave=%s is not applied.",
                getattr(installed, _ONE_WAVE_ATTR, False),
                one_wave,
            )
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
        return _gen_xqa_module(False, *args, one_wave=one_wave, **kwargs)

    setattr(gen_xqa_module, _PATCHED_ATTR, True)
    setattr(gen_xqa_module, _ONE_WAVE_ATTR, one_wave)
    fi_xqa.gen_xqa_module = gen_xqa_module
    logger.info_once(
        "MuseSpark SM120: XQA decode merges only the sub-sequences with work "
        "(VLLM_MUSE_SPARK_SM120_XQA_ACTIVE_MERGE=1, bitwise the stock output); the "
        "shadow modules build at their first use."
    )
    if one_wave:
        logger.info_once(
            "MuseSpark SM120: the XQA spec-dec verify launch of 5 to 8 query tokens is "
            "sized to one wave of CTAs (VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE=1, "
            "rounding-level: the partial results merge in another order)."
        )
    return True


def shadow_xqa_module(*module_args: Any, one_wave: bool = False) -> Any:
    """The shadow module of a configuration, built even where ``install`` did not run.

    ``module_args`` are the arguments of ``flashinfer.xqa._get_xqa_module_cached``
    (input dtype, KV dtype, page size, head dim, head group ratio, sliding window,
    output dtype, q_seq_len, ragged Q). The module has its own name and torch op, so
    it works next to the stock module of the same configuration. Raises when it
    cannot be the shadow, and in a process that called ``install`` (the module of a
    configuration that ``install`` built has the same name, and FlashInfer would
    register its torch op a second time). ``one_wave`` is the one-wave variant of a
    spec-dec module (a module of its own, next to the plain shadow module).
    """
    if getattr(fi_xqa.gen_xqa_module, _PATCHED_ATTR, False):
        raise RuntimeError(
            "shadow_xqa_module cannot be used after install(): the installed hook "
            "serves the shadow modules through flashinfer.xqa itself"
        )
    with _swap_lock:
        module = _shadow_modules.get((module_args, one_wave))
        if module is None:
            installed_gen = fi_xqa.gen_xqa_module
            fi_xqa.gen_xqa_module = functools.partial(
                _gen_xqa_module, True, one_wave=one_wave
            )
            try:
                module = _STOCK_FACTORY(*module_args)
            finally:
                fi_xqa.gen_xqa_module = installed_gen
            _shadow_modules[(module_args, one_wave)] = module
        return module


@contextlib.contextmanager
def use_shadow_modules(one_wave: bool = False) -> Iterator[None]:
    """Inside, ``flashinfer.xqa`` runs the shadow module of every configuration.

    For tests and benchmarks that compare the shadow with the stock module in one
    process (outside, the stock modules run again). ``one_wave`` selects the one-wave
    variant of the spec-dec modules.
    """
    stock_factory = fi_xqa._get_xqa_module_cached
    fi_xqa._get_xqa_module_cached = functools.partial(
        shadow_xqa_module, one_wave=one_wave
    )
    try:
        yield
    finally:
        fi_xqa._get_xqa_module_cached = stock_factory
