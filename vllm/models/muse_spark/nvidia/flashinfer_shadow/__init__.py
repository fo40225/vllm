# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Branch-owned shadow builds of FlashInfer JIT modules (SM120).

The FlashInfer that the runtime images ship is never edited, but a kernel bug or a
missing optimization in one of its JIT modules can be fixed in the branch: a shadow
is a patched copy of ONE source file of the installed package (a directory per
FlashInfer module, e.g. ``xqa/mha.cu``), compiled through FlashInfer's own JIT with
the flags of the module that it replaces. Nothing else of the module changes, so the
Python side of FlashInfer keeps working unchanged.

How a shadow is built:

* The stock ``JitSpecNvcc`` is generated first (``derive_spec``), so the compile flags,
  the arch list and the other sources stay what FlashInfer chose for this device.
  Only the replaced source and the module name differ. The name carries a digest of
  the shadow sources (``<stock name>_muse_<tag>_<digest>``), so the build directory,
  the lock and the op name never collide with the stock module, and a stock and a
  shadow module can live in one process. The source is compiled from a copy in the
  FlashInfer workspace at a path set by its digest (``materialize``), so that the
  source trees of the branch do not rebuild each other's module.
* The headers next to the replaced file come from the installed package (its
  directory is put on the include path), which is only sound while they are the bytes
  that the shadow was derived from: every shadow ships an md5 manifest of the stock
  files it depends on (``stock.md5``, md5sum format, paths relative to
  ``flashinfer/data``) and is skipped, with a warning, when the installed FlashInfer
  is another version or one of the files differs.
* The module is compiled at the first use, like the stock one, and cached in the
  FlashInfer workspace. When that build fails the stock spec is returned instead, and
  the failure is remembered in the build directory (delete ``FAILED_MARKER`` there to
  try again), so that the next process does not compile again.

To add a shadow of another module (for example the FA2 ``prefill.cuh``): copy the
file from the installed package, record the md5 of every stock file it needs, patch,
and write a hook like ``xqa_active_merge.py`` that wraps the ``gen_*_module`` of that
module. ``include_dirs`` puts a directory in front of the stock include path.
"""

import dataclasses
import functools
import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# The FlashInfer version every manifest in this package was taken from.
STOCK_FLASHINFER_VERSION = "0.7.0.post1"
# Left in the build directory of a shadow whose compilation failed.
FAILED_MARKER = ".muse_shadow_failed"


def flashinfer_data_dir() -> Path:
    """``flashinfer/data`` of the installed package (csrc, include, ...)."""
    from flashinfer.jit import env as jit_env

    return Path(jit_env.FLASHINFER_DATA)


def read_manifest(path: Path) -> dict[str, str]:
    """An md5sum listing: ``{path relative to flashinfer/data: md5}``."""
    manifest: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            manifest[rel.strip()] = digest
    return manifest


@functools.cache
def stock_mismatch(manifest_path: Path) -> str | None:
    """Why the installed FlashInfer is not the one of the manifest, None if it is."""
    import flashinfer

    if flashinfer.__version__ != STOCK_FLASHINFER_VERSION:
        return (
            f"FlashInfer {flashinfer.__version__} is installed, the shadow was "
            f"derived from {STOCK_FLASHINFER_VERSION}"
        )
    data_dir = flashinfer_data_dir()
    for rel, digest in read_manifest(manifest_path).items():
        stock = data_dir / rel
        if not stock.is_file():
            return f"{stock} is missing"
        if hashlib.md5(stock.read_bytes()).hexdigest() != digest:
            return f"{stock} differs from FlashInfer {STOCK_FLASHINFER_VERSION}"
    return None


def source_digest(sources: Sequence[Path]) -> str:
    """Short digest of the shadow sources, the unique part of the module name."""
    md5 = hashlib.md5()
    for source in sources:
        md5.update(source.read_bytes())
    return md5.hexdigest()[:8]


def materialize(source: Path) -> Path:
    """A copy of ``source`` in the FlashInfer workspace, at a path set by its content.

    The build then does not depend on which source tree of the branch runs: ninja
    recompiles a module when the path of a source changes, and the dev trees and the
    deployed one would otherwise rebuild each other's module.
    """
    from flashinfer.jit import env as jit_env

    data = source.read_bytes()
    dest = (
        Path(jit_env.FLASHINFER_WORKSPACE_DIR)
        / "muse_shadow"
        / hashlib.md5(data).hexdigest()[:8]
        / source.name
    )
    if not dest.is_file() or dest.read_bytes() != data:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    return dest


def derive_spec(
    stock_spec: Any,
    *,
    tag: str,
    replace: Mapping[Path, Path],
    include_dirs: Sequence[Path] = (),
) -> Any | None:
    """The shadow of ``stock_spec`` (a ``JitSpecNvcc``) with sources replaced.

    ``replace`` maps a stock source path to its shadow (compiled from a copy in the
    FlashInfer workspace, see ``materialize``); None when the spec compiles none of
    them. ``include_dirs`` go in front of the include directories of the stock spec.
    """
    sources = [Path(s) for s in stock_spec.sources]
    shadows = {s: replace[s] for s in sources if s in replace}
    if not shadows:
        return None
    digest = source_digest(list(shadows.values()))
    return dataclasses.replace(
        stock_spec,
        name=f"{stock_spec.name}_muse_{tag}_{digest}",
        sources=[materialize(shadows[s]) if s in shadows else s for s in sources],
        extra_include_dirs=[*include_dirs, *(stock_spec.extra_include_dirs or ())],
    )


def build_or_stock(shadow_spec: Any, stock_spec: Any, *, strict: bool = False) -> Any:
    """Compile and load ``shadow_spec`` now, so that a failure can fall back.

    Returns ``shadow_spec`` when its module builds (the caller's own
    ``build_and_load`` is then a ninja no-op) and ``stock_spec`` when it does not.
    ``strict`` raises instead (tests and benchmarks must not compare the stock
    module with itself).
    """
    marker = shadow_spec.build_dir / FAILED_MARKER
    if marker.exists():
        if strict:
            raise RuntimeError(f"an earlier build of {shadow_spec.name} failed")
        logger.warning_once(
            "Shadow FlashInfer module %s failed to build earlier (%s); using the "
            "stock module.",
            shadow_spec.name,
            marker,
        )
        return stock_spec
    try:
        shadow_spec.build_and_load()
    except Exception as exc:
        if strict:
            raise
        logger.warning(
            "Shadow FlashInfer module %s failed to build (%s: %s); using the stock "
            "module.",
            shadow_spec.name,
            type(exc).__name__,
            exc,
        )
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{type(exc).__name__}: {exc}\n")
        except OSError:
            pass
        return stock_spec
    return shadow_spec
