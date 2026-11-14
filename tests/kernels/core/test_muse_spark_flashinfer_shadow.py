# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The shadow FlashInfer build mechanism (``muse_spark/nvidia/flashinfer_shadow``).

Covers what keeps a shadow honest without compiling or running a kernel: the md5
manifest against the installed FlashInfer, the module name and sources of a derived
spec, the content-addressed copy that is compiled, and the fallback to the stock
module when the shadow does not build.
"""

import dataclasses
import hashlib
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Requires a GPU (FlashInfer detects the JIT arch)", allow_module_level=True
    )

flashinfer = pytest.importorskip("flashinfer")

from vllm.models.muse_spark.nvidia import flashinfer_shadow as shadow  # noqa: E402


@dataclasses.dataclass
class FakeSpec:
    """The fields of a ``JitSpecNvcc`` that the mechanism reads."""

    name: str
    sources: list
    extra_include_dirs: list | None
    build_dir: Path
    fail: bool = False
    built: int = 0

    def build_and_load(self):
        self.built += 1
        if self.fail:
            raise RuntimeError("nvcc failed")
        return object()


def _workspace(monkeypatch, tmp_path):
    from flashinfer.jit import env as jit_env

    monkeypatch.setattr(jit_env, "FLASHINFER_WORKSPACE_DIR", tmp_path / "ws")


def _manifest(tmp_path, rel: str, digest: str, name: str = "stock.md5") -> Path:
    path = tmp_path / name
    path.write_text(f"{digest}  {rel}\n")
    return path


def test_manifest_matches_the_installed_files(monkeypatch, tmp_path):
    data_dir = shadow.flashinfer_data_dir()
    rel = "csrc/xqa/mha.cu"
    digest = hashlib.md5((data_dir / rel).read_bytes()).hexdigest()
    monkeypatch.setattr(shadow, "STOCK_FLASHINFER_VERSION", flashinfer.__version__)
    assert shadow.read_manifest(_manifest(tmp_path, rel, digest)) == {rel: digest}
    assert shadow.stock_mismatch(_manifest(tmp_path, rel, digest, "ok.md5")) is None


def test_changed_missing_or_other_version_is_detected(monkeypatch, tmp_path):
    data_dir = shadow.flashinfer_data_dir()
    rel = "csrc/xqa/mha.cu"
    digest = hashlib.md5((data_dir / rel).read_bytes()).hexdigest()
    monkeypatch.setattr(shadow, "STOCK_FLASHINFER_VERSION", flashinfer.__version__)
    reason = shadow.stock_mismatch(_manifest(tmp_path, rel, "0" * 32, "changed.md5"))
    assert reason is not None and str(data_dir / rel) in reason
    reason = shadow.stock_mismatch(
        _manifest(tmp_path, "csrc/xqa/no_such_file.cu", digest, "missing.md5")
    )
    assert reason is not None and "missing" in reason
    monkeypatch.setattr(shadow, "STOCK_FLASHINFER_VERSION", "0.0.1")
    reason = shadow.stock_mismatch(_manifest(tmp_path, rel, digest, "version.md5"))
    assert reason is not None and "0.0.1" in reason


def test_derive_spec_renames_and_replaces_only_the_shadowed_source(
    monkeypatch, tmp_path
):
    _workspace(monkeypatch, tmp_path)
    stock_src = Path("/stock/csrc/xqa/mha.cu")
    other_src = Path("/stock/csrc/xqa/xqa_wrapper.cu")
    shadow_src = tmp_path / "mha.cu"
    shadow_src.write_text("// shadow\n")
    stock = FakeSpec("xqa_a", [stock_src, other_src], [Path("/inc")], tmp_path)
    spec = shadow.derive_spec(
        stock, tag="t", replace={stock_src: shadow_src}, include_dirs=[Path("/xqa")]
    )
    digest = shadow.source_digest([shadow_src])
    assert spec.name == f"xqa_a_muse_t_{digest}"
    assert spec.sources[1] == other_src
    # A copy at a path set by its content, not the shadow file itself.
    assert spec.sources[0] != shadow_src
    assert spec.sources[0].parent.name == digest
    assert spec.sources[0].read_text() == "// shadow\n"
    assert spec.extra_include_dirs == [Path("/xqa"), Path("/inc")]
    assert stock.name == "xqa_a" and stock.sources[0] == stock_src
    # Nothing to replace: no shadow.
    unrelated = FakeSpec("other", [other_src], None, tmp_path)
    replace = {stock_src: shadow_src}
    assert shadow.derive_spec(unrelated, tag="t", replace=replace) is None


def test_materialize_is_content_addressed_and_idempotent(monkeypatch, tmp_path):
    _workspace(monkeypatch, tmp_path)
    src = tmp_path / "a.cu"
    src.write_text("one")
    first = shadow.materialize(src)
    mtime = first.stat().st_mtime_ns
    assert shadow.materialize(src) == first and first.stat().st_mtime_ns == mtime
    src.write_text("two")
    second = shadow.materialize(src)
    assert second != first
    assert second.read_text() == "two" and first.read_text() == "one"


def test_failed_build_falls_back_to_stock_and_is_remembered(tmp_path):
    stock = FakeSpec("stock", [], None, tmp_path / "stock")
    bad = FakeSpec("bad", [], None, tmp_path / "bad", fail=True)
    assert shadow.build_or_stock(bad, stock) is stock
    assert bad.built == 1 and (bad.build_dir / shadow.FAILED_MARKER).is_file()
    # The next process does not compile again.
    assert shadow.build_or_stock(bad, stock) is stock and bad.built == 1
    with pytest.raises(RuntimeError, match="earlier build"):
        shadow.build_or_stock(bad, stock, strict=True)
    fresh = FakeSpec("fresh", [], None, tmp_path / "fresh", fail=True)
    with pytest.raises(RuntimeError, match="nvcc failed"):
        shadow.build_or_stock(fresh, stock, strict=True)
    assert not (fresh.build_dir / shadow.FAILED_MARKER).exists()


def test_good_build_keeps_the_shadow(tmp_path):
    stock = FakeSpec("stock", [], None, tmp_path / "stock")
    good = FakeSpec("good", [], None, tmp_path / "good")
    assert shadow.build_or_stock(good, stock) is good and good.built == 1
