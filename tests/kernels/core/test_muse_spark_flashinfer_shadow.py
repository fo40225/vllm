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
import subprocess
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
    fail: BaseException | None = None
    built: int = 0
    extra_cuda_cflags: list | None = None

    def build_and_load(self):
        self.built += 1
        if self.fail is not None:
            raise self.fail
        return object()


def _ninja_failure(output: str | None, returncode: int = 1) -> RuntimeError:
    """What FlashInfer's ``run_ninja`` raises when ninja exits with an error."""
    try:
        raise subprocess.CalledProcessError(returncode, "ninja", output=output)
    except subprocess.CalledProcessError as cause:
        try:
            raise RuntimeError("Ninja build failed.") from cause
        except RuntimeError as exc:
            return exc


NVCC_ERROR = _ninja_failure("mha.cu(10): error: identifier 'foo' is undefined")


@pytest.fixture(autouse=True)
def _toolchain(monkeypatch):
    monkeypatch.setattr(shadow, "toolchain", lambda: "CUDA 13.3 at /usr/local/cuda ")


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


def test_derive_spec_adds_compile_flags_after_the_stock_ones(monkeypatch, tmp_path):
    _workspace(monkeypatch, tmp_path)
    stock_src = Path("/stock/csrc/xqa/mha.cu")
    shadow_src = tmp_path / "mha.cu"
    shadow_src.write_text("// shadow\n")
    replace = {stock_src: shadow_src}
    stock = FakeSpec("xqa_a", [stock_src], None, tmp_path, extra_cuda_cflags=["-DA=1"])
    plain = shadow.derive_spec(stock, tag="t", replace=replace)
    assert plain.extra_cuda_cflags == ["-DA=1"]
    flagged = shadow.derive_spec(stock, tag="u", replace=replace, cuda_cflags=["-DB=2"])
    assert flagged.extra_cuda_cflags == ["-DA=1", "-DB=2"]
    assert stock.extra_cuda_cflags == ["-DA=1"]
    # The flags are not in the digest of the sources: the tag keeps the names apart.
    assert flagged.name != plain.name
    # A spec without flags of its own.
    bare = FakeSpec("xqa_b", [stock_src], None, tmp_path)
    flagged = shadow.derive_spec(bare, tag="u", replace=replace, cuda_cflags=["-DB=2"])
    assert flagged.extra_cuda_cflags == ["-DB=2"]


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


def test_compiler_failure_falls_back_to_stock_and_is_remembered(tmp_path):
    stock = FakeSpec("stock", [], None, tmp_path / "stock")
    bad = FakeSpec("bad", [], None, tmp_path / "bad", fail=NVCC_ERROR)
    assert shadow.build_or_stock(bad, stock) is stock
    marker = bad.build_dir / shadow.FAILED_MARKER
    assert bad.built == 1 and marker.is_file()
    assert marker.read_text().startswith("toolchain: CUDA 13.3 at /usr/local/cuda \n")
    # The next process with this toolchain does not compile again.
    assert shadow.build_or_stock(bad, stock) is stock and bad.built == 1
    with pytest.raises(RuntimeError, match="earlier build"):
        shadow.build_or_stock(bad, stock, strict=True)
    fresh = FakeSpec("fresh", [], None, tmp_path / "fresh", fail=NVCC_ERROR)
    with pytest.raises(RuntimeError, match="Ninja build failed"):
        shadow.build_or_stock(fresh, stock, strict=True)
    assert not (fresh.build_dir / shadow.FAILED_MARKER).exists()


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(OSError(28, "No space left on device"), id="oserror"),
        pytest.param(RuntimeError("Could not find nvcc"), id="no-nvcc"),
        pytest.param(TimeoutError("lock"), id="lock-timeout"),
        pytest.param(_ninja_failure(None), id="verbose-ninja-has-no-output"),
        pytest.param(_ninja_failure("error: x", returncode=-9), id="ninja-killed"),
        pytest.param(
            _ninja_failure("nvcc error : 'cicc' died due to signal 9"), id="oom-killed"
        ),
        pytest.param(
            _ninja_failure("fatal error: cannot write: No space left on device"),
            id="disk-full",
        ),
        pytest.param(_ninja_failure("ninja: build stopped"), id="no-diagnostic"),
    ],
)
def test_a_failure_of_the_machine_is_not_remembered(tmp_path, failure):
    """A helper process without nvcc or a full disk must not switch production off."""
    stock = FakeSpec("stock", [], None, tmp_path / "stock")
    flaky = FakeSpec("flaky", [], None, tmp_path / "flaky", fail=failure)
    assert shadow.build_or_stock(flaky, stock) is stock
    assert not (flaky.build_dir / shadow.FAILED_MARKER).exists()
    # The next process tries again and gets the shadow.
    flaky.fail = None
    assert shadow.build_or_stock(flaky, stock) is flaky and flaky.built == 2


def test_marker_of_another_toolchain_is_ignored_and_removed(monkeypatch, tmp_path):
    stock = FakeSpec("stock", [], None, tmp_path / "stock")
    spec = FakeSpec("spec", [], None, tmp_path / "spec", fail=NVCC_ERROR)
    assert shadow.build_or_stock(spec, stock) is stock
    marker = spec.build_dir / shadow.FAILED_MARKER
    monkeypatch.setattr(shadow, "toolchain", lambda: "CUDA 13.4 at /opt/cuda ")
    spec.fail = None  # the other toolchain compiles it
    assert shadow.build_or_stock(spec, stock) is spec and spec.built == 2
    assert not marker.exists()
    # Without a known toolchain nothing is remembered or honored.
    monkeypatch.setattr(shadow, "toolchain", lambda: None)
    spec.fail = NVCC_ERROR
    assert shadow.build_or_stock(spec, stock) is stock and not marker.exists()
    marker.write_text("toolchain: None\n")
    assert shadow.build_or_stock(spec, stock) is stock and spec.built == 4


def test_good_build_keeps_the_shadow(tmp_path):
    stock = FakeSpec("stock", [], None, tmp_path / "stock")
    good = FakeSpec("good", [], None, tmp_path / "good")
    assert shadow.build_or_stock(good, stock) is good and good.built == 1
