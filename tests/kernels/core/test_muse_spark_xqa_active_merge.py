# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for the XQA decode shadow kernel that merges the active CTAs only.

The shadow ``mha.cu`` (``flashinfer_shadow/xqa``) returns early in the CTAs that have
no tile and merges only the partials of the others. It has to be bit-for-bit the
FlashInfer kernel, so every test runs the stock module and the shadow module on the
same inputs, through ``xqa_batch_decode_with_kv_cache`` as the vLLM FlashInfer backend
calls it for the Muse Spark decode (TP8 rank: 16 query and 2 KV heads of 64, pages
of 16 tokens, fp8 KV, a block table as wide as the 131072 token context), and compares
the bits of the outputs. The two modules have their own names and torch ops, so they
run side by side in one process.
"""

import contextlib
import random
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

import vllm.envs as envs
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

pytest.importorskip("flashinfer")

from flashinfer.decode import xqa_batch_decode_with_kv_cache  # noqa: E402

from vllm.models.muse_spark.nvidia import flashinfer_shadow as shadow  # noqa: E402
from vllm.models.muse_spark.nvidia.flashinfer_shadow import (  # noqa: E402
    xqa_active_merge as am,
)
from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    _make_xqa_draft_block_mask,
    _make_xqa_ragged_draft_block_mask,
)

if (_reason := am.unavailable_reason()) is not None:
    pytest.skip(f"No shadow XQA module: {_reason}", allow_module_level=True)

SHADOW_XQA = Path(am.__file__).resolve().parent / "xqa"
DEVICE = torch.device("cuda")
PAGE = 16
HEAD_DIM = 64
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
MAX_LEN = 131072
TABLE_WIDTH = MAX_LEN // PAGE
# The KV pool is shared by every sequence (block tables alias its pages): the bits
# of the attention do not depend on which page holds which tokens.
POOL_PAGES = 16384
WORKSPACE_BYTES = 394 * 1024 * 1024  # the vLLM FlashInfer workspace
SWA = 2047  # window_left of the 2048 token sliding window
WINDOWS = {"swa": SWA, "full": -1}
SENTINEL = 0.5


def _module(variant: str):
    """Run the XQA calls inside with the stock or the shadow module."""
    return am.use_shadow_modules() if variant == "shadow" else contextlib.nullcontext()


class Rig:
    """Inputs shared by the stock and the shadow call, and their private state."""

    def __init__(self, seed: int = 0):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        self.gen = gen
        pool = torch.randn(
            POOL_PAGES, PAGE, NUM_KV_HEADS, 2 * HEAD_DIM, device=DEVICE, generator=gen
        )
        self.pool = (pool * 0.5).to(torch.float8_e4m3fn).contiguous()
        # kv_cache_permute.split(head_size, dim=-1): K and V are strided halves.
        self.k_cache, self.v_cache = self.pool.split(HEAD_DIM, dim=-1)
        self.workspace = {
            v: torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=DEVICE)
            for v in ("stock", "shadow")
        }

    def query(self, tokens: int) -> torch.Tensor:
        return torch.randn(
            tokens,
            NUM_Q_HEADS,
            HEAD_DIM,
            device=DEVICE,
            dtype=torch.bfloat16,
            generator=self.gen,
        )

    def block_tables(self, batch: int) -> torch.Tensor:
        return torch.randint(
            0,
            POOL_PAGES,
            (batch, TABLE_WIDTH),
            device=DEVICE,
            dtype=torch.int32,
            generator=self.gen,
        )

    def call(
        self,
        variant,
        query,
        out,
        block_tables,
        seq_lens,
        *,
        window_left,
        q_len=1,
        mask=None,
        q_cu_seq_lens=None,
        pdl=True,
        workspace=None,
    ):
        with _module(variant):
            xqa_batch_decode_with_kv_cache(
                query=query,
                kv_cache=(self.k_cache, self.v_cache),
                workspace_buffer=workspace
                if workspace is not None
                else self.workspace[variant],
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=MAX_LEN,
                bmm1_scale=0.125,
                bmm2_scale=1.0,
                window_left=window_left,
                out=out,
                sinks=None,
                kv_layout="NHD",
                enable_pdl=pdl,
                q_len_per_req=q_len,
                mask=mask,
                q_cu_seq_lens=q_cu_seq_lens,
            )


@pytest.fixture(scope="module")
def rig():
    return Rig()


def _mask(q_len: int, batch: int) -> torch.Tensor | None:
    """The causal draft mask of a uniform batch, as the backend builds it."""
    if q_len == 1:
        return None
    per_req = _make_xqa_draft_block_mask(q_len, True, DEVICE)
    return per_req.unsqueeze(0).expand(batch, -1, -1).contiguous()


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16)


def _assert_same(stock, shadow, what: str) -> None:
    """Bitwise equality, NaN payloads included."""
    a, b = _bits(stock), _bits(shadow)
    if not torch.equal(a, b):
        bad = (a != b).nonzero()
        first = tuple(bad[0].tolist())
        raise AssertionError(
            f"{what}: {len(bad)} of {a.numel()} elements differ, first at {first}: "
            f"stock {stock[first].item()} shadow {shadow[first].item()}"
        )


def _run_pair(rig, ctxs, *, window, q_len=1, pdl=True, tables=None, check_data=True):
    """One launch of both modules on the contexts ``ctxs``; asserts equal bits."""
    batch = len(ctxs)
    seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
    tables = rig.block_tables(batch) if tables is None else tables
    query = rig.query(batch * q_len)
    mask = _mask(q_len, batch)
    outs = {}
    for variant in ("stock", "shadow"):
        out = torch.full_like(query, SENTINEL)
        rig.call(
            variant,
            query,
            out,
            tables,
            seq_lens,
            window_left=WINDOWS[window],
            q_len=q_len,
            mask=mask,
            pdl=pdl,
        )
        outs[variant] = out
    torch.cuda.synchronize()
    _assert_same(outs["stock"], outs["shadow"], f"{window} q{q_len} ctx={ctxs}")
    if check_data:
        assert torch.isfinite(outs["shadow"].float()).all()
        assert not (outs["shadow"] == SENTINEL).all()
    return outs["shadow"]


def _sweep_contexts(q_len: int) -> list[int]:
    ctxs = set(range(max(1, q_len), 700))
    # The tile (256 tokens) and the page (16 tokens) boundaries up to the model length.
    for k in range(1, MAX_LEN // 256 + 1):
        ctxs.update(256 * k + d for d in (-1, 0, 1))
    # The window (2048) against the tiles, and the lengths of the 94 CTA wraps.
    for base in (2048, 4096, 24064, 24320):
        ctxs.update(base + d for d in range(-20, 21))
    rnd = random.Random(q_len)
    ctxs.update(rnd.randint(1, MAX_LEN) for _ in range(150))
    return sorted(c for c in ctxs if q_len <= c <= MAX_LEN)


@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_single_sequence_sweep(rig, window, q_len):
    """Batch 1 (94 CTAs per KV head): every length around the tile boundaries."""
    for ctx in _sweep_contexts(q_len):
        _run_pair(rig, [ctx], window=window, q_len=q_len)


@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_every_length_of_one_sequence(rig, window, q_len):
    """Batch 1: each of the lengths 1 to 131072 (queued, compared per chunk)."""
    chunk = 4096
    tables = rig.block_tables(1)
    query = rig.query(q_len)
    mask = _mask(q_len, 1)
    for start in range(q_len, MAX_LEN + 1, chunk):
        stop = min(start + chunk, MAX_LEN + 1)
        lens = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        shape = (len(lens) * q_len, *query.shape[1:])
        outs = {
            v: torch.full(shape, SENTINEL, device=DEVICE, dtype=query.dtype)
            for v in ("stock", "shadow")
        }
        for i in range(len(lens)):
            for variant, out in outs.items():
                rig.call(
                    variant,
                    query,
                    out[i * q_len : (i + 1) * q_len],
                    tables,
                    lens[i : i + 1],
                    window_left=WINDOWS[window],
                    q_len=q_len,
                    mask=mask,
                )
        torch.cuda.synchronize()
        a, b = _bits(outs["stock"]), _bits(outs["shadow"])
        if not torch.equal(a, b):
            bad = (a != b).flatten(1).any(1).nonzero().flatten()[0].item() // q_len
            raise AssertionError(
                f"{window} q{q_len}: first difference at ctx {start + bad}"
            )


BATCH_SIZES = [2, 3, 4, 5, 7, 8, 16, 32, 64, 128]


def _random_contexts(rnd: random.Random, batch: int, min_len: int) -> list[int]:
    """Mixed lengths: some short, some near the tile boundaries, some long."""
    ctxs = []
    for _ in range(batch):
        kind = rnd.random()
        if kind < 0.35:
            ctx = rnd.randint(min_len, 600)
        elif kind < 0.6:
            ctx = 256 * rnd.randint(1, 512) + rnd.randint(-2, 2)
        else:
            ctx = int(2 ** rnd.uniform(9, 17))
        ctxs.append(min(max(ctx, min_len), MAX_LEN))
    return ctxs


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_heterogeneous_batches(rig, batch, window, q_len, pdl):
    """Every request of a batch has its own number of active CTAs."""
    rnd = random.Random(batch * 100 + q_len)
    for _ in range(6):
        ctxs = _random_contexts(rnd, batch, q_len)
        _run_pair(rig, ctxs, window=window, q_len=q_len, pdl=pdl)


@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_edge_batch(rig, window, q_len):
    ctxs = [8, 255, 256, 257, 2047, 2049, 4096, 131072]
    _run_pair(rig, ctxs, window=window, q_len=q_len)
    _run_pair(rig, ctxs[::-1], window=window, q_len=q_len)
    _run_pair(rig, [MAX_LEN] * 16, window=window, q_len=q_len)
    _run_pair(rig, [max(q_len, 1)] * 16, window=window, q_len=q_len)


@pytest.mark.parametrize("window", ["swa", "full"])
def test_empty_sequences(rig, window):
    """Padded slots of a CUDA graph batch carry length 0 (NaN rows in both kernels)."""
    for ctxs in ([0], [0, 0, 0, 0], [0, 1, 0, 300], [5000, 0, 70000, 0, 257, 0, 0, 1]):
        outs = {}
        seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
        tables = rig.block_tables(len(ctxs))
        query = rig.query(len(ctxs))
        for variant in ("stock", "shadow"):
            out = torch.full_like(query, SENTINEL)
            rig.call(
                variant, query, out, tables, seq_lens, window_left=WINDOWS[window]
            )
            outs[variant] = out
        torch.cuda.synchronize()
        _assert_same(outs["stock"], outs["shadow"], f"{window} ctx={ctxs}")


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_ragged_draft_batches(rig, batch, window, pdl):
    """Spec-decode verify with a different draft length per request (0 included)."""
    rnd = random.Random(batch)
    max_q = 8
    for _ in range(8):
        q_lens = [rnd.randint(0, max_q) for _ in range(batch)]
        q_lens[0] = max_q
        ctxs = [max(n, 1) + c for n, c in zip(q_lens, _random_contexts(rnd, batch, 1))]
        ctxs = [min(c, MAX_LEN) for c in ctxs]
        total = sum(q_lens)
        q_cu = torch.tensor(
            [0] + list(torch.tensor(q_lens).cumsum(0)), device=DEVICE, dtype=torch.int32
        )
        mask = _make_xqa_ragged_draft_block_mask(q_lens, max_q, True, DEVICE)
        seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
        tables = rig.block_tables(batch)
        query = rig.query(total)
        outs = {}
        for variant in ("stock", "shadow"):
            out = torch.full_like(query, SENTINEL)
            rig.call(
                variant,
                query,
                out,
                tables,
                seq_lens,
                window_left=WINDOWS[window],
                q_len=max_q,
                mask=mask,
                q_cu_seq_lens=q_cu,
                pdl=pdl,
            )
            outs[variant] = out
        torch.cuda.synchronize()
        _assert_same(outs["stock"], outs["shadow"], f"ragged {window} {q_lens} {ctxs}")


@pytest.mark.parametrize(
    "head_dim,kv_dtype,page_size,q_len,window",
    [
        (128, torch.bfloat16, 16, 8, "swa"),
        (128, torch.bfloat16, 16, 1, "full"),
        (64, torch.bfloat16, 16, 1, "swa"),
        (64, torch.float8_e4m3fn, 32, 1, "full"),
        (64, torch.float8_e4m3fn, 64, 8, "swa"),
    ],
)
def test_other_modules_of_the_process(
    rig, head_dim, kv_dtype, page_size, q_len, window
):
    """``install`` also serves the XQA modules of other models (drafters, other KV)."""
    gen = torch.Generator(device=DEVICE).manual_seed(head_dim + page_size)
    pages = 4096
    pool = torch.randn(
        pages, page_size, NUM_KV_HEADS, 2 * head_dim, device=DEVICE, generator=gen
    )
    pool = (pool * 0.5).to(kv_dtype).contiguous()
    k_cache, v_cache = pool.split(head_dim, dim=-1)
    width = MAX_LEN // page_size
    rnd = random.Random(head_dim * 7 + q_len)
    for batch in (1, 3, 8):
        for _ in range(8):
            ctxs = _random_contexts(rnd, batch, q_len)
            seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
            tables = torch.randint(
                0,
                pages,
                (batch, width),
                device=DEVICE,
                dtype=torch.int32,
                generator=gen,
            )
            query = torch.randn(
                batch * q_len,
                NUM_Q_HEADS,
                head_dim,
                device=DEVICE,
                dtype=torch.bfloat16,
                generator=gen,
            )
            outs = {}
            for variant in ("stock", "shadow"):
                out = torch.full_like(query, SENTINEL)
                with _module(variant):
                    xqa_batch_decode_with_kv_cache(
                        query=query,
                        kv_cache=(k_cache, v_cache),
                        workspace_buffer=rig.workspace[variant],
                        block_tables=tables,
                        seq_lens=seq_lens,
                        max_seq_len=MAX_LEN,
                        bmm1_scale=head_dim**-0.5,
                        bmm2_scale=1.0,
                        window_left=WINDOWS[window],
                        out=out,
                        kv_layout="NHD",
                        q_len_per_req=q_len,
                        mask=_mask(q_len, batch),
                    )
                outs[variant] = out
            torch.cuda.synchronize()
            _assert_same(outs["stock"], outs["shadow"], f"hd{head_dim} ctx={ctxs}")


class GraphInputs:
    """What the graphs of both variants read; the test rewrites it in place."""

    def __init__(self, rig, batch, q_len, layers):
        self.seq_lens = torch.full((batch,), MAX_LEN, device=DEVICE, dtype=torch.int32)
        self.tables = rig.block_tables(batch)
        self.mask = _mask(q_len, batch)
        self.queries = [rig.query(batch * q_len) for _ in range(layers)]
        # The layers alternate sliding window and full attention like the model does,
        # so the semaphores and the scratch of one workspace serve both modules.
        self.windows = [SWA if i % 2 == 0 else -1 for i in range(layers)]


class GraphRun:
    """A CUDA graph of the layers that share one workspace, for one variant."""

    def __init__(self, rig, variant, inputs, q_len):
        self.workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=DEVICE)
        self.outs = [torch.full_like(q, SENTINEL) for q in inputs.queries]

        def run():
            for query, out, window in zip(
                inputs.queries, self.outs, inputs.windows, strict=True
            ):
                rig.call(
                    variant,
                    query,
                    out,
                    inputs.tables,
                    inputs.seq_lens,
                    window_left=window,
                    q_len=q_len,
                    mask=inputs.mask,
                    workspace=self.workspace,
                )

        run()  # builds the modules outside the capture
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            run()


@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("batch", [1, 3, 8])
def test_graph_replay_with_changing_contexts(rig, batch, q_len):
    """Stale semaphores or scratch across replays and layers show up as a difference.

    The graphs are captured at the model length and replayed with other lengths and
    other block tables written in place, the way the model runner reuses a graph. The
    counts of active CTAs differ from replay to replay (and from request to request),
    and each replay is run twice.
    """
    layers = 3
    inputs = GraphInputs(rig, batch, q_len, layers)
    stock = GraphRun(rig, "stock", inputs, q_len)
    shadow = GraphRun(rig, "shadow", inputs, q_len)
    rnd = random.Random(batch * 10 + q_len)
    steps = [[MAX_LEN] * batch, [max(q_len, 1)] * batch, [300] * batch]
    steps += [_random_contexts(rnd, batch, q_len) for _ in range(10)]
    steps += [[MAX_LEN] * batch, [256 + rnd.randint(0, 2)] * batch]
    for step, ctxs in enumerate(steps):
        inputs.seq_lens.copy_(torch.tensor(ctxs, device=DEVICE, dtype=torch.int32))
        inputs.tables.copy_(rig.block_tables(batch))
        for rep in range(2):
            stock.graph.replay()
            shadow.graph.replay()
            torch.cuda.synchronize()
            for layer in range(layers):
                _assert_same(
                    stock.outs[layer],
                    shadow.outs[layer],
                    f"q{q_len} step {step} rep {rep} layer {layer} ctx={ctxs}",
                )
    del stock, shadow
    torch.cuda.empty_cache()


# The source and the hook of the shadow.


def test_manifest_lists_the_files_of_the_installed_package():
    manifest = shadow.read_manifest(SHADOW_XQA / "stock.md5")
    data_dir = shadow.flashinfer_data_dir()
    assert "csrc/xqa/mha.cu" in manifest
    assert all((data_dir / rel).is_file() for rel in manifest)


def test_shadow_source_is_the_stock_file_plus_the_patch(tmp_path):
    if shutil.which("patch") is None:
        pytest.skip("patch is not installed")
    stock = shadow.flashinfer_data_dir() / "csrc/xqa/mha.cu"
    patched = tmp_path / "mha.cu"
    subprocess.run(
        [
            "patch",
            "-s",
            "-o",
            str(patched),
            str(stock),
            str(SHADOW_XQA / "mha_active_merge.patch"),
        ],
        check=True,
    )
    assert patched.read_bytes() == (SHADOW_XQA / "mha.cu").read_bytes()


def test_hook_returns_the_shadow_spec_or_the_stock_spec(monkeypatch):
    """``gen_xqa_module`` of FlashInfer, as the hook wraps it (no compilation)."""
    args = (
        torch.bfloat16,
        torch.float8_e4m3fn,
        16,
        64,
        8,
        True,
        torch.bfloat16,
        8,
        False,
    )
    stock = am._STOCK_GEN(*args)
    built = []
    monkeypatch.setattr(
        shadow,
        "build_or_stock",
        lambda spec, stock_spec, strict=False: built.append(spec) or spec,
    )
    spec = am._gen_xqa_module(False, *args)
    assert built == [spec]
    assert spec.name.startswith(stock.name + "_muse_activemerge_")
    assert spec.extra_cuda_cflags == stock.extra_cuda_cflags
    assert [p.name for p in spec.sources] == [p.name for p in stock.sources]
    assert spec.sources[0] != stock.sources[0]
    # Not the shadow's FlashInfer: the stock spec, or an error where it is required.
    monkeypatch.setattr(am, "unavailable_reason", lambda: "another FlashInfer")
    assert am._gen_xqa_module(False, *args).name == stock.name
    with pytest.raises(RuntimeError, match="another FlashInfer"):
        am._gen_xqa_module(True, *args)


def test_install_wraps_gen_xqa_module_once(monkeypatch):
    original = am.fi_xqa.gen_xqa_module
    monkeypatch.setattr(am.fi_xqa, "gen_xqa_module", original)  # restored afterwards
    assert am.install()
    wrapped = am.fi_xqa.gen_xqa_module
    assert wrapped is not original and getattr(wrapped, am._PATCHED_ATTR)
    assert am.install() and am.fi_xqa.gen_xqa_module is wrapped


def test_install_declines_for_another_flashinfer(monkeypatch):
    original = am.fi_xqa.gen_xqa_module
    monkeypatch.setattr(am, "unavailable_reason", lambda: "FlashInfer 9.9")
    assert not am.install() and am.fi_xqa.gen_xqa_module is original


def test_shadow_module_is_a_distinct_build_of_the_patched_source():
    args = (
        torch.bfloat16,
        torch.float8_e4m3fn,
        16,
        64,
        8,
        True,
        torch.bfloat16,
        1,
        False,
    )
    assert am.shadow_xqa_module(*args) is am.shadow_xqa_module(*args)
    stock = am._STOCK_GEN(*args)
    spec = am._derive(stock)
    assert spec.name.startswith(stock.name + "_muse_activemerge_")
    assert spec.jit_library_path.is_file() and spec.jit_library_path != (
        stock.jit_library_path
    )
    assert "nbActiveSubSeq" in spec.sources[0].read_text()
    # Next to the module of the stock build, in the same process.
    assert am.shadow_xqa_module(*args) is not am.fi_xqa._get_xqa_module_cached(*args)


def test_flag_is_on_by_default(monkeypatch):
    knob = "VLLM_MUSE_SPARK_SM120_XQA_ACTIVE_MERGE"
    monkeypatch.delenv(knob, raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_XQA_ACTIVE_MERGE is True
    monkeypatch.setenv(knob, "0")
    assert envs.VLLM_MUSE_SPARK_SM120_XQA_ACTIVE_MERGE is False


def test_model_hook_installs_and_survives_a_failure(monkeypatch):
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xModel

    calls = []
    monkeypatch.setattr(am, "install", lambda: calls.append("install") or True)
    MuseSpark1xModel._configure_xqa_active_merge(None)
    assert calls == ["install"]

    def broken():
        raise RuntimeError("cannot patch FlashInfer")

    monkeypatch.setattr(am, "install", broken)
    MuseSpark1xModel._configure_xqa_active_merge(None)  # warns, runs on stock
