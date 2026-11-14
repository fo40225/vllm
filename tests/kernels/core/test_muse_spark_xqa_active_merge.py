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

The last part of the file covers the one-wave split count of the spec-dec verify
launch (``one_wave``, ``VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE``): a third module,
built from the same source, that is not bitwise the shadow where the context has more
tiles than the new split count, and is compared with the shadow within a rounding-level
tolerance.
"""

import contextlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
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
# The first bytes of the workspace hold the semaphores of the multi-block merge (the
# rest is scratch); a launch has to leave them all zero for the next launch.
SEMAPHORE_BYTES = 8 * 1024 * 1024


def _module(variant: str):
    """Run the XQA calls inside with the stock, the shadow or the one-wave module."""
    if variant == "shadow":
        return am.use_shadow_modules()
    if variant == "wave":
        return am.use_shadow_modules(one_wave=True)
    return contextlib.nullcontext()


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
            for v in ("stock", "shadow", "wave")
        }
        # Attention sinks: one fp32 logit per query head, as the sink layers pass them.
        self.sinks = 1.5 * torch.randn(
            NUM_Q_HEADS, device=DEVICE, dtype=torch.float32, generator=gen
        )

    def query(self, tokens: int) -> torch.Tensor:
        return torch.randn(
            tokens,
            NUM_Q_HEADS,
            HEAD_DIM,
            device=DEVICE,
            dtype=torch.bfloat16,
            generator=self.gen,
        )

    def sinks_for(self, sinks: bool) -> torch.Tensor | None:
        return self.sinks if sinks else None

    def assert_semaphores_idle(self, what: str) -> None:
        """Every launch so far returned the semaphores of both modules to zero."""
        for variant, workspace in self.workspace.items():
            _assert_idle(workspace, f"{variant} {what}")

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
        sinks=None,
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
                sinks=sinks,
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


def _assert_idle(workspace: torch.Tensor, what: str) -> None:
    """The semaphores are zero (they reset themselves when the last CTA merges)."""
    stuck = workspace[:SEMAPHORE_BYTES].view(torch.int32).nonzero()
    if len(stuck):
        raise AssertionError(
            f"{what}: {len(stuck)} semaphores are not zero after the launch, first "
            f"at {stuck[0].item()}"
        )


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


def _run_pair(
    rig, ctxs, *, window, q_len=1, pdl=True, tables=None, check_data=True, sinks=False
):
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
            sinks=rig.sinks_for(sinks),
        )
        outs[variant] = out
    torch.cuda.synchronize()
    what = f"{window} q{q_len}{' sinks' if sinks else ''} ctx={ctxs}"
    _assert_same(outs["stock"], outs["shadow"], what)
    rig.assert_semaphores_idle(what)
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


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_single_sequence_sweep(rig, window, q_len, sinks):
    """Batch 1 (94 CTAs per KV head): every length around the tile boundaries."""
    for ctx in _sweep_contexts(q_len):
        _run_pair(rig, [ctx], window=window, q_len=q_len, sinks=sinks)


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_every_length_of_one_sequence(rig, window, q_len, sinks):
    """Batch 1: each of the lengths 1 to 131072 (queued, compared per chunk).

    With sinks the lengths stop at 20480 (the whole range is covered without them).
    """
    chunk = 4096
    last = 20480 if sinks else MAX_LEN
    tables = rig.block_tables(1)
    query = rig.query(q_len)
    mask = _mask(q_len, 1)
    for start in range(q_len, last + 1, chunk):
        stop = min(start + chunk, last + 1)
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
                    sinks=rig.sinks_for(sinks),
                )
        torch.cuda.synchronize()
        rig.assert_semaphores_idle(f"{window} q{q_len} ctx {start}..{stop - 1}")
        a, b = _bits(outs["stock"]), _bits(outs["shadow"])
        if not torch.equal(a, b):
            bad = (a != b).flatten(1).any(1).nonzero().flatten()[0].item() // q_len
            raise AssertionError(
                f"{window} q{q_len}: first difference at ctx {start + bad}"
            )


# The grid of a batch has 188 // (2 * batch) CTAs per KV head (94 at batch 1): 4 and 3
# CTAs at the batches 23, 24 and 31 are where it holds fewer than the 4 warp groups
# that share the merge, and 1 CTA from batch 48 on (no merge at all).
BATCH_SIZES = [2, 3, 4, 5, 7, 8, 16, 23, 24, 31, 32, 64, 128]


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


@pytest.mark.parametrize("pdl,sinks", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_heterogeneous_batches(rig, batch, window, q_len, pdl, sinks):
    """Every request of a batch has its own number of active CTAs."""
    rnd = random.Random(batch * 100 + q_len)
    for _ in range(6):
        ctxs = _random_contexts(rnd, batch, q_len)
        _run_pair(rig, ctxs, window=window, q_len=q_len, pdl=pdl, sinks=sinks)


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_edge_batch(rig, window, q_len, sinks):
    ctxs = [8, 255, 256, 257, 2047, 2049, 4096, 131072]
    _run_pair(rig, ctxs, window=window, q_len=q_len, sinks=sinks)
    _run_pair(rig, ctxs[::-1], window=window, q_len=q_len, sinks=sinks)
    _run_pair(rig, [MAX_LEN] * 16, window=window, q_len=q_len, sinks=sinks)
    _run_pair(rig, [max(q_len, 1)] * 16, window=window, q_len=q_len, sinks=sinks)


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_empty_sequences(rig, window, sinks):
    """Padded slots of a CUDA graph batch carry length 0 (NaN rows in both kernels).

    With sinks an empty row is what the sink alone makes of it, and it has to be the
    stock one too.
    """
    for ctxs in ([0], [0, 0, 0, 0], [0, 1, 0, 300], [5000, 0, 70000, 0, 257, 0, 0, 1]):
        outs = {}
        seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
        tables = rig.block_tables(len(ctxs))
        query = rig.query(len(ctxs))
        for variant in ("stock", "shadow"):
            out = torch.full_like(query, SENTINEL)
            rig.call(
                variant,
                query,
                out,
                tables,
                seq_lens,
                window_left=WINDOWS[window],
                sinks=rig.sinks_for(sinks),
            )
            outs[variant] = out
        torch.cuda.synchronize()
        _assert_same(outs["stock"], outs["shadow"], f"{window} ctx={ctxs}")
        rig.assert_semaphores_idle(f"{window} empty ctx={ctxs}")


@pytest.mark.parametrize("pdl,sinks", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_ragged_draft_batches(rig, batch, window, pdl, sinks):
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
                sinks=rig.sinks_for(sinks),
            )
            outs[variant] = out
        torch.cuda.synchronize()
        _assert_same(outs["stock"], outs["shadow"], f"ragged {window} {q_lens} {ctxs}")
        rig.assert_semaphores_idle(f"ragged {window} {q_lens} {ctxs}")


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
            rig.assert_semaphores_idle(f"hd{head_dim} ctx={ctxs}")


OTHER_DRAFT_LENGTHS = [2, 4, 5, 7]  # drafters with 1 to 6 tokens (q_len 8 is above)


def _edge_contexts(q_len: int) -> list[int]:
    """A reduced sweep: the tile, page and window edges and some random lengths."""
    ctxs = set(range(q_len, q_len + 6))
    for base in (255, 511, 2047, 2303, 4095, 24063, 24319, MAX_LEN - 256):
        ctxs.update(base + d for d in (-1, 0, 1, 2))
    ctxs.update({MAX_LEN - 1, MAX_LEN})
    rnd = random.Random(q_len)
    ctxs.update(rnd.randint(1, MAX_LEN) for _ in range(40))
    return sorted(c for c in ctxs if q_len <= c <= MAX_LEN)


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", OTHER_DRAFT_LENGTHS)
@pytest.mark.parametrize("window", ["swa", "full"])
def test_other_draft_lengths_single_sequence(rig, window, q_len, sinks):
    """Every XQA module of the process is a shadow, so a drafter with 1 to 6 tokens
    (q_len 2 to 7, uniform) runs modules that no other test of this file builds. The
    one with q_len 5 and 7 has an incomplete second block of query rows."""
    for ctx in _edge_contexts(q_len):
        _run_pair(rig, [ctx], window=window, q_len=q_len, sinks=sinks)


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", OTHER_DRAFT_LENGTHS)
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", [2, 3, 8, 23, 24, 31, 64])
def test_other_draft_lengths_batches(rig, batch, window, q_len, sinks):
    rnd = random.Random(batch * 100 + q_len)
    for _ in range(4):
        ctxs = _random_contexts(rnd, batch, q_len)
        _run_pair(rig, ctxs, window=window, q_len=q_len, sinks=sinks)


class GraphInputs:
    """What the graphs of both variants read; the test rewrites it in place."""

    def __init__(self, rig, batch, q_len, layers, sinks=False):
        self.sinks = rig.sinks_for(sinks)
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
                    sinks=inputs.sinks,
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


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [1, 8])
@pytest.mark.parametrize("batch", [1, 3, 8, 24])
def test_graph_replay_with_changing_contexts(rig, batch, q_len, sinks):
    """Stale semaphores or scratch across replays and layers show up as a difference.

    The graphs are captured at the model length and replayed with other lengths and
    other block tables written in place, the way the model runner reuses a graph. The
    counts of active CTAs differ from replay to replay (and from request to request),
    and each replay is run twice.
    """
    layers = 3
    inputs = GraphInputs(rig, batch, q_len, layers, sinks)
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
            # A stale semaphore would corrupt the next replay (or the next layer).
            _assert_idle(stock.workspace, f"stock q{q_len} step {step} rep {rep}")
            _assert_idle(shadow.workspace, f"shadow q{q_len} step {step} rep {rep}")
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


def test_shadow_module_is_refused_after_install(monkeypatch):
    """The module of the installed hook has the name of the shadow module."""
    original = am.fi_xqa.gen_xqa_module
    monkeypatch.setattr(am.fi_xqa, "gen_xqa_module", original)  # restored afterwards
    assert am.install()
    args = (torch.bfloat16, torch.float8_e4m3fn, 16, 64, 8, True, torch.bfloat16, 1)
    with pytest.raises(RuntimeError, match="after install"):
        am.shadow_xqa_module(*args, False)
    with pytest.raises(RuntimeError, match="after install"):
        with am.use_shadow_modules():
            am.fi_xqa._get_xqa_module_cached(*args, False)


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
    monkeypatch.setattr(am, "install", lambda **_: calls.append("install") or True)
    MuseSpark1xModel._configure_xqa_active_merge(None)
    assert calls == ["install"]

    def broken(**_):
        raise RuntimeError("cannot patch FlashInfer")

    monkeypatch.setattr(am, "install", broken)
    MuseSpark1xModel._configure_xqa_active_merge(None)  # warns, runs on stock


# The one-wave split count of the spec-dec verify launch (``one_wave``).
#
# A launch of 5 to 8 query tokens per request has 2 blocks of 32 query rows per KV head
# (8 query heads per KV head), and the module built with ``one_wave`` divides the SMs by
# the batch, the KV heads and the blocks, where the shadow divides by the first two only
# (twice as many CTAs as SMs at 1 CTA per SM). It does so where that grid fills at least
# 90% of the SMs (a global layer that streams many tiles per CTA is slower on a grid
# that leaves SMs idle) and for the sliding window layers at every batch (they read 9
# or 10 tiles per sequence). The split count sets the order of the merge of the partial
# results, so the output is not bitwise the shadow's where a sequence has more tiles
# than the new split count (and is where it has not, or where the split count is kept).

SM_COUNT = torch.cuda.get_device_properties(0).multi_processor_count
GROUP = NUM_Q_HEADS // NUM_KV_HEADS
ROWS_PER_BLOCK = 32
TILE = 256
MIN_FILL_PERCENT = 90  # kOneWaveMinFillPercent of the launcher
FEW_TILES = 16  # kOneWaveFewTiles of the launcher
# The largest difference between the one-wave and the shadow output of a request, in
# bf16 ulps of the largest element of that request's shadow output. The merge order
# changes the rounding of the bf16 output: the measured maximum is 3 ulps over random
# batches of 1 to 16 requests (the median of the requests that differ at all is 1.5 to
# 2) and 4 over each of the lengths 1 to 131072 of one request.
WAVE_MAX_ULPS = 6
SPEC_ARGS = (
    torch.bfloat16,
    torch.float8_e4m3fn,
    PAGE,
    HEAD_DIM,
    GROUP,
    True,
    torch.bfloat16,
    8,
    False,
)
PLAIN_ARGS = (*SPEC_ARGS[:7], 1, False)


def _blocks(q_len: int) -> int:
    return -(-q_len * GROUP // ROWS_PER_BLOCK) if q_len > 1 else 1


def _expected_grid(
    batch: int, q_len: int, one_wave: bool, window: str = "full"
) -> tuple[int, int, int]:
    """The grid of ``launchMHAFlashInfer`` of the shadow (or the one-wave module)."""
    blocks = _blocks(q_len)
    splits = max(1, SM_COUNT // (batch * NUM_KV_HEADS))
    if one_wave and blocks > 1:
        ctas = batch * NUM_KV_HEADS * blocks
        wave = max(1, SM_COUNT // ctas)
        few_tiles = window == "swa" and -(-(SWA + 1 + q_len) // TILE) + 1 <= FEW_TILES
        if few_tiles or wave * ctas * 100 >= SM_COUNT * MIN_FILL_PERCENT:
            splits = wave
    return (min(splits, MAX_LEN // TILE), NUM_KV_HEADS * blocks, batch)


def _launched_grid(
    rig, variant: str, batch: int, q_len: int, window: str = "full"
) -> tuple[int, ...]:
    """The grid of the ``kernel_mha`` launch, as the profiler reports it."""
    from torch.profiler import ProfilerActivity, profile

    seq_lens = torch.full((batch,), 4096, device=DEVICE, dtype=torch.int32)
    tables = rig.block_tables(batch)
    query = rig.query(batch * q_len)
    out = torch.full_like(query, SENTINEL)

    def call():
        rig.call(
            variant,
            query,
            out,
            tables,
            seq_lens,
            window_left=WINDOWS[window],
            q_len=q_len,
            mask=_mask(q_len, batch),
        )

    call()  # builds the module outside the profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as tmp:
        trace = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    grids = {
        tuple(e["args"]["grid"])
        for e in events
        if e.get("cat") == "kernel" and "mha" in e.get("name", "").lower()
    }
    assert len(grids) == 1, grids
    return grids.pop()


@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("q_len", [1, 2, 4, 5, 6, 8])
@pytest.mark.parametrize(
    "batch", [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 16, 23, 24, 31, 32, 47, 48, 64]
)
def test_one_wave_launch_grid(rig, batch, q_len, window):
    """The shadow launches the grid of the stock launcher, the one-wave module a wave
    where the policy applies (and the stock grid elsewhere)."""
    shadow_grid = _launched_grid(rig, "shadow", batch, q_len, window)
    wave_grid = _launched_grid(rig, "wave", batch, q_len, window)
    assert shadow_grid == _expected_grid(batch, q_len, False, window)
    assert wave_grid == _expected_grid(batch, q_len, True, window)
    assert wave_grid[0] <= shadow_grid[0]
    if _blocks(q_len) == 1:
        assert wave_grid == shadow_grid  # nothing to fit: the same launch
    elif wave_grid != shadow_grid and batch * NUM_KV_HEADS * _blocks(q_len) <= SM_COUNT:
        # One wave (1 CTA per SM) instead of the shadow's more than one.
        assert math.prod(wave_grid) <= SM_COUNT < math.prod(shadow_grid)


def test_one_wave_policy_of_the_grid():
    """What the launcher decides, from the numbers of the measurements (188 SMs):
    batch 1 to 5 fit one wave and fill 94 to 100% of the SMs; the global layer keeps the
    stock split from batch 6 on, where the grid leaves more than 10% of the SMs idle
    (batch 6: 168 CTAs, batch 8: 160, batch 16: 128), the sliding window layer never."""
    if SM_COUNT != 188:
        pytest.skip("The numbers are those of 188 SMs")
    batches = (1, 2, 3, 4, 5)
    assert [_expected_grid(b, 8, True)[0] for b in batches] == [47, 23, 15, 11, 9]
    assert [_expected_grid(b, 8, False)[0] for b in batches] == [94, 47, 31, 23, 18]
    for batch in (6, 7, 8, 10, 12, 16, 24):
        assert _expected_grid(batch, 8, True)[0] == _expected_grid(batch, 8, False)[0]
    for batch in (9, 11, 15, 23, 47):
        assert _expected_grid(batch, 8, True)[0] < _expected_grid(batch, 8, False)[0]
    swa = [_expected_grid(b, 8, True, "swa")[0] for b in (6, 8, 16, 24)]
    assert swa == [7, 5, 2, 1]


def _per_request(q_lens: list[int]) -> torch.Tensor:
    """The request of every query row."""
    return torch.repeat_interleave(
        torch.arange(len(q_lens), device=DEVICE),
        torch.tensor(q_lens, device=DEVICE),
    )


def _wave_excess(shadow: torch.Tensor, wave: torch.Tensor, q_lens: list[int]):
    """Per request: max |wave - shadow| in bf16 ulps of the max |shadow| (float32)."""
    a, b = shadow.float(), wave.float()
    req = _per_request(q_lens)
    peak = torch.zeros(len(q_lens), device=DEVICE).scatter_reduce_(
        0, req, a.abs().amax((1, 2)), "amax", include_self=True
    )
    diff = torch.zeros(len(q_lens), device=DEVICE).scatter_reduce_(
        0, req, (a - b).abs().amax((1, 2)), "amax", include_self=True
    )
    ulp = torch.exp2(torch.floor(torch.log2(peak.clamp_min(2.0**-100))) - 7)
    return diff / ulp


def _assert_close(shadow, wave, q_lens: list[int], what: str) -> None:
    """The one-wave output is the shadow's up to the merge order (rounding level)."""
    assert torch.isfinite(shadow.float()).all(), f"{what}: the shadow is not finite"
    assert torch.isfinite(wave.float()).all(), f"{what}: the one-wave is not finite"
    excess = _wave_excess(shadow, wave, q_lens)
    if not (excess <= WAVE_MAX_ULPS).all():
        worst = int(excess.argmax())
        raise AssertionError(
            f"{what}: request {worst} differs by {excess[worst].item():.1f} ulps "
            f"(limit {WAVE_MAX_ULPS})"
        )


def _run_wave_pair(
    rig,
    ctxs,
    *,
    window,
    q_len,
    pdl=True,
    sinks=False,
    tables=None,
    bitwise=False,
    query=None,
):
    """One launch of the shadow and of the one-wave module on the contexts ``ctxs``."""
    batch = len(ctxs)
    seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
    tables = rig.block_tables(batch) if tables is None else tables
    query = rig.query(batch * q_len) if query is None else query
    mask = _mask(q_len, batch)
    outs = {}
    for variant in ("shadow", "wave"):
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
            sinks=rig.sinks_for(sinks),
        )
        outs[variant] = out
    torch.cuda.synchronize()
    what = f"{window} q{q_len}{' sinks' if sinks else ''} ctx={ctxs}"
    if bitwise:
        _assert_same(outs["shadow"], outs["wave"], what)
    else:
        _assert_close(outs["shadow"], outs["wave"], [q_len] * batch, what)
    rig.assert_semaphores_idle(what)
    assert not (outs["wave"] == SENTINEL).all()
    return outs["shadow"], outs["wave"]


def test_one_wave_is_off_by_default(monkeypatch):
    knob = "VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE"
    monkeypatch.delenv(knob, raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE is False
    monkeypatch.setenv(knob, "1")
    assert envs.VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE is True
    # The module of the default is the plain shadow: no flag, no tag.
    stock = am._STOCK_GEN(*SPEC_ARGS)
    spec = am._derive(stock)
    assert "_muse_activemerge_onewave_" not in spec.name
    assert spec.extra_cuda_cflags == stock.extra_cuda_cflags
    # The launcher of the source keeps the stock split count unless the flag is set.
    source = (SHADOW_XQA / "mha.cu").read_text()
    assert "#define MUSE_XQA_SPEC_ONE_WAVE 0" in source


def test_one_wave_spec_is_a_module_of_its_own():
    stock = am._STOCK_GEN(*SPEC_ARGS)
    plain = am._derive(stock)
    wave = am._derive(stock, one_wave=True)
    assert wave.name.startswith(stock.name + "_muse_activemerge_onewave_")
    assert wave.name != plain.name and wave.jit_library_path != plain.jit_library_path
    assert wave.extra_cuda_cflags == [*stock.extra_cuda_cflags, am._ONE_WAVE_FLAG]
    assert [p.name for p in wave.sources] == [p.name for p in stock.sources]
    # Without a spec-dec launch (q_len 1) there is nothing to size: the plain module.
    stock1 = am._STOCK_GEN(*PLAIN_ARGS)
    assert am._derive(stock1, one_wave=True).name == am._derive(stock1).name
    assert (
        am._derive(stock1, one_wave=True).extra_cuda_cflags == stock1.extra_cuda_cflags
    )


def test_shadow_module_one_wave_is_a_distinct_cached_build():
    plain = am.shadow_xqa_module(*SPEC_ARGS)
    wave = am.shadow_xqa_module(*SPEC_ARGS, one_wave=True)
    assert wave is not plain
    assert wave is am.shadow_xqa_module(*SPEC_ARGS, one_wave=True)
    assert plain is am.shadow_xqa_module(*SPEC_ARGS)


def test_install_one_wave_builds_the_one_wave_spec_dec_modules(monkeypatch):
    original = am.fi_xqa.gen_xqa_module
    monkeypatch.setattr(am.fi_xqa, "gen_xqa_module", original)  # restored afterwards
    built = []
    monkeypatch.setattr(
        shadow,
        "build_or_stock",
        lambda spec, stock_spec, strict=False: built.append(spec) or spec,
    )
    assert am.install(one_wave=True)
    hook = am.fi_xqa.gen_xqa_module
    assert getattr(hook, am._ONE_WAVE_ATTR) is True
    assert "_muse_activemerge_onewave_" in hook(*SPEC_ARGS).name
    assert "_muse_activemerge_onewave_" not in hook(*PLAIN_ARGS).name
    # Installed once: another call neither wraps again nor changes the mode.
    assert am.install(one_wave=False) and am.fi_xqa.gen_xqa_module is hook
    assert getattr(hook, am._ONE_WAVE_ATTR) is True


def test_install_without_one_wave_builds_the_plain_modules(monkeypatch):
    original = am.fi_xqa.gen_xqa_module
    monkeypatch.setattr(am.fi_xqa, "gen_xqa_module", original)  # restored afterwards
    monkeypatch.setattr(
        shadow, "build_or_stock", lambda spec, stock, strict=False: spec
    )
    assert am.install()
    hook = am.fi_xqa.gen_xqa_module
    assert getattr(hook, am._ONE_WAVE_ATTR) is False
    assert "_muse_activemerge_onewave_" not in hook(*SPEC_ARGS).name


def test_model_hook_passes_the_one_wave_flag(monkeypatch):
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xModel

    calls = []
    monkeypatch.setattr(am, "install", lambda **kw: calls.append(kw) or True)
    knob = "VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE"
    monkeypatch.delenv(knob, raising=False)
    MuseSpark1xModel._configure_xqa_active_merge(None)
    monkeypatch.setenv(knob, "1")
    MuseSpark1xModel._configure_xqa_active_merge(None)
    assert calls == [{"one_wave": False}, {"one_wave": True}]


WAVE_BATCHES = [1, 2, 3, 4, 5, 6, 8, 12, 16, 23, 24, 31, 32, 47, 48, 64]


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [2, 3, 4])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_one_wave_does_not_change_launches_of_one_block(rig, window, q_len, sinks):
    """Up to 4 query tokens (1 block of rows) the grid is the same: bitwise equal."""
    rnd = random.Random(q_len)
    for batch in (1, 2, 3, 8, 24):
        for _ in range(4):
            ctxs = _random_contexts(rnd, batch, q_len)
            _run_wave_pair(
                rig, ctxs, window=window, q_len=q_len, sinks=sinks, bitwise=True
            )


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", [1, 2, 3, 4, 5, 8, 16])
def test_one_wave_is_bitwise_where_the_merge_order_is_the_same(
    rig, batch, window, q_len, sinks
):
    """A sequence with no more tiles than the one-wave split count merges one partial
    per tile in the order of the tiles under either split count."""
    splits = _expected_grid(batch, q_len, True, window)[0]
    limit = splits * TILE
    if window == "swa":
        limit = min(limit, 2048)  # a longer context slides: the tiles that remain
    rnd = random.Random(batch * 10 + q_len)
    for _ in range(8):
        ctxs = [rnd.randint(q_len, max(q_len, limit)) for _ in range(batch)]
        ctxs[0] = max(q_len, limit)
        _run_wave_pair(
            rig, ctxs, window=window, q_len=q_len, sinks=sinks, bitwise=True
        )


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("batch", [6, 7, 8, 10, 12, 16, 24, 31])
def test_one_wave_keeps_the_stock_split_of_a_grid_that_would_not_fill_the_sms(
    rig, batch, q_len, sinks
):
    """A global layer at these batches launches the shadow's grid: bitwise equal."""
    assert _expected_grid(batch, q_len, True) == _expected_grid(batch, q_len, False)
    rnd = random.Random(batch * 1000 + q_len)
    for _ in range(6):
        ctxs = _random_contexts(rnd, batch, q_len)
        _run_wave_pair(rig, ctxs, window="full", q_len=q_len, sinks=sinks, bitwise=True)


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [5, 6, 7, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_one_wave_single_sequence_sweep(rig, window, q_len, sinks):
    """Batch 1 (47 CTAs per block): every length around the tile boundaries."""
    for ctx in _sweep_contexts(q_len):
        _run_wave_pair(rig, [ctx], window=window, q_len=q_len, sinks=sinks)


@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_one_wave_every_length_of_one_sequence(rig, window, q_len):
    """Batch 1: each of the lengths 1 to 131072 against the shadow (per chunk)."""
    chunk = 4096
    tables = rig.block_tables(1)
    query = rig.query(q_len)
    mask = _mask(q_len, 1)
    worst = 0.0
    for start in range(q_len, MAX_LEN + 1, chunk):
        stop = min(start + chunk, MAX_LEN + 1)
        lens = torch.arange(start, stop, device=DEVICE, dtype=torch.int32)
        shape = (len(lens) * q_len, *query.shape[1:])
        outs = {
            v: torch.full(shape, SENTINEL, device=DEVICE, dtype=query.dtype)
            for v in ("shadow", "wave")
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
        rig.assert_semaphores_idle(f"{window} q{q_len} ctx {start}..{stop - 1}")
        excess = _wave_excess(outs["shadow"], outs["wave"], [q_len] * len(lens))
        assert torch.isfinite(outs["wave"].float()).all()
        if not (excess <= WAVE_MAX_ULPS).all():
            bad = int(excess.argmax())
            raise AssertionError(
                f"{window} q{q_len}: ctx {start + bad} differs by "
                f"{excess[bad].item():.1f} ulps (limit {WAVE_MAX_ULPS})"
            )
        worst = max(worst, excess.max().item())
    print(f"one-wave every length {window} q{q_len}: worst {worst:.1f} ulps")


@pytest.mark.parametrize("pdl,sinks", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("q_len", [5, 6, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", WAVE_BATCHES)
def test_one_wave_heterogeneous_batches(rig, batch, window, q_len, pdl, sinks):
    """Every request of a batch has its own number of tiles and active CTAs."""
    rnd = random.Random(batch * 100 + q_len)
    for _ in range(6):
        ctxs = _random_contexts(rnd, batch, q_len)
        _run_wave_pair(rig, ctxs, window=window, q_len=q_len, pdl=pdl, sinks=sinks)


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_one_wave_edge_batch(rig, window, q_len, sinks):
    ctxs = [8, 255, 256, 257, 2047, 2049, 4096, 131072]
    _run_wave_pair(rig, ctxs, window=window, q_len=q_len, sinks=sinks)
    _run_wave_pair(rig, ctxs[::-1], window=window, q_len=q_len, sinks=sinks)
    for batch in (1, 4, 16):
        _run_wave_pair(rig, [MAX_LEN] * batch, window=window, q_len=q_len, sinks=sinks)
        _run_wave_pair(rig, [q_len] * batch, window=window, q_len=q_len, sinks=sinks)


@pytest.mark.parametrize("pdl,sinks", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("window", ["swa", "full"])
def test_one_wave_ragged_draft_batches(rig, batch, window, pdl, sinks):
    """The verify batch of dynamic speculation: a draft length per request (0 too)."""
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
        for variant in ("shadow", "wave"):
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
                sinks=rig.sinks_for(sinks),
            )
            outs[variant] = out
        torch.cuda.synchronize()
        what = f"ragged {window} {q_lens} {ctxs}"
        _assert_close(outs["shadow"], outs["wave"], q_lens, what)
        rig.assert_semaphores_idle(what)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", [1, 4, 16])
def test_one_wave_is_deterministic(rig, batch, window, q_len, pdl):
    """The same launch gives the same bits again, with other launches in between.

    The CTA that merges is the last one to finish, which differs from launch to launch
    and with PDL; the order of the merge is the order of the partials, not of the CTAs.
    """
    rnd = random.Random(batch * 7 + q_len)
    ctxs = [min(MAX_LEN, 40000 + rnd.randint(0, 90000)) for _ in range(batch)]
    ctxs[0] = MAX_LEN
    seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
    tables = rig.block_tables(batch)
    query = rig.query(batch * q_len)
    mask = _mask(q_len, batch)
    first = None
    for rep in range(12):
        out = torch.full_like(query, SENTINEL)
        rig.call(
            "wave",
            query,
            out,
            tables,
            seq_lens,
            window_left=WINDOWS[window],
            q_len=q_len,
            mask=mask,
            pdl=pdl,
        )
        if rep % 3 == 2:  # an unrelated launch that leaves other scratch behind
            other = torch.full_like(query, SENTINEL)
            rig.call(
                "wave",
                query,
                other,
                rig.block_tables(batch),
                torch.full_like(seq_lens, 70000),
                window_left=-1,
                q_len=q_len,
                mask=mask,
                pdl=pdl,
            )
        torch.cuda.synchronize()
        if first is None:
            first = out
        else:
            _assert_same(first, out, f"{window} q{q_len} b{batch} pdl={pdl} rep {rep}")
    rig.assert_semaphores_idle(f"{window} q{q_len} b{batch} deterministic")


@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", [1, 3])
def test_one_wave_kv_heads_with_equal_data_give_equal_bits(rig, batch, window, q_len):
    """Two KV heads that hold the same data and queries (what a TP rank pair that holds
    the same head would compute) give the same bits: nothing depends on which CTA a head
    runs in, or on the order in which the CTAs of a launch finish."""
    pool = rig.pool.clone()
    pool[:, :, 1] = pool[:, :, 0]
    k_cache, v_cache = pool.split(HEAD_DIM, dim=-1)
    ctxs = [MAX_LEN - 1000 * i for i in range(batch)]
    seq_lens = torch.tensor(ctxs, device=DEVICE, dtype=torch.int32)
    tables = rig.block_tables(batch)
    query = rig.query(batch * q_len)
    query[:, GROUP:] = query[:, :GROUP]  # the 8 query heads of each KV head
    out = torch.full_like(query, SENTINEL)
    with _module("wave"):
        xqa_batch_decode_with_kv_cache(
            query=query,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=rig.workspace["wave"],
            block_tables=tables,
            seq_lens=seq_lens,
            max_seq_len=MAX_LEN,
            bmm1_scale=0.125,
            bmm2_scale=1.0,
            window_left=WINDOWS[window],
            out=out,
            kv_layout="NHD",
            enable_pdl=True,
            q_len_per_req=q_len,
            mask=_mask(q_len, batch),
        )
    torch.cuda.synchronize()
    _assert_same(out[:, :GROUP], out[:, GROUP:], f"{window} q{q_len} b{batch}")
    assert not (out == SENTINEL).any()


def _reference(rig, query, tables, ctxs, q_len, window):
    """The attention of the verify rows in float32 from the fp8 KV (no rounding)."""
    out = torch.empty(query.shape, dtype=torch.float32, device=DEVICE)
    rows = torch.arange(q_len, device=DEVICE)
    for i, ctx in enumerate(ctxs):
        pages = tables[i, : -(-ctx // PAGE)].long()
        kv = rig.pool[pages].reshape(-1, NUM_KV_HEADS, 2 * HEAD_DIM)[:ctx].float()
        k, v = kv[..., :HEAD_DIM], kv[..., HEAD_DIM:]
        q = query[i * q_len : (i + 1) * q_len].float()
        q = q.view(q_len, NUM_KV_HEADS, GROUP, HEAD_DIM)
        scores = torch.einsum("rghd,tgd->rght", q, k) * 0.125
        t = torch.arange(ctx, device=DEVICE)
        pos = ctx - q_len + rows  # the position of draft row r
        visible = t[None, :] <= pos[:, None]
        if window == "swa":
            visible &= t[None, :] >= pos[:, None] - SWA
        scores = scores.masked_fill(~visible[:, None, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        o = torch.einsum("rght,tgd->rghd", probs, v)
        out[i * q_len : (i + 1) * q_len] = o.reshape(q_len, NUM_Q_HEADS, HEAD_DIM)
    return out


def _error_stats(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    """Mean |error|, max |error| and mean (signed) error against ``ref``."""
    err = out.float() - ref
    return err.abs().mean().item(), err.abs().max().item(), err.mean().item()


@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("window", ["swa", "full"])
@pytest.mark.parametrize("batch", [1, 3])
def test_one_wave_error_against_a_float32_reference(rig, batch, window, q_len):
    """The one-wave kernel is as accurate as the shadow, and unbiased.

    Both outputs are bf16 of an attention with partial results merged in 94 or 47
    parts (more than one tile per part at these lengths); the error against the exact
    attention of the fp8 KV is the same size, and the signed mean of the error is
    noise, not a bias.
    """
    rnd = random.Random(batch * 31 + q_len)
    sums = {v: [0.0, 0.0, 0.0] for v in ("shadow", "wave")}
    count = 0
    for _ in range(6):
        ctxs = [rnd.randint(30000, MAX_LEN) for _ in range(batch)]
        tables = rig.block_tables(batch)
        query = rig.query(batch * q_len)
        shadow_out, wave_out = _run_wave_pair(
            rig, ctxs, window=window, q_len=q_len, tables=tables, query=query
        )
        ref = _reference(rig, query, tables, ctxs, q_len, window)
        for name, out in (("shadow", shadow_out), ("wave", wave_out)):
            mean_abs, max_abs, mean = _error_stats(out, ref)
            sums[name][0] += mean_abs
            sums[name][1] = max(sums[name][1], max_abs)
            sums[name][2] += mean
        count += 1
    stats = {
        name: [mean_abs / count, max_abs, mean / count]
        for name, (mean_abs, max_abs, mean) in sums.items()
    }
    shadow_stats, wave_stats = stats["shadow"], stats["wave"]
    print(
        f"error vs fp32 {window} q{q_len} b{batch}: "
        f"shadow {shadow_stats} wave {wave_stats}"
    )
    assert wave_stats[0] <= 1.25 * shadow_stats[0]
    assert wave_stats[1] <= 2.0 * shadow_stats[1]
    assert abs(wave_stats[2]) <= 0.1 * wave_stats[0]


class WaveGraphs:
    """A CUDA graph of the shadow and of the one-wave module over the same inputs."""

    def __init__(self, rig, inputs, q_len, layers):
        self.runs = {v: GraphRun(rig, v, inputs, q_len) for v in ("shadow", "wave")}
        self.eager_workspace = torch.zeros(
            WORKSPACE_BYTES, dtype=torch.uint8, device=DEVICE
        )
        self.rig, self.inputs, self.q_len, self.layers = rig, inputs, q_len, layers

    def eager_wave(self):
        """The same layers launched eagerly (no graph) with the one-wave module."""
        outs = []
        for query, window in zip(self.inputs.queries, self.inputs.windows, strict=True):
            out = torch.full_like(query, SENTINEL)
            self.rig.call(
                "wave",
                query,
                out,
                self.inputs.tables,
                self.inputs.seq_lens,
                window_left=window,
                q_len=self.q_len,
                mask=self.inputs.mask,
                workspace=self.eager_workspace,
                sinks=self.inputs.sinks,
            )
            outs.append(out)
        return outs


@pytest.mark.parametrize("sinks", [False, True])
@pytest.mark.parametrize("q_len", [5, 8])
@pytest.mark.parametrize("batch", [1, 3, 8, 24])
def test_one_wave_graph_replay_with_changing_contexts(rig, batch, q_len, sinks):
    """The one-wave graph, replayed with other lengths and block tables written in place
    (the number of tiles and of active CTAs differs from replay to replay), is the
    eager one-wave launch bit for bit, replay after replay, and the shadow's output
    within the merge-order tolerance. The semaphores are idle after every replay."""
    layers = 3
    inputs = GraphInputs(rig, batch, q_len, layers, sinks)
    graphs = WaveGraphs(rig, inputs, q_len, layers)
    shadow, wave = graphs.runs["shadow"], graphs.runs["wave"]
    rnd = random.Random(batch * 10 + q_len)
    steps = [[MAX_LEN] * batch, [max(q_len, 1)] * batch, [300] * batch]
    steps += [_random_contexts(rnd, batch, q_len) for _ in range(10)]
    steps += [[MAX_LEN] * batch, [256 + rnd.randint(0, 2)] * batch]
    for step, ctxs in enumerate(steps):
        inputs.seq_lens.copy_(torch.tensor(ctxs, device=DEVICE, dtype=torch.int32))
        inputs.tables.copy_(rig.block_tables(batch))
        eager = graphs.eager_wave()
        for rep in range(2):
            shadow.graph.replay()
            wave.graph.replay()
            torch.cuda.synchronize()
            for layer in range(layers):
                what = f"q{q_len} step {step} rep {rep} layer {layer} ctx={ctxs}"
                _assert_same(eager[layer], wave.outs[layer], what)
                _assert_close(
                    shadow.outs[layer], wave.outs[layer], [q_len] * batch, what
                )
            _assert_idle(shadow.workspace, f"shadow q{q_len} step {step} rep {rep}")
            _assert_idle(wave.workspace, f"wave q{q_len} step {step} rep {rep}")
            _assert_idle(graphs.eager_workspace, f"eager q{q_len} step {step}")
    del graphs, shadow, wave
    torch.cuda.empty_cache()


_RANK_SCRIPT = """
import json, sys, torch
from flashinfer.decode import xqa_batch_decode_with_kv_cache
from vllm.models.muse_spark.nvidia.flashinfer_shadow import xqa_active_merge as am
from vllm.v1.attention.backends.flashinfer import _make_xqa_draft_block_mask

PAGE, HEAD_DIM, MAX_LEN = 16, 64, 131072
cases, path = json.loads(sys.argv[1]), sys.argv[2]
gen = torch.Generator().manual_seed(1234)
pool = torch.randn(4096, PAGE, 2, 2 * HEAD_DIM, generator=gen) * 0.5
pool = pool.to(torch.float8_e4m3fn).cuda()
k_cache, v_cache = pool.split(HEAD_DIM, dim=-1)
workspace = torch.zeros(394 * 1024 * 1024, dtype=torch.uint8, device="cuda")
sinks = (1.5 * torch.randn(16, generator=gen)).cuda()
outs = []
with am.use_shadow_modules(one_wave=True):
    for q_len, window_left, ctxs in cases:
        batch = len(ctxs)
        tables = torch.randint(
            0, 4096, (batch, MAX_LEN // PAGE), generator=gen, dtype=torch.int32
        ).cuda()
        query = torch.randn(batch * q_len, 16, HEAD_DIM, generator=gen)
        query = query.to(torch.bfloat16).cuda()
        mask = _make_xqa_draft_block_mask(q_len, True, query.device)
        mask = mask.unsqueeze(0).expand(batch, -1, -1).contiguous()
        out = torch.zeros_like(query)
        xqa_batch_decode_with_kv_cache(
            query=query, kv_cache=(k_cache, v_cache), workspace_buffer=workspace,
            block_tables=tables, seq_lens=torch.tensor(ctxs, dtype=torch.int32).cuda(),
            max_seq_len=MAX_LEN, bmm1_scale=0.125, bmm2_scale=1.0,
            window_left=window_left, out=out, sinks=sinks, kv_layout="NHD",
            enable_pdl=True, q_len_per_req=q_len, mask=mask,
        )
        outs.append(out.cpu())
torch.cuda.synchronize()
torch.save(outs, path)
"""
RANK_CASES = [
    [8, -1, [118000]],
    [8, 2047, [100000]],
    [8, -1, [131072, 40000, 5000, 700]],
    [6, -1, [64001, 90000, 30000]],
    [5, -1, [20000] * 16],
    [8, 2047, [131072] * 8],
]


def test_one_wave_is_identical_on_two_devices_and_in_two_processes(tmp_path):
    """What the TP ranks rely on: every GPU gives the same bits for the same inputs.

    The ranks are processes of their own, each with its own device, so the check runs
    the same launches in fresh processes, twice on the first GPU (two cold starts) and
    once on the second GPU, and compares the bits.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible else [str(i) for i in range(2)]
    if torch.cuda.device_count() < 2:
        pytest.skip("Needs two GPUs")
    results = {}
    for tag, device in (("a0", devices[0]), ("a1", devices[1]), ("b0", devices[0])):
        path = tmp_path / f"{tag}.pt"
        subprocess.run(
            [sys.executable, "-c", _RANK_SCRIPT, json.dumps(RANK_CASES), str(path)],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": device},
            check=True,
            timeout=1800,
        )
        results[tag] = torch.load(path)
    for tag in ("a1", "b0"):
        for case, ref, other in zip(RANK_CASES, results["a0"], results[tag]):
            _assert_same(ref, other, f"{tag} vs a0 {case}")
