# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XQA spec-dec verify (``kernel_mha``, q_len 5 to 8) of Muse Spark: one-wave split.

Calls ``xqa_batch_decode_with_kv_cache`` the way the vLLM FlashInfer backend does for
the verify step of a TP8 rank of the model (16 query and 2 KV heads of 64, pages of 16
tokens, fp8 KV, bf16 query and output, NHD layout, no sinks, scale 1/8, a block table of
``cdiv(131072, 16)`` = 8192 columns, the causal draft mask) and times two builds of the
shadow kernel on the same inputs:

  shadow  the active-merge shadow module (what runs today): the split count is
          SMs / (batch x KV heads), the 2 token blocks of a q_len 5 to 8 launch make
          the grid 2 waves of CTAs
  wave    the same module built with ``one_wave``
          (VLLM_MUSE_SPARK_SM120_XQA_SPEC_ONE_WAVE): SMs / (batch x KV heads x token
          blocks), one wave, where that grid fills 90% of the SMs (always for the
          sliding window layers); the shadow's split count elsewhere

The stock FlashInfer module is only run for the output check (``shadow`` is bitwise the
stock output, ``wave`` differs at rounding level where the split counts give another
merge order).

Metrics, in us per call, the CUPTI duration of ``kernel_mha`` inside a CUDA graph,
``enable_pdl=False``:
  cold  a 256 MB read-only L2 flush (a reduction over a 256 MB buffer, which leaves
        clean lines) before every call: the production-like one;
  fill  the same with a 256 MB fill (leaves dirty lines that compete with the KV reads);
  rot   different page tables over disjoint pages, back to back without a flush (every
        call misses L2 like the 16 distinct layers of the model do).
The rounds of the two variants are interleaved (shadow, wave, wave, shadow) and
averaged.
``grid`` is read from the profiler trace; ``tb_s`` = batch x ctx x 256 B / cold us.

    python benchmarks/kernels/benchmark_muse_spark_xqa_wave_fit.py --out rows.jsonl
"""

import argparse
import contextlib
import itertools
import json
import os
import statistics
import sys
import tempfile
import time

import torch
from flashinfer.decode import xqa_batch_decode_with_kv_cache

from vllm.models.muse_spark.nvidia.flashinfer_shadow import xqa_active_merge as am
from vllm.v1.attention.backends.flashinfer import _make_xqa_draft_block_mask

PAGE = 16
HEAD_DIM = 64
NUM_Q_HEADS = 16
NUM_KV_HEADS = 2
MAX_MODEL_LEN = 131072
PROD_PAGES = MAX_MODEL_LEN // PAGE  # the block table of vLLM
WORKSPACE_BYTES = 394 * 1024 * 1024  # VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE default
MAX_NUM_REQS = 128
WINDOW_SWA = 2047  # sliding_window 2048 -> window_left = 2047
BYTES_PER_TOKEN = NUM_KV_HEADS * 2 * HEAD_DIM  # K + V, fp8
POOL_PAGES = 1 << 21  # 8 GiB of KV pool (4096 B per page)
ROT_TARGET_BYTES = 640 * 1024 * 1024  # 5x L2
ROT_MAX_CALLS = 160

CTXS = [16384, 32768, 65536, 118000]
BATCHES = [1, 2, 3, 4]
QLENS = [8]
WINDOWS = {"full": -1, "swa": WINDOW_SWA}
VARIANTS = ("shadow", "wave")
ROUNDS = ("shadow", "wave", "wave", "shadow")


def cdiv(a, b):
    return (a + b - 1) // b


class Case:
    """The tensors of one (batch, ctx, q_len, window) configuration."""

    def __init__(self, pool, workspace, batch, ctx, q_len, window_left):
        dev = pool.device
        self.batch, self.ctx, self.q_len = batch, ctx, q_len
        self.window_left = window_left
        self.pages = cdiv(ctx, PAGE)
        self.bytes = batch * ctx * BYTES_PER_TOKEN
        per_call_pages = batch * self.pages
        want = cdiv(ROT_TARGET_BYTES, self.bytes)
        self.k_rot = max(1, min(ROT_MAX_CALLS, want, pool.shape[0] // per_call_pages))
        perm = torch.randperm(pool.shape[0], device=dev)
        perm = perm[: self.k_rot * per_call_pages].to(torch.int32)
        perm = perm.view(self.k_rot, batch, self.pages)
        self.tables = []
        for k in range(self.k_rot):
            table = torch.zeros(batch, PROD_PAGES, dtype=torch.int32, device=dev)
            table[:, : self.pages] = perm[k]
            self.tables.append(table)
        self.seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
        self.kv = pool.split(HEAD_DIM, dim=-1)  # K and V are strided halves
        self.query = torch.randn(
            batch * q_len, NUM_Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16
        )
        self.out = {v: torch.empty_like(self.query) for v in ("stock", *VARIANTS)}
        per_req = _make_xqa_draft_block_mask(q_len, True, dev)
        self.mask = per_req.unsqueeze(0).expand(MAX_NUM_REQS, -1, -1)[:batch]
        self.mask = self.mask.contiguous()
        self.workspace = workspace

    def call(self, variant, table_idx=0, enable_pdl=False):
        if variant == "stock":
            module = contextlib.nullcontext()
        else:
            module = am.use_shadow_modules(one_wave=variant == "wave")
        with module:
            xqa_batch_decode_with_kv_cache(
                query=self.query,
                kv_cache=self.kv,
                workspace_buffer=self.workspace,
                block_tables=self.tables[table_idx],
                seq_lens=self.seq_lens,
                max_seq_len=MAX_MODEL_LEN,
                bmm1_scale=0.125,
                bmm2_scale=1.0,
                window_left=self.window_left,
                out=self.out[variant],
                sinks=None,
                kv_layout="NHD",
                enable_pdl=enable_pdl,
                q_len_per_req=self.q_len,
                mask=self.mask,
                q_cu_seq_lens=None,
            )


def capture(fn, n):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fn(0)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            fn(i)
    return g


def kernel_durations(g, replays, names_seen, grids=None):
    """CUPTI durations (us) of the ``kernel_mha`` launches of ``replays`` replays."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            g.replay()
        torch.cuda.synchronize()
    durs = []
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA and "mha" in ev.name.lower():
            names_seen.add(ev.name[:80])
            durs.append(ev.device_time)
    if grids is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.json")
            prof.export_chrome_trace(path)
            with open(path) as f:
                events = json.load(f)["traceEvents"]
        for ev in events:
            if ev.get("cat") == "kernel" and "mha" in ev.get("name", "").lower():
                grids.add(tuple(ev["args"]["grid"]))
    return durs


def time_flush(case, variant, flush_fn, names_seen, grids, calls=24, replays=12):
    def one(i):
        flush_fn()
        case.call(variant, 0)

    g = capture(one, calls)
    durs = kernel_durations(g, replays, names_seen, grids)
    del g
    if len(durs) != calls * replays:
        print(f"WARN flush: {len(durs)} events != {calls * replays}", file=sys.stderr)
    return statistics.median(durs)


def time_rot(case, variant, names_seen):
    g = capture(lambda i: case.call(variant, i), case.k_rot)
    durs = kernel_durations(g, 6 if case.k_rot < 8 else 3, names_seen)
    del g
    return statistics.median(durs)


def run_case(case, flush_read, flush_fill, names_seen, row):
    for v in ("stock", *VARIANTS):  # eager: builds the modules, compares the outputs
        case.out[v].fill_(0.5)
        case.call(v, 0)
    torch.cuda.synchronize()
    stock = case.out["stock"].view(torch.int16)
    shadow = case.out["shadow"].view(torch.int16)
    row["shadow_bitwise_stock"] = bool(torch.equal(stock, shadow))
    wave = case.out["wave"].view(torch.int16)
    row["wave_bitwise_shadow"] = bool(torch.equal(shadow, wave))
    diff = (case.out["wave"].float() - case.out["shadow"].float()).abs()
    row["wave_max_abs_diff"] = diff.max().item()
    cold = {v: [] for v in VARIANTS}
    fill = {v: [] for v in VARIANTS}
    rot = {v: [] for v in VARIANTS}
    grids = {v: set() for v in VARIANTS}
    for v in ROUNDS:
        cold[v].append(time_flush(case, v, flush_read, names_seen, grids[v]))
        fill[v].append(time_flush(case, v, flush_fill, names_seen, grids[v]))
        rot[v].append(time_rot(case, v, names_seen))
    for v in VARIANTS:
        row[f"grid_{v}"] = sorted(grids[v])
        row[f"cold_us_{v}"] = statistics.mean(cold[v])
        row[f"cold_rounds_{v}"] = cold[v]
        row[f"fill_us_{v}"] = statistics.mean(fill[v])
        row[f"rot_us_{v}"] = statistics.mean(rot[v])
        row[f"tb_s_{v}"] = case.bytes / row[f"cold_us_{v}"] / 1e6
    for m in ("cold", "fill", "rot"):
        row[f"{m}_save_us"] = row[f"{m}_us_shadow"] - row[f"{m}_us_wave"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ctxs", default="")
    ap.add_argument("--batches", default="")
    ap.add_argument("--qlens", default="")
    ap.add_argument("--windows", default="full", help="comma list of: full, swa")
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    reason = am.unavailable_reason()
    if reason is not None:
        sys.exit(f"no shadow XQA module: {reason}")
    import flashinfer

    print("device", torch.cuda.get_device_name(0), "flashinfer", flashinfer.__version__)
    shape = (POOL_PAGES, PAGE, NUM_KV_HEADS, 2 * HEAD_DIM)
    pool = torch.empty(shape, device=dev, dtype=torch.float8_e4m3fn)
    chunk = 1 << 16
    for i in range(0, POOL_PAGES, chunk):
        data = torch.randn(chunk, *shape[1:], device=dev) * 0.5
        pool[i : i + chunk] = data.to(torch.float8_e4m3fn)
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    flush_buf = torch.zeros(256 * 1024 * 1024 // 8, dtype=torch.int64, device=dev)
    flush_u8 = flush_buf.view(torch.uint8)

    def flush_read():
        flush_buf.sum()

    def flush_fill():
        flush_u8.fill_(1)

    names_seen: set[str] = set()
    ctxs = [int(x) for x in args.ctxs.split(",")] if args.ctxs else CTXS
    batches = [int(x) for x in args.batches.split(",")] if args.batches else BATCHES
    qlens = [int(x) for x in args.qlens.split(",")] if args.qlens else QLENS
    windows = args.windows.split(",")
    if args.quick:
        ctxs, batches, qlens = [16384, 118000], [1, 4], [8]
    t0 = time.time()
    with open(args.out, "w") as f:
        grid = itertools.product(qlens, windows, batches, ctxs)
        for q_len, window, batch, ctx in grid:
            row = {"q_len": q_len, "window": window, "batch": batch, "ctx": ctx}
            case = Case(pool, workspace, batch, ctx, q_len, WINDOWS[window])
            row["bytes"] = case.bytes
            row["k_rot"] = case.k_rot
            run_case(case, flush_read, flush_fill, names_seen, row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(
                f"[{time.time() - t0:6.0f}s] q{q_len} {window:4s} b{batch:2d} "
                f"ctx{ctx:7d} "
                f"grid {row['grid_shadow']}->{row['grid_wave']} "
                f"cold {row['cold_us_shadow']:7.2f}->{row['cold_us_wave']:7.2f} "
                f"fill {row['fill_us_shadow']:7.2f}->{row['fill_us_wave']:7.2f} "
                f"rot {row['rot_us_shadow']:7.2f}->{row['rot_us_wave']:7.2f} "
                f"shadow=stock {row['shadow_bitwise_stock']} "
                f"wave=shadow {row['wave_bitwise_shadow']} "
                f"maxdiff {row['wave_max_abs_diff']:.3g}",
                flush=True,
            )
            del case
    print("kernel names:", sorted(names_seen), flush=True)


if __name__ == "__main__":
    main()
