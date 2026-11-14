# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XQA decode (``kernel_mha``) of Muse Spark: stock against the active-merge shadow.

Calls ``xqa_batch_decode_with_kv_cache`` the way the vLLM FlashInfer backend does for
a TP8 rank of the model (16 query and 2 KV heads of 64, pages of 16 tokens, fp8 KV,
bf16 query and output, NHD layout, no sinks, scale 1/8, a block table of
``cdiv(131072, 16)`` = 8192 columns) and times three variants of every case:

  stock   the FlashInfer module on the full-width block table (what runs today)
  shadow  the active-merge shadow module on the same inputs
  bound   the stock module on a block table that is only as wide as the context: the
          grid shrinks to the tiles in use, the best that the shadow can reach (it
          keeps the grid of the full-width table and lets the idle CTAs return)

Metrics, in us per call:
  hot   median over CUDA graph replays of (graph time / 62 back-to-back calls), the
        CUDA graph node gap included, KV L2-resident at short contexts;
  cold  median CUPTI duration of ``kernel_mha`` inside a CUDA graph with a 256 MB
        L2-flushing fill before every call.
Both run with ``enable_pdl=False`` (consecutive kernels do not overlap); a few rows
repeat the hot metric with the production default (``enable_pdl=True``). The rounds
of the variants are interleaved (stock, shadow, bound, bound, shadow, stock) and
averaged. The outputs of shadow and bound are compared bit for bit with stock.

    python benchmarks/kernels/benchmark_muse_spark_xqa_active_merge.py --out rows.jsonl
"""

import argparse
import contextlib
import itertools
import json
import statistics
import sys
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
FULL_PAGES = MAX_MODEL_LEN // PAGE  # the block table of vLLM
WORKSPACE_BYTES = 394 * 1024 * 1024  # VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE default
MAX_NUM_REQS = 128
WINDOW_SWA = 2047  # sliding_window 2048 -> window_left = 2047

BATCHES = [1, 2, 4, 8, 16]
CTXS = [174, 512, 1024, 4096, 16384, 32768, 118000]
QLENS = [1, 8]
WINDOWS = [("swa2048", WINDOW_SWA), ("full", -1)]
VARIANTS = ("stock", "shadow", "bound")
ROUNDS = ("stock", "shadow", "bound", "bound", "shadow", "stock")

CALLS_COLD = 32  # flush + call pairs per graph
REPLAYS_COLD = 20
CALLS_HOT = 62  # one decode step worth of layers
REPLAYS_HOT = 30


def cdiv(a, b):
    return (a + b - 1) // b


class Case:
    """The tensors of one (batch, ctx, q_len, window) configuration, per variant."""

    def __init__(self, pool, workspace, batch, ctx, q_len, window_left):
        dev = pool.device
        self.q_len = q_len
        self.window_left = window_left
        pages = cdiv(ctx, PAGE)
        perm = torch.randperm(pool.shape[0], device=dev)[: batch * pages]
        perm = perm.to(torch.int32).view(batch, pages)
        # The variants read the same pages; only the width of the table differs.
        self.tables = {
            "full": torch.zeros(batch, FULL_PAGES, dtype=torch.int32, device=dev),
            "bound": perm.contiguous(),
        }
        self.tables["full"][:, :pages] = perm
        self.seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
        self.kv = pool.split(HEAD_DIM, dim=-1)  # K and V are strided halves
        self.query = torch.randn(
            batch * q_len, NUM_Q_HEADS, HEAD_DIM, device=dev, dtype=torch.bfloat16
        )
        self.out = {v: torch.empty_like(self.query) for v in VARIANTS}
        if q_len > 1:
            per_req = _make_xqa_draft_block_mask(q_len, True, dev)
            self.mask = per_req.unsqueeze(0).expand(MAX_NUM_REQS, -1, -1)[:batch]
            self.mask = self.mask.contiguous()
        else:
            self.mask = None
        self.workspace = workspace

    def call(self, variant, enable_pdl=False):
        shadow = variant == "shadow"
        with am.use_shadow_modules() if shadow else contextlib.nullcontext():
            xqa_batch_decode_with_kv_cache(
                query=self.query,
                kv_cache=self.kv,
                workspace_buffer=self.workspace,
                block_tables=self.tables["bound" if variant == "bound" else "full"],
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
        fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    return g


def time_hot(case, variant, enable_pdl):
    g = capture(lambda: case.call(variant, enable_pdl), CALLS_HOT)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPLAYS_HOT):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / CALLS_HOT)
    del g
    return statistics.median(samples)


def time_cold(case, variant, flush, enable_pdl, names_seen):
    def one():
        flush.fill_(1)
        case.call(variant, enable_pdl)

    g = capture(one, CALLS_COLD)
    for _ in range(2):
        g.replay()
    torch.cuda.synchronize()
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(REPLAYS_COLD):
            g.replay()
        torch.cuda.synchronize()
    durs = []
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA and "mha" in ev.name.lower():
            names_seen.add(ev.name[:90])
            durs.append(ev.device_time)  # us
    del g
    if len(durs) != CALLS_COLD * REPLAYS_COLD:
        print(f"WARN: kernel_mha events {len(durs)}", file=sys.stderr)
    if not durs:
        return float("nan")
    return statistics.median(durs)


def run_case(case, flush, names_seen, row):
    """Check the bits of the shadow and the bound, then time the three variants."""
    for v in VARIANTS:  # eager: builds the modules, compares the outputs
        case.out[v].fill_(0.5)
        case.call(v)
    torch.cuda.synchronize()
    stock_bits = case.out["stock"].view(torch.int16)
    for v in ("shadow", "bound"):
        same = torch.equal(stock_bits, case.out[v].view(torch.int16))
        row[f"bitwise_equal_{v}"] = bool(same)
    cold = {v: [] for v in VARIANTS}
    hot = {v: [] for v in VARIANTS}
    for v in ROUNDS:
        cold[v].append(time_cold(case, v, flush, False, names_seen))
        hot[v].append(time_hot(case, v, False))
    for v in VARIANTS:
        row[f"cold_us_{v}"] = statistics.mean(cold[v])
        row[f"cold_rounds_{v}"] = cold[v]
        row[f"hot_us_{v}"] = statistics.mean(hot[v])
        row[f"hot_rounds_{v}"] = hot[v]
    # The production default (PDL) for the short context rows of c1.
    if row["batch"] == 1 and row["ctx"] in (512, 1024) and row["q_len"] == 1:
        for v in VARIANTS:
            row[f"hot_pdl_us_{v}"] = time_hot(case, v, True)
    for m in ("hot", "cold"):
        stock = row[f"{m}_us_stock"]
        row[f"{m}_save_shadow"] = stock - row[f"{m}_us_shadow"]
        row[f"{m}_save_bound"] = stock - row[f"{m}_us_bound"]


def describe(row, elapsed):
    eq = f"{row['bitwise_equal_shadow']}/{row['bitwise_equal_bound']}"
    times = " ".join(
        f"{m} {row[f'{m}_us_stock']:6.2f} -> {row[f'{m}_us_shadow']:6.2f} "
        f"(bound {row[f'{m}_us_bound']:6.2f})"
        for m in ("hot", "cold")
    )
    return (
        f"[{elapsed:6.0f}s] q{row['q_len']} {row['window']:7s} b{row['batch']:2d} "
        f"ctx{row['ctx']:6d} {times} eq={eq}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    reason = am.unavailable_reason()
    if reason is not None:
        sys.exit(f"no shadow XQA module: {reason}")
    import flashinfer

    print("device", torch.cuda.get_device_name(0), "flashinfer", flashinfer.__version__)
    max_pages = max(BATCHES) * cdiv(max(CTXS), PAGE)
    pool = torch.randn(max_pages, PAGE, NUM_KV_HEADS, 2 * HEAD_DIM, device=dev) * 0.5
    pool = pool.to(torch.float8_e4m3fn).contiguous()
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device=dev)
    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    names_seen: set[str] = set()

    batches, ctxs = ([1, 16], [512, 118000]) if args.quick else (BATCHES, CTXS)
    configs = itertools.product(QLENS, WINDOWS, batches, ctxs)
    t0 = time.time()
    with open(args.out, "w") as f:
        for q_len, (wname, wleft), batch, ctx in configs:
            row = {"batch": batch, "ctx": ctx, "q_len": q_len, "window": wname}
            case = Case(pool, workspace, batch, ctx, q_len, wleft)
            run_case(case, flush, names_seen, row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(describe(row, time.time() - t0), flush=True)
            del case
    print("kernel names:", sorted(names_seen))


if __name__ == "__main__":
    main()
