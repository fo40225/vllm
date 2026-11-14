# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark decode attention input: in-place Q plus a copy, against a dense Q.

One layer of the decode graph of a TP8 rank of the model (16 query and 2 KV heads of
64, fp8 KV cache of 16-token pages in the HND layout of the fused insert, bf16 query
and output, active-merge XQA, QKV rows of 2304 elements: q, k, v and the output gate):

  copy   the fused QK-norm / RoPE / KV insert writes Q in place into the QKV row, and
         the FlashInfer backend makes the strided view dense (``.contiguous()``, an
         aten ``direct_copy`` kernel for more than one row) before XQA
  dense  the fused op writes Q into a dense buffer (``q_out``,
         VLLM_MUSE_SPARK_SM120_DENSE_Q=1) that XQA reads as is

Per batch size (the CUDA graph capture sizes of the decode path) it reports the median
over CUDA graph replays of (graph time / 62 layers) in us, the CUDA graph node gap and
the production PDL default included, the kernels of one layer in each variant (names and
counts from the profiler), and whether the XQA outputs and the KV caches are bit for bit
the same.

    python benchmarks/kernels/benchmark_muse_spark_dense_q.py --out rows.jsonl
"""

import argparse
import json
import statistics
import sys
import time
from collections import Counter

import torch
from flashinfer.decode import xqa_batch_decode_with_kv_cache

from vllm.models.muse_spark.nvidia.flashinfer_shadow import xqa_active_merge as am
from vllm.models.muse_spark.nvidia.ops import muse_c

PAGE = 16
HEAD_DIM = 64
NQ = 16
NKV = 2
QSZ = NQ * HEAD_DIM
ROW = 2 * (QSZ + NKV * HEAD_DIM)  # q | k | v | o gate
MAX_MODEL_LEN = 131072
FULL_PAGES = MAX_MODEL_LEN // PAGE  # the block table of vLLM
WORKSPACE_BYTES = 394 * 1024 * 1024
WINDOW_SWA = 2047

BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
CTXS = [640, 4096]
VARIANTS = ("copy", "dense")
ROUNDS = ("copy", "dense", "dense", "copy", "copy", "dense")
LAYERS = 62
REPLAYS = 30


def cdiv(a, b):
    return (a + b - 1) // b


class Case:
    def __init__(self, op, workspace, batch, ctx, window_left):
        dev = torch.device("cuda:0")
        self.op = op
        self.batch = batch
        self.window_left = window_left
        self.workspace = workspace
        pages = cdiv(ctx, PAGE)
        num_blocks = batch * pages + 1
        perm = torch.randperm(num_blocks, device=dev)[: batch * pages]
        perm = perm.to(torch.int32).view(batch, pages)
        self.table = torch.zeros(batch, FULL_PAGES, dtype=torch.int32, device=dev)
        self.table[:, :pages] = perm
        self.seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
        pool = torch.randn(num_blocks, NKV, PAGE, 2 * HEAD_DIM, device=dev) * 0.5
        self.pool0 = pool.to(torch.float8_e4m3fn).contiguous().view(torch.uint8)
        self.pool = {v: self.pool0.clone() for v in VARIANTS}
        last = ctx - 1
        self.slots = (perm[:, last // PAGE].long() * PAGE + last % PAGE).contiguous()
        self.positions = torch.full((batch,), last, dtype=torch.int64, device=dev)
        self.qkv0 = torch.randn(batch, ROW, dtype=torch.bfloat16, device=dev)
        self.qkv = {v: self.qkv0.clone() for v in VARIANTS}
        self.out = {
            v: torch.empty(batch, NQ, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            for v in VARIANTS
        }
        self.cos_sin = _cos_sin(8192, HEAD_DIM, dev)
        self.scales = torch.ones(1, dtype=torch.float32, device=dev)

    def layer(self, variant):
        """The calls of one layer, as the model makes them."""
        qkv, pool = self.qkv[variant], self.pool[variant]
        q_out = None
        if variant == "dense":
            q_out = torch.empty(
                self.batch, QSZ, dtype=torch.bfloat16, device=qkv.device
            )
        extra = () if q_out is None else (q_out,)
        self.op(
            qkv,
            self.positions,
            NQ,
            NKV,
            HEAD_DIM,
            1e-5,
            self.cos_sin,
            HEAD_DIM,
            self.slots,
            pool.view(torch.float8_e4m3fn),
            PAGE,
            "fp8",
            self.scales,
            self.scales,
            None,
            None,
            *extra,
        )
        q = qkv[:, :QSZ] if q_out is None else q_out
        q = q.view(-1, NQ, HEAD_DIM)
        # FlashInferImpl.forward: decode_query.contiguous() before XQA.
        q = q.contiguous()
        kv = pool.view(torch.float8_e4m3fn).split(HEAD_DIM, dim=-1)
        xqa_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv,
            workspace_buffer=self.workspace,
            block_tables=self.table,
            seq_lens=self.seq_lens,
            max_seq_len=MAX_MODEL_LEN,
            bmm1_scale=0.125,
            bmm2_scale=1.0,
            window_left=self.window_left,
            out=self.out[variant],
            sinks=None,
            kv_layout="HND",
            q_len_per_req=1,
        )


def _cos_sin(max_pos, rotary_dim, device):
    exponent = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    inv = 1.0 / (500_000.0 ** (exponent / rotary_dim))
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1)


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


def time_graph(case, variant):
    g = capture(lambda: case.layer(variant), LAYERS)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(REPLAYS):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        samples.append(s.elapsed_time(e) * 1000.0 / LAYERS)
    del g
    return statistics.median(samples)


def kernels_of_one_layer(case, variant):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    g = capture(lambda: case.layer(variant), 1)
    g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    names = Counter()
    for ev in prof.events():
        if ev.device_type == DeviceType.CUDA:
            names[ev.name[:60]] += 1
    del g
    return dict(names)


def run_case(case, row):
    # Eager, same inputs: the outputs and the caches must be the same bits.
    for v in VARIANTS:
        case.qkv[v].copy_(case.qkv0)
        case.pool[v].copy_(case.pool0)
        case.out[v].fill_(0.5)
        case.layer(v)
    torch.cuda.synchronize()
    out_copy = case.out["copy"].view(torch.int16)
    out_dense = case.out["dense"].view(torch.int16)
    row["bitwise_out"] = bool(torch.equal(out_copy, out_dense))
    row["bitwise_cache"] = bool(torch.equal(case.pool["copy"], case.pool["dense"]))
    row["qkv_k_v_gate_equal"] = bool(
        torch.equal(case.qkv["copy"][:, QSZ:], case.qkv["dense"][:, QSZ:])
    )
    us = {v: [] for v in VARIANTS}
    for v in ROUNDS:
        us[v].append(time_graph(case, v))
    for v in VARIANTS:
        row[f"us_{v}"] = statistics.mean(us[v])
        row[f"rounds_{v}"] = us[v]
    row["save_us"] = row["us_copy"] - row["us_dense"]
    if row["batch"] in (1, 8, 64):
        row["kernels_copy"] = kernels_of_one_layer(case, "copy")
        row["kernels_dense"] = kernels_of_one_layer(case, "dense")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    if (reason := muse_c.preflight()) is not None:
        sys.exit(f"_muse_C is unavailable: {reason}")
    muse_c.load()
    op = torch.ops._muse_C.fused_muse_spark_qknorm_rope_kv_insert
    if not any(a.name == "q_out" for a in op.default._schema.arguments):
        sys.exit("the _muse_C op has no q_out")
    if (reason := am.unavailable_reason()) is not None:
        sys.exit(f"no shadow XQA module: {reason}")
    workspace = torch.zeros(WORKSPACE_BYTES, dtype=torch.uint8, device="cuda:0")

    batches, ctxs = ([1, 64], [640]) if args.quick else (BATCHES, CTXS)
    t0 = time.time()
    with open(args.out, "w") as f, am.use_shadow_modules():
        for ctx in ctxs:
            for batch in batches:
                row = {"batch": batch, "ctx": ctx}
                case = Case(op, workspace, batch, ctx, WINDOW_SWA)
                run_case(case, row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"[{time.time() - t0:5.0f}s] ctx{ctx:5d} b{batch:3d}: copy "
                    f"{row['us_copy']:6.2f} us  dense {row['us_dense']:6.2f} us  "
                    f"save {row['save_us']:5.2f} us/layer  out={row['bitwise_out']} "
                    f"cache={row['bitwise_cache']}",
                    flush=True,
                )
                del case


if __name__ == "__main__":
    main()
