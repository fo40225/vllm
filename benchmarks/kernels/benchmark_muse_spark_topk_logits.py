# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling from the top-K logits of every TP rank (Muse Spark) against the full path.

One device plays the role of a TP rank and fakes the all-gathers (a concatenation of
the rank's own words, a copy of its own shard), so what is timed is the compute around
the gathers, not the gathers. Per decode batch size T it times, in us per step:

  old_*    the soft cap of the gathered [T, 202048] logits, then the sampler: greedy
           (``gumbel_sample`` at temperature 0) or t1k64 (``apply_top_k_top_p`` with
           top_k 64, then ``gumbel_sample``; ``old_t1k64_fi``: FlashInfer's top-k
           sampler, the route of a batch without greedy rows and explicit seeds
           where it is supported), as the V2 sampler runs them
  new_*    the local top-K of the [T, 25256] shard, the soft cap of the gathered
           candidates and ``topk_merge_sample``, greedy or t1k64 (``*_threshold``: the
           t1k64 step that keeps every tie of the top_k-th logit, as FlashInfer's
           sampler does: 8 more words per row from ``fill_tie_closure``)
  and each stage alone. The bytes a rank sends per step are printed with them: the
  all-gather of the full path moves T * 25256 * 2 B per rank, the candidate path
  T * 2 * K * 2 B.

Metrics: ``graph`` = median CUDA-graph replay time / repeats (the GPU time of the
kernels back to back); ``eager`` = median wall time between two events around the
launches of one step with the L2 flushed first (launch gaps included, the way the
sampler runs: eagerly, outside the model's CUDA graph).

    python benchmarks/kernels/benchmark_muse_spark_topk_logits.py --out rows.jsonl
"""

import argparse
import json
import statistics

import torch

from vllm.models.muse_spark.common.layers import MulTanhSoftCap
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
)
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.topk_logits import (
    TIE_EXTRA_WORDS,
    TieDraw,
    TopKLogits,
    fill_tie_closure,
    local_topk_candidates,
    topk_merge_sample,
)

VOCAB = 202048
TP = 8
MULTIPLIER = 0.17677669529663687
SOFT_CAP = 20.0
REPEATS = 20


def time_graph(fn, repeats: int = REPEATS, rounds: int = 30) -> float:
    fn()
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            fn()
    for _ in range(3):
        graph.replay()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / repeats)
    return statistics.median(times)


def time_eager(fn, flush: torch.Tensor, rounds: int = 100) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    times = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        flush.fill_(1)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 128, 256])
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    k = args.k
    cap = MulTanhSoftCap(MULTIPLIER, SOFT_CAP)
    print(f"soft cap op: {'fused CUDA' if cap._fused_op is not None else 'torch'}")
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    shard = VOCAB // TP
    rows_out = []
    for t in args.tokens:
        g = torch.Generator(device="cuda").manual_seed(t)
        full = (torch.randn(t, VOCAB, generator=g, device="cuda") * 3).to(torch.bfloat16)
        local = full[:, :shard].contiguous()
        idx = torch.arange(t, device="cuda")
        zeros = torch.zeros(t, dtype=torch.float32, device="cuda")
        ones = torch.ones(t, dtype=torch.float32, device="cuda")
        top_k_all = torch.full((t,), 64, dtype=torch.int32, device="cuda")
        top_k_none = torch.full((t,), VOCAB, dtype=torch.int32, device="cuda")
        seeds = torch.randint(-(1 << 60), 1 << 60, (t,), device="cuda")
        pos = torch.randint(1, 10000, (t,), device="cuda")
        words_one = local_topk_candidates(local, k)
        words = torch.cat([words_one] * TP).view(torch.bfloat16)
        capped = cap(words)
        topk = TopKLogits(words, capped, t, k, TP, shard)
        ties = TieDraw(idx, seeds, pos)
        words_t_one = local_topk_candidates(local, k, extra_words=TIE_EXTRA_WORDS)
        fill_tie_closure(local, words_t_one, k, ties, token_base=0)
        words_t = torch.cat([words_t_one] * TP).view(torch.bfloat16)
        topk_t = TopKLogits(words_t, cap(words_t), t, k, TP, shard, threshold=True)
        sink = {}

        def old_cap():
            sink["cap"] = cap(full)

        def old_greedy():
            logits = cap(full)
            sink["tok"] = gumbel_sample(logits, idx, zeros, seeds, pos, False, False)

        def old_t1k64():
            logits = cap(full)
            logits = apply_top_k_top_p(logits, top_k_all, None)
            sink["tok"] = gumbel_sample(logits, idx, ones, seeds, pos, False, False)

        def old_t1k64_fi():
            logits = cap(full)
            sink["tok"] = flashinfer_sample(logits, top_k_all, None)

        def new_local():
            sink["w"] = local_topk_candidates(local, k)

        def new_cap():
            sink["c"] = cap(words)

        def new_merge_greedy():
            sink["tok"] = topk_merge_sample(topk, idx, zeros, top_k_none, seeds, pos)

        def new_merge_t1k64():
            sink["tok"] = topk_merge_sample(topk, idx, ones, top_k_all, seeds, pos)

        def new_greedy():
            w = local_topk_candidates(local, k)
            wg = torch.cat([w] * TP).view(torch.bfloat16)
            tk = TopKLogits(wg, cap(wg), t, k, TP, shard)
            sink["tok"] = topk_merge_sample(tk, idx, zeros, top_k_none, seeds, pos)

        def new_t1k64():
            w = local_topk_candidates(local, k)
            wg = torch.cat([w] * TP).view(torch.bfloat16)
            tk = TopKLogits(wg, cap(wg), t, k, TP, shard)
            sink["tok"] = topk_merge_sample(tk, idx, ones, top_k_all, seeds, pos)

        def new_closure():
            fill_tie_closure(local, words_t_one, k, ties, token_base=0)

        def new_merge_threshold():
            sink["tok"] = topk_merge_sample(topk_t, idx, ones, top_k_all, seeds, pos)

        def new_t1k64_threshold():
            w = local_topk_candidates(local, k, extra_words=TIE_EXTRA_WORDS)
            fill_tie_closure(local, w, k, ties, token_base=0)
            wg = torch.cat([w] * TP).view(torch.bfloat16)
            tk = TopKLogits(wg, cap(wg), t, k, TP, shard, threshold=True)
            sink["tok"] = topk_merge_sample(tk, idx, ones, top_k_all, seeds, pos)

        row = {
            "tokens": t,
            "k": k,
            "bytes_per_rank_full": t * shard * 2,
            "bytes_per_rank_topk": t * 2 * k * 2,
            "bytes_per_rank_topk_threshold": t * (2 * k + TIE_EXTRA_WORDS) * 2,
        }
        for name, fn in [
            ("old_cap", old_cap),
            ("old_greedy", old_greedy),
            ("old_t1k64", old_t1k64),
            *([("old_t1k64_fi", old_t1k64_fi)] if flashinfer_sampler_supported() else []),
            ("new_local", new_local),
            ("new_cap", new_cap),
            ("new_merge_greedy", new_merge_greedy),
            ("new_merge_t1k64", new_merge_t1k64),
            ("new_greedy", new_greedy),
            ("new_t1k64", new_t1k64),
            ("new_closure", new_closure),
            ("new_merge_threshold", new_merge_threshold),
            ("new_t1k64_threshold", new_t1k64_threshold),
        ]:
            row[f"{name}_graph_us"] = round(time_graph(fn), 2)
            row[f"{name}_eager_us"] = round(time_eager(fn, flush), 2)
        print(json.dumps(row))
        rows_out.append(row)
    if args.out:
        with open(args.out, "w") as f:
            for row in rows_out:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
