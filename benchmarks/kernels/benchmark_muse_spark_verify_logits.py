# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The verify step of Muse Spark speculative decoding on 8 TP ranks: from the gathered
logits (the full path) against from the top-K logits of every rank.

Both paths run, per rank and through the TP group of vLLM (so through the host
communicator plugin when it is on the PYTHONPATH and enabled), what a verify step does
between the last hidden states and the sampled tokens of ``num_reqs`` requests with 7
draft tokens each (rows = 8 * num_reqs):

  old  lm_head GEMM [rows, 8192] x [8192, 25256] -> all-gather of the bf16 shard on the
       last dim -> soft cap (fp32 [rows, 202048]) -> ``apply_sampling_params`` as the
       rejection sampler runs it (fp32 copy and top-k mask for the sampled scenario,
       nothing for greedy) -> ``rejection_sample`` (block method, draft logits)
  new  the same GEMM -> ``local_topk_candidates`` (K = 1 for greedy, 64 for sampled) ->
       all-gather of the candidate words on dim 0 -> soft cap of the words ->
       ``topk_verify``

and prints the median CUDA-event time of the whole step and of its stages, rank 0, in
us, for the scenarios ``greedy`` (every request temperature 0) and ``t1k64`` (every
request temperature 1, top_k 64), eager, the way the sampler runs (launch gaps
included), with the L2 flushed and the ranks synchronized before each step. The drafts
are right for about 3 of 7 steps; the outputs of the two paths are compared once per
configuration (greedy: identical; t1k64: the share of requests with the same tokens).

    runsrc_topo.sh benchmarks/kernels/benchmark_muse_spark_verify_logits.py \\
        --out rows.jsonl
"""

import argparse
import json
import statistics
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

VOCAB = 202048
TP = 8
HIDDEN = 8192
STEPS = 7
MULTIPLIER = 0.17677669529663687
SOFT_CAP = 20.0


def _inputs(num_reqs: int, weight: torch.Tensor, cap, gather, logit_std: float):
    """The hidden states, the drafts (tokens and draft logits) and the request state of
    ``num_reqs`` requests with ``STEPS`` draft tokens each, identical on every rank."""
    dev = "cuda"
    rows = num_reqs * (STEPS + 1)
    g = torch.Generator(device=dev).manual_seed(7 + num_reqs)
    hidden = torch.randn(rows, HIDDEN, generator=g, device=dev).to(torch.bfloat16)
    hidden *= logit_std / (HIDDEN**0.5 * 0.02)
    full = cap(gather(torch.nn.functional.linear(hidden, weight), dim=-1))
    # Drafts: the argmax with probability 3/4 per step, else one of the 8 best tokens
    # drawn with weights of the target plus noise; the draft distribution lives on
    # those 8.
    top = full.topk(8, dim=-1).indices
    vals = full.gather(1, top) + 0.3 * torch.randn(top.shape, generator=g, device=dev)
    pick = torch.multinomial(vals.softmax(-1), 1, generator=g).squeeze(1)
    pick_argmax = torch.rand(rows, generator=g, device=dev) < 0.75
    pick = torch.where(pick_argmax, torch.zeros_like(pick), pick)
    draft_token = top.gather(1, pick[:, None]).squeeze(1)
    draft_sampled = torch.zeros(rows, dtype=torch.int64, device=dev)
    row_idx = torch.arange(rows, device=dev)
    local_pos = row_idx % (STEPS + 1)
    not_last = local_pos < STEPS
    draft_sampled[row_idx[not_last] + 1] = draft_token[not_last]
    draft_logits = torch.full(
        (num_reqs, STEPS, VOCAB), float("-inf"), dtype=torch.float32, device=dev
    )
    req_of_row = row_idx // (STEPS + 1)
    for c in range(top.shape[1]):
        draft_logits[req_of_row[not_last], local_pos[not_last], top[not_last, c]] = (
            vals[not_last, c]
        )
    idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device=dev)
    base_pos = torch.randint(1, 100000, (num_reqs,), generator=g, device=dev)
    return SimpleNamespace(
        rows=rows,
        hidden=hidden,
        draft_sampled=draft_sampled,
        draft_logits=draft_logits,
        idx_mapping=idx_mapping,
        expanded_idx=idx_mapping.repeat_interleave(STEPS + 1),
        expanded_pos=local_pos.to(torch.int32),
        cu=torch.arange(num_reqs + 1, dtype=torch.int32, device=dev) * (STEPS + 1),
        pos=base_pos[req_of_row] + local_pos,
        seeds=torch.randint(-(1 << 60), 1 << 60, (num_reqs,), generator=g, device=dev),
    )


def _steps(ctx, num_reqs: int, sampled_rows: bool, weight, cap, gather, marks, sink):
    """The two verify steps as closures: ``old`` (full logits) and ``new`` (top-K)."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.sample.topk_logits import (
        TopKLogits,
        local_topk_candidates,
    )
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )
    from vllm.v1.worker.gpu.spec_decode.topk_rejection import topk_verify

    dev = "cuda"
    temperature = torch.full((num_reqs,), 1.0 if sampled_rows else 0.0, device=dev)
    top_k = torch.full((num_reqs,), 64 if sampled_rows else VOCAB, device=dev)
    top_k = top_k.to(torch.int32)
    k = 64 if sampled_rows else 1
    top_k_rows = top_k[ctx.expanded_idx.long()]

    def mark():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        marks.append(e)

    def old_step():
        local = torch.nn.functional.linear(ctx.hidden, weight)
        mark()
        logits = gather(local, dim=-1)
        mark()
        logits = cap(logits)
        if sampled_rows:
            # apply_sampling_params: fp32 copy, temperature 1 skipped, top-k.
            logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
            apply_temperature(logits, ctx.expanded_idx, temperature)
            logits = apply_top_k_top_p(logits, top_k_rows, None)
        mark()
        sink["old"] = rejection_sample(
            logits,
            ctx.draft_logits,
            ctx.draft_sampled,
            ctx.cu,
            ctx.pos,
            ctx.idx_mapping,
            ctx.expanded_idx,
            ctx.expanded_pos,
            temperature,
            ctx.seeds,
            STEPS,
            None,
            use_fp64=False,
            use_block_verification=True,
        )
        mark()

    def new_step():
        local = torch.nn.functional.linear(ctx.hidden, weight)
        mark()
        words = local_topk_candidates(local, k)
        words = gather(words.view(torch.bfloat16), dim=0)
        mark()
        topk = TopKLogits(
            words=words,
            capped=cap(words.contiguous()),
            num_rows=ctx.rows,
            k=k,
            tp_size=TP,
            shard_size=VOCAB // TP,
        )
        mark()
        sink["new"] = topk_verify(
            topk,
            ctx.draft_sampled,
            ctx.pos,
            ctx.cu,
            ctx.idx_mapping,
            ctx.expanded_idx,
            ctx.expanded_pos,
            temperature,
            top_k,
            ctx.seeds,
            STEPS,
            draft_logits=ctx.draft_logits,
            has_sampled=sampled_rows,
        )
        mark()

    return mark, {"old": old_step, "new": new_step}


def _worker(rank: int, port: int, args) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    from vllm.models.muse_spark.common.layers import MulTanhSoftCap
    from vllm.plugins import load_general_plugins

    torch.cuda.set_device(rank)
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=TP))
    with set_current_vllm_config(config):
        load_general_plugins()
        init_distributed_environment(
            world_size=TP,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=TP)
    cap = MulTanhSoftCap(MULTIPLIER, SOFT_CAP)
    dev = "cuda"
    g_rank = torch.Generator(device=dev).manual_seed(1000 + rank)
    weight = (torch.randn(VOCAB // TP, HIDDEN, generator=g_rank, device=dev) * 0.02).to(
        torch.bfloat16
    )
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    rows_out = []
    for num_reqs in args.reqs:
        ctx = _inputs(
            num_reqs, weight, cap, tensor_model_parallel_all_gather, args.logit_std
        )
        for scenario in args.scenarios:
            sampled_rows = scenario != "greedy"
            marks: list = []
            sink: dict = {}
            mark, steps = _steps(
                ctx,
                num_reqs,
                sampled_rows,
                weight,
                cap,
                tensor_model_parallel_all_gather,
                marks,
                sink,
            )
            row = {"scenario": scenario, "reqs": num_reqs, "rows": ctx.rows}
            for name, step in steps.items():
                for _ in range(5):
                    marks.clear()
                    step()
                torch.cuda.synchronize()
                times: list[list[float]] = []
                for _ in range(args.iters):
                    torch.distributed.barrier()
                    flush.fill_(1)
                    torch.cuda.synchronize()
                    marks.clear()
                    mark()
                    step()
                    marks[-1].synchronize()
                    times.append(
                        [
                            marks[i].elapsed_time(marks[i + 1]) * 1000
                            for i in range(len(marks) - 1)
                        ]
                    )
                stage = [statistics.median(col) for col in zip(*times)]
                total = statistics.median(sum(t) for t in times)
                row[f"{name}_total_us"] = round(total, 1)
                row[f"{name}_stage_us"] = [round(x, 1) for x in stage]
            old_s, old_n = sink["old"]
            new_s, new_n = sink["new"]
            same = [
                old_s[r, : int(old_n[r])].tolist() == new_s[r, : int(new_n[r])].tolist()
                for r in range(num_reqs)
            ]
            row["same_tokens_share"] = round(sum(same) / num_reqs, 4)
            row["mean_accepted"] = round(float(old_n.float().mean()) - 1, 2)
            row["gain_us"] = round(row["old_total_us"] - row["new_total_us"], 1)
            if rank == 0:
                print(json.dumps(row), flush=True)
                rows_out.append(row)
            torch.distributed.barrier()
    if rank == 0 and args.out:
        with open(args.out, "w") as f:
            for row in rows_out:
                f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reqs", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["greedy", "t1k64"],
        choices=["greedy", "t1k64"],
    )
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument(
        "--logit-std",
        type=float,
        default=8.0,
        help="std of the bf16 logits (the larger, the more peaked the target)",
    )
    parser.add_argument("--port", type=int, default=29870)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.port, args), nprocs=TP)


if __name__ == "__main__":
    main()
