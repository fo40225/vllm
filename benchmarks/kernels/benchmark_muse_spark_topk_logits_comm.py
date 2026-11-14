# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The all-gathers of the Muse Spark logits on 8 TP ranks: the full path against the
top-K candidates.

Times, through the TP group of vLLM (so through the host communicator plugin when it
is on the PYTHONPATH and enabled), per decode batch size T:

  full   ``tensor_model_parallel_all_gather`` of the [T, 25256] bf16 logits shard on the
         last dim, what ``LogitsProcessor`` does
  topk   the all-gather on dim 0 of the [T, 2 * K] 16-bit candidate words of
         ``MuseSpark1xForCausalLM.compute_topk_logits``

in us per call (median of CUDA-event times of back-to-back calls on the stream, the
way the model issues them: eagerly), rank 0.

    python benchmarks/kernels/benchmark_muse_spark_topk_logits_comm.py --out rows.jsonl
"""

import argparse
import json
import statistics

import torch
import torch.multiprocessing as mp

VOCAB = 202048
TP = 8
K = 64


def _worker(rank: int, port: int, tokens: list[int], out: str | None) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
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
    shard = VOCAB // TP
    rows = []
    for t in tokens:
        logits = torch.randn(t, shard, device="cuda").to(torch.bfloat16)
        words = torch.randn(t, 2 * K, device="cuda").to(torch.bfloat16)
        calls = {
            "full": lambda: tensor_model_parallel_all_gather(logits, dim=-1),
            "topk": lambda: tensor_model_parallel_all_gather(words, dim=0),
        }
        row = {"tokens": t}
        for name, call in calls.items():
            for _ in range(20):
                call()
            torch.cuda.synchronize()
            torch.distributed.barrier()
            times = []
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(20):
                start.record()
                for _ in range(10):
                    call()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 100)
            row[f"{name}_us"] = round(statistics.median(times), 2)
            sent = logits if name == "full" else words
            row[f"{name}_bytes_per_rank"] = sent.numel() * sent.element_size()
        if rank == 0:
            print(json.dumps(row), flush=True)
            rows.append(row)
        torch.distributed.barrier()
    if rank == 0 and out:
        with open(out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 128, 256])
    parser.add_argument("--port", type=int, default=29850)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    mp.spawn(_worker, args=(args.port, args.tokens, args.out), nprocs=TP)


if __name__ == "__main__":
    main()
