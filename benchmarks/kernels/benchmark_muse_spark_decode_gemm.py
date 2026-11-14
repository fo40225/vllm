# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Skinny decode GEMMs of Muse Spark: cuBLAS (F.linear) against ``decode_gemm``.

The three projections of a TP8 rank whose cuBLAS GEMM ends in a split-K reduction
kernel (``cublasLt::splitKreduce_kernel``) in the decode graphs: the sharded pooler
(``[T, 8192] x [512, 8192]^T``), the sharded unpooler (``[T, 4096] x [1024, 4096]^T``)
and the QKV projection (``[T, 8192] x [2304, 8192]^T``), at the CUDA graph capture
sizes, against ``_muse_C::decode_gemm`` (one kernel, K split over a thread block
cluster).

Per (projection, rows) it reports, in us per call:
  hot   median over CUDA graph replays of (graph time / 62 back-to-back calls) with one
        weight (L2 resident, like the weights that the L2 prefetch of the model has
        pulled in), the graph node gap included;
  cold  the same with 62 different weights (the weights come from HBM);
the kernels cuBLAS launches for the case (names and CUPTI durations), and the accuracy
of both products against the exact fp64 product of the same bf16 operands: mean and max
error in bf16 ulps of the exact value, the fraction of elements that are correctly
rounded up to the bound of an fp32 accumulation (4e-6 of sum |x||w|), and the largest
error relative to that bound. The order of the additions of ``decode_gemm`` is fixed,
so its bits are those of every run.

    python benchmarks/kernels/benchmark_muse_spark_decode_gemm.py --out rows.jsonl
"""

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from vllm.models.muse_spark.nvidia.ops import muse_c

SHAPES = {  # name: (N, K) of a TP8 rank
    "pooler": (512, 8192),
    "unpooler": (1024, 4096),
    "qkv": (2304, 8192),
}
ROWS = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 96, 128]
LAYERS = 62
REPLAYS = 30


def capture(fn, n):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(n):
            fn(i)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(n):
            fn(i)
    return g


def time_graph(fn):
    g = capture(fn, LAYERS)
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


def kernels_of(fn):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    g = capture(fn, 1)
    g.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        g.replay()
        torch.cuda.synchronize()
    names = [
        (ev.name[:64], round(ev.device_time, 2))
        for ev in prof.events()
        if ev.device_type == DeviceType.CUDA
    ]
    del g
    return names


def ulp(ref):
    exp = torch.floor(torch.log2(ref.abs().clamp_min(2.0**-100)))
    return torch.exp2(exp - 7)


def accuracy(got, x, w):
    ref = x.double() @ w.double().t()
    err = (got.double() - ref).abs() / ulp(ref)
    slack = 4e-6 * (x.double().abs() @ w.double().abs().t()) / ulp(ref)
    return {
        "mean_ulp": float(err.mean()),
        "max_ulp": float(err.max()),
        "correctly_rounded": float((err < 0.5 + slack).float().mean()),
        "max_over_bound": float((err / (0.5 + slack)).max()),
    }


class Buffers:
    """One weight set (hot) or one per layer (cold)."""

    def __init__(self, n, k, rows, hot):
        sets = 1 if hot else LAYERS
        self.sets = sets
        self.x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        self.w = [
            (torch.randn(n, k, device="cuda") * 0.02).bfloat16() for _ in range(sets)
        ]
        self.out = [
            torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
            for _ in range(sets)
        ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    args = ap.parse_args()
    muse_c.load()
    op = torch.ops._muse_C.decode_gemm
    scratch = torch.empty(4 << 20, dtype=torch.float32, device="cuda")
    torch.manual_seed(0)
    t0 = time.time()
    with open(args.out, "w") as f:
        for name, (n, k) in SHAPES.items():
            for rows in [int(v) for v in args.rows.split(",")]:
                hot, cold = Buffers(n, k, rows, True), Buffers(n, k, rows, False)
                row = {"proj": name, "rows": rows, "n": n, "k": k}

                def cublas(b):
                    return lambda i: F.linear(b.x, b.w[i % b.sets])

                def own(b):
                    return lambda i: op(
                        b.x,
                        b.w[i % b.sets],
                        b.out[i % b.sets],
                        scratch,
                        -1,
                        False,
                        False,
                    )

                row["cublas_hot_us"] = time_graph(cublas(hot))
                row["cublas_cold_us"] = time_graph(cublas(cold))
                row["own_hot_us"] = time_graph(own(hot))
                row["own_cold_us"] = time_graph(own(cold))
                row["cublas_kernels"] = kernels_of(cublas(hot))
                row["own_kernels"] = kernels_of(own(hot))
                # Accuracy and determinism on fresh data.
                x = torch.randn(rows, k, device="cuda").bfloat16()
                w = (torch.randn(n, k, device="cuda") * 0.02).bfloat16()
                got = torch.empty(rows, n, device="cuda", dtype=torch.bfloat16)
                op(x, w, got, scratch, -1, False, False)
                again = torch.empty_like(got)
                op(x, w, again, scratch, -1, False, False)
                row["own_deterministic"] = bool(torch.equal(got, again))
                row["own_accuracy"] = accuracy(got, x, w)
                row["cublas_accuracy"] = accuracy(F.linear(x, w), x, w)
                for m in ("hot", "cold"):
                    row[f"save_{m}_us"] = row[f"cublas_{m}_us"] - row[f"own_{m}_us"]
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(
                    f"[{time.time() - t0:5.0f}s] {name:8s} T={rows:3d}  hot "
                    f"{row['cublas_hot_us']:6.2f} -> {row['own_hot_us']:6.2f}  cold "
                    f"{row['cublas_cold_us']:6.2f} -> {row['own_cold_us']:6.2f}  "
                    f"err mean/max ulp {row['cublas_accuracy']['mean_ulp']:.2f}/"
                    f"{row['cublas_accuracy']['max_ulp']:.0f} -> "
                    f"{row['own_accuracy']['mean_ulp']:.2f}/"
                    f"{row['own_accuracy']['max_ulp']:.0f}",
                    flush=True,
                )
                del hot, cold


if __name__ == "__main__":
    main()
