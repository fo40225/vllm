# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark large-batch decode MoE: the plan on the router's aux stream.

One MoE layer of the decode graph of a TP8 + EP8 rank at a padded batch of 9 to 128
rows:

  main stream  the sharded pooler GEMM, the all-gather (a sleep of --ag-us: the
               collective is not part of this benchmark), the pre-expert RMSNorm
  aux stream   the fp32 cast copy of the router input, the fp32 router GEMM of the
               8192-wide input, a sleep of --route-us (the routing kernel: sigmoid, bias,
               top-k) that hands over the ids, and, in the ``aux_plan`` variants, the plan
               of the muse large-batch kernel
  then         the stream join and ``nvfp4_decode_moe_large``

Variants (``--variants``):

  base            the preparation launch (``k_prep``) builds the plan and quantizes
  aux_plan        the plan is built on the aux stream behind the top-k
                  (VLLM_MUSE_SPARK_SM120_AUX_PLAN=1); ``k_prep`` only quantizes
  norm_quant      the pre-expert RMSNorm and the quantization in one launch
                  (VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT=1); the kernel only builds
                  its plan in the preparation launch
  norm_quant_aux  both: norm + quantization in one launch, the plan on the aux stream
                  (no preparation launch at all)
  aux_plan_xq     UPPER BOUND of folding the quantization into the producer of the
                  activations: the plan as in ``aux_plan``, the quantization done once
                  outside the graph and handed to the kernel (no preparation launch
                  at all; the real fold costs a little inside the producer)

It reports, per batch size, the median over CUDA graph replays of (graph time / layers)
in us, ABBA-ordered over --rounds, and the standalone durations of the plan launch and the
preparation launch. Real routing statistics are not modelled: the ids are random
distinct experts per token over 256 experts (32 local), as in the unit tests.

    python benchmarks/kernels/benchmark_muse_spark_aux_plan.py --out rows.jsonl
"""

import argparse
import json
import re
import statistics

import torch
import torch.nn.functional as F

from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_rmsnorm_fwd
from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.models.muse_spark.nvidia.ops.muse_decode_moe import MuseDecodeMoE

HIDDEN = 8192
LATENT = 4096
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
FLAG_EXTERNAL_PLAN = 32
FLAG_EXTERNAL_XQ = 64
BATCHES = [9, 16, 24, 32, 48, 64, 96, 128]
LAYERS = 20


def _calibrate_sleep() -> float:
    """cycles of ``torch.cuda._sleep`` per microsecond."""
    cycles = 1_000_000
    torch.cuda._sleep(cycles)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(cycles)
    end.record()
    torch.cuda.synchronize()
    return cycles / (start.elapsed_time(end) * 1000.0)


class Layer:
    """Random expert weights of one EP8 rank (the contents do not change the timing:
    the kernel reads the same bytes for any weights)."""

    def __init__(self):
        dev = torch.device("cuda")
        self.w13 = torch.randint(
            0, 256, (E_LOCAL, 2 * LATENT, LATENT // 2), dtype=torch.uint8, device=dev
        )
        self.w2 = torch.randint(
            0, 256, (E_LOCAL, LATENT, LATENT // 2), dtype=torch.uint8, device=dev
        )
        ratio = torch.rand(E_LOCAL, 2 * LATENT, LATENT // 16, device=dev) * 0.5 + 0.5
        sf13 = (448 * ratio).to(torch.float8_e4m3fn)
        ratio = torch.rand(E_LOCAL, LATENT, LATENT // 16, device=dev) * 0.5 + 0.5
        sf2 = (448 * ratio).to(torch.float8_e4m3fn)
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            swizzle_blockscale,
        )

        self.sf13 = swizzle_blockscale(sf13)
        self.sf2 = swizzle_blockscale(sf2)
        self.prm = torch.rand(E_LOCAL, 4, device=dev) * 1e-3
        self.gs = torch.tensor([0.05, 1.0], device=dev)
        self.expert_map = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
        self.expert_map[:E_LOCAL] = torch.arange(E_LOCAL, dtype=torch.int32, device=dev)
        self.engine = MuseDecodeMoE(
            self.w13, self.sf13, self.w2, self.sf2, prm=self.prm, gs=self.gs
        )


class Case:
    def __init__(self, layer: Layer, t: int, ag_cycles: int, route_cycles: int):
        dev = torch.device("cuda")
        self.layer = layer
        self.t = t
        self.ag_cycles = ag_cycles
        self.route_cycles = route_cycles
        self.ids = torch.stack(
            [torch.randperm(E_GLOBAL, device=dev)[:TOP_K] for _ in range(t)]
        ).to(torch.int32)
        self.x = torch.randn(t, HIDDEN, dtype=torch.bfloat16, device=dev)
        self.pooler_w = (
            torch.randn(512, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02
        )
        self.norm_w = torch.randn(LATENT, dtype=torch.bfloat16, device=dev) * 0.1
        self.router_w = (
            torch.randn(E_GLOBAL, HIDDEN, dtype=torch.float32, device=dev) * 0.02
        )
        self.gathered = torch.zeros(t, LATENT, dtype=torch.bfloat16, device=dev)
        self.side = torch.cuda.Stream()
        self.fork, self.join = torch.cuda.Event(), torch.cuda.Event()
        self.pooled_ref = None

    def _pooler_chain(self, quantize: bool = False):
        shard = F.linear(self.x, self.pooler_w)  # [T, 512], this rank's columns
        self.gathered[:, :512].copy_(shard)
        torch.cuda._sleep(self.ag_cycles)  # the all-gather
        if quantize:
            return self.layer.engine.rmsnorm_quantize(
                self.gathered, self.norm_w, 1e-6, 1.0
            )
        return fused_rmsnorm_fwd(
            self.gathered, self.norm_w, 1e-6, gain_center=1.0, use_kahan_summation=True
        )

    def _router_chain(self):
        x32 = self.x.float()  # the cast copy of the router input
        logits = x32 @ self.router_w.t()
        torch.cuda._sleep(self.route_cycles)  # the routing kernel
        routed = torch.empty_like(self.ids)
        routed.copy_(self.ids)  # its ids (a dependency on the logits is not needed)
        return routed, logits

    def layer_step(self, variant: str):
        eng, emap = self.layer.engine, self.layer.expert_map
        self.fork.record()
        with torch.cuda.stream(self.side):
            self.fork.wait()
            ids, _ = self._router_chain()
            if variant in ("aux_plan", "aux_plan_xq", "norm_quant_aux"):
                eng.build_plan(ids, emap)
            self.join.record(self.side)
        pooled = self._pooler_chain(quantize=variant.startswith("norm_quant"))
        self.join.wait()
        if variant == "aux_plan_xq":
            eng.mark_input_quantized(pooled)
        return eng.run(pooled, ids, emap)


def _graph(case: Case, variant: str):
    engine = case.layer.engine
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        if variant == "aux_plan_xq":
            _prequantize(case)
        case.layer_step(variant)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    if variant == "aux_plan_xq":
        _prequantize(case)
        engine.discard_input()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAYERS):
            case.layer_step(variant)
    torch.cuda.synchronize()
    return graph


def _chain_graph(case: Case, which: str):
    """Only the main-stream chain or only the aux-stream chain of the layer."""
    fn = case._pooler_chain if which == "main" else case._router_chain
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(LAYERS):
            fn()
    torch.cuda.synchronize()
    return graph


def _prequantize(case: Case) -> None:
    """The activations the stand-in norm produces, quantized into the scratch once (the
    bytes do not change between layers: the same input every time)."""
    pooled = fused_rmsnorm_fwd(
        case.gathered, case.norm_w, 1e-6, gain_center=1.0, use_kahan_summation=True
    )
    case.layer.engine.input_scratch(case.t)
    data, sf = case.layer.engine.input_scratch(case.t)
    torch.ops._muse_C.nvfp4_decode_act_quant(
        pooled, case.layer.engine.input_scale, data, sf, False
    )


def _kernel_table(graph) -> dict[str, float]:
    """Mean duration (us) of the muse kernels in one replay of the graph, per layer."""
    from torch.profiler import ProfilerActivity, profile

    graph.replay()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    out: dict[str, list[float]] = {}
    for e in prof.events():
        match = re.search(r"k_prep|k_plan|k_persistent", e.name)
        if match:
            out.setdefault(match.group(0), []).append(e.device_time)
    return {k: round(statistics.mean(v), 2) for k, v in out.items()}


def _time(graph, replays: int) -> float:
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / LAYERS)
    return statistics.median(out)


def _kernel_us(fn, repeat=30) -> float:
    """Median duration of one launch of ``fn`` alone (a CUDA graph of 20 launches)."""
    graph = torch.cuda.CUDAGraph()
    fn()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    return _time_n(graph, repeat, 20)


def _time_n(graph, replays, n) -> float:
    graph.replay()
    torch.cuda.synchronize()
    out = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) * 1000.0 / n)
    return statistics.median(out)


@torch.inference_mode()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--batches", type=lambda s: [int(v) for v in s.split(",")], default=BATCHES
    )
    ap.add_argument(
        "--variants", default="base,aux_plan,norm_quant,norm_quant_aux,aux_plan_xq"
    )
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--replays", type=int, default=30)
    ap.add_argument("--ag-us", type=float, default=17.0, help="the pooler all-gather")
    ap.add_argument("--route-us", type=float, default=5.5, help="the routing kernel")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    muse_c.load()
    variants = args.variants.split(",")
    cycles_per_us = _calibrate_sleep()
    layer = Layer()
    rows = []
    for t in args.batches:
        case = Case(
            layer,
            t,
            int(args.ag_us * cycles_per_us),
            int(args.route_us * cycles_per_us),
        )
        graphs = {v: _graph(case, v) for v in variants}
        # ABBA rounds: the variants in alternating order.
        order = variants + variants[::-1]
        samples = {v: [] for v in variants}
        for r in range(args.rounds):
            for v in order:
                samples[v].append(_time(graphs[v], args.replays))
        row = {"T": t}
        for v in variants:
            row[v] = round(statistics.median(samples[v]), 2)
            row[v + "_kernels_us"] = _kernel_table(graphs[v])
        for which in ("main", "aux"):
            row[f"{which}_chain_us"] = round(_time(_chain_graph(case, which), 20), 2)
        # The standalone launches of the plan and of the preparation launch.
        u = torch.randn(t, LATENT, dtype=torch.bfloat16, device="cuda")
        ids = torch.stack(
            [torch.randperm(E_GLOBAL, device="cuda")[:TOP_K] for _ in range(t)]
        ).to(torch.int32)
        eng = layer.engine
        row["plan_launch_us"] = round(
            _kernel_us(lambda: eng.build_plan(ids, layer.expert_map)), 2
        )
        data, sf = eng.input_scratch(t)
        row["quant_launch_us"] = round(
            _kernel_us(
                lambda: torch.ops._muse_C.nvfp4_decode_act_quant(
                    u, eng.input_scale, data, sf, False
                )
            ),
            2,
        )
        for v in variants:
            if v != "base" and "base" in variants:
                row[f"delta_{v}"] = round(row[v] - row["base"], 2)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del graphs
    if args.out:
        with open(args.out, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
