# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch (PDL) of the Muse Spark Triton kernels.

A kernel launched as a programmatic dependent may start before its predecessor
has finished, so it must wait (``gdc_wait``) before it reads anything a kernel of
the step writes. Every test puts a deliberately slow producer in front of the
kernel under test: the producer lets its dependents launch at once and writes the
kernel's inputs only after a long spin, over a poison value. A kernel that reads
before its wait sees the poison and misses the reference, which is the result of
the same kernel launched without PDL (``VLLM_MUSE_SPARK_SM120_PDL=0``).
"""

import pytest
import torch

from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
)
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
    fused_rmsnorm_fwd,
)
from vllm.models.muse_spark.common.ops.fused_routing import (
    fused_musespark1x_routing,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

if not (current_platform.is_cuda() and current_platform.is_arch_support_pdl()):
    pytest.skip("Requires a GPU with PDL support", allow_module_level=True)

HIDDEN, EXPERTS, TOP_K = 8192, 256, 8
LATENT = 4096
# About 60 us of dependent FMAs on an SM120.
SPIN = 20_000
POISON = 7.0


@triton.jit
def _slow_copy_kernel(dst_ptr, src_ptr, n, SPIN: tl.constexpr, BLOCK: tl.constexpr):
    # Let the dependents of this kernel launch right now ...
    tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(SPIN):
        acc = acc * 0.999 + 1.0
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # ... and write the data only after the spin. acc stays below 1e3, so the
    # second mask term is always true; it keeps the spin from being removed.
    mask = (offs < n) & (acc < 1e30)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


def slow_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src)`` that finishes late but triggers its dependents early."""
    assert dst.is_contiguous() and src.is_contiguous() and dst.dtype == src.dtype
    block = 1024
    _slow_copy_kernel[(triton.cdiv(src.numel(), block),)](
        dst.view(-1), src.view(-1), src.numel(), SPIN=SPIN, BLOCK=block
    )


def poison_like(t: torch.Tensor) -> torch.Tensor:
    return torch.full_like(t, POISON)


def run(step, outputs_of, use_graph: bool, poisons):
    """Runs ``step`` eagerly, or replayed from a CUDA graph over re-poisoned inputs.

    ``poisons`` are (tensor, poison value) pairs written back before the run, so
    that only the producer inside ``step`` can make the right data appear.
    """
    for t, p in poisons:
        t.copy_(p)
    step()  # compiles, and is the eager result
    torch.cuda.synchronize()
    if not use_graph:
        for t, p in poisons:
            t.copy_(p)
        got = step()
        torch.cuda.synchronize()
        return outputs_of(got)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = step()
    for _ in range(3):
        for t, p in poisons:
            t.copy_(p)
        graph.replay()
        torch.cuda.synchronize()
    return outputs_of(got)


def boundary_inputs(rows: int):
    g = torch.Generator(device="cuda").manual_seed(rows)
    branch = torch.randn(rows, HIDDEN, device="cuda", generator=g).bfloat16()
    stream = torch.randn(rows, HIDDEN, device="cuda", generator=g)
    kw = dict(
        post_weight=(torch.randn(HIDDEN, device="cuda", generator=g) * 0.1).bfloat16(),
        pre_weight=(torch.randn(HIDDEN, device="cuda", generator=g) * 0.1).bfloat16(),
        post_eps=1e-8,
        pre_eps=1e-5,
        alpha=torch.rand(HIDDEN, device="cuda", generator=g),
        beta=torch.rand(HIDDEN, device="cuda", generator=g),
        post_gain_center=1.0,
        pre_gain_center=1.0,
        post_input_scale=True,
        out_dtype=torch.bfloat16,
    )
    return branch, stream, kw


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 300])
@torch.inference_mode()
def test_boundary_waits_for_its_producer(monkeypatch, rows: int, use_graph: bool):
    branch_src, stream, kw = boundary_inputs(rows)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "0")
    ref = fused_norm_residual_norm_fwd(branch_src, stream, **kw)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "1")

    branch = poison_like(branch_src)

    def step():
        slow_copy(branch, branch_src)
        return fused_norm_residual_norm_fwd(branch, stream, **kw)

    out, new_stream = run(
        step, lambda r: tuple(x.clone() for x in r), use_graph, [(branch, POISON)]
    )
    assert torch.equal(out, ref[0])
    assert torch.equal(new_stream, ref[1])


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_boundary_waits_for_a_stream_written_by_the_step(
    monkeypatch, rows: int, use_graph: bool
):
    """The residual stream is also a product of the step (the last boundary)."""
    branch, stream_src, kw = boundary_inputs(rows)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "0")
    ref = fused_norm_residual_norm_fwd(branch, stream_src, **kw)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "1")

    stream = poison_like(stream_src)

    def step():
        slow_copy(stream, stream_src)
        return fused_norm_residual_norm_fwd(branch, stream, **kw)

    out, new_stream = run(
        step, lambda r: tuple(x.clone() for x in r), use_graph, [(stream, POISON)]
    )
    assert torch.equal(out, ref[0])
    assert torch.equal(new_stream, ref[1])


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_rmsnorm_and_routing_chain_waits_for_the_producer(
    monkeypatch, rows: int, use_graph: bool
):
    """producer -> pre-expert-norm-like RMSNorm -> router top-k, all dependents."""
    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    gating_src = torch.randn(rows, EXPERTS, device="cuda", generator=g)
    norm_weight = (torch.randn(EXPERTS, device="cuda", generator=g) * 0.1).bfloat16()
    bias = torch.randn(EXPERTS, device="cuda", generator=g)
    weights = torch.empty(rows, TOP_K, device="cuda")
    ids = torch.empty(rows, TOP_K, device="cuda", dtype=torch.int32)

    def chain(gating):
        scores = fused_rmsnorm_fwd(
            gating, norm_weight, 1e-6, gain_center=1.0, out_dtype=torch.float32
        )
        return fused_musespark1x_routing(
            scores, bias, TOP_K, 1.0, topk_weights=weights, topk_ids=ids
        )

    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "0")
    ref = tuple(x.clone() for x in chain(gating_src))
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "1")

    gating = poison_like(gating_src)

    def step():
        slow_copy(gating, gating_src)
        return chain(gating)

    got = run(
        step, lambda r: tuple(x.clone() for x in r), use_graph, [(gating, POISON)]
    )
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1], ref[1])


@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_finalize_waits_for_the_experts(monkeypatch, rows: int, use_graph: bool):
    g = torch.Generator(device="cuda").manual_seed(200 + rows)
    num_perm = 8 * rows
    routed_src = torch.randn(num_perm, LATENT, device="cuda", generator=g).bfloat16()
    tw_src = torch.rand(rows, TOP_K, device="cuda", generator=g)
    idx_src = torch.arange(num_perm, device="cuda", dtype=torch.int32).view(rows, 8)
    idx_src[:, 3] = -1  # a remote route
    weight = (torch.randn(LATENT, device="cuda", generator=g) * 0.1).bfloat16()
    out = torch.empty(rows, LATENT, device="cuda", dtype=torch.bfloat16)

    def finalize(routed, tw, idx):
        return fused_post_expert_finalize(
            routed, weight, tw, 1e-8, 1.0, True, out, permuted_idx=idx
        )

    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "0")
    ref = finalize(routed_src, tw_src, idx_src).clone()
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", "1")

    routed, tw = poison_like(routed_src), poison_like(tw_src)
    idx = torch.full_like(idx_src, 5)

    def step():
        slow_copy(routed, routed_src)
        slow_copy(tw, tw_src)
        slow_copy(idx, idx_src)
        return finalize(routed, tw, idx)

    got = run(
        step,
        lambda r: r.clone(),
        use_graph,
        [(routed, POISON), (tw, POISON), (idx, torch.full_like(idx_src, 5))],
    )
    assert torch.equal(got, ref)


@pytest.mark.parametrize("rows", [1, 8, 300])
@torch.inference_mode()
def test_pdl_launches_are_bitwise_identical_to_plain_launches(monkeypatch, rows: int):
    """Without a producer in front, PDL on and off give the same bits."""
    branch, stream, kw = boundary_inputs(rows)
    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_PDL", flag)
        for _ in range(2):
            results[flag] = fused_norm_residual_norm_fwd(branch, stream, **kw)
    assert torch.equal(results["0"][0], results["1"][0])
    assert torch.equal(results["0"][1], results["1"][1])
