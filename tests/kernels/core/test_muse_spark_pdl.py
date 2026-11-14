# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch (PDL) of the Muse Spark kernels.

A kernel launched as a programmatic dependent may start before its predecessor
has finished, so it must wait (``gdc_wait``) before it reads anything a kernel of
the step writes. Every test puts a deliberately slow producer in front of the
kernel under test: the producer lets its dependents launch at once and writes ALL
the kernel's inputs in one kernel only after a long spin, over a poison value (see
``tests/kernels/muse_spark_pdl_utils.py``). A kernel that reads any of them before
its wait sees the poison and misses the reference, which is the result of the same
kernel launched without PDL (``VLLM_MUSE_SPARK_SM120_PDL=0``).

PDL is limited to SM120 (``muse_pdl_enabled``): the launch-parameter tests at the
end check that the other GPUs keep their previous launches, and run anywhere.
"""

import subprocess
import sys

import pytest
import torch

from tests.kernels.muse_spark_pdl_utils import slow_copy
from vllm.models.muse_spark.common.ops import fused_finalize as finalize_module
from vllm.models.muse_spark.common.ops import fused_rmsnorm as rmsnorm_module
from vllm.models.muse_spark.common.ops import fused_routing as routing_module
from vllm.models.muse_spark.common.ops import muse_pdl_enabled
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

FLAG = "VLLM_MUSE_SPARK_SM120_PDL"
requires_sm120 = pytest.mark.skipif(
    not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    ),
    reason="Requires an SM120 GPU (PDL is limited to it)",
)

HIDDEN, EXPERTS, TOP_K = 8192, 256, 8
LATENT = 4096
POISON = 7.0


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


def boundary_inputs(rows: int, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(rows)
    branch = torch.randn(rows, HIDDEN, device=device, generator=g).bfloat16()
    stream = torch.randn(rows, HIDDEN, device=device, generator=g)
    kw = dict(
        post_weight=(torch.randn(HIDDEN, device=device, generator=g) * 0.1).bfloat16(),
        pre_weight=(torch.randn(HIDDEN, device=device, generator=g) * 0.1).bfloat16(),
        post_eps=1e-8,
        pre_eps=1e-5,
        alpha=torch.rand(HIDDEN, device=device, generator=g),
        beta=torch.rand(HIDDEN, device=device, generator=g),
        post_gain_center=1.0,
        pre_gain_center=1.0,
        post_input_scale=True,
        out_dtype=torch.bfloat16,
    )
    return branch, stream, kw


@requires_sm120
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 300])
@torch.inference_mode()
def test_boundary_waits_for_its_producer(monkeypatch, rows: int, use_graph: bool):
    """The branch (an all-reduce or all-gather output) and the residual stream
    (written by the previous boundary) both come from the step."""
    branch_src, stream_src, kw = boundary_inputs(rows)
    monkeypatch.setenv(FLAG, "0")
    ref = fused_norm_residual_norm_fwd(branch_src, stream_src, **kw)
    monkeypatch.setenv(FLAG, "1")

    branch, stream = poison_like(branch_src), poison_like(stream_src)

    def step():
        slow_copy((branch, branch_src), (stream, stream_src))
        return fused_norm_residual_norm_fwd(branch, stream, **kw)

    out, new_stream = run(
        step,
        lambda r: tuple(x.clone() for x in r),
        use_graph,
        [(branch, POISON), (stream, POISON)],
    )
    assert torch.equal(out, ref[0])
    assert torch.equal(new_stream, ref[1])


@requires_sm120
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

    monkeypatch.setenv(FLAG, "0")
    ref = tuple(x.clone() for x in chain(gating_src))
    monkeypatch.setenv(FLAG, "1")

    gating = poison_like(gating_src)

    def step():
        slow_copy((gating, gating_src))
        return chain(gating)

    got = run(
        step, lambda r: tuple(x.clone() for x in r), use_graph, [(gating, POISON)]
    )
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1], ref[1])


@requires_sm120
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("rows", [1, 8])
@torch.inference_mode()
def test_finalize_waits_for_the_experts(monkeypatch, rows: int, use_graph: bool):
    """The rows, the top-k weights and the route map are all written by the step."""
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

    monkeypatch.setenv(FLAG, "0")
    ref = finalize(routed_src, tw_src, idx_src).clone()
    monkeypatch.setenv(FLAG, "1")

    routed, tw = poison_like(routed_src), poison_like(tw_src)
    idx = torch.full_like(idx_src, 5)

    def step():
        slow_copy((routed, routed_src), (tw, tw_src), (idx, idx_src))
        return finalize(routed, tw, idx)

    got = run(
        step,
        lambda r: r.clone(),
        use_graph,
        [(routed, POISON), (tw, POISON), (idx, torch.full_like(idx_src, 5))],
    )
    assert torch.equal(got, ref)


@requires_sm120
@pytest.mark.parametrize("kernel", ["boundary", "finalize", "chain"])
@pytest.mark.parametrize("rows", [1, 8, 300])
@torch.inference_mode()
def test_pdl_launches_are_bitwise_identical_to_plain_launches(
    monkeypatch, kernel: str, rows: int
):
    """Without a producer in front, PDL on and off give the same bits."""
    g = torch.Generator(device="cuda").manual_seed(300 + rows)
    branch, stream, kw = boundary_inputs(rows)
    routed = torch.randn(8 * rows, LATENT, device="cuda", generator=g).bfloat16()
    tw = torch.rand(rows, TOP_K, device="cuda", generator=g)
    idx = torch.arange(8 * rows, device="cuda", dtype=torch.int32).view(rows, 8)
    idx[:, 5] = -1
    weight = (torch.randn(LATENT, device="cuda", generator=g) * 0.1).bfloat16()
    gating = torch.randn(rows, EXPERTS, device="cuda", generator=g)
    bias = torch.randn(EXPERTS, device="cuda", generator=g)
    norm_weight = (torch.randn(EXPERTS, device="cuda", generator=g) * 0.1).bfloat16()

    def launch():
        if kernel == "boundary":
            return fused_norm_residual_norm_fwd(branch, stream, **kw)
        if kernel == "finalize":
            out = torch.empty(rows, LATENT, device="cuda", dtype=torch.bfloat16)
            return (
                fused_post_expert_finalize(
                    routed, weight, tw, 1e-8, 1.0, True, out, permuted_idx=idx
                ),
            )
        scores = fused_rmsnorm_fwd(
            gating, norm_weight, 1e-6, gain_center=1.0, out_dtype=torch.float32
        )
        return fused_musespark1x_routing(scores, bias, TOP_K, 1.0)

    results = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        for _ in range(2):
            results[flag] = tuple(x.clone() for x in launch())
    for off, on in zip(results["0"], results["1"], strict=True):
        assert torch.equal(off, on)


@triton.jit
def _consume_kernel(dst_ptr, src_ptr, n, BLOCK: tl.constexpr):
    # A successor like the attention kernel: it waits for its predecessor
    # before it reads what that wrote.
    tl.extra.cuda.gdc_wait()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@requires_sm120
@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("use_graph", [False, True])
@pytest.mark.parametrize("tokens", [1, 8, 300])
@torch.inference_mode()
def test_qknorm_op_waits_and_its_successor_sees_the_final_rows(
    monkeypatch, tokens: int, use_graph: bool, flag: str
):
    """producer -> QK-norm/RoPE/KV-insert op -> a waiting consumer. The op triggers
    its successor at its start (flag 1) or at its end (flag 0, read by the op's
    launcher); either way the bits are those of the plain launch."""
    from vllm.models.muse_spark.nvidia.ops import muse_c

    if (reason := muse_c.preflight()) is not None:
        pytest.skip(f"_muse_C is unavailable: {reason}")
    muse_c.load()
    op = torch.ops._muse_C.fused_muse_spark_qknorm_rope_kv_insert
    nq, nkv, head_dim = 16, 2, 64
    width = (nq + 2 * nkv) * head_dim
    g = torch.Generator(device="cuda").manual_seed(400 + tokens)
    qkv_src = torch.randn(tokens, width, device="cuda", generator=g).bfloat16()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    out = torch.empty_like(qkv_src)

    def norm(qkv):
        # NoPE, no cache bound: the op only normalizes q and k in place.
        op(qkv, positions, nq, nkv, head_dim, 1e-5, None, 0, None, None, 0, "auto")

    def consume():
        _consume_kernel[(triton.cdiv(out.numel(), 1024),)](
            out.view(-1), qkv.view(-1), out.numel(), BLOCK=1024, launch_pdl=True
        )

    monkeypatch.setenv(FLAG, "0")
    ref = qkv_src.clone()
    norm(ref)

    monkeypatch.setenv(FLAG, flag)
    qkv = poison_like(qkv_src)

    def step():
        slow_copy((qkv, qkv_src))
        norm(qkv)
        consume()
        return out

    got = run(step, lambda r: r.clone(), use_graph, [(qkv, POISON)])
    assert torch.equal(got, ref)


@requires_sm120
def test_synchronize_or_exit_ends_a_process_with_a_hung_kernel():
    """The watchdog of the tests whose kernel hangs for good when it skips a wait:
    a kernel that outlives the deadline ends the process with status 3."""
    code = """
import torch
import tests.kernels.muse_spark_pdl_utils as utils

dst = torch.zeros(1024, device="cuda")
src = torch.ones(1024, device="cuda")
utils.SPIN = 500_000_000  # many minutes
utils.slow_copy((dst, src))
utils.synchronize_or_exit("the hung launch", timeout_s=1.0)
print("NOT REACHED")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 3, (result.returncode, result.stdout, result.stderr)
    assert "the hung launch did not finish" in result.stderr
    assert "NOT REACHED" not in result.stdout


# --- PDL is limited to SM120 --------------------------------------------------


class _KernelSpy:
    """Stands in for a Triton kernel and records the keyword arguments of a launch."""

    def __init__(self):
        self.launches: list[dict] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append(kwargs)

        return launch


def _pretend_gpu(monkeypatch, family: int, flag: str = "1") -> None:
    """Make the platform look like an SM<family> GPU (any SM90+ GPU supports PDL)."""
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_arch_support_pdl", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda capability, device_id=0: capability == family,
    )


def _launch_params(monkeypatch) -> dict[str, dict]:
    """The keyword arguments of one launch of every kernel the flag touches."""
    spies = {
        "boundary": (rmsnorm_module, "_norm_residual_norm_fwd_kernel"),
        "finalize": (finalize_module, "_post_expert_finalize_kernel"),
        "routing": (routing_module, "_routing_kernel"),
    }
    spies = {name: (mod, attr, _KernelSpy()) for name, (mod, attr) in spies.items()}
    for mod, attr, spy in spies.values():
        monkeypatch.setattr(mod, attr, spy)
    # The tensors below live on the CPU: nothing is launched.
    monkeypatch.setattr(
        rmsnorm_module, "supports_norm_residual_norm", lambda branch, stream: True
    )

    rows = 2
    branch, stream, kw = boundary_inputs(rows, device="cpu")
    fused_norm_residual_norm_fwd(branch, stream, **kw)
    routed = torch.zeros(8 * rows, LATENT, dtype=torch.bfloat16)
    tw = torch.zeros(rows, TOP_K)
    idx = torch.zeros(rows, TOP_K, dtype=torch.int32)
    out = torch.zeros(rows, LATENT, dtype=torch.bfloat16)
    fused_post_expert_finalize(routed, None, tw, 1e-8, 1.0, True, out, permuted_idx=idx)
    gating, bias = torch.zeros(rows, EXPERTS), torch.zeros(EXPERTS)
    fused_musespark1x_routing(gating, bias, TOP_K, 1.0)
    return {name: spy.launches[-1] for name, (_, _, spy) in spies.items()}


def test_sm120_launches_use_pdl(monkeypatch):
    _pretend_gpu(monkeypatch, family=120)
    assert muse_pdl_enabled()
    params = _launch_params(monkeypatch)
    for name in ("boundary", "finalize"):
        assert params[name]["USE_PDL"] is True, name
        assert params[name]["launch_pdl"] is True, name
    assert params["routing"]["USE_PDL"] is True
    assert params["routing"]["launch_pdl"] is True
    assert params["routing"]["EARLY_TRIGGER"] is True


@pytest.mark.parametrize("family", [90, 100])
def test_sm90_and_sm100_keep_their_previous_launches(monkeypatch, family: int):
    """Both support PDL, but the new dependents and early triggers are SM120 only:
    the boundary and finalize kernels stay plain launches and the routing kernel
    keeps its (previously existing) late trigger."""
    _pretend_gpu(monkeypatch, family=family)
    assert not muse_pdl_enabled()
    params = _launch_params(monkeypatch)
    for name in ("boundary", "finalize"):
        assert params[name]["USE_PDL"] is False, name
        assert "launch_pdl" not in params[name], name
    assert params["routing"]["USE_PDL"] is True
    assert params["routing"]["launch_pdl"] is True
    assert params["routing"]["EARLY_TRIGGER"] is False


def test_the_flag_turns_the_sm120_launches_back_into_the_previous_ones(monkeypatch):
    _pretend_gpu(monkeypatch, family=120, flag="0")
    assert not muse_pdl_enabled()
    params = _launch_params(monkeypatch)
    for name in ("boundary", "finalize"):
        assert params[name]["USE_PDL"] is False, name
        assert "launch_pdl" not in params[name], name
    assert params["routing"]["EARLY_TRIGGER"] is False
