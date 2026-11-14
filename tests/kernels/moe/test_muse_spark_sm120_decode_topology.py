# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 decode topology of a MuseSpark MoE layer, on non-zero data, in a CUDA graph.

The model captures the layer as one graph with these edges: the pooler GEMM and the
pre-expert norm on the main stream, the router GEMM and the top-k routing on an aux
stream joined by an event, the muse decode kernel (a programmatic dependent of the
norm, whose predecessor edge is the event join) and the finalize (a programmatic
dependent of the decode kernel). The other tests cover pieces of it: the aux-stream
graph test of ``test_muse_spark_sm120_nvfp4_moe.py`` runs the CUTLASS engine at a toy
shape (it cannot select the muse engine), and the persistent-kernel race test of
``test_muse_spark_sm120_decode_moe.py`` is single-stream. Here the layer has the
kernel's shapes (hidden = intermediate = 4096, 32 local experts, top-k 8), the muse
engine runs in the graph, and the aux-stream graph is compared bit for bit with a
graph that runs the same ops in order, over many replays on fresh inputs, with PDL on
and off.
"""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.moe.muse_spark_utils import run_moe
from tests.kernels.moe.test_muse_spark_sm120_decode_engine import (
    ENGINE_AGREEMENT_REL_L2,
    _build_layer,
)
from tests.kernels.moe.test_muse_spark_sm120_decode_moe import (
    E_LOCAL,
    H,
    _Layer,
    _rel_l2,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_rmsnorm_fwd
from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.models.muse_spark.nvidia.ops.muse_decode_moe import (
    MUSE_MAX_TOKENS,
    MUSE_SMALL_MAX_TOKENS,
)
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

_unusable = muse_c.preflight()
if _unusable is not None:
    pytest.skip(
        f"The _muse_C extension cannot be built: {_unusable}", allow_module_level=True
    )

PDL_ENV = "VLLM_MUSE_SPARK_SM120_PDL"
# Hidden size of the model: the pooler maps 2 * H to H.
D_MODEL = 2 * H
NORM_EPS = 1e-6
REPLAYS = 60
# The first replays are also run eagerly (CUTLASS engine, exact GEMM1) and compared.
EAGER_CHECKS = 3


@pytest.fixture(scope="module", autouse=True)
def _extension() -> None:
    muse_c.load()


@pytest.fixture(scope="module")
def layer() -> _Layer:
    with torch.inference_mode():
        yield _Layer(diff_s=True, seed=5)


def _random_input(num_tokens: int, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(num_tokens, D_MODEL, device="cuda", generator=generator).to(
        torch.bfloat16
    )


@pytest.mark.parametrize("pdl", ["1", "0"])
@torch.inference_mode()
def test_aux_stream_routing_graph_with_the_muse_engine_replays_like_a_serial_graph(
    dist_init, workspace_init, layer, pdl, monkeypatch
) -> None:
    monkeypatch.setenv(PDL_ENV, pdl)
    router, runner, vllm_config = _build_layer(layer)
    quant_method = runner.routed_experts.quant_method
    assert quant_method.use_muse_decode and quant_method.exact_gemm1
    assert runner.supports_routed_forward

    generator = torch.Generator(device="cuda").manual_seed(0)
    pooler_weight = (
        torch.randn(H, D_MODEL, device="cuda", generator=generator) * 0.02
    ).to(torch.bfloat16)
    router_weight = (
        torch.randn(E_LOCAL, D_MODEL, device="cuda", generator=generator) * 0.05
    ).float()
    norm_weight = torch.ones(H, device="cuda", dtype=torch.bfloat16)
    aux = torch.cuda.Stream()
    events = (torch.cuda.Event(), torch.cuda.Event())

    def pool_and_norm(x: torch.Tensor) -> torch.Tensor:
        return fused_rmsnorm_fwd(F.linear(x, pooler_weight), norm_weight, NORM_EPS)

    def route(x: torch.Tensor):
        return runner.select_experts(x, F.linear(x.float(), router_weight))

    def serial(x: torch.Tensor) -> torch.Tensor:
        pooled = pool_and_norm(x)
        weights, ids = route(x)
        return runner.forward_routed(pooled, weights, ids)

    def overlapped(x: torch.Tensor) -> torch.Tensor:
        pooled, (weights, ids) = maybe_execute_in_parallel(
            lambda: pool_and_norm(x), lambda: route(x), *events, aux
        )
        return runner.forward_routed(pooled, weights, ids)

    def capture(forward, x_static: torch.Tensor):
        for _ in range(3):
            forward(x_static)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward(x_static)
        return graph, out

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        for num_tokens in (1, 2, 4, MUSE_SMALL_MAX_TOKENS, 24, MUSE_MAX_TOKENS):
            x_serial = _random_input(num_tokens, generator)
            x_aux = x_serial.clone()
            serial_graph, serial_out = capture(serial, x_serial)
            aux_graph, aux_out = capture(overlapped, x_aux)
            nonzero = 0
            for replay in range(REPLAYS):
                new_x = _random_input(num_tokens, generator)
                x_serial.copy_(new_x)
                x_aux.copy_(new_x)
                serial_graph.replay()
                aux_graph.replay()
                torch.accelerator.synchronize()
                assert torch.equal(aux_out, serial_out), (num_tokens, replay)
                nonzero += int(bool((serial_out != 0).any()))
                if replay < EAGER_CHECKS:
                    # Outside a capture the layer runs the CUTLASS engine (exact
                    # GEMM1), which computes the rows of the muse kernel.
                    eager = serial(new_x)
                    torch.accelerator.synchronize()
                    assert _rel_l2(aux_out, eager) <= ENGINE_AGREEMENT_REL_L2, (
                        num_tokens,
                        replay,
                    )
            # The data is not trivially zero (a zero output would match anything).
            assert nonzero == REPLAYS and bool(torch.isfinite(serial_out).all())


@pytest.mark.parametrize("pdl", ["1", "0"])
@torch.inference_mode()
def test_aux_stream_routing_eager_run_is_the_runners_own(
    dist_init, workspace_init, layer, pdl, monkeypatch
) -> None:
    """Routing finished on the aux stream, then ``forward_routed``, is bit for bit
    the runner's own ``forward`` for the production layer (CUTLASS engine, exact
    GEMM1, real shapes), where the toy-shape test cannot say."""
    monkeypatch.setenv(PDL_ENV, pdl)
    router, runner, vllm_config = _build_layer(layer)
    generator = torch.Generator(device="cuda").manual_seed(1)
    router_weight = (
        torch.randn(E_LOCAL, D_MODEL, device="cuda", generator=generator) * 0.05
    ).float()
    aux = torch.cuda.Stream()
    events = (torch.cuda.Event(), torch.cuda.Event())
    for num_tokens in (1, 8, 33):
        x = _random_input(num_tokens, generator)
        pooled = torch.randn(num_tokens, H, device="cuda", generator=generator).to(
            torch.bfloat16
        )
        logits = F.linear(x.float(), router_weight)
        expected = run_moe(runner, vllm_config, pooled, logits).clone()
        with set_current_vllm_config(vllm_config), set_forward_context(
            None, vllm_config
        ):
            _, (weights, ids) = maybe_execute_in_parallel(
                lambda: pooled.clone(),
                lambda: runner.select_experts(x, F.linear(x.float(), router_weight)),
                *events,
                aux,
            )
            out = runner.forward_routed(pooled, weights, ids)
        torch.accelerator.synchronize()
        assert torch.equal(out, expected), num_tokens


@pytest.mark.parametrize("pdl", ["1", "0"])
@torch.inference_mode()
def test_aux_plan_and_norm_quant_graph_replays_like_the_plain_aux_graph(
    dist_init, workspace_init, layer, pdl, monkeypatch
) -> None:
    """The plan of the large-batch kernel built on the aux stream behind the top-k
    (``build_decode_plan``) and the quantization folded into the pre-expert norm
    (``norm_quant_decode_input``), through the real runner and its modular experts: the
    graph is bit for bit the graph without them, over many replays on fresh inputs, and
    the large-batch kernel really takes both (flags 32 and 64) in the captured graph."""
    monkeypatch.setenv(PDL_ENV, pdl)
    router, runner, vllm_config = _build_layer(layer)
    generator = torch.Generator(device="cuda").manual_seed(2)
    pooler_weight = (
        torch.randn(H, D_MODEL, device="cuda", generator=generator) * 0.02
    ).to(torch.bfloat16)
    router_weight = (
        torch.randn(E_LOCAL, D_MODEL, device="cuda", generator=generator) * 0.05
    ).float()
    norm_weight = (torch.randn(H, device="cuda", generator=generator) * 0.3).to(
        torch.bfloat16
    )
    aux = torch.cuda.Stream()
    events = (torch.cuda.Event(), torch.cuda.Event())
    flags_seen: list[int] = []
    original = torch.ops._muse_C.nvfp4_decode_moe_large

    depth = [0]

    def spy(*args):
        # (The expert weights are Parameters, whose __torch_function__ calls the op again.)
        if depth[0] == 0:
            flags_seen.append(args[-1])
        depth[0] += 1
        try:
            return original(*args)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(torch.ops._muse_C, "nvfp4_decode_moe_large", spy)

    def make(plan: bool, quant: bool):
        def pool_and_norm(x: torch.Tensor) -> torch.Tensor:
            pooled = F.linear(x, pooler_weight)
            if quant:
                fused = runner.norm_quant_decode_input(
                    pooled, norm_weight, NORM_EPS, 1.0
                )
                if fused is not None:
                    return fused
            return fused_rmsnorm_fwd(pooled, norm_weight, NORM_EPS, gain_center=1.0)

        def route(x: torch.Tensor):
            weights, ids = runner.select_experts(x, F.linear(x.float(), router_weight))
            if plan:
                runner.build_decode_plan(ids)
            return weights, ids

        def forward(x: torch.Tensor) -> torch.Tensor:
            pooled, (weights, ids) = maybe_execute_in_parallel(
                lambda: pool_and_norm(x), lambda: route(x), *events, aux
            )
            return runner.forward_routed(pooled, weights, ids)

        return forward

    def capture(forward, x_static: torch.Tensor):
        for _ in range(3):
            forward(x_static)
        torch.accelerator.synchronize()
        flags_seen.clear()  # (the first forward pass prepares the engine: warm-ups)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward(x_static)
        return graph, out

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        for num_tokens in (MUSE_SMALL_MAX_TOKENS + 1, 24, 64, MUSE_MAX_TOKENS):
            graphs = {}
            for plan, quant in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                x = _random_input(num_tokens, generator)
                graph, out = capture(make(plan, quant), x)
                # The capture is the only launch with the engine; eager forwards are CUTLASS.
                assert len(flags_seen) == 1, (num_tokens, plan, quant)
                assert bool(flags_seen[0] & 32) == plan, (num_tokens, plan, quant)
                assert bool(flags_seen[0] & 64) == quant, (num_tokens, plan, quant)
                graphs[(plan, quant)] = (graph, out, x)
                # Nothing is left prepared after the layer.
                experts = runner.routed_experts.quant_method.moe_kernel.fused_experts
                assert experts.muse._plan_key is None and experts.muse._xq_key is None
            for replay in range(REPLAYS):
                new_x = _random_input(num_tokens, generator)
                for _, _, x in graphs.values():
                    x.copy_(new_x)
                for graph, _, _ in graphs.values():
                    graph.replay()
                torch.accelerator.synchronize()
                reference = graphs[(False, False)][1]
                assert bool(torch.isfinite(reference).all())
                assert bool((reference != 0).any())
                for key, (_, out, _) in graphs.items():
                    assert torch.equal(out, reference), (num_tokens, replay, key)


@pytest.mark.parametrize("pdl", ["1", "0"])
@torch.inference_mode()
def test_chained_layers_share_the_plan_and_the_quantization_scratch_in_a_graph(
    dist_init, workspace_init, layer, pdl, monkeypatch
) -> None:
    """Several layers back to back in one graph, as the model's decode graph runs them:
    the plan, the route -> row map and the quantized input are static scratch shared by
    every layer, so the aux-stream plan of layer L + 1 must wait for the experts of
    layer L, and its quantization for the finalize of layer L. The chain with the plan
    on the aux stream and the quantizing norm is bit for bit the chain without them,
    over many replays on fresh inputs (a layer's plan overwriting the map the previous
    layer still reads would show here)."""
    monkeypatch.setenv(PDL_ENV, pdl)
    router, runner, vllm_config = _build_layer(layer)
    generator = torch.Generator(device="cuda").manual_seed(3)
    layers = []
    for _ in range(3):
        layers.append(
            (
                (torch.randn(H, D_MODEL, device="cuda", generator=generator) * 0.02).to(
                    torch.bfloat16
                ),
                (
                    torch.randn(E_LOCAL, D_MODEL, device="cuda", generator=generator)
                    * 0.05
                ).float(),
                (torch.randn(H, device="cuda", generator=generator) * 0.3).to(
                    torch.bfloat16
                ),
            )
        )
    aux = torch.cuda.Stream()
    events = (torch.cuda.Event(), torch.cuda.Event())
    flags_seen: list[int] = []
    original = torch.ops._muse_C.nvfp4_decode_moe_large
    depth = [0]

    def spy(*args):
        if depth[0] == 0:
            flags_seen.append(args[-1])
        depth[0] += 1
        try:
            return original(*args)
        finally:
            depth[0] -= 1

    monkeypatch.setattr(torch.ops._muse_C, "nvfp4_decode_moe_large", spy)

    def make(prepared: bool):
        def one_layer(x, pooler_weight, router_weight, norm_weight):
            def pool_and_norm() -> torch.Tensor:
                pooled = F.linear(x, pooler_weight)
                if prepared:
                    fused = runner.norm_quant_decode_input(
                        pooled, norm_weight, NORM_EPS, 1.0
                    )
                    if fused is not None:
                        return fused
                return fused_rmsnorm_fwd(pooled, norm_weight, NORM_EPS, gain_center=1.0)

            def route():
                weights, ids = runner.select_experts(
                    x, F.linear(x.float(), router_weight)
                )
                if prepared:
                    runner.build_decode_plan(ids)
                return weights, ids

            pooled, (weights, ids) = maybe_execute_in_parallel(
                pool_and_norm, route, *events, aux
            )
            return runner.forward_routed(pooled, weights, ids)

        def forward(x: torch.Tensor) -> torch.Tensor:
            for weights in layers:
                out = one_layer(x, *weights)
                # The next layer's input depends on this layer's output.
                x = (0.5 * x.float() + 0.5 * torch.cat([out, out], dim=1).float()).to(
                    torch.bfloat16
                )
            return x

        return forward

    def capture(forward, x_static: torch.Tensor):
        for _ in range(3):
            forward(x_static)
        torch.accelerator.synchronize()
        flags_seen.clear()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward(x_static)
        return graph, out

    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        for num_tokens in (MUSE_SMALL_MAX_TOKENS + 1, 24, 64, MUSE_MAX_TOKENS):
            graphs = {}
            for prepared in (False, True):
                x = _random_input(num_tokens, generator)
                graph, out = capture(make(prepared), x)
                assert len(flags_seen) == len(layers), (num_tokens, prepared)
                assert all(bool(f & 32) == prepared for f in flags_seen)
                assert all(bool(f & 64) == prepared for f in flags_seen)
                graphs[prepared] = (graph, out, x)
            for replay in range(REPLAYS):
                new_x = _random_input(num_tokens, generator)
                for _, _, x in graphs.values():
                    x.copy_(new_x)
                for graph, _, _ in graphs.values():
                    graph.replay()
                torch.accelerator.synchronize()
                reference = graphs[False][1]
                assert bool(torch.isfinite(reference).all())
                assert torch.equal(graphs[True][1], reference), (num_tokens, replay)
