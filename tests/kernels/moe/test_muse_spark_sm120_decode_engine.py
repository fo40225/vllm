# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 integration of the muse decode MoE kernel into MuseSpark's native-FP4 layer.

Covers the engine switch (``VLLM_MUSE_SPARK_SM120_DECODE_ENGINE``), the weight
layouts it implies, the experts that hand the kernel's rows to the post-expert
finalizer, CUDA graph replays, and the fallbacks. The kernel itself is tested in
``test_muse_spark_sm120_decode_moe.py``; the CUTLASS and b12x engines in
``test_muse_spark_sm120_nvfp4_moe.py``. The layers here have the kernel's shapes
(hidden = intermediate = 4096, top-k 8) and 32 local experts (the kernel takes any
number).
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    make_runner,
    nvfp4_quant_config,
    run_moe,
)
from tests.kernels.moe.test_muse_spark_sm120_decode_moe import (
    E_GLOBAL,
    E_LOCAL,
    FQ_REL_L2,
    FQ_REL_L2_FEW_ROWS,
    H,
    TOP_K,
    _finalize,
    _ids_cases,
    _Layer,
    _local_ids,
    _rel_l2,
    _routing,
)
from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale
from vllm.models.muse_spark.common.ops.fused_finalize import fused_post_expert_finalize
from vllm.models.muse_spark.nvidia import sm120_nvfp4_moe
from vllm.models.muse_spark.nvidia.muse_moe_runner import MusePostExpertFinalize
from vllm.models.muse_spark.nvidia.ops import muse_c, muse_decode_moe
from vllm.models.muse_spark.nvidia.ops.muse_decode_moe import (
    MUSE_MAX_TOKENS,
    MuseDecodeMoE,
)
from vllm.models.muse_spark.nvidia.sm120_nvfp4_moe import (
    MuseSparkSm120NvFp4Experts,
    decode_kernel_params,
    make_sm120_nvfp4_quant_config,
    select_decode_engine,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

_unusable = muse_c.preflight()
if _unusable is not None:
    pytest.skip(
        f"The _muse_C extension cannot be built: {_unusable}", allow_module_level=True
    )

ENGINE_ENV = "VLLM_MUSE_SPARK_SM120_DECODE_ENGINE"
# Final output of the muse engine and of the CUTLASS path against the exact
# reference: both round like the reference (see the kernel tests), the CUTLASS
# GEMM1 keeping the gate and up scales apart like the kernel does, so only a few
# intermediate codes flip.
REPLAY_REL_L2 = 3e-3
# The muse engine and the exact CUTLASS path compute the same rows bit for bit;
# what is left between the two layer outputs is the finalizer's.
ENGINE_AGREEMENT_REL_L2 = 1e-6


@pytest.fixture(scope="module", autouse=True)
def _extension() -> None:
    muse_c.load()


@pytest.fixture(scope="module", params=[True, False], ids=["diff_s", "same_s"])
def layer(request) -> _Layer:
    with torch.inference_mode():
        yield _Layer(diff_s=request.param)


@pytest.fixture(scope="module")
def same_s_layer() -> _Layer:
    with torch.inference_mode():
        yield _Layer(diff_s=False, seed=3)


def _checkpoint(layer: _Layer) -> dict[str, torch.Tensor]:
    """The checkpoint tensors of the layer: [gate; up] rows, linear block scales,
    and global input scales that are the same for every expert."""
    return {
        "w13_weight": layer.w13,
        "w13_weight_scale": layer.w13_sf,
        "w13_weight_scale_2": torch.stack([layer.s1, layer.s3], 1),
        "w13_input_scale": layer.a13.expand(E_LOCAL, 2).contiguous(),
        "w2_weight": layer.w2,
        "w2_weight_scale": layer.w2_sf,
        "w2_weight_scale_2": layer.s2,
        "w2_input_scale": layer.a2.expand(E_LOCAL).contiguous(),
    }


def _make_norm(layer: _Layer) -> MuseSpark1xRMSNormWithInputScale:
    norm = MuseSpark1xRMSNormWithInputScale(
        H,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(layer.gamma)
    return norm


def _swap_halves(t: torch.Tensor) -> torch.Tensor:
    half = t.shape[1] // 2
    return torch.cat([t[:, half:], t[:, :half]], dim=1)


# ------------------------------------------------------------------------------
# The engine switch
# ------------------------------------------------------------------------------
def test_engine_env_var_default_and_validation(monkeypatch) -> None:
    monkeypatch.delenv(ENGINE_ENV, raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_DECODE_ENGINE == "auto"
    for value in ("auto", "muse", "b12x", "cutlass"):
        monkeypatch.setenv(ENGINE_ENV, value)
        assert envs.VLLM_MUSE_SPARK_SM120_DECODE_ENGINE == value
    monkeypatch.setenv(ENGINE_ENV, "MUSE")
    assert envs.VLLM_MUSE_SPARK_SM120_DECODE_ENGINE.lower() == "muse"
    monkeypatch.setenv(ENGINE_ENV, "triton")
    with pytest.raises(ValueError, match=ENGINE_ENV):
        _ = envs.VLLM_MUSE_SPARK_SM120_DECODE_ENGINE


class _Host:
    """A host and a config for ``select_decode_engine``: the verdicts of the muse
    and b12x probes (None = usable) and the extension preflight, and what the
    selection logs."""

    def __init__(self, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self.muse: str | None = None
        self.b12x: str | None = None
        self.preflight: str | None = None
        self.logs: list[tuple[str, str]] = []
        monkeypatch.setattr(sm120_nvfp4_moe, "muse_unusable_reason", lambda: self.muse)
        monkeypatch.setattr(sm120_nvfp4_moe, "b12x_unusable_reason", lambda: self.b12x)
        monkeypatch.setattr(muse_c, "preflight", lambda: self.preflight)
        logs = self.logs

        class Logger:
            def info_once(self, msg, *args):
                logs.append(("info", msg % args))

            def warning_once(self, msg, *args):
                logs.append(("warning", msg % args))

        monkeypatch.setattr(sm120_nvfp4_moe, "logger", Logger())
        monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
        monkeypatch.delenv(ENGINE_ENV, raising=False)
        self.graphs()

    def graphs(self, mode=CUDAGraphMode.FULL_AND_PIECEWISE, enforce_eager=False):
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=mode),
            model_config=SimpleNamespace(enforce_eager=enforce_eager),
        )
        self.monkeypatch.setattr(
            sm120_nvfp4_moe, "get_current_vllm_config_or_none", lambda: config
        )

    @staticmethod
    def moe_config(
        hidden=H, intermediate=H, top_k=TOP_K, local=E_LOCAL, dtype=torch.bfloat16
    ):
        return SimpleNamespace(
            hidden_dim=hidden,
            intermediate_size_per_partition=intermediate,
            experts_per_token=top_k,
            num_local_experts=local,
            in_dtype=dtype,
        )

    def select(self, engine=None, backend="auto", **config) -> str:
        if engine is not None:
            self.monkeypatch.setenv(ENGINE_ENV, engine)
        self.logs.clear()
        return select_decode_engine(backend, self.moe_config(**config))

    def levels(self) -> list[str]:
        return [level for level, _ in self.logs]


def test_decode_engine_selection(monkeypatch) -> None:
    host = _Host(monkeypatch)
    # Everything usable: auto takes muse, a forced engine is honoured.
    assert host.select() == "muse"
    assert host.select("auto") == "muse"
    assert host.select("muse") == "muse"
    assert host.select("b12x") == "b12x"
    assert host.select("cutlass") == "cutlass"
    assert "warning" not in host.levels()
    # Whichever engine was chosen is logged once, with the request.
    assert host.select("MUSE") == "muse"
    assert any("decode engine muse" in msg for _, msg in host.logs)

    # --moe-backend cutlass, and no CUDA graphs, turn both engines off.
    for engine in ("auto", "muse", "b12x"):
        assert host.select(engine, backend="cutlass") == "cutlass"
    host.graphs(CUDAGraphMode.NONE)
    assert host.select("auto") == "cutlass"
    assert host.select("muse") == "cutlass"
    host.graphs(CUDAGraphMode.FULL_AND_PIECEWISE, enforce_eager=True)
    assert host.select("auto") == "cutlass"
    host.graphs()
    monkeypatch.setattr(
        sm120_nvfp4_moe, "get_current_vllm_config_or_none", lambda: None
    )
    assert host.select("auto") == "cutlass"
    host.graphs()

    # Another layer shape or dtype: the kernel does not apply; auto goes to b12x
    # quietly, a forced muse says why.
    for config in (
        {"hidden": 512},
        {"intermediate": 2048},
        {"top_k": 4},
        {"local": 0},
        {"dtype": torch.float16},
    ):
        assert host.select("auto", **config) == "b12x"
        assert "warning" not in host.levels()
        assert host.select("muse", **config) == "b12x"
        assert host.levels().count("warning") == 1
        assert host.select("b12x", **config) == "b12x"
    # Any number of local experts is served (the EP size is not fixed).
    for local in (1, 8, 64):
        assert host.select("auto", local=local) == "muse"

    # The extension cannot be built: neither engine, b12x needs it as well.
    host.muse = "nvcc was not found"
    host.preflight = "nvcc was not found"
    assert host.select("auto") == "cutlass"
    assert "warning" in host.levels()
    assert host.select("muse") == "cutlass"
    host.preflight = None
    # Only the muse engine is unusable (an earlier layer failed to set it up).
    assert host.select("auto") == "b12x"
    assert "warning" not in host.levels()
    assert host.select("muse") == "b12x"
    assert host.levels().count("warning") == 1
    assert "nvcc was not found" in [m for lv, m in host.logs if lv == "warning"][0]
    host.muse = None
    # Only b12x is unusable: auto does not care, a forced b12x falls to CUTLASS.
    host.b12x = "b12x is not installed"
    assert host.select("auto") == "muse"
    assert host.select("muse") == "muse"
    assert host.select("b12x") == "cutlass"
    assert host.levels().count("warning") == 1
    # Neither is usable, and the layer does not fit the kernel either.
    assert host.select("auto", hidden=512) == "cutlass"
    assert "warning" in host.levels()
    host.b12x = None

    # Batch-invariant mode excludes both engines: they round differently from
    # the CUTLASS path of larger batches, which would make a request's output
    # depend on the number of its companions.
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    for requested in ("auto", "muse", "b12x", "cutlass"):
        assert host.select(requested) == "cutlass"
    assert "batch-invariant mode" in " ".join(m for _, m in host.logs)
    host.logs.clear()
    host.select("auto")
    assert "warning" not in host.levels()


# ------------------------------------------------------------------------------
# The engine of one layer
# ------------------------------------------------------------------------------
class _Engine:
    """A ``MuseDecodeMoE`` over the layer's checkpoint-order weights."""

    def __init__(self, layer: _Layer, **kwargs) -> None:
        self.layer = layer
        self.engine = MuseDecodeMoE(
            layer.w13,
            layer.sf13,
            layer.w2,
            layer.sf2,
            prm=layer.prm,
            gs=layer.gs,
            **kwargs,
        )

    def run(self, u, ids):
        return self.engine.run(u, ids, self.layer.expert_map)


@torch.inference_mode()
def test_engine_serves_every_padded_token_count(layer: _Layer) -> None:
    engine = _Engine(layer)
    ids_all = _ids_cases(8, 5)["random"]
    u_all = _routing(8, 5)[0]
    rows8, perm8 = engine.run(u_all, ids_all)
    assert rows8.shape == (8 * TOP_K, H) and perm8.shape == (8 * TOP_K,)
    rows8, perm8 = rows8.clone(), perm8.clone()
    for t in range(1, MUSE_MAX_TOKENS + 1):
        rows, perm = engine.run(u_all[:t], ids_all[:t])
        assert rows.shape == (t * TOP_K, H) and perm.shape == (t * TOP_K,)
        assert rows.dtype == torch.bfloat16 and perm.dtype == torch.int32
        assert rows.is_contiguous() and perm.is_contiguous()
        # A token's rows do not depend on the other tokens of the batch, nor on
        # the padding to 1, 2, 4 or 8.
        assert torch.equal(perm, perm8[: t * TOP_K])
        got = rows[perm >= 0].view(torch.int16)
        want = rows8[: t * TOP_K][perm >= 0].view(torch.int16)
        assert torch.equal(got, want), t
    with pytest.raises(RuntimeError, match="1..8"):
        engine.run(torch.zeros(9, H, dtype=torch.bfloat16, device="cuda"), ids_all)


@torch.inference_mode()
def test_engine_reads_the_weights_in_place(layer: _Layer) -> None:
    engine = _Engine(layer).engine
    assert engine.reads(layer.w13, layer.sf13, layer.w2, layer.sf2)
    assert not engine.reads(layer.w13.clone(), layer.sf13, layer.w2, layer.sf2)
    assert not engine.reads(layer.w13, layer.sf13, layer.w2, layer.sf2.clone())
    # Not a copy: no weight-sized allocation at all.
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    _Engine(layer)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before < (8 << 20)


@torch.inference_mode()
def test_engine_rejects_other_shapes(layer: _Layer) -> None:
    with pytest.raises(ValueError, match="hidden=4096"):
        MuseDecodeMoE(
            layer.w13[:, :512],
            layer.sf13[:, :512],
            layer.w2[:, :, :256],
            layer.sf2[:, :, :32],
            prm=layer.prm,
            gs=layer.gs,
        )
    assert muse_decode_moe.muse_shape_mismatch(H, H, TOP_K, 1) is None
    assert muse_decode_moe.muse_shape_mismatch(H, H, TOP_K, 0) is not None


@torch.inference_mode()
def test_engine_zeroes_remote_rows_on_request(layer: _Layer) -> None:
    """Rows that are all-reduced before the norm must be zero where a route is
    remote; otherwise they are not written."""
    u = _routing(4, 9)[0]
    ids = _local_ids(4, [0, 2, 5, 8], 9)
    for zero in (False, True):
        engine = _Engine(layer, zero_non_local=zero)
        # Poison the shared rows first.
        outputs = muse_decode_moe._get_outputs(layer.w13.device, 4)
        outputs.rows.fill_(float("nan"))
        rows, perm = engine.run(u, ids)
        remote = perm < 0
        assert remote.any() and not remote.all()
        assert torch.isfinite(rows[~remote]).all()
        if zero:
            assert torch.count_nonzero(rows[remote]) == 0
        else:
            assert torch.isnan(rows[remote]).all()


@torch.inference_mode()
def test_kernel_takes_any_number_of_local_experts(layer: _Layer) -> None:
    """The kernel reads the number of local experts from the weights. Fewer
    experts than 32 (an unmapped id is non-local), or more (copies of the same
    32 experts), give the same rows as the 32 do."""
    u = _routing(4, 3)[0]
    ids = _local_ids(4, [3, 4, 5, 6], 3)
    full = _Engine(layer)
    rows, perm = full.run(u, ids)
    rows, perm = rows.clone(), perm.clone()

    # The first 8 experts only: routes to the others become non-local.
    few = MuseDecodeMoE(
        layer.w13[:8],
        layer.sf13[:8],
        layer.w2[:8],
        layer.sf2[:8],
        prm=layer.prm[:8],
        gs=layer.gs,
    )
    emap = layer.expert_map.clone()
    emap[8:] = -1
    got_rows, got_perm = few.run(u, ids, emap)
    local8 = (layer.expert_map[ids.long()] >= 0) & (layer.expert_map[ids.long()] < 8)
    assert torch.equal(got_perm >= 0, local8.view(-1))
    assert local8.any()
    want = rows[got_perm >= 0].view(torch.int16)
    assert torch.equal(got_rows[got_perm >= 0].view(torch.int16), want)
    # 64 experts: local expert e + 32 is a copy of expert e, and the map sends the
    # global expert e to it.
    many = MuseDecodeMoE(
        layer.w13.repeat(2, 1, 1),
        layer.sf13.repeat(2, 1, 1),
        layer.w2.repeat(2, 1, 1),
        layer.sf2.repeat(2, 1, 1),
        prm=layer.prm.repeat(2, 1),
        gs=layer.gs,
    )
    emap = torch.full_like(layer.expert_map, -1)
    emap[:E_LOCAL] = torch.arange(
        E_LOCAL, 2 * E_LOCAL, dtype=torch.int32, device="cuda"
    )
    got_rows, got_perm = many.run(u, ids, emap)
    assert torch.equal(got_perm, perm)
    assert torch.equal(
        got_rows[perm >= 0].view(torch.int16), rows[perm >= 0].view(torch.int16)
    )


# ------------------------------------------------------------------------------
# The experts, with expert parallelism: kernel rows -> finalizer
# ------------------------------------------------------------------------------
def _moe_config(tp_size: int = 1) -> FusedMoEConfig:
    ep_size = E_GLOBAL // E_LOCAL
    return FusedMoEConfig(
        num_experts=E_GLOBAL,
        experts_per_token=TOP_K,
        hidden_dim=H,
        intermediate_size=H * tp_size,
        num_local_experts=E_LOCAL,
        num_logical_experts=E_GLOBAL,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=tp_size,
            pcp_size=1,
            dp_size=1,
            ep_size=ep_size,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=True,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        activation=MoEActivation.SILU,
        device=torch.device("cuda"),
        routing_method=RoutingMethodType.TopK,
        in_dtype=torch.bfloat16,
        max_num_tokens=64,
    )


class _Experts:
    """The experts of one EP rank (the layer's 32 experts of 256, [gate; up]
    weights) with the muse engine on. ``apply`` takes the muse path as a captured
    graph would: ``_capturing`` is patched for eager calls."""

    def __init__(self, layer: _Layer, norm: torch.nn.Module) -> None:
        self.layer = layer
        ckpt = _checkpoint(layer)
        _, _, quant_config = make_sm120_nvfp4_quant_config(
            layer.w13_sf,
            layer.w2_sf,
            ckpt["w13_weight_scale_2"],
            ckpt["w13_input_scale"],
            layer.s2,
            ckpt["w2_input_scale"],
        )
        moe_config = _moe_config()
        self.kernel = mk.FusedMoEKernel(
            make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
            MuseSparkSm120NvFp4Experts(
                moe_config=moe_config, quant_config=quant_config
            ),
        )
        self.experts = self.kernel.fused_experts
        self.finalizer = MusePostExpertFinalize(norm, moe_config)
        self.experts.post_gemm2_hook = self.finalizer
        self.experts.up_gate_layout = False
        self.experts.use_muse_decode = True
        self.experts.decode_scales = layer.scales
        self.experts._ensure_muse(layer.w13, layer.w2, capturing=False)
        assert self.experts.muse is not None

    def apply(self, x, ids, weights, *, muse: bool = True) -> torch.Tensor:
        with pytest.MonkeyPatch.context() as patch:
            if muse:
                patch.setattr(sm120_nvfp4_moe, "_capturing", lambda: True)
            return self.kernel.apply(
                hidden_states=x,
                w1=self.layer.w13,
                w2=self.layer.w2,
                topk_weights=weights,
                topk_ids=ids,
                activation=MoEActivation.SILU,
                global_num_experts=E_GLOBAL,
                expert_map=self.layer.expert_map,
                apply_router_weight_on_input=False,
            )


@pytest.fixture
def experts(dist_init, workspace_init, layer):
    return _Experts(layer, _make_norm(layer))


@pytest.fixture
def handed_over(monkeypatch) -> list[SimpleNamespace]:
    """Copies of what the finalizer receives, per call, before it runs."""
    seen: list[SimpleNamespace] = []
    finalize = MusePostExpertFinalize.__call__

    def spy(self, unfinalized, out):
        seen.append(
            SimpleNamespace(
                rows=unfinalized.gemm2_permuted.clone(),
                weights=unfinalized.expert_weights.clone(),
                idx=unfinalized.expanded_idx_to_permuted_idx.clone(),
                route_major=unfinalized.route_major,
            )
        )
        finalize(self, unfinalized, out)

    monkeypatch.setattr(MusePostExpertFinalize, "__call__", spy)
    return seen


@pytest.mark.parametrize("t", [1, 2, 3, 4, 8])
@torch.inference_mode()
def test_experts_match_the_reference_and_hand_over_route_major_rows(
    experts: _Experts, handed_over, t: int
) -> None:
    layer = experts.layer
    u, _, weights = _routing(t, 7)
    for name, ids in _ids_cases(t, 11).items():
        handed_over.clear()
        out = experts.apply(u, ids, weights)
        (seen,) = handed_over
        # Route-major, with the kernel's permute map: the identity apart from -1.
        assert seen.route_major
        assert seen.rows.shape == (t * TOP_K, H)
        assert seen.idx.shape == (t, TOP_K) and seen.idx.dtype == torch.int32
        local = (layer.expert_map[ids.long()] >= 0).view(t, TOP_K)
        want_idx = torch.where(
            local,
            torch.arange(t * TOP_K, dtype=torch.int32, device="cuda").view(t, TOP_K),
            -1,
        )
        assert torch.equal(seen.idx, want_idx), name
        assert torch.equal(seen.weights, weights)
        rows_ref, valid = layer.reference_rows(u, ids, "fq")
        ref = _finalize(rows_ref, weights, layer.gamma, valid)
        assert torch.isfinite(out).all(), name
        if valid.any():
            bound = FQ_REL_L2_FEW_ROWS if name == "same_expert" else FQ_REL_L2
            assert _rel_l2(out, ref) <= bound, (name, _rel_l2(out, ref))
        else:
            assert torch.count_nonzero(out) == 0, name


@torch.inference_mode()
def test_experts_ignore_the_rows_of_remote_routes(experts: _Experts, monkeypatch):
    """The kernel leaves the rows of remote routes stale, and the finalizer must
    skip them: poisoning them changes nothing."""
    t = 8
    u, _, weights = _routing(t, 13)
    ids = _local_ids(t, [0, 1, 2, 3, 4, 5, 6, 8], 13)
    clean = experts.apply(u, ids, weights).clone()
    assert torch.isfinite(clean).all()

    apply_muse = MuseSparkSm120NvFp4Experts._apply_muse

    def poisoned(self, *args):
        unfinalized = apply_muse(self, *args)
        remote = unfinalized.expanded_idx_to_permuted_idx.view(-1) < 0
        unfinalized.gemm2_permuted[remote] = float("nan")
        return unfinalized

    monkeypatch.setattr(MuseSparkSm120NvFp4Experts, "_apply_muse", poisoned)
    got = experts.apply(u, ids, weights)
    assert torch.equal(got, clean)


@torch.inference_mode()
def test_every_finalize_variant_consumes_the_rows(
    experts: _Experts, handed_over
) -> None:
    """The finalizers that run on SM120 (Triton, the torch fallback) and the CuTe
    DSL one (which the SM100 deployments use, and which compiles for SM120 too)
    take the kernel's rows, permute map and weights as they are, with the stale
    rows of remote routes poisoned. The fused all-reduce variants (MNNVL
    multicast, TP) need several ranks; they take the same three tensors and skip
    ``-1`` routes the same way."""
    t = 6
    u, _, weights = _routing(t, 17)
    ids = _local_ids(t, [0, 1, 2, 4, 6, 8], 17)
    out = experts.apply(u, ids, weights).clone()
    (seen,) = handed_over
    remote = seen.idx.view(-1) < 0
    assert remote.any() and not remote.all()
    seen.rows[remote] = float("nan")
    finalizer = experts.finalizer
    unfinalized = SimpleNamespace(
        gemm2_permuted=seen.rows,
        expert_weights=seen.weights,
        expanded_idx_to_permuted_idx=seen.idx,
        route_major=True,
    )
    native = torch.empty_like(out)
    finalizer._finalize_native(unfinalized, native)
    assert torch.isfinite(native).all()
    assert _rel_l2(native, out) <= 1e-2
    norm = finalizer._norm
    triton = torch.empty_like(out)
    fused_post_expert_finalize(
        seen.rows,
        norm.weight if norm.has_weight else None,
        seen.weights,
        norm.variance_epsilon,
        norm.fused_gain_center,
        norm.applies_weight_before_norm,
        triton,
        permuted_idx=seen.idx,
    )
    assert torch.equal(triton, out)
    cutedsl = pytest.importorskip(
        "vllm.models.muse_spark.nvidia.ops.cute_dsl.moe_finalize_rmsnorm"
    )
    if not cutedsl.HAS_CUTEDSL:
        pytest.skip("nvidia-cutlass-dsl is not installed")
    fused = torch.empty_like(out)
    try:
        cutedsl.moe_finalize_with_rmsnorm(
            seen.rows,
            seen.weights,
            seen.idx,
            finalizer._effective_weight(seen.rows.dtype, seen.rows.device),
            fused,
            topk=TOP_K,
            eps=norm.variance_epsilon,
            gain_center=norm.fused_gain_center,
            weight_before_norm=norm.applies_weight_before_norm,
        )
    except Exception as e:  # pragma: no cover - depends on the DSL build
        pytest.skip(f"the CuTe DSL finalize does not run here: {type(e).__name__}")
    assert torch.isfinite(fused).all()
    assert _rel_l2(fused, out) <= 1e-2


@pytest.mark.parametrize("t", [1, 3, 8])
@torch.inference_mode()
def test_experts_eager_and_graph_are_bitwise_equal(experts: _Experts, t: int) -> None:
    layer = experts.layer
    u, _, weights = _routing(t, 23)
    ids = _ids_cases(t, 29)["random"]
    expected = experts.apply(u, ids, weights).clone()
    assert expected.abs().max() > 0
    u_static, ids_static, w_static = u.clone(), ids.clone(), weights.clone()
    # Every call outside the capture, so that nothing is set up inside it.
    experts.apply(u_static, ids_static, w_static)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = experts.kernel.apply(
            hidden_states=u_static,
            w1=layer.w13,
            w2=layer.w2,
            topk_weights=w_static,
            topk_ids=ids_static,
            activation=MoEActivation.SILU,
            global_num_experts=E_GLOBAL,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=False,
        )
    graph.replay()
    assert torch.equal(out_static, expected)
    # New routes through the same graph match the eager run on them.
    for seed in range(5):
        u2, ids2, w2 = _routing(t, 100 + seed)
        want = experts.apply(u2, ids2, w2).clone()
        u_static.copy_(u2)
        ids_static.copy_(ids2)
        w_static.copy_(w2)
        graph.replay()
        assert torch.equal(out_static, want), seed
    # The graph is the muse engine: counters are back to zero.
    assert not bool(experts.experts.muse._scratch.ctl.any())


# ------------------------------------------------------------------------------
# The layer: runner -> quant method -> experts -> finalizer
# ------------------------------------------------------------------------------
@pytest.fixture
def engine_calls(monkeypatch) -> list[str]:
    """The engine (``cutlass``, ``b12x`` or ``muse``) of every ``apply``."""
    seen: list[str] = []

    def spy(label, apply_engine):
        def spied(self, *args):
            seen.append(label)
            return apply_engine(self, *args)

        return spied

    for label in ("cutlass", "b12x", "muse"):
        name = f"_apply_{label}"
        monkeypatch.setattr(
            MuseSparkSm120NvFp4Experts,
            name,
            spy(label, getattr(MuseSparkSm120NvFp4Experts, name)),
        )
    return seen


@pytest.fixture
def quant_kernels(monkeypatch) -> list[str]:
    """The kernel that quantizes the intermediate of every CUTLASS ``apply``:
    ``split`` with the exact GEMM1; with the folded one ``stock`` for [gate; up]
    weights and ``muse`` for [up; gate]."""
    seen: list[str] = []
    for label, owner, name in (
        ("stock", ops, "silu_and_mul_scaled_fp4_experts_quant"),
        ("muse", muse_c, "mul_and_silu_scaled_fp4_experts_quant"),
        ("split", muse_c, "silu_and_mul_split_scaled_fp4_experts_quant"),
    ):

        def spy(*args, _kernel=getattr(owner, name), _label=label, **kwargs):
            seen.append(_label)
            return _kernel(*args, **kwargs)

        monkeypatch.setattr(owner, name, spy)
    return seen


def _build_layer(layer: _Layer, moe_backend: str = "auto"):
    norm = _make_norm(layer)
    expert_bias = torch.linspace(-0.5, 0.5, E_LOCAL, dtype=torch.float32, device="cuda")
    return make_runner(
        norm,
        expert_bias,
        _checkpoint(layer),
        nvfp4_quant_config(),
        hidden_size=H,
        intermediate_size=H,
        top_k=TOP_K,
        moe_backend=moe_backend,
    )


def _inputs(num_tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, H, device="cuda", generator=g).to(torch.bfloat16)
    logits = torch.randn(
        num_tokens, E_LOCAL, dtype=torch.float32, device="cuda", generator=g
    )
    return x, logits


@pytest.mark.parametrize("num_tokens", [1, 3, 8, 16])
@torch.inference_mode()
def test_layer_runs_the_muse_engine_in_graphs(
    dist_init,
    workspace_init,
    engine_calls,
    quant_kernels,
    layer,
    num_tokens,
    monkeypatch,
) -> None:
    monkeypatch.delenv(ENGINE_ENV, raising=False)
    router, runner, vllm_config = _build_layer(layer)
    routed = runner.routed_experts
    quant_method = routed.quant_method
    experts = quant_method.moe_kernel.fused_experts
    assert type(experts) is MuseSparkSm120NvFp4Experts
    # auto: the muse engine, on the checkpoint's layout; CUTLASS keeps the gate
    # and up scales apart.
    assert quant_method.decode_engine == "muse"
    assert quant_method.use_muse_decode and not quant_method.use_b12x_decode
    assert experts.use_muse_decode and not experts.use_b12x_decode
    assert not experts.up_gate_layout
    assert experts.muse is None and experts.b12x is None
    assert torch.equal(routed.w13_weight.data, layer.w13)
    assert torch.equal(
        routed.w13_weight_scale.data.view(torch.uint8),
        layer.sf13.view(torch.uint8),
    )
    # The exact scales of the kernel, per expert.
    prm, gs = decode_kernel_params(experts.decode_scales)
    torch.testing.assert_close(prm, layer.prm)
    torch.testing.assert_close(gs, layer.gs)

    x, logits = _inputs(num_tokens, 1)
    out_eager = run_moe(runner, vllm_config, x, logits).clone()
    # An eager forward pass is CUTLASS, whatever the batch size; it prepares the
    # engine on the stored weights.
    assert engine_calls == ["cutlass"]
    assert quant_kernels == ["split"]
    assert experts.muse is not None
    assert experts.muse.reads(
        routed.w13_weight.data,
        experts.w1_scale,
        routed.w2_weight.data,
        experts.w2_scale,
    )
    topk_weights, topk_ids = router._compute_routing(x, logits, None)
    rows_ref, valid = layer.reference_rows(x, topk_ids.to(torch.int32), "fq")
    ref = _finalize(rows_ref, topk_weights, layer.gamma, valid)
    assert _rel_l2(out_eager, ref) <= REPLAY_REL_L2

    # The captured graph runs the muse engine for up to eight tokens (it replays
    # no Python), and CUTLASS beyond that, and is checked against the eager run:
    # the engine and the CUTLASS path agree, whatever the scales.
    uses_muse = num_tokens <= MUSE_MAX_TOKENS
    x2, logits2 = _inputs(num_tokens, 2)
    assert_cudagraph_replays(
        runner,
        vllm_config,
        x,
        logits,
        x2,
        logits2,
        max_rel_l2=ENGINE_AGREEMENT_REL_L2 if uses_muse else None,
    )
    assert engine_calls.count("muse") == int(uses_muse)
    assert engine_calls[0] == "cutlass"
    assert quant_kernels.count("muse") == 0 and quant_kernels.count("stock") == 0
    assert not bool(muse_decode_moe._scratch[x.device].ctl.any())
    if uses_muse:
        # The replay is exactly the muse engine: compare with an eager call of it.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(sm120_nvfp4_moe, "_capturing", lambda: True)
            eager_muse = run_moe(runner, vllm_config, x2, logits2).clone()
        graph_out = _graph_forward(runner, vllm_config, x2, logits2)
        assert torch.equal(graph_out, eager_muse)
        topk_weights2, topk_ids2 = router._compute_routing(x2, logits2, None)
        rows_ref2, valid2 = layer.reference_rows(x2, topk_ids2.to(torch.int32), "fq")
        ref2 = _finalize(rows_ref2, topk_weights2, layer.gamma, valid2)
        assert _rel_l2(graph_out, ref2) <= FQ_REL_L2


def _graph_forward(runner, vllm_config, x, logits) -> torch.Tensor:
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context

    x_static, logits_static = x.clone(), logits.clone()
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(hidden_states=x_static, router_logits=logits_static)
        graph.replay()
    return out.clone()


@pytest.mark.parametrize("engine", ["muse", "b12x", "cutlass"])
@torch.inference_mode()
def test_layer_layout_follows_the_selected_engine(
    dist_init, workspace_init, engine_calls, quant_kernels, same_s_layer, engine,
    monkeypatch,
) -> None:
    layer = same_s_layer
    monkeypatch.setenv(ENGINE_ENV, engine)
    router, runner, vllm_config = _build_layer(layer)
    routed = runner.routed_experts
    quant_method = routed.quant_method
    experts = quant_method.moe_kernel.fused_experts
    assert quant_method.decode_engine == engine
    assert experts.use_muse_decode == (engine == "muse")
    assert experts.use_b12x_decode == (engine == "b12x")
    # Only b12x reorders w13 to [up; gate]; the others keep the checkpoint's order.
    # Whatever the layout, the CUTLASS path quantizes the intermediate with the
    # kernel of the exact GEMM1.
    up_gate = engine == "b12x"
    assert experts.up_gate_layout == up_gate
    stored = _swap_halves(layer.w13) if up_gate else layer.w13
    stored_sf = _swap_halves(layer.w13_sf) if up_gate else layer.w13_sf
    assert torch.equal(routed.w13_weight.data, stored)
    assert torch.equal(
        routed.w13_weight_scale.data.view(torch.uint8),
        swizzle_blockscale(stored_sf).view(torch.uint8),
    )
    x, logits = _inputs(5, 4)
    out = run_moe(runner, vllm_config, x, logits)
    assert engine_calls == ["cutlass"]
    assert quant_kernels == ["split"]
    topk_weights, topk_ids = router._compute_routing(x, logits, None)
    rows_ref, valid = layer.reference_rows(x, topk_ids.to(torch.int32), "fq")
    ref = _finalize(rows_ref, topk_weights, layer.gamma, valid)
    # Same weights, same scales (same_s), whichever layout: CUTLASS rounds like
    # the reference.
    assert _rel_l2(out, ref) <= REPLAY_REL_L2
    assert (experts.muse is not None) == (engine == "muse")
    assert (experts.b12x is not None) == up_gate


@torch.inference_mode()
def test_layer_without_the_extension_runs_cutlass_only(
    dist_init, workspace_init, engine_calls, quant_kernels, same_s_layer, monkeypatch
) -> None:
    """The extension passes the preflight but cannot be built or loaded: the layer
    keeps the checkpoint's layout and runs CUTLASS everywhere, in graphs too."""
    layer = same_s_layer
    monkeypatch.delenv(ENGINE_ENV, raising=False)

    def failing_load():
        raise RuntimeError("no compiler today")

    monkeypatch.setattr(muse_c, "load", failing_load)
    # An earlier test may have loaded the library in this process.
    monkeypatch.setattr(muse_c, "_loaded", False)
    router, runner, vllm_config = _build_layer(layer)
    routed = runner.routed_experts
    quant_method = routed.quant_method
    experts = quant_method.moe_kernel.fused_experts
    # Chosen at construction, demoted where the weights are processed.
    assert quant_method.decode_engine == "cutlass"
    assert not quant_method.use_muse_decode and not experts.use_muse_decode
    assert not experts.use_b12x_decode and not experts.up_gate_layout
    assert torch.equal(routed.w13_weight.data, layer.w13)
    x, logits = _inputs(2, 6)
    out = run_moe(runner, vllm_config, x, logits).clone()
    assert torch.isfinite(out).all()
    assert experts.muse is None and experts.b12x is None
    x2, logits2 = _inputs(2, 7)
    assert_cudagraph_replays(runner, vllm_config, x, logits, x2, logits2)
    assert set(engine_calls) == {"cutlass"}
    assert set(quant_kernels) == {"stock"}


@torch.inference_mode()
def test_layer_without_a_toolchain_never_selects_a_decode_engine(
    dist_init, workspace_init, same_s_layer, monkeypatch
) -> None:
    monkeypatch.delenv(ENGINE_ENV, raising=False)
    monkeypatch.setattr(muse_c, "preflight", lambda: "nvcc was not found")
    router, runner, vllm_config = _build_layer(same_s_layer)
    quant_method = runner.routed_experts.quant_method
    assert quant_method.decode_engine == "cutlass"
    assert not quant_method.moe_kernel.fused_experts.up_gate_layout
    # Neither does the exact GEMM1, which needs the extension as well.
    assert not quant_method.exact_gemm1
    assert not quant_method.moe_kernel.fused_experts.exact_gemm1


@torch.inference_mode()
def test_layer_setup_failure_demotes_it_to_cutlass(
    dist_init, workspace_init, engine_calls, same_s_layer, monkeypatch
) -> None:
    """The engine cannot be prepared by the first forward pass: the layer keeps
    the CUTLASS grouped GEMMs (the [gate; up] layout cannot serve b12x), warns once
    and stops trying; graphs capture without the engine."""
    layer = same_s_layer
    monkeypatch.delenv(ENGINE_ENV, raising=False)
    monkeypatch.setattr(muse_decode_moe, "_setup_failure", None)
    attempts = []

    def failing(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("scratch allocation failed")

    monkeypatch.setattr(sm120_nvfp4_moe, "MuseDecodeMoE", failing)
    router, runner, vllm_config = _build_layer(layer)
    experts = runner.routed_experts.quant_method.moe_kernel.fused_experts
    assert experts.use_muse_decode
    x, logits = _inputs(2, 8)
    run_moe(runner, vllm_config, x, logits)
    assert attempts == [1]
    assert experts.muse is None and not experts.use_muse_decode
    assert muse_decode_moe.muse_unusable_reason() is not None
    x2, logits2 = _inputs(2, 9)
    assert_cudagraph_replays(runner, vllm_config, x, logits, x2, logits2)
    assert set(engine_calls) == {"cutlass"}
    assert attempts == [1]


@torch.inference_mode()
def test_capture_without_a_prepared_engine_raises(
    dist_init, workspace_init, same_s_layer, monkeypatch
) -> None:
    monkeypatch.delenv(ENGINE_ENV, raising=False)
    router, runner, vllm_config = _build_layer(same_s_layer)
    experts = runner.routed_experts.quant_method.moe_kernel.fused_experts
    assert experts.use_muse_decode and experts.muse is None
    x, logits = _inputs(2, 10)
    monkeypatch.setattr(sm120_nvfp4_moe, "_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="prepared"):
        run_moe(runner, vllm_config, x, logits)
    assert experts.muse is None and experts.use_muse_decode


@torch.inference_mode()
def test_layer_weight_reload_keeps_the_layout_and_prepares_the_engine_late(
    dist_init, workspace_init, engine_calls, same_s_layer, monkeypatch
) -> None:
    """The reload path (layerwise reload, refit) runs ``process_weights_after_
    loading`` on temporary tensors holding the checkpoint again, then copies the
    result into the persistent ones. The engine is never bound to the temporaries:
    the next eager forward pass prepares it on the persistent tensors, with the
    scales of the reloaded checkpoint."""
    layer = same_s_layer
    monkeypatch.delenv(ENGINE_ENV, raising=False)
    router, runner, vllm_config = _build_layer(layer)
    routed = runner.routed_experts
    quant_method = routed.quant_method
    first = quant_method.moe_kernel.fused_experts
    x, logits = _inputs(3, 11)
    before = run_moe(runner, vllm_config, x, logits).clone()
    assert first.muse is not None

    prepared: list[MuseDecodeMoE] = []
    engine_cls = sm120_nvfp4_moe.MuseDecodeMoE

    def spy(*args, **kwargs):
        prepared.append(engine_cls(*args, **kwargs))
        return prepared[-1]

    monkeypatch.setattr(sm120_nvfp4_moe, "MuseDecodeMoE", spy)
    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    persistent = {name: getattr(routed, name) for name in names}
    pointers = {name: t.data_ptr() for name, t in persistent.items()}
    temporaries = {
        "w13_weight": layer.w13.clone(),
        "w13_weight_scale": layer.w13_sf.clone(),
        "w2_weight": layer.w2.clone(),
        "w2_weight_scale": layer.w2_sf.clone(),
    }
    for name, tensor in temporaries.items():
        setattr(routed, name, torch.nn.Parameter(tensor, requires_grad=False))
    quant_method.process_weights_after_loading(routed)
    experts = quant_method.moe_kernel.fused_experts
    assert experts is not first and experts.use_muse_decode
    assert quant_method.decode_engine == "muse"
    # The checkpoint's order stays, and nothing is bound to the temporaries.
    assert torch.equal(routed.w13_weight.data, layer.w13)
    assert not experts.up_gate_layout
    assert experts.muse is None and prepared == []
    assert experts.decode_scales is not None
    for name, tensor in persistent.items():
        tensor.data.copy_(getattr(routed, name))
        setattr(routed, name, tensor)
        assert tensor.data_ptr() == pointers[name]
    assert experts.muse is None and prepared == []
    del temporaries

    after = run_moe(runner, vllm_config, x, logits)
    torch.testing.assert_close(after, before)
    assert len(prepared) == 1 and experts.muse is prepared[0]
    assert experts.muse.reads(
        routed.w13_weight.data,
        experts.w1_scale,
        routed.w2_weight.data,
        experts.w2_scale,
    )
    assert engine_calls == ["cutlass", "cutlass"]
    run_moe(runner, vllm_config, x, logits)
    assert len(prepared) == 1
    # And inside a graph, the engine is the one prepared on the persistent weights.
    x2, logits2 = _inputs(3, 12)
    assert_cudagraph_replays(
        runner, vllm_config, x, logits, x2, logits2, max_rel_l2=REPLAY_REL_L2
    )
    assert engine_calls.count("muse") == 1
