# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 sequence-parallel prefill of MuseSpark: when it applies, the runner entry
that leaves the expert-parallel sum to the caller, the projections on a rank's rows,
and the sharded FFN branch against the replicated one under expert parallelism."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

import vllm.envs as envs
from tests.kernels.moe.muse_spark_utils import (
    make_runner,
    nvfp4_quant_config,
    run_moe,
)
from vllm.config import set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import GateLinear
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.common.ops.sequence_parallel import is_sequence_parallel_active
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.models.muse_spark.nvidia import model as muse_model
from vllm.platforms import current_platform

FLAG = "VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL"
MIN_TOKENS = "VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS"


def _vllm_config(**overrides) -> SimpleNamespace:
    parallel = dict(
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
        tensor_parallel_size=8,
        use_sequence_parallel_moe=False,
    )
    parallel.update(overrides.pop("parallel", {}))
    text = SimpleNamespace(
        mlp_layer_types=overrides.pop("mlp_layer_types", ["sparse"] * 4)
    )
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**parallel),
        kernel_config=SimpleNamespace(moe_backend=overrides.pop("moe_backend", "auto")),
        model_config=SimpleNamespace(hf_text_config=text),
        # The largest CUDA-graph batch of the reference --max-num-seqs 128 launch.
        compilation_config=SimpleNamespace(
            max_cudagraph_capture_size=overrides.pop("max_graph_tokens", 256)
        ),
    )


@pytest.fixture
def sm120_platform(monkeypatch):
    monkeypatch.setattr(
        muse_model,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_device_capability_family=lambda f: f == 120
        ),
    )


def test_sm120_sequence_parallel_flag_values(monkeypatch) -> None:
    # Unset, "" and "auto" are the auto mode (None); 0 and 1 override it.
    for value, expected in (
        (None, None),
        ("", None),
        ("auto", None),
        ("0", False),
        ("1", True),
    ):
        if value is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, value)
        assert envs.VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL is expected, value


def _communicator_advertises(monkeypatch, advertised: bool) -> list[int]:
    asked: list[int] = []

    def advertises() -> bool:
        asked.append(1)
        return advertised

    monkeypatch.setattr(muse_model, "tp_reduce_scatter_matches_all_reduce", advertises)
    return asked


def test_sm120_sequence_parallel_auto_follows_the_communicator(
    monkeypatch, sm120_platform
) -> None:
    config = _vllm_config()
    monkeypatch.delenv(FLAG, raising=False)
    _communicator_advertises(monkeypatch, True)
    assert muse_model.is_sm120_sequence_parallel_enabled(config)
    assert muse_model.sm120_sequence_parallel_is_gated()
    # NCCL (or any communicator that rounds the reduce-scatter and the all-reduce
    # differently) keeps the stream replicated unless it is asked for.
    _communicator_advertises(monkeypatch, False)
    assert not muse_model.is_sm120_sequence_parallel_enabled(config)
    monkeypatch.setenv(FLAG, "1")
    assert muse_model.is_sm120_sequence_parallel_enabled(config)
    assert not muse_model.sm120_sequence_parallel_is_gated()
    _communicator_advertises(monkeypatch, True)
    monkeypatch.setenv(FLAG, "0")
    assert not muse_model.is_sm120_sequence_parallel_enabled(config)


def test_sm120_sequence_parallel_auto_is_silent_where_it_cannot_apply(
    monkeypatch, sm120_platform
) -> None:
    warnings: list[tuple] = []
    monkeypatch.setattr(
        muse_model.logger, "warning_once", lambda *args: warnings.append(args)
    )
    monkeypatch.delenv(FLAG, raising=False)
    asked = _communicator_advertises(monkeypatch, True)
    for overrides in (
        dict(parallel=dict(enable_expert_parallel=False)),
        dict(parallel=dict(tensor_parallel_size=1)),
        dict(mlp_layer_types=["sparse", "dense"]),
        # Backends that may hand the routing to monolithic experts.
        dict(moe_backend="flashinfer_b12x"),
        dict(moe_backend="marlin"),
    ):
        assert not muse_model.is_sm120_sequence_parallel_enabled(
            _vllm_config(**overrides)
        )
    assert muse_model.is_sm120_sequence_parallel_enabled(
        _vllm_config(moe_backend="cutlass")
    )
    assert not warnings
    # The communicator is only asked where the layout and the backend allow the
    # stream at all (the cutlass launch above).
    assert len(asked) == 1
    # A request for the stream on a backend it cannot feed is not silent.
    monkeypatch.setenv(FLAG, "1")
    assert muse_model.is_sm120_sequence_parallel_enabled(
        _vllm_config(moe_backend="flashinfer_b12x")
    )


def test_tp_reduce_scatter_capability(monkeypatch) -> None:
    from vllm.distributed.device_communicators.base_device_communicator import (
        DeviceCommunicatorBase,
    )
    from vllm.models.common.ops import sequence_parallel as sp_ops

    # A communicator promises nothing unless it says so: NCCL rounds differently.
    assert DeviceCommunicatorBase.reduce_scatter_all_gather_matches_all_reduce is False
    for communicator, expected in (
        (SimpleNamespace(reduce_scatter_all_gather_matches_all_reduce=True), True),
        (SimpleNamespace(reduce_scatter_all_gather_matches_all_reduce=False), False),
        (SimpleNamespace(), False),
        (None, False),
    ):
        monkeypatch.setattr(
            sp_ops,
            "get_tp_group",
            lambda communicator=communicator: SimpleNamespace(
                device_communicator=communicator
            ),
        )
        assert sp_ops.tp_reduce_scatter_matches_all_reduce() is expected


class _Projection(torch.nn.Module):
    """A linear layer that computes the rows of fewer than ``split_below`` rows
    differently (in the last bits), as a split-K cuBLAS kernel does."""

    def __init__(self, in_features: int, out_features: int, split_below: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(in_features)
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features, generator=g), requires_grad=False
        )
        self.split_below = split_below
        self.calls = 0

    def forward(self, x: torch.Tensor):
        self.calls += 1
        out = x @ self.weight.t()
        if x.shape[0] < self.split_below:
            out = out * (1 + 2**-20)
        return out, None


def _row_gate(monkeypatch, split_below=(0, 0, 0), world_size=1):
    monkeypatch.setattr(
        muse_model,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=world_size, cpu_group=None),
    )
    projections = [
        _Projection(32, 16, split_below[0]),
        _Projection(16, 32, split_below[1]),
        _Projection(32, 8, split_below[2]),
    ]
    feed_forward = SimpleNamespace(
        tp_size=8,
        pooler=projections[0],
        unpooler=projections[1],
        router=projections[2],
    )
    return muse_model._SequenceParallelRowGate(feed_forward), projections


def test_sequence_parallel_row_gate_admits_only_matching_sizes(monkeypatch) -> None:
    # Rows of 1024 / 8 = 128 are computed like the full batch, those of 512 / 8 = 64
    # are not (one of the three projections splits).
    gate, projections = _row_gate(monkeypatch, split_below=(0, 100, 0))
    assert gate(1024)
    assert not gate(512)
    assert gate(2048)
    # One verdict per size.
    calls = [p.calls for p in projections]
    assert gate(1024) and not gate(512)
    assert [p.calls for p in projections] == calls
    # A size whose last rank holds padding only compares the rows that exist.
    gate, _ = _row_gate(monkeypatch)
    assert gate(1001) and gate(1003) and gate(1009)
    # Every projection counts.
    for index in range(3):
        split = [0, 0, 0]
        split[index] = 100
        gate, _ = _row_gate(monkeypatch, split_below=tuple(split))
        assert not gate(512), index


def test_sequence_parallel_row_gate_ranks_agree(monkeypatch) -> None:
    gate, _ = _row_gate(monkeypatch, world_size=8)
    reduced: list[torch.Tensor] = []

    def all_reduce(tensor, op=None, group=None):
        assert op == torch.distributed.ReduceOp.MIN
        reduced.append(tensor.clone())
        tensor.zero_()  # another rank of the group found a difference

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    # The local verdict is true, the group's is not: the rank takes the group's.
    assert not gate(1024)
    assert reduced[0].item() == 1
    # A group of one never reduces.
    gate, _ = _row_gate(monkeypatch, world_size=1)
    assert gate(1024) and len(reduced) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        dict(parallel=dict(enable_expert_parallel=False)),
        dict(parallel=dict(tensor_parallel_size=1)),
        dict(parallel=dict(pipeline_parallel_size=2)),
        dict(parallel=dict(data_parallel_size=2)),
        dict(parallel=dict(use_sequence_parallel_moe=True)),
        dict(moe_backend="deep_gemm_mega_moe"),
        dict(mlp_layer_types=["sparse", "dense", "sparse"]),
        dict(mlp_layer_types=None),
    ],
)
def test_sm120_sequence_parallel_needs_the_replicated_ep_layout(
    monkeypatch, sm120_platform, overrides
) -> None:
    monkeypatch.setenv(FLAG, "1")
    warnings: list[str] = []
    monkeypatch.setattr(
        muse_model.logger,
        "warning_once",
        lambda msg, *args: warnings.append(msg % args),
    )
    assert not muse_model.is_sm120_sequence_parallel_enabled(_vllm_config(**overrides))
    # A flag that is set but cannot apply says why, instead of being ignored silently.
    assert len(warnings) == 1 and FLAG in warnings[0], warnings


def test_sm120_sequence_parallel_flag_off_is_silent(
    monkeypatch, sm120_platform
) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        muse_model.logger, "warning_once", lambda *args: warnings.append(args)
    )
    monkeypatch.delenv(FLAG, raising=False)
    assert not muse_model.is_sm120_sequence_parallel_enabled(
        _vllm_config(parallel=dict(enable_expert_parallel=False))
    )
    assert not warnings


def test_sm120_sequence_parallel_is_for_sm120_only(monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    for family in (90, 100):
        monkeypatch.setattr(
            muse_model,
            "current_platform",
            SimpleNamespace(
                is_cuda=lambda: True,
                is_device_capability_family=lambda f, family=family: f == family,
            ),
        )
        assert not muse_model.is_sm120_sequence_parallel_enabled(_vllm_config())


def test_sm120_sequence_parallel_threshold(monkeypatch) -> None:
    config = _vllm_config()
    monkeypatch.delenv(MIN_TOKENS, raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS == 512
    assert muse_model.sm120_sequence_parallel_min_tokens(config) == 512
    monkeypatch.setenv(MIN_TOKENS, "4096")
    assert muse_model.sm120_sequence_parallel_min_tokens(config) == 4096
    # Batches of up to 256 tokens use the fused all-reduce and the sharded
    # projections, and run in CUDA graphs on the --max-num-seqs 128 launch.
    monkeypatch.setenv(MIN_TOKENS, "1")
    assert muse_model.sm120_sequence_parallel_min_tokens(config) == 257


@pytest.mark.parametrize(
    "max_graph_tokens, env_min, expected",
    [
        # Default launch (--max-num-seqs 1024 on a 96 GB GPU): graphs up to 512
        # tokens, so the default minimum of 512 must move up to 513.
        (512, None, 513),
        (512, 300, 513),
        (512, 1, 513),
        (512, 513, 513),
        (512, 4096, 4096),
        # The reference launch and the floor of the fused all-reduce paths.
        (256, None, 512),
        (256, 1, 257),
        (128, 1, 257),
        # Eager (no graphs) and an explicit wide capture range.
        (0, 1, 257),
        (None, 1, 257),
        (1024, None, 1025),
    ],
)
def test_sm120_sequence_parallel_never_takes_cuda_graph_batches(
    monkeypatch, max_graph_tokens, env_min, expected
) -> None:
    """The stream must stay out of every batch the CUDA-graph dispatcher can serve
    (num_tokens <= the largest capture size), whatever the launch and the env."""
    if env_min is None:
        monkeypatch.delenv(MIN_TOKENS, raising=False)
    else:
        monkeypatch.setenv(MIN_TOKENS, str(env_min))
    config = _vllm_config(max_graph_tokens=max_graph_tokens)
    min_tokens = muse_model.sm120_sequence_parallel_min_tokens(config)
    assert min_tokens == expected
    assert min_tokens > (max_graph_tokens or 0)
    assert min_tokens > muse_model._SHARDED_PROJECTION_MAX_TOKENS


def test_sequence_parallel_activation_threshold(dist_init) -> None:
    # Default: one token per rank (a world of one here).
    assert is_sequence_parallel_active(True, 1)
    assert not is_sequence_parallel_active(True, 0)
    # The SM120 stream shards from its own minimum on.
    assert not is_sequence_parallel_active(True, 1023, 1024)
    assert is_sequence_parallel_active(True, 1024, 1024)
    assert is_sequence_parallel_active(True, 8192, 1024)
    assert not is_sequence_parallel_active(False, 8192, 1024)


# ------------------------------------------------------------------------------
# The runner's entry without the combine
# ------------------------------------------------------------------------------
requires_sm120 = pytest.mark.skipif(
    not current_platform.is_device_capability_family(120), reason="Requires SM120"
)

E, N, K, TOPK = 16, 256, 512, 8


def _norm() -> MuseSpark1xRMSNormWithInputScale:
    norm = MuseSpark1xRMSNormWithInputScale(
        K,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(torch.linspace(-0.2, 0.2, K, device="cuda"))
    return norm


def _bf16_runner():
    g = torch.Generator(device="cuda").manual_seed(1)
    weights = {
        "w13_weight": torch.randn(
            E, 2 * N, K, dtype=torch.bfloat16, device="cuda", generator=g
        )
        / 10,
        "w2_weight": torch.randn(
            E, K, N, dtype=torch.bfloat16, device="cuda", generator=g
        )
        / 10,
    }
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    return make_runner(
        _norm(),
        expert_bias,
        weights,
        None,
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
    )


def _nvfp4_runner():
    from tests.kernels.moe.test_muse_spark_sm120_nvfp4_moe import _Problem

    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    return make_runner(
        _norm(),
        expert_bias,
        _Problem().checkpoint(),
        nvfp4_quant_config(),
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
    )


@requires_sm120
@pytest.mark.parametrize("make", [_bf16_runner, _nvfp4_runner])
@pytest.mark.parametrize("num_tokens", [40, 700])
@torch.inference_mode()
def test_forward_routed_partial_is_forward_without_the_combine(
    dist_init, workspace_init, make, num_tokens
) -> None:
    """With one rank there is nothing to combine, so the entry that takes the
    routing from the caller must return exactly what ``forward`` does."""
    router, runner, vllm_config = make()
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(num_tokens, K, device="cuda", generator=g).to(torch.bfloat16)
    logits = torch.randn(num_tokens, E, dtype=torch.float32, device="cuda", generator=g)
    expected = run_moe(runner, vllm_config, x, logits)
    topk_weights, topk_ids = router._compute_routing(x, logits, torch.int32)
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        actual = runner.forward_routed_partial(x, topk_weights, topk_ids)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    assert torch.equal(actual, expected)


@requires_sm120
@torch.inference_mode()
def test_forward_routed_partial_leaves_the_expert_parallel_sum_to_the_caller(
    dist_init, workspace_init, monkeypatch
) -> None:
    """``forward`` all-reduces over the expert-parallel group when there is one;
    ``forward_routed_partial`` must not."""
    router, runner, vllm_config = _bf16_runner()
    calls: list[tuple[int, ...]] = []
    from vllm.model_executor.layers.fused_moe.runner import moe_runner

    monkeypatch.setattr(
        moe_runner,
        "tensor_model_parallel_all_reduce",
        lambda t: calls.append(tuple(t.shape)) or t,
    )
    monkeypatch.setattr(type(runner.moe_config), "ep_size", property(lambda self: 2))
    x = torch.randn(40, K, device="cuda").to(torch.bfloat16)
    logits = torch.randn(40, E, dtype=torch.float32, device="cuda")
    topk_weights, topk_ids = router._compute_routing(x, logits, torch.int32)
    run_moe(runner, vllm_config, x, logits)
    assert calls == [(40, K)]
    calls.clear()
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        runner.forward_routed_partial(x, topk_weights, topk_ids)
    assert calls == []


# ------------------------------------------------------------------------------
# The projections on a rank's rows
# ------------------------------------------------------------------------------
@requires_sm120
@torch.inference_mode()
def test_projections_on_local_rows_stay_accurate(dist_init) -> None:
    """The sharded stream runs the router, pooler and unpooler on T / 8 rows. cuBLAS
    picks its kernel by the row count, so these rows are not bitwise those of the full
    batch (one reason the sharded stream is opt-in, and why the auto mode checks each
    batch size, see ``_SequenceParallelRowGate``). Without
    ``_fp32_split_k_reduction`` the bf16 GEMMs of the small batch are also less exact
    (split-K slices reduced in bf16: 1.2-1.7x the error of a single rounding against an
    fp32 reference on SM120); with it they must stay at the single rounding."""
    torch.manual_seed(0)
    with set_current_vllm_config(_gemm_vllm_config()), torch.device("cuda"):
        pooler = ReplicatedLinear(
            8192, 4096, bias=False, params_dtype=torch.bfloat16, quant_config=None
        )
        unpooler = ReplicatedLinear(
            4096, 8192, bias=False, params_dtype=torch.bfloat16, quant_config=None
        )
        router = GateLinear(
            8192,
            256,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
        )
    for module in (pooler, unpooler, router):
        for p in module.parameters():
            torch.nn.init.normal_(p, std=0.02)
    # (module, input features, max relative L2 error against the fp32 product); the
    # bf16 outputs of an fp32-accumulated GEMM are at 1.66e-3.
    cases = ((pooler, 8192, 2e-3), (unpooler, 4096, 2e-3), (router, 8192, 1e-4))
    for module, in_features, tolerance in cases:
        weight = module.weight.float()
        for num_tokens in (8192, 6000, 2048, 1024):
            x = torch.randn(num_tokens, in_features, device="cuda").to(torch.bfloat16)
            expected = x.float() @ weight.t()
            rows = (num_tokens + 7) // 8
            for start in (0, num_tokens - rows):
                with muse_model._fp32_split_k_reduction():
                    actual = module(x[start : start + rows].contiguous())[0].float()
                reference = expected[start : start + rows]
                rel_l2 = (actual - reference).norm() / reference.norm()
                assert rel_l2 < tolerance, f"{num_tokens} tokens: rel L2 {rel_l2:.2e}"


def test_fp32_split_k_reduction_restores_the_flag() -> None:
    matmul = torch.backends.cuda.matmul
    for initial in (True, False):
        matmul.allow_bf16_reduced_precision_reduction = initial
        with muse_model._fp32_split_k_reduction():
            assert not matmul.allow_bf16_reduced_precision_reduction
        assert matmul.allow_bf16_reduced_precision_reduction == initial
        with pytest.raises(RuntimeError):
            with muse_model._fp32_split_k_reduction():
                raise RuntimeError
        assert matmul.allow_bf16_reduced_precision_reduction == initial
    matmul.allow_bf16_reduced_precision_reduction = True


@requires_sm120
@pytest.mark.parametrize("num_tokens", [512, 1024, 1503, 3000, 4096, 8192])
def test_sequence_parallel_row_gate_matches_the_real_projections(
    num_tokens: int, dist_init
) -> None:
    """The verdict of the gate (on its own random inputs) agrees with the rows of the
    real-shape pooler, unpooler and fp32 router on other inputs: the sizes it admits
    are bitwise equal on a rank's rows, the sizes it refuses differ somewhere."""
    torch.manual_seed(7)
    with set_current_vllm_config(_gemm_vllm_config()), torch.device("cuda"):
        pooler = ReplicatedLinear(8192, 4096, bias=False, params_dtype=torch.bfloat16)
        unpooler = ReplicatedLinear(4096, 8192, bias=False, params_dtype=torch.bfloat16)
        router = GateLinear(
            8192, 256, bias=False, out_dtype=torch.float32, params_dtype=torch.float32
        )
    for module in (pooler, unpooler, router):
        for p in module.parameters():
            p.data.normal_(std=0.02)
    gate = muse_model._SequenceParallelRowGate(
        SimpleNamespace(tp_size=8, pooler=pooler, unpooler=unpooler, router=router)
    )
    rows = -(-num_tokens // 8)
    equal = True
    for module in (pooler, unpooler, router):
        weight = module.weight
        x = torch.randn(num_tokens, weight.shape[1], device="cuda").to(weight.dtype)
        full = module(x)[0]
        padded = F.pad(x, (0, 0, 0, rows * 8 - num_tokens))
        for rank in range(8):
            valid = min(rows, num_tokens - rank * rows)
            with muse_model._fp32_split_k_reduction():
                local = module(padded[rank * rows : (rank + 1) * rows])[0]
            equal &= torch.equal(
                full[rank * rows : rank * rows + valid], local[:valid]
            )
    assert gate(num_tokens) == equal, num_tokens


def _gemm_vllm_config():
    from vllm.config import VllmConfig

    return VllmConfig()


# ------------------------------------------------------------------------------
# The sharded FFN branch against the replicated one, on real expert parallelism
# ------------------------------------------------------------------------------
HIDDEN = 1024
# The expert paths of the model: NVFP4 W4A4 (CUTLASS grouped GEMMs, layers 1-60),
# BF16 (Triton, layers 0 and 61) and the Marlin W4A16 route-major experts.
BRANCH_KINDS = ("nvfp4_cutlass", "bf16_triton", "nvfp4_marlin")
# 257 is the smallest batch the stream takes (33 rows per rank, the last padded).
BRANCH_TOKENS = (1024, 700, 257)


def _marlin_checkpoint() -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cuda").manual_seed(3)

    def packed(*shape):
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device="cuda", generator=g
        )

    def rand(*shape):
        return torch.rand(*shape, dtype=torch.float32, device="cuda", generator=g)

    return {
        "w13_weight": packed(E, 2 * N, K // 2),
        "w13_weight_scale": torch.ones(
            E, 2 * N, K // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "w13_weight_scale_2": rand(E, 2) * 0.01 + 0.01,
        "w13_input_scale": rand(E, 2) + 0.5,
        "w2_weight": packed(E, K, N // 2),
        "w2_weight_scale": torch.ones(
            E, K, N // 16, dtype=torch.float8_e4m3fn, device="cuda"
        ),
        "w2_weight_scale_2": rand(E) * 0.01 + 0.01,
        "w2_input_scale": rand(E) + 0.5,
    }


def _branch_experts(kind: str, expert_bias, parallel):
    """(router, runner, vllm_config) of the expert path ``kind`` under ``parallel``."""
    if kind == "bf16_triton":
        g = torch.Generator(device="cuda").manual_seed(1)
        ckpt = {
            "w13_weight": torch.randn(
                E, 2 * N, K, dtype=torch.bfloat16, device="cuda", generator=g
            )
            / 10,
            "w2_weight": torch.randn(
                E, K, N, dtype=torch.bfloat16, device="cuda", generator=g
            )
            / 10,
        }
        quant_config, backend = None, "auto"
    elif kind == "nvfp4_marlin":
        ckpt = _marlin_checkpoint()
        quant_config, backend = nvfp4_quant_config(), "marlin"
    else:
        from tests.kernels.moe.test_muse_spark_sm120_nvfp4_moe import _Problem

        ckpt, quant_config, backend = (
            _Problem().checkpoint(),
            nvfp4_quant_config(),
            "cutlass",
        )
    return make_runner(
        _norm(),
        expert_bias,
        ckpt,
        quant_config,
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
        moe_backend=backend,
        parallel_config=parallel,
    )


def _sharded_branch_worker(rank: int, world: int, port: int, out_path: str) -> None:
    from vllm.config import KernelConfig, ParallelConfig, VllmConfig
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    from vllm.distributed.parallel_state import graph_capture
    from vllm.v1.worker.workspace import init_workspace_manager

    torch.cuda.set_device(rank)
    parallel = ParallelConfig(tensor_parallel_size=world, enable_expert_parallel=True)
    config = VllmConfig(
        parallel_config=parallel, kernel_config=KernelConfig(moe_backend="cutlass")
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
    init_workspace_manager(torch.device("cuda", rank))
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")

    results = {}
    for kind in BRANCH_KINDS:
        router, runner, vllm_config = _branch_experts(kind, expert_bias, parallel)
        with set_current_vllm_config(vllm_config), torch.device("cuda"):
            pooler = ReplicatedLinear(
                HIDDEN, K, bias=False, params_dtype=torch.bfloat16, quant_config=None
            )
            unpooler = ReplicatedLinear(
                K, HIDDEN, bias=False, params_dtype=torch.bfloat16, quant_config=None
            )
            gate = GateLinear(
                HIDDEN,
                E,
                bias=False,
                out_dtype=torch.float32,
                params_dtype=torch.float32,
            )
            norm = MuseSpark1xRMSNorm(
                K,
                eps=1e-6,
                has_weight=True,
                is_post_norm=False,
                zero_centered_gamma=True,
                post_norm_gain_center_type="one",
                residual_id=2,
                output_norm_gain_center_type="zero",
            )
        g = torch.Generator(device="cuda").manual_seed(5)
        for module, std in ((pooler, HIDDEN**-0.5), (unpooler, K**-0.5), (gate, 0.03)):
            for p in module.parameters():
                p.data.copy_(torch.randn(p.shape, generator=g, device="cuda") * std)
        norm.weight.data.copy_(torch.randn(K, generator=g, device="cuda") * 0.05)
        # The members of MuseSpark1xMoE that forward_sequence_parallel uses.
        layer = SimpleNamespace(
            sm120_sequence_parallel=True,
            pooler=pooler,
            unpooler=unpooler,
            pre_expert_norm=norm,
            moe_router=router,
            router=gate,
            experts=runner,
        )

        def sharded(local, num_tokens, layer=layer, router_input=None):
            return muse_model.MuseSpark1xMoE.forward_sequence_parallel(
                layer, local, num_tokens, router_input=router_input
            )

        for num_tokens in BRANCH_TOKENS:
            x = torch.randn(num_tokens, HIDDEN, generator=g, device="cuda")
            x = x.to(torch.bfloat16)
            with (
                set_current_vllm_config(vllm_config),
                set_forward_context(None, vllm_config),
                torch.inference_mode(),
            ):
                # The replicated branch, as every rank runs it for any batch.
                reference = unpooler(
                    runner.forward(
                        hidden_states=norm(pooler(x)[0]), router_logits=gate(x)[0]
                    )
                )[0]
                # The sharded one: this rank's rows only, padded to a multiple of
                # the world size.
                rows = -(-num_tokens // world)
                padded = F.pad(x, (0, 0, 0, rows * world - num_tokens))
                local = padded[rank * rows : (rank + 1) * rows].contiguous()
                first = sharded(local, num_tokens)
                second = sharded(local, num_tokens)
                # The boundary kernel's fp32 copy of the rows is what the router
                # would cast itself.
                widened = sharded(local, num_tokens, router_input=local.float())
                output = tensor_model_parallel_all_gather(first, 0)[:num_tokens]
                # The same branch captured in a CUDA graph, as a decode-sized batch
                # of a launch with large capture sizes would run it.
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with graph_capture(device=torch.device("cuda", rank)) as capture:
                    with torch.cuda.graph(graph, stream=capture.stream):
                        captured = sharded(local, num_tokens)
                graph.replay()
                torch.cuda.synchronize()
            reference = reference.float()
            diff = output.float() - reference
            results[f"{kind}:{num_tokens}"] = dict(
                rel_l2=float(diff.norm() / reference.norm()),
                max_row_rel_l2=float(
                    (diff.norm(dim=1) / reference.norm(dim=1).clamp_min(1e-12)).max()
                ),
                nonzero=float((reference != 0).float().mean()),
                finite=bool(torch.isfinite(output).all()),
                deterministic=bool(torch.equal(first, second)),
                router_input_equal=bool(torch.equal(first, widened)),
                graph_equal=bool(torch.equal(captured, first)),
                graph_rel_l2=float(
                    (captured.float() - first.float()).norm()
                    / first.float().norm().clamp_min(1e-12)
                ),
            )
            torch.distributed.barrier()
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@requires_sm120
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 or more GPUs")
@pytest.mark.parametrize("world", [2, 8])
def test_sharded_branch_matches_the_replicated_one(tmp_path, world: int) -> None:
    """Every rank's rows of the sharded branch, gathered, are the replicated branch up
    to rounding: random experts (NVFP4 CUTLASS, BF16 Triton, Marlin W4A16) under
    expert parallelism (E / world local experts per rank), the real router, pooler,
    unpooler and runner, and batches that are and are not a multiple of the world
    size. A misrouted, misordered or unreduced row would show as an O(1) error. The
    branch also has to replay from a CUDA graph unchanged."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} GPUs")
    out_path = str(tmp_path / "results.json")
    port = 29600 + os.getpid() % 1000 + world
    mp.spawn(_sharded_branch_worker, args=(world, port, out_path), nprocs=world)
    with open(out_path) as f:
        results = json.load(f)
    assert len(results) == len(BRANCH_KINDS) * len(BRANCH_TOKENS)
    for key, result in results.items():
        assert result["finite"], (key, result)
        # The Marlin experts are not run-to-run reproducible on their own (the
        # replicated stream included), so only the other two have to repeat bitwise.
        if not key.startswith("nvfp4_marlin"):
            assert result["router_input_equal"], (key, result)
            assert result["deterministic"], (key, result)
            assert result["graph_equal"], (key, result)
        else:
            assert result["graph_rel_l2"] < 1e-2, (key, result)
        # Measured on 8 SM120 GPUs: 3.9e-3 to 4.0e-3 and 5.3e-3 to 6.1e-3, the
        # rounding of the collectives and of the pooled rows (which also moves
        # some FP4 codes of the experts' input).
        assert result["nonzero"] > 0.99, (key, result)
        assert result["rel_l2"] < 1e-2, (key, result)
        assert result["max_row_rel_l2"] < 2e-2, (key, result)
