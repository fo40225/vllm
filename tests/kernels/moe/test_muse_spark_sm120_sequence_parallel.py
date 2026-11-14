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


def test_sm120_sequence_parallel_is_opt_in(monkeypatch, sm120_platform) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    config = _vllm_config()
    assert not muse_model.is_sm120_sequence_parallel_enabled(config)
    monkeypatch.setenv(FLAG, "1")
    assert muse_model.is_sm120_sequence_parallel_enabled(config)


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
    assert not muse_model.is_sm120_sequence_parallel_enabled(_vllm_config(**overrides))


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
    monkeypatch.delenv(MIN_TOKENS, raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS == 512
    assert muse_model.sm120_sequence_parallel_min_tokens() == 512
    monkeypatch.setenv(MIN_TOKENS, "4096")
    assert muse_model.sm120_sequence_parallel_min_tokens() == 4096
    # Batches of up to 256 tokens run in CUDA graphs and use the fused all-reduce.
    monkeypatch.setenv(MIN_TOKENS, "1")
    assert muse_model.sm120_sequence_parallel_min_tokens() == 257


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
    batch (one reason the sharded stream is opt-in). Without
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


def _gemm_vllm_config():
    from vllm.config import VllmConfig

    return VllmConfig()


# ------------------------------------------------------------------------------
# The sharded FFN branch against the replicated one, on real expert parallelism
# ------------------------------------------------------------------------------
HIDDEN = 1024


def _sharded_branch_worker(rank: int, world: int, port: int, out_path: str) -> None:
    from tests.kernels.moe.test_muse_spark_sm120_nvfp4_moe import _Problem
    from vllm.config import KernelConfig, ParallelConfig, VllmConfig
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
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
    router, runner, vllm_config = make_runner(
        _norm(),
        expert_bias,
        _Problem().checkpoint(),
        nvfp4_quant_config(),
        hidden_size=K,
        intermediate_size=N,
        top_k=TOPK,
        moe_backend="cutlass",
        parallel_config=parallel,
    )
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

    results = {}
    for num_tokens in (1024, 700):
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
            # The sharded one: this rank's rows only, padded to a multiple of world.
            rows = -(-num_tokens // world)
            padded = F.pad(x, (0, 0, 0, rows * world - num_tokens))
            local = padded[rank * rows : (rank + 1) * rows].contiguous()
            first = muse_model.MuseSpark1xMoE.forward_sequence_parallel(
                layer, local, num_tokens
            )
            second = muse_model.MuseSpark1xMoE.forward_sequence_parallel(
                layer, local, num_tokens
            )
            output = tensor_model_parallel_all_gather(first, 0)[:num_tokens]
        reference = reference.float()
        diff = output.float() - reference
        results[num_tokens] = dict(
            rel_l2=float(diff.norm() / reference.norm()),
            max_row_rel_l2=float(
                (diff.norm(dim=1) / reference.norm(dim=1).clamp_min(1e-12)).max()
            ),
            nonzero=float((reference != 0).float().mean()),
            finite=bool(torch.isfinite(output).all()),
            deterministic=bool(torch.equal(first, second)),
        )
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@requires_sm120
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 or more GPUs")
@pytest.mark.parametrize("world", [2, 8])
def test_sharded_branch_matches_the_replicated_one(tmp_path, world: int) -> None:
    """Every rank's rows of the sharded branch, gathered, are the replicated branch up
    to rounding: random NVFP4 experts under expert parallelism (E / world local experts
    per rank), the real router, pooler, unpooler and runner, a batch that is and one
    that is not a multiple of the world size. A misrouted, misordered or unreduced row
    would show as an O(1) error."""
    if torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} GPUs")
    out_path = str(tmp_path / "results.json")
    port = 29600 + os.getpid() % 1000 + world
    mp.spawn(_sharded_branch_worker, args=(world, port, out_path), nprocs=world)
    with open(out_path) as f:
        results = json.load(f)
    for num_tokens, result in results.items():
        assert result["finite"] and result["deterministic"], (num_tokens, result)
        # Measured on 8 SM120 GPUs: 3.8e-3 and 5.4e-3, the rounding of the collectives
        # and of the pooled rows (which also moves some FP4 codes of the experts' input).
        assert result["nonzero"] > 0.99, (num_tokens, result)
        assert result["rel_l2"] < 2e-2, (num_tokens, result)
        assert result["max_row_rel_l2"] < 5e-2, (num_tokens, result)
