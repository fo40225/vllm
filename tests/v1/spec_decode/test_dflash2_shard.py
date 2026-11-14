# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP sharding of the DFlash2 draft's aux fc (modes 1 and 2).

CPU tests: the column shards hold the right rows of the checkpoint weight and their
gathered outputs are the replicated outputs (the TP collectives are faked). 8-GPU tests:
the real shards and collectives on the real shapes agree with the replicated draft
within rounding, are identical on all ranks, and a CUDA graph replay gives the eager
bits; the target's real forward with the combiner under the sequence-parallel stream and
r2e microbatches returns the projected taps of every row in token order.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models.qwen3_dflash import (
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    draft_column_shard_enabled,
)
from vllm.model_executor.models.qwen3_dflash2 import (
    DFlash2Qwen3ForCausalLM,
    DFlash2Qwen3Model,
)
from vllm.model_executor.models.utils import AutoWeightsLoader

TP = 8
FC_FLAG = "VLLM_MUSE_SPARK_SM120_DRAFT_FC_SHARD"


class FakeTP:
    """A TP group of ``TP`` ranks in one process: ``all_gather`` hands out the shards
    recorded by the ranks (``record`` first, then ``replay``)."""

    def __init__(self, rank: int, world_size: int = TP):
        self.rank_in_group = rank
        self.world_size = world_size
        self.recorded: dict[int, torch.Tensor] = {}
        self.replaying = False

    def all_gather(self, shard: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if not self.replaying:
            self.recorded[self.rank_in_group] = shard.clone()
            return torch.cat([shard] * self.world_size, dim=dim)
        return torch.cat([self.recorded[r] for r in range(self.world_size)], dim=dim)


def set_tp(monkeypatch, rank: int, world_size: int = TP) -> FakeTP:
    group = FakeTP(rank, world_size)
    monkeypatch.setattr(parallel_state, "_TP", group)
    return group


# --------------------------------------------------------------------------------
# The gate.
# --------------------------------------------------------------------------------


def test_the_gate(monkeypatch):
    set_tp(monkeypatch, 0)
    assert not draft_column_shard_enabled(False, 2048, "x")
    assert draft_column_shard_enabled(True, 2048, "x")
    assert not draft_column_shard_enabled(True, 2047, "x")  # no even split
    set_tp(monkeypatch, 0, world_size=1)
    assert not draft_column_shard_enabled(True, 2048, "x")  # nothing to shard over


def test_the_gate_does_not_read_the_group_when_off(monkeypatch):
    monkeypatch.setattr(parallel_state, "_TP", None)
    assert not draft_column_shard_enabled(False, 2048, "x")


HIDDEN = 64


# --------------------------------------------------------------------------------
# The aux fc.
# --------------------------------------------------------------------------------


def fc_model(
    monkeypatch, sharded: bool, rank: int = 0, mode: str | None = None
) -> DFlash2Qwen3Model:
    """The ``_build_fc`` of a DFlash2 model, without building the model."""
    set_tp(monkeypatch, rank)
    monkeypatch.setenv(FC_FLAG, mode or ("1" if sharded else "0"))
    model = object.__new__(DFlash2Qwen3Model)
    nn.Module.__init__(model)
    model.quant_config = None
    return model


def build_fc(model: DFlash2Qwen3Model) -> nn.Module:
    return model._build_fc(
        input_size=5 * HIDDEN,
        output_size=HIDDEN,
        params_dtype=torch.float32,
        prefix="model.fc",
    )


def test_the_fc_is_a_column_shard_when_the_flag_is_set(monkeypatch):
    assert isinstance(build_fc(fc_model(monkeypatch, False)), ReplicatedLinear)
    fc = build_fc(fc_model(monkeypatch, True))
    assert isinstance(fc, ColumnParallelLinear) and fc.gather_output
    assert fc.weight.shape == (HIDDEN // TP, 5 * HIDDEN)
    assert fc.input_size == 5 * HIDDEN
    assert fc.params_dtype == torch.float32


def test_the_other_drafters_keep_a_replicated_fc(monkeypatch):
    """The hook is on the DFlash base class: DFlash itself never shards."""
    set_tp(monkeypatch, 0)
    monkeypatch.setenv(FC_FLAG, "1")
    model = object.__new__(DFlashQwen3Model)
    nn.Module.__init__(model)
    model.quant_config = None
    assert isinstance(build_fc(model), ReplicatedLinear)


def test_the_sharded_fc_gathers_to_the_replicated_result(monkeypatch):
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(HIDDEN, 5 * HIDDEN, generator=generator)
    taps = [torch.randn(11, HIDDEN, generator=generator) for _ in range(5)]
    tap_input = torch.cat(taps, dim=-1)

    def combine(fc: nn.Module) -> torch.Tensor:
        model = SimpleNamespace(model=SimpleNamespace(use_aux_hidden_state=True, fc=fc))
        return DFlashQwen3ForCausalLM.combine_hidden_states(model, tap_input)

    replicated = build_fc(fc_model(monkeypatch, False))
    AutoWeightsLoader(nn.ModuleDict({"fc": replicated})).load_weights(
        [("fc.weight", weight)]
    )
    expected = combine(replicated)

    fcs = []
    for rank in range(TP):
        fc = build_fc(fc_model(monkeypatch, True, rank))
        AutoWeightsLoader(nn.ModuleDict({"fc": fc})).load_weights(
            [("fc.weight", weight)]
        )
        width = HIDDEN // TP
        torch.testing.assert_close(fc.weight, weight[rank * width : (rank + 1) * width])
        fcs.append(fc)
    recorded = {}
    groups = {}
    for rank, fc in enumerate(fcs):
        groups[rank] = set_tp(monkeypatch, rank)
        combine(fc)
        recorded[rank] = groups[rank].recorded[rank]
    for rank, fc in enumerate(fcs):
        groups[rank].recorded = recorded
        groups[rank].replaying = True
        monkeypatch.setattr(parallel_state, "_TP", groups[rank])
        torch.testing.assert_close(combine(fc), expected, rtol=1e-4, atol=1e-4)


# --------------------------------------------------------------------------------
# Mode 2: the aux projection runs in the target's forward.
# --------------------------------------------------------------------------------


def combiner_for(monkeypatch, rank: int, weight: torch.Tensor, mode: str = "2"):
    """``make_aux_combiner`` of a draft with a replicated fc holding ``weight``."""
    fc = build_fc(fc_model(monkeypatch, False, rank, mode))
    AutoWeightsLoader(nn.ModuleDict({"fc": fc})).load_weights([("fc.weight", weight)])
    causal = object.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(causal)
    causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    return causal.make_aux_combiner(torch.float32), fc


@pytest.mark.parametrize("mode", ["0", "1"])
def test_only_mode_2_hands_the_projection_to_the_target(monkeypatch, mode):
    weight = torch.zeros(HIDDEN, 5 * HIDDEN)
    combiner, _ = combiner_for(monkeypatch, 0, weight, mode)
    assert combiner is None


def test_mode_2_keeps_the_weight_replicated(monkeypatch):
    weight = torch.randn(HIDDEN, 5 * HIDDEN)
    combiner, fc = combiner_for(monkeypatch, 3, weight)
    assert combiner is not None
    assert isinstance(fc, ReplicatedLinear)
    torch.testing.assert_close(fc.weight, weight)


def test_a_quantized_fc_is_never_handed_to_the_target(monkeypatch):
    """The combiner slices the weight by rows and calls F.linear: not a packed one."""
    weight = torch.zeros(HIDDEN, 5 * HIDDEN)
    fc = build_fc(fc_model(monkeypatch, False, 0, "2"))
    AutoWeightsLoader(nn.ModuleDict({"fc": fc})).load_weights([("fc.weight", weight)])
    causal = object.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(causal)
    causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    assert causal.make_aux_combiner(torch.float32) is not None
    fc.quant_method = object()  # anything but UnquantizedLinearMethod
    assert causal.make_aux_combiner(torch.float32) is None


def _warnings(monkeypatch) -> list[str]:
    import vllm.model_executor.models.qwen3_dflash2 as dflash2

    messages: list[str] = []
    monkeypatch.setattr(
        dflash2.logger,
        "warning_once",
        lambda message, *args, **kwargs: messages.append(message % args),
    )
    return messages


def test_a_fallback_of_mode_2_is_logged(monkeypatch):
    """A launch that sets the flag but cannot run it must not look like one that does:
    the success line is the speculator's, so the fallback needs its own line."""
    messages = _warnings(monkeypatch)
    weight = torch.zeros(HIDDEN, 5 * HIDDEN)
    fc = build_fc(fc_model(monkeypatch, False, 0, "2"))
    AutoWeightsLoader(nn.ModuleDict({"fc": fc})).load_weights([("fc.weight", weight)])
    causal = object.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(causal)

    causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    assert causal.make_aux_combiner(torch.float32) is not None
    assert messages == []

    fc.quant_method = object()
    assert causal.make_aux_combiner(torch.float32) is None
    assert len(messages) == 1 and "quantized" in messages[0]

    causal.model = SimpleNamespace(use_aux_hidden_state=False)
    assert causal.make_aux_combiner(torch.float32) is None
    assert len(messages) == 2 and "no aux hidden states" in messages[1]


def test_a_width_that_does_not_divide_is_logged(monkeypatch):
    messages = _warnings(monkeypatch)
    fc = SimpleNamespace(
        quant_method=UnquantizedLinearMethod(), output_size=HIDDEN + 1, input_size=3
    )
    causal = object.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(causal)
    causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    set_tp(monkeypatch, 0)
    monkeypatch.setenv(FC_FLAG, "2")

    assert causal.make_aux_combiner(torch.float32) is None
    assert messages and "does not divide" in messages[-1]


@pytest.mark.parametrize("mode", ["0", "1", "2"])
def test_nothing_is_said_where_there_is_nothing_to_split_over(monkeypatch, mode):
    messages = _warnings(monkeypatch)
    set_tp(monkeypatch, 0, world_size=1)
    monkeypatch.setenv(FC_FLAG, mode)
    causal = object.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(causal)
    causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=None)

    assert causal.make_aux_combiner(torch.float32) is None
    assert messages == []


def test_the_taps_are_cast_to_the_dtype_of_the_fc_like_the_replicated_path(
    monkeypatch,
):
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(HIDDEN, 5 * HIDDEN, generator=generator)
    taps = [
        torch.randn(3, HIDDEN, generator=generator).to(torch.bfloat16) for _ in range(5)
    ]
    _, fc = combiner_for(monkeypatch, 0, weight)
    assert fc.params_dtype == torch.float32
    causal = object.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(causal)
    causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    # The speculator's dtype (bf16) differs from the fc's (fp32), as the F.linear of
    # the replicated path would see them: the combiner casts the concat to the fc's.
    bf16_combiner = causal.make_aux_combiner(torch.bfloat16)

    out = bf16_combiner(taps, True)

    assert out.dtype == torch.float32
    torch.testing.assert_close(out, fc(torch.cat(taps, dim=-1).float()))


def test_taps_of_the_wrong_width_are_refused(monkeypatch):
    weight = torch.zeros(HIDDEN, 5 * HIDDEN)
    combiner, _ = combiner_for(monkeypatch, 0, weight)
    taps = [torch.zeros(3, HIDDEN) for _ in range(4)]
    with pytest.raises(ValueError, match="target_layer_ids"):
        combiner(taps, True)


def test_an_unknown_mode_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="expected 0, 1 or 2"):
        build_fc(fc_model(monkeypatch, False, mode="3"))


def test_local_rows_are_projected_with_the_full_weight(monkeypatch):
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn(HIDDEN, 5 * HIDDEN, generator=generator)
    taps = [torch.randn(3, HIDDEN, generator=generator) for _ in range(5)]
    combiner, _ = combiner_for(monkeypatch, 5, weight)

    out = combiner(taps, True)

    assert out.shape == (3, HIDDEN)
    torch.testing.assert_close(
        out, torch.cat(taps, dim=-1) @ weight.T, rtol=1e-4, atol=1e-4
    )


def test_all_rows_are_projected_with_the_rank_slice_and_gathered(monkeypatch):
    generator = torch.Generator().manual_seed(3)
    weight = torch.randn(HIDDEN, 5 * HIDDEN, generator=generator)
    taps = [torch.randn(11, HIDDEN, generator=generator) for _ in range(5)]
    expected = torch.cat(taps, dim=-1) @ weight.T
    width = HIDDEN // TP

    groups = {}
    combiners = {}
    for rank in range(TP):
        combiners[rank], _ = combiner_for(monkeypatch, rank, weight)
        groups[rank] = parallel_state._TP
    recorded = {}
    for rank in range(TP):
        monkeypatch.setattr(parallel_state, "_TP", groups[rank])
        combiners[rank](taps, False)
        recorded[rank] = groups[rank].recorded[rank]
    for rank in range(TP):
        # Each rank contributed its rows [rank * width, (rank + 1) * width) of fc.
        torch.testing.assert_close(
            recorded[rank],
            expected[:, rank * width : (rank + 1) * width],
            rtol=1e-4,
            atol=1e-4,
        )
        groups[rank].recorded = recorded
        groups[rank].replaying = True
        monkeypatch.setattr(parallel_state, "_TP", groups[rank])
        torch.testing.assert_close(
            combiners[rank](taps, False), expected, rtol=1e-4, atol=1e-4
        )


# --------------------------------------------------------------------------------
# Real shapes on 8 GPUs.
# --------------------------------------------------------------------------------

REAL_HIDDEN = 8192
REAL_FC_IN = 5 * REAL_HIDDEN
FC_ROWS = (8, 128, 512, 1024, 2051)
# Mode 2: the decode rows that stay under the sequence-parallel floor, and the
# sequence-parallel batches (a verify batch, a chunk with a ragged tail).
COMBINE_ROWS = (8, 128, 383)
COMBINE_SP_ROWS = (512, 1024, 8185)


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.float() - reference).abs().max() / reference.abs().max())


def _worker(rank: int, world: int, port: int, out_path: str) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    from vllm.distributed.parallel_state import graph_capture
    from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard

    torch.cuda.set_device(rank)
    if os.environ.get("MUSE_TOPO_ENABLE") == "1":
        # The host plugin patches the TP communicator (the collectives of production).
        from vllm.plugins import load_general_plugins

        load_general_plugins()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world)

    def build(sharded: bool):
        os.environ[FC_FLAG] = "1" if sharded else "0"
        with set_current_vllm_config(config), torch.device("cuda"):
            model = object.__new__(DFlash2Qwen3Model)
            nn.Module.__init__(model)
            model.quant_config = None
            return model._build_fc(
                input_size=REAL_FC_IN,
                output_size=REAL_HIDDEN,
                params_dtype=torch.bfloat16,
                prefix="model.fc",
            )

    g = torch.Generator(device="cuda").manual_seed(11)
    fc_weight = (
        torch.randn(REAL_HIDDEN, REAL_FC_IN, generator=g, device="cuda")
        / REAL_FC_IN**0.5
    ).to(torch.bfloat16)

    arms = {}
    for name, sharded in (("replicated", False), ("sharded", True)):
        fc = build(sharded)
        AutoWeightsLoader(nn.ModuleDict({"fc": fc})).load_weights(
            [("fc.weight", fc_weight)]
        )
        arms[name] = fc
    assert isinstance(arms["sharded"], ColumnParallelLinear)
    results = {}
    with torch.inference_mode():
        for rows in FC_ROWS:
            taps = torch.randn(rows, REAL_FC_IN, generator=g, device="cuda").to(
                torch.bfloat16
            )
            reference = taps.float() @ fc_weight.float().T
            outputs = {name: fc(taps).clone() for name, fc in arms.items()}
            sharded_fc = arms["sharded"]
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            static_taps = taps.clone()
            with (
                graph_capture(device=torch.device("cuda", rank)) as capture,
                torch.cuda.graph(graph, stream=capture.stream),
            ):
                captured = sharded_fc(static_taps)
            graph.replay()
            torch.cuda.synchronize()
            everyone = tensor_model_parallel_all_gather(
                outputs["sharded"].reshape(1, -1), 0
            )
            results[f"fc_{rows}"] = dict(
                sharded_error=_relative_error(outputs["sharded"], reference),
                replicated_error=_relative_error(outputs["replicated"], reference),
                graph_equal=bool(torch.equal(captured, outputs["sharded"])),
                same_on_all_ranks=bool((everyone == everyone[0]).all()),
            )
        # Mode 2: the projection as the target's forward runs it.
        os.environ[FC_FLAG] = "2"
        with set_current_vllm_config(config), torch.device("cuda"):
            model = object.__new__(DFlash2Qwen3Model)
            nn.Module.__init__(model)
            model.quant_config = None
            fc2 = model._build_fc(
                input_size=REAL_FC_IN,
                output_size=REAL_HIDDEN,
                params_dtype=torch.bfloat16,
                prefix="model.fc",
            )
        AutoWeightsLoader(nn.ModuleDict({"fc": fc2})).load_weights(
            [("fc.weight", fc_weight)]
        )
        assert isinstance(fc2, ReplicatedLinear)
        causal = object.__new__(DFlash2Qwen3ForCausalLM)
        nn.Module.__init__(causal)
        causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc2)
        combine = causal.make_aux_combiner(torch.bfloat16)
        assert combine is not None
        for rows in COMBINE_ROWS + COMBINE_SP_ROWS:
            sp = rows in COMBINE_SP_ROWS
            taps = [
                torch.randn(rows, REAL_HIDDEN, generator=g, device="cuda").to(
                    torch.bfloat16
                )
                for _ in range(5)
            ]
            reference = torch.cat(taps, dim=-1).float() @ fc_weight.float().T
            replicated = arms["replicated"](torch.cat(taps, dim=-1)).clone()

            def run(taps=taps, sp=sp, rows=rows):
                if not sp:
                    return combine(taps, False)
                local = combine([sp_shard(tap) for tap in taps], True)
                return sp_all_gather(local)[:rows]

            eager = run().clone()
            graph_equal = True
            if not sp:  # the sequence-parallel stream is never captured
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with (
                    graph_capture(device=torch.device("cuda", rank)) as capture,
                    torch.cuda.graph(graph, stream=capture.stream),
                ):
                    captured = run()
                graph.replay()
                torch.cuda.synchronize()
                graph_equal = bool(torch.equal(captured, eager))
            everyone = tensor_model_parallel_all_gather(eager.reshape(1, -1), 0)
            results[f"combine_{'sp_' if sp else ''}{rows}"] = dict(
                sharded_error=_relative_error(eager, reference),
                replicated_error=_relative_error(replicated, reference),
                graph_equal=graph_equal,
                same_on_all_ranks=bool((everyone == everyone[0]).all()),
            )
        torch.distributed.barrier()
    if rank == 0:
        topo = getattr(get_tp_group().device_communicator, "topo_comm", None)
        # Which collectives ran: the host plugin's (MUSE_TOPO_ENABLE=1) or NCCL.
        results["_collectives"] = "muse_topo" if topo is not None else "nccl"
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires 8 GPUs")
def test_the_real_shards_agree_with_the_replicated_draft_on_8_ranks(tmp_path):
    """Real shapes (hidden 8192, fc 40960 -> 8192), real
    collectives (the host plugin's when MUSE_TOPO_ENABLE=1): the sharded outputs agree
    with the fp32 reference to bf16 rounding (as the replicated ones, within 4x of their
    error), are identical on all ranks, and a captured CUDA graph replays the eager
    bits."""
    out_path = str(tmp_path / "results.json")
    port = 29800 + os.getpid() % 1000
    mp.spawn(_worker, args=(8, port, out_path), nprocs=8)
    with open(out_path) as f:
        results = json.load(f)
    print("collectives:", results.pop("_collectives"))
    for name, result in results.items():
        print(name, result)
    assert set(results) == (
        {f"fc_{r}" for r in FC_ROWS}
        | {f"combine_{r}" for r in COMBINE_ROWS}
        | {f"combine_sp_{r}" for r in COMBINE_SP_ROWS}
    )
    for name, result in results.items():
        assert result["graph_equal"], (name, result)
        assert result["same_on_all_ranks"], (name, result)
        # bf16 outputs (max error against the fp32 reference, relative to the largest
        # element): the rounding of the output alone is 2^-9 = 0.2 %; measured 0.3 %
        # .. 0.7 % for the replicated GEMMs and 0.3 % .. 1.0 % for the shards, which
        # are another GEMM shape (the library picks another tiling and reduction).
        assert result["sharded_error"] <= 4 * result["replicated_error"] + 1e-3, (
            name,
            result,
        )
        assert result["sharded_error"] < 1.5e-2, (name, result)


# --------------------------------------------------------------------------------
# The combiner in the target's real forward: sequence-parallel stream, r2e microbatches.
# --------------------------------------------------------------------------------

FORWARD_LAYERS = 6
# Whole batches on the sequence-parallel stream (a verify batch, a chunk with a ragged
# tail); microbatched batches as r2e splits them (both halves on the stream, a ragged
# split, an odd split, and one half under the stream's floor: 3000 + 300 tokens, whose
# microbatches take different paths and issue different numbers of collectives).
FORWARD_WHOLE = ((512,), (1024,), (8185,))
FORWARD_SPLIT = ((512, 1024), (4096, 4089), (1025, 1025), (3000, 300))


class _FakeLayer:
    """A layer that adds 1/3 to the stream and, like the real ones, takes part in the
    sequence-parallel stream (the rows it is handed are the rank's own rows)."""

    is_sequence_parallel = True

    def __call__(self, positions, x):
        return x + 1.0 / 3.0


def _forward_worker(rank: int, world: int, port: int, out_path: str) -> None:
    import threading

    import vllm.models.muse_spark.nvidia.model as model_ops
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    from vllm.v1.worker.gpu.ubatch_utils import merge_ubatch_outputs
    from vllm.v1.worker.ubatching import dbo_abort, make_ubatch_contexts

    torch.cuda.set_device(rank)
    if os.environ.get("MUSE_TOPO_ENABLE") == "1":
        # The host plugin patches the TP communicator (the collectives of production).
        from vllm.plugins import load_general_plugins

        load_general_plugins()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world)
    model_ops.get_pp_group = lambda: SimpleNamespace(
        is_first_rank=True, is_last_rank=True, world_size=1
    )

    g = torch.Generator(device="cuda").manual_seed(23)
    fc_weight = (
        torch.randn(REAL_HIDDEN, REAL_FC_IN, generator=g, device="cuda")
        / REAL_FC_IN**0.5
    ).to(torch.bfloat16)
    os.environ[FC_FLAG] = "2"
    with set_current_vllm_config(config), torch.device("cuda"):
        draft = object.__new__(DFlash2Qwen3Model)
        nn.Module.__init__(draft)
        draft.quant_config = None
        fc = draft._build_fc(
            input_size=REAL_FC_IN,
            output_size=REAL_HIDDEN,
            params_dtype=torch.bfloat16,
            prefix="model.fc",
        )
    AutoWeightsLoader(nn.ModuleDict({"fc": fc})).load_weights(
        [("fc.weight", fc_weight)]
    )
    causal = object.__new__(DFlash2Qwen3ForCausalLM)
    nn.Module.__init__(causal)
    causal.model = SimpleNamespace(use_aux_hidden_state=True, fc=fc)
    combiner = causal.make_aux_combiner(torch.bfloat16)
    assert combiner is not None

    def target(with_combiner: bool):
        """A Muse Spark model with 5 taps and fake layers, its real forward."""
        model = object.__new__(model_ops.MuseSpark1xModel)
        model.start_layer, model.end_layer = 0, FORWARD_LAYERS
        model.layers = [_FakeLayer() for _ in range(FORWARD_LAYERS)]
        model.residual_dtype = torch.float32
        model.default_dtype = torch.bfloat16
        model.aux_hidden_state_dtype = torch.bfloat16
        model.aux_hidden_state_layers = tuple(range(1, 6))
        model._aux_upstream_total_cached = 0
        model.is_sequence_parallel = True
        model.is_sequence_parallel_input = True
        model.sp_min_tokens = 384
        model.sp_row_gate = None
        model.fuse_across_layers = True
        model._l2_prefetcher = None
        model._expert_prefetch = None
        model.norm = lambda x, out_dtype=None: x.to(out_dtype or x.dtype)
        model.set_aux_combiner(combiner if with_combiner else None)
        return model

    comm_stream = torch.cuda.Stream(priority=-1)

    def run(model, x: torch.Tensor, sizes: tuple[int, ...]):
        """One whole forward, or r2e's microbatches of ``sizes`` on threads that hand
        off at every collective (the real contexts and comm stream), merged as the
        runner merges them."""
        positions = torch.arange(x.shape[0], device="cuda")
        if len(sizes) == 1:
            return model.forward(None, positions, None, inputs_embeds=x)
        bounds = [0]
        for size in sizes:
            bounds.append(bounds[-1] + size)
        barrier = threading.Barrier(len(sizes) + 1)
        contexts = make_ubatch_contexts(
            num_micro_batches=len(sizes),
            compute_stream=torch.cuda.current_stream(),
            comm_stream=comm_stream,
            forward_contexts=[object() for _ in sizes],
            ready_barrier=barrier,
            overlap_collectives=True,
        )
        outputs: dict[int, object] = {}
        errors: dict[int, BaseException] = {}

        def microbatch(context, rows: slice) -> None:
            try:
                torch.cuda.set_device(rank)
                with torch.inference_mode(), context:
                    outputs[context.id] = model.forward(
                        None, positions[rows], None, inputs_embeds=x[rows]
                    )
            except BaseException as error:  # noqa: BLE001
                errors[context.id] = error
                dbo_abort()

        threads = [
            threading.Thread(
                target=microbatch,
                args=(context, slice(bounds[i], bounds[i + 1])),
            )
            for i, context in enumerate(contexts)
        ]
        for thread in threads:
            thread.start()
        barrier.wait()
        contexts[0].cpu_wait_event.set()
        for thread in threads:
            thread.join()
        assert not errors, errors
        return merge_ubatch_outputs([outputs.pop(i) for i in range(len(sizes))])

    def projection_reference(taps: list[torch.Tensor], sizes: tuple[int, ...]):
        """The bits the combiner must give: per microbatch, a stream batch projects
        each rank's block of ceil(rows / world) rows with the full weight, one under
        the floor projects all rows with every rank's slice of the weight."""
        pieces, start = [], 0
        for size in sizes:
            hidden = torch.cat([tap[start : start + size] for tap in taps], dim=-1)
            start += size
            if size >= 384:
                chunk = -(-size // world)
                padded = torch.nn.functional.pad(
                    hidden, (0, 0, 0, chunk * world - size)
                )
                blocks = [fc(padded[b * chunk : (b + 1) * chunk]) for b in range(world)]
                pieces.append(torch.cat(blocks, dim=0)[:size])
            else:
                rows = REAL_HIDDEN // world
                slices = [
                    torch.nn.functional.linear(
                        hidden, fc_weight[r * rows : (r + 1) * rows]
                    )
                    for r in range(world)
                ]
                pieces.append(torch.cat(slices, dim=-1))
        return torch.cat(pieces, dim=0)

    results = {}
    compute_stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(compute_stream):
        for sizes in FORWARD_WHOLE + FORWARD_SPLIT:
            total = sum(sizes)
            x = torch.randn(total, REAL_HIDDEN, generator=g, device="cuda").to(
                torch.bfloat16
            )
            # What the taps and the output are, row by row: the fake layers add 1/3 to
            # the fp32 stream one after the other, the taps are cast after layers 1..5.
            stream = x.float()
            taps_reference = []
            for layer in range(1, FORWARD_LAYERS + 1):
                stream = stream + 1.0 / 3.0
                if layer <= 5:
                    taps_reference.append(stream.to(torch.bfloat16))
            output_reference = stream.to(torch.bfloat16)

            output, taps = run(target(False), x, sizes)
            plain = dict(
                plain_taps_exact=len(taps) == 5
                and all(torch.equal(t, r) for t, r in zip(taps, taps_reference)),
                plain_output_exact=bool(torch.equal(output, output_reference)),
            )
            del output, taps

            output, aux = run(target(True), x, sizes)
            torch.cuda.synchronize()
            fp32_reference = (
                torch.cat(taps_reference, dim=-1).float() @ fc_weight.float().T
            )
            exact_reference = projection_reference(taps_reference, sizes)
            everyone = tensor_model_parallel_all_gather(aux[0].reshape(1, -1), 0)
            results["+".join(map(str, sizes))] = dict(
                aux_count=len(aux),
                aux_shape=list(aux[0].shape),
                error=_relative_error(aux[0], fp32_reference),
                exact=bool(torch.equal(aux[0], exact_reference)),
                output_exact=bool(torch.equal(output, output_reference)),
                same_on_all_ranks=bool((everyone == everyone[0]).all()),
                **plain,
            )
            del output, aux, everyone
        torch.distributed.barrier()
    if rank == 0:
        topo = getattr(get_tp_group().device_communicator, "topo_comm", None)
        results["_collectives"] = "muse_topo" if topo is not None else "nccl"
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires 8 GPUs")
def test_the_combiner_in_the_real_forward_under_sp_and_microbatches(tmp_path):
    """The target's real ``MuseSpark1xModel.forward`` (fake layers, real collectives
    and real r2e microbatch contexts, the host plugin's collectives when
    MUSE_TOPO_ENABLE=1) with the real combiner of a DFlash2 draft (fc 40960 -> 8192):
    whole sequence-parallel batches of 512 / 1024 / 8185 tokens and r2e splits of them
    (also a ragged one and one whose microbatches take different paths). The one aux
    hidden state it returns is the projection of the taps of every row, in token order,
    bit for bit what the ranks' blocks give, identical on all ranks; without the
    combiner the 5 taps come back as before."""
    out_path = str(tmp_path / "forward.json")
    port = 29900 + os.getpid() % 1000
    mp.spawn(_forward_worker, args=(8, port, out_path), nprocs=8)
    with open(out_path) as f:
        results = json.load(f)
    print("collectives:", results.pop("_collectives"))
    for name, result in results.items():
        print(name, result)
    assert set(results) == {
        "+".join(map(str, sizes)) for sizes in FORWARD_WHOLE + FORWARD_SPLIT
    }
    for name, result in results.items():
        total = sum(int(n) for n in name.split("+"))
        assert result["aux_count"] == 1, (name, result)
        assert result["aux_shape"] == [total, REAL_HIDDEN], (name, result)
        # A misplaced row would be off by the size of the values, not of the rounding.
        assert result["error"] < 1.5e-2, (name, result)
        assert result["exact"], (name, result)
        assert result["output_exact"], (name, result)
        assert result["same_on_all_ranks"], (name, result)
        assert result["plain_taps_exact"], (name, result)
        assert result["plain_output_exact"], (name, result)


# --------------------------------------------------------------------------------
# The speculator hands the projection to the target.
# --------------------------------------------------------------------------------


def _speculator(monkeypatch, draft_combiner, holder, pp_world_size=1):
    import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator
    from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator

    monkeypatch.setattr(
        dflash_speculator,
        "get_pp_group",
        lambda: SimpleNamespace(world_size=pp_world_size),
    )
    speculator = object.__new__(DFlash2Speculator)
    speculator.dtype = torch.bfloat16
    draft = SimpleNamespace(make_aux_combiner=lambda dtype: draft_combiner)
    target = SimpleNamespace(get_language_model=lambda: SimpleNamespace(model=holder))
    return speculator, target, draft


def test_the_target_gets_the_combiner_and_propose_takes_its_result(monkeypatch):
    registered = []
    holder = SimpleNamespace(set_aux_combiner=registered.append)
    combiner = object()
    speculator, target, draft = _speculator(monkeypatch, combiner, holder)
    assert not speculator.aux_combined_by_target

    speculator._hand_aux_projection_to_target(target, draft)

    assert registered == [combiner] and speculator.aux_combined_by_target
    projected = torch.zeros(4, HIDDEN)
    assert speculator.combine_aux_hidden_states([projected]) is projected


def test_no_combiner_keeps_the_projection_in_propose(monkeypatch):
    registered = []
    holder = SimpleNamespace(set_aux_combiner=registered.append)
    speculator, target, draft = _speculator(monkeypatch, None, holder)

    speculator._hand_aux_projection_to_target(target, draft)

    assert registered == [] and not speculator.aux_combined_by_target


def test_a_drafter_without_the_offer_keeps_the_projection_in_propose(monkeypatch):
    registered = []
    holder = SimpleNamespace(set_aux_combiner=registered.append)
    speculator, target, _ = _speculator(monkeypatch, object(), holder)

    speculator._hand_aux_projection_to_target(target, SimpleNamespace())

    assert registered == [] and not speculator.aux_combined_by_target


@pytest.mark.parametrize("holder_has_hook, pp", [(False, 1), (True, 2)])
def test_a_target_that_cannot_run_it_keeps_the_projection(
    monkeypatch, holder_has_hook, pp
):
    registered = []
    holder = (
        SimpleNamespace(set_aux_combiner=registered.append)
        if holder_has_hook
        else SimpleNamespace()
    )
    speculator, target, draft = _speculator(monkeypatch, object(), holder, pp)

    speculator._hand_aux_projection_to_target(target, draft)

    assert registered == [] and not speculator.aux_combined_by_target
