# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layers that take a static expert placement and the model's hook (CPU only).

Which layers can run a placement (the two expert paths of the model that locate their
routes through ``expert_map``), what the constructor does under the flag (installs the
table, or leaves the layer linear with the reason), the guards (EPLB, MegaMoE), and the
loader premises a placement relies on. The table and its install are in
``test_muse_spark_expert_placement.py``.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
from tests.kernels.core.test_muse_spark_expert_placement import (
    EP,
    EXPERTS,
    LAYERS,
    PER_RANK,
    SHIPPED,
    _manager,
    _Routed,
    _small_json,
    _small_table,
    _snapshot_dir,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight
from vllm.model_executor.model_loader.reload.meta import SKIP_LOAD_TENSORS
from vllm.models.muse_spark.nvidia import model as muse_model
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseSparkModelOptNvFp4FusedMoE,
    MuseSparkUnquantizedFusedMoE,
    expert_placement_blocker,
)
from vllm.models.muse_spark.nvidia.ops.expert_placement import ExpertPlacementTable


# ------------------------------------------------------------------------------
# The layers that cannot take a placement
# ------------------------------------------------------------------------------
def _quant_method(cls, **attrs):
    method = object.__new__(cls)
    for name, value in attrs.items():
        setattr(method, name, value)
    return method


def _routed(quant_method, *, eplb=False, all2all=False, ag_rs=False):
    parallel = SimpleNamespace(
        enable_eplb=eplb, use_all2all_kernels=all2all, use_ag_rs_all2all_kernels=ag_rs
    )
    return SimpleNamespace(
        quant_method=quant_method,
        moe_config=SimpleNamespace(moe_parallel_config=parallel),
    )


def _sm120_native() -> MuseSparkModelOptNvFp4FusedMoE:
    return _quant_method(MuseSparkModelOptNvFp4FusedMoE, use_sm120_native=True)


def _bf16_triton() -> MuseSparkUnquantizedFusedMoE:
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts

    return _quant_method(
        MuseSparkUnquantizedFusedMoE, use_route_major=True, experts_cls=TritonExperts
    )


def test_the_two_expert_paths_of_the_model_take_a_placement() -> None:
    assert expert_placement_blocker(_routed(_sm120_native())) is None
    assert expert_placement_blocker(_routed(_bf16_triton())) is None
    # An all2all that is the allgather + reduce-scatter pair still replicates tokens.
    assert (
        expert_placement_blocker(_routed(_sm120_native(), all2all=True, ag_rs=True))
        is None
    )


def test_experts_that_read_the_rank_number_do_not() -> None:
    marlin_or_trtllm = _quant_method(
        MuseSparkModelOptNvFp4FusedMoE, use_sm120_native=False
    )
    assert "SM120 native" in expert_placement_blocker(_routed(marlin_or_trtllm))
    trtllm_bf16 = _quant_method(
        MuseSparkUnquantizedFusedMoE, use_route_major=False, experts_cls=object
    )
    assert "Triton" in expert_placement_blocker(_routed(trtllm_bf16))
    wrong_experts = _quant_method(
        MuseSparkUnquantizedFusedMoE, use_route_major=True, experts_cls=object
    )
    assert "Triton" in expert_placement_blocker(_routed(wrong_experts))
    assert "unknown" in expert_placement_blocker(_routed(object()))


def test_eplb_and_dispatching_all2all_do_not() -> None:
    assert "EPLB" in expert_placement_blocker(_routed(_sm120_native(), eplb=True))
    assert "all2all" in expert_placement_blocker(
        _routed(_bf16_triton(), all2all=True, ag_rs=False)
    )


# ------------------------------------------------------------------------------
# The model's hook
# ------------------------------------------------------------------------------
class _TableFile:
    """A packaged-style table of 3 layers, EP4, 16 experts, for the snapshot."""

    def __init__(self, tmp_path: Path):
        self.path = tmp_path / "table.json"
        self.path.write_text(json.dumps(_small_json()))


def _moe_stub(manager: ExpertMapManager, quant_method=None):
    routed = _Routed(manager)
    routed.quant_method = quant_method or _sm120_native()
    routed.moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            use_all2all_kernels=False,
            use_ag_rs_all2all_kernels=False,
        )
    )
    return SimpleNamespace(
        experts=SimpleNamespace(routed_experts=routed),
        num_experts=16,
        top_k=2,
        expert_placement=None,
    ), routed


def _vllm_config(model: str, layers: int = 3):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model=model,
            revision=None,
            hf_text_config=SimpleNamespace(num_hidden_layers=layers),
        )
    )


def _install(stub, config, layer_idx: int) -> None:
    muse_model.MuseSpark1xMoE._install_expert_placement(stub, config, layer_idx)


@pytest.fixture(autouse=True)
def flag_on(monkeypatch) -> None:
    """The hook acts under the flag; the tests that want it off say so."""
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", "1")


def test_flag_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", raising=False)
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT is False
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE is None
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", "1")
    monkeypatch.setenv(
        "VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", "/some/table.json"
    )
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT is True
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE == "/some/table.json"
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", "")
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE is None


def test_the_constructor_runs_the_hook() -> None:
    import inspect

    source = inspect.getsource(muse_model.MuseSpark1xMoE.__init__)
    assert "self._install_expert_placement(vllm_config, layer_idx)" in source


def test_without_the_flag_the_hook_leaves_the_layer_alone(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", "0")
    monkeypatch.setenv(
        "VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", str(_TableFile(tmp_path).path)
    )
    warnings = _warnings(monkeypatch)
    stub, routed = _moe_stub(_manager(4, 2, 16))
    stock = routed.expert_map_manager.expert_map.clone()
    _install(stub, _vllm_config(str(tmp_path)), layer_idx=1)
    assert stub.expert_placement is None and routed.updates == 0
    assert torch.equal(routed.expert_map_manager.expert_map, stock)
    assert routed.expert_map_manager.placement_strategy == "linear"
    assert not warnings


def test_hook_installs_the_table_and_records_the_local_experts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(
        "VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", str(_TableFile(tmp_path).path)
    )
    stub, routed = _moe_stub(_manager(4, 2, 16))
    _install(stub, _vllm_config(str(tmp_path)), layer_idx=1)
    table = _small_table()
    assert stub.expert_placement == table.local_experts(1, 2)
    assert torch.equal(routed.expert_map_manager.expert_map, table.expert_map(1, 2))
    assert routed.updates == 1


def test_hook_uses_the_packaged_table_of_the_snapshot(tmp_path) -> None:
    ep_rank = 5
    stub, routed = _moe_stub(_manager(EP, ep_rank, EXPERTS))
    stub.num_experts, stub.top_k = EXPERTS, 8
    config = _vllm_config(str(_snapshot_dir(tmp_path)), layers=LAYERS)
    _install(stub, config, layer_idx=17)
    table = ExpertPlacementTable.load(SHIPPED)
    assert stub.expert_placement == table.local_experts(17, ep_rank)
    assert len(stub.expert_placement) == PER_RANK
    installed = routed.expert_map_manager.expert_map
    assert torch.equal(installed, table.expert_map(17, ep_rank))
    # Layers 0 and 61 (BF16) have their entries too.
    for layer_idx in (0, 61):
        stub, routed = _moe_stub(_manager(EP, ep_rank, EXPERTS), _bf16_triton())
        stub.num_experts, stub.top_k = EXPERTS, 8
        _install(stub, config, layer_idx=layer_idx)
        assert stub.expert_placement == table.local_experts(layer_idx, ep_rank)


def _warnings(monkeypatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(
        muse_model.logger, "warning_once", lambda msg, *args: seen.append(msg % args)
    )
    return seen


@pytest.mark.parametrize(
    "case", ["no snapshot", "other snapshot", "ep size", "drafter layer", "blocker"]
)
def test_hook_leaves_the_layer_linear_with_the_reason(
    case, tmp_path, monkeypatch
) -> None:
    warnings = _warnings(monkeypatch)
    ep_size, experts, layers, layer_idx = EP, EXPERTS, LAYERS, 3
    quant_method = None
    model = str(_snapshot_dir(tmp_path))
    if case == "no snapshot":
        model = str(tmp_path)
    elif case == "other snapshot":
        model = str(_snapshot_dir(tmp_path / "x", "1234567890abcdef"))
    elif case == "ep size":
        ep_size = 4
    elif case == "drafter layer":
        layer_idx = LAYERS  # beyond the table
    elif case == "blocker":
        quant_method = _quant_method(
            MuseSparkModelOptNvFp4FusedMoE, use_sm120_native=False
        )
    manager = _manager(ep_size, 1, experts)
    stub, routed = _moe_stub(manager, quant_method)
    stub.num_experts, stub.top_k = experts, 8
    stock = manager.expert_map.clone()
    _install(stub, _vllm_config(model, layers), layer_idx=layer_idx)
    assert stub.expert_placement is None
    assert routed.updates == 0 and torch.equal(manager.expert_map, stock)
    assert manager.placement_strategy == "linear"
    assert len(warnings) == 1 and "keeps the linear expert placement" in warnings[0]


def test_hook_warns_for_a_table_that_cannot_be_read(tmp_path, monkeypatch) -> None:
    warnings = _warnings(monkeypatch)
    monkeypatch.setenv(
        "VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", str(tmp_path / "nope.json")
    )
    stub, routed = _moe_stub(_manager(4, 1, 16))
    _install(stub, _vllm_config(str(tmp_path)), layer_idx=0)
    assert stub.expert_placement is None and routed.updates == 0
    assert "cannot be read" in warnings[0]


def test_eplb_cannot_change_the_experts_of_a_placed_model() -> None:
    placed = SimpleNamespace(
        feed_forward=object.__new__(muse_model.MuseSpark1xMoE)
    )
    placed.feed_forward.expert_placement = (1, 2, 3)
    linear = SimpleNamespace(feed_forward=object.__new__(muse_model.MuseSpark1xMoE))
    linear.feed_forward.expert_placement = None

    def metadata(layers):
        model = SimpleNamespace(
            num_local_physical_experts=32,
            model=SimpleNamespace(layers=layers),
        )
        muse_model.MuseSpark1xForCausalLM.update_physical_experts_metadata(
            model, 256, 32
        )

    with pytest.raises(RuntimeError, match="EXPERT_PLACEMENT"):
        metadata([linear, placed])


def test_megamoe_layers_return_before_the_hook() -> None:
    """MegaMoE takes its experts from the rank number: its constructor returns ahead of
    the placement hook (with a warning under the flag), so such a layer stays linear."""
    import inspect

    source = inspect.getsource(muse_model.MuseSpark1xMoE.__init__)
    assert source.index("MuseSparkMegaMoEExperts(") < source.index(
        "self._install_expert_placement("
    )
    assert "apply to MegaMoE" in source


# ------------------------------------------------------------------------------
# What a placement relies on
# ------------------------------------------------------------------------------
def test_the_ep_weight_filter_leaves_the_consolidated_expert_tensors_alone() -> None:
    """The loader may skip per-expert checkpoint tensors it thinks another rank owns,
    by the linear or the round-robin map. The Muse checkpoint holds every layer's
    experts as one [E, ...] tensor, which it never matches: all of them reach
    ``RoutedExperts.weight_loader`` and the placement's map picks the local ones."""
    local = set(range(32))
    for name in (
        "model.layers.7.feed_forward.experts.w13_weight",
        "model.layers.7.feed_forward.experts.w13_weight_scale",
        "model.layers.7.feed_forward.experts.w2_weight",
    ):
        assert not should_skip_weight(name, local)


def test_a_layerwise_weight_reload_keeps_the_installed_map() -> None:
    """``_expert_map`` is never moved to the meta device or reloaded, so the layer a
    reload re-materializes still holds the placement's map."""
    assert "_expert_map" in SKIP_LOAD_TENSORS
