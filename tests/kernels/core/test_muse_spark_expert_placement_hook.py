# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The layers that take a static expert placement and the model's hook (CPU only).

Which layers can run a placement (the two expert paths of the model that locate their
routes through ``expert_map``), what the constructor does under the flag (installs the
table, or leaves the layer linear with the reason), the guards (EPLB, MegaMoE), and the
loader premises a placement relies on. The table and its install are in
``test_muse_spark_expert_placement.py``.
"""

import ast
import inspect
import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

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
from vllm.models.muse_spark.nvidia.ops.expert_placement import (
    ExpertPlacementTable,
    check_placement_agreement,
    install_expert_placement,
)


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


def test_flag_is_on_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", raising=False)
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT is True
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE is None
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", "0")
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT is False
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", "1")
    monkeypatch.setenv(
        "VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", "/some/table.json"
    )
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT is True
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE == "/some/table.json"
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE", "")
    assert envs.VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE is None


def _constructor() -> ast.FunctionDef:
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(muse_model.MuseSpark1xMoE.__init__))
    )
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    return function


def _is_hook_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self._install_expert_placement"
        and [ast.unparse(a) for a in node.args] == ["vllm_config", "layer_idx"]
    )


def test_the_constructor_runs_the_hook_once_after_the_experts_are_built() -> None:
    """The syntax of the constructor, not its text: the hook is one statement of the
    function's own body (no branch, loop or ``try`` that a layer could skip), behind
    every assignment of ``self.experts`` (the placement replaces the map of the built
    experts) and nothing loads weights in the constructor (the placement has to be in
    before the weights land)."""
    function = _constructor()
    calls = [n for n in ast.walk(function) if _is_hook_call(n)]
    assert len(calls) == 1
    top = [
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Expr) and _is_hook_call(n.value)
    ]
    assert len(top) == 1, "the hook is not an unconditional statement of the body"
    hook = function.body[top[0]]
    experts = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Assign | ast.AnnAssign)
        and "self.experts" in ast.unparse(n).split("=")[0]
    ]
    assert len(experts) == 2  # MegaMoE's and the FusedMoE runner
    assert max(n.lineno for n in experts) < hook.lineno
    loads = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call) and "load_weights" in ast.unparse(n.func)
    ]
    assert not loads


def test_the_hook_is_the_last_statement_of_the_constructor() -> None:
    """Nothing the constructor does after the hook can take the experts' map from
    the layer (a check that rereads or replaces it would see the linear one)."""
    function = _constructor()
    last = function.body[-1]
    assert isinstance(last, ast.Expr) and _is_hook_call(last.value)


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
    "case",
    [
        "no snapshot",
        "other snapshot",
        "ep size",
        "drafter layer",
        "blocker",
        "round robin",
    ],
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
    strategy = "round_robin" if case == "round robin" else "linear"
    manager = _manager(ep_size, 1, experts, strategy)
    assert manager.placement_strategy == strategy
    stub, routed = _moe_stub(manager, quant_method)
    stub.num_experts, stub.top_k = experts, 8
    stock = manager.expert_map.clone()
    _install(stub, _vllm_config(model, layers), layer_idx=layer_idx)
    assert stub.expert_placement is None
    assert routed.updates == 0 and torch.equal(manager.expert_map, stock)
    assert manager.placement_strategy == strategy
    # The warning names the placement the layer keeps, not always "linear".
    assert len(warnings) == 1
    assert f"keeps its {strategy} expert placement" in warnings[0]


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
    placed = SimpleNamespace(feed_forward=object.__new__(muse_model.MuseSpark1xMoE))
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
    """MegaMoE takes its experts from the rank number: the constructor returns from the
    MegaMoE branch (with a warning under the flag) ahead of the placement hook, so such
    a layer stays linear."""
    function = _constructor()
    mega = [
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self.use_mega_moe"
        and isinstance(n.body[-1], ast.Return)
        and "apply to MegaMoE" in ast.unparse(n)
        and "MuseSparkMegaMoEExperts(" in ast.unparse(n)
    ]
    hook = [
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Expr) and _is_hook_call(n.value)
    ]
    assert len(mega) == 1 and len(hook) == 1 and mega[0] < hook[0]


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


def test_the_ep_weight_filter_is_off_under_a_placement(monkeypatch) -> None:
    """The filter derives the experts of a rank from the linear / round-robin strategy
    and would skip per-expert tensors of the experts the rank owns under the placement:
    with the flag it does not compute a set at all (without it, it does)."""
    import vllm.config
    import vllm.distributed as distributed
    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    monkeypatch.setattr(distributed, "get_tensor_model_parallel_rank", lambda: 3)
    model_config = SimpleNamespace(
        is_moe=True,
        get_num_experts=lambda: 256,
        architectures=["MuseSparkForConditionalGeneration"],
    )
    parallel = SimpleNamespace(
        tensor_parallel_size=8,
        data_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=True,
        enable_ep_weight_filter=True,
        enable_eplb=False,
        expert_placement_strategy="linear",
    )
    monkeypatch.setattr(
        vllm.config,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=parallel),
    )

    def local_experts() -> set[int] | None:
        loader = DefaultModelLoader(LoadConfig())
        loader._init_ep_weight_filter(model_config)
        return loader.local_expert_ids

    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", "0")
    assert local_experts() == set(range(96, 128))
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT", "1")
    assert local_experts() is None
    # The flag is on by default: another model keeps the filter.
    model_config.architectures = ["DeepseekV3ForCausalLM"]
    assert local_experts() == set(range(96, 128))


# ------------------------------------------------------------------------------
# The ranks must agree on the placement (a split holds experts twice or never)
# ------------------------------------------------------------------------------
WORLD = 4
# 0: MoE, 1: dense layer, 2: MegaMoE, 3: MoE (EP4, 16 experts, table of 4 layers).
MOE_LAYERS = (0, 3)


def _moe(rank: int, table: ExpertPlacementTable | None, layer_idx: int):
    """A MuseSpark1xMoE that went through the constructor's hook with ``table``."""
    moe = object.__new__(muse_model.MuseSpark1xMoE)
    moe.use_mega_moe = False
    moe.num_experts = 16
    moe.expert_placement = None
    routed = _Routed(_manager(WORLD, rank, 16))
    if table is not None:
        moe.expert_placement = install_expert_placement(routed, table, layer_idx)
    moe.experts = SimpleNamespace(routed_experts=routed)
    return moe


def _mega_moe() -> muse_model.MuseSpark1xMoE:
    moe = object.__new__(muse_model.MuseSpark1xMoE)
    moe.use_mega_moe = True
    moe.num_experts = 16
    moe.expert_placement = None
    return moe


def _model(rank: int, tables: dict[int, ExpertPlacementTable | None]):
    """The layers of one rank; ``tables[i]`` is what its hook found for layer ``i``."""
    feed_forward = [
        _moe(rank, tables.get(0), 0),
        object(),
        _mega_moe(),
        _moe(rank, tables.get(3), 3),
    ]
    return SimpleNamespace(
        start_layer=0,
        end_layer=len(feed_forward),
        layers=[SimpleNamespace(feed_forward=ff) for ff in feed_forward],
    )


def _table(shift: int = 0) -> ExpertPlacementTable:
    """A valid table of 4 layers; another ``shift`` is another table (each rank of it
    owns the experts that rank ``shift`` before it owns in the shift 0 table)."""
    data = _small_json(num_layers=4)
    for entry in data["layers"].values():
        entry["rank_of"] = [(r + shift) % WORLD for r in entry["rank_of"]]
        del entry["new_id"]
    return ExpertPlacementTable.from_json(data, source=f"shift{shift}.json")


def _agreement_worker(rank: int, world: int, port: int) -> None:
    """``_check_expert_placement`` over a real gloo group, a model per scenario."""
    torch.distributed.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", world_size=world, rank=rank
    )
    muse_model.get_ep_group = lambda: SimpleNamespace(
        world_size=world, cpu_group=torch.distributed.group.WORLD
    )
    table, other = _table(), _table(shift=1)
    last = rank == world - 1
    first = rank == 0
    both = {0: table, 3: table}
    # The same table on every rank: partition in both layers, nothing to say.
    ok = {
        "every rank took the table": both,
        "no rank took it (flag on, layers blocked everywhere)": {},
        "layer 0 only": {0: table},
    }
    # (what this rank's hook found, the words the error has to hold)
    bad = {
        "the last rank kept layer 3 linear": (
            {0: table, 3: None if last else table},
            "layer 3: 3 of 4 EP ranks took the placement table",
        ),
        "the first rank kept both linear": (
            {} if first else both,
            "layer 0: 3 of 4 EP ranks took the placement table",
        ),
        "another table on the last rank": (
            {0: other if last else table, 3: table},
            "layer 0: ",
        ),
    }
    for tables in ok.values():
        muse_model.MuseSpark1xModel._check_expert_placement(_model(rank, tables))
    for name, (tables, words) in bad.items():
        try:
            muse_model.MuseSpark1xModel._check_expert_placement(_model(rank, tables))
        except RuntimeError as e:
            # Every rank raises, the one that kept the layer linear and the others.
            assert words in str(e), (rank, name, str(e))
            assert "EXPERT_PLACEMENT" in str(e)
        else:
            raise AssertionError((rank, name, "no error"))
        # The group is still usable (every rank went through the one collective).
        muse_model.MuseSpark1xModel._check_expert_placement(_model(rank, both))
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def test_the_placement_decision_is_agreed_over_the_ep_cpu_group() -> None:
    """A rank that kept a layer linear while the others took the table (a failed read
    of the file, another table on another node) is an error on every rank, before the
    weights load, and a model on which all agree is accepted. The check is on the EP
    group, not the TP group: here only ``get_ep_group`` is patched."""
    port = 31700 + os.getpid() % 1000
    mp.spawn(_agreement_worker, args=(WORLD, port), nprocs=WORLD)


def test_one_ep_rank_has_nothing_to_agree_on(monkeypatch) -> None:
    """EP1 (or no layer that could be placed) takes no collective and just logs."""
    monkeypatch.setattr(
        muse_model,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=1, cpu_group=None),
    )
    seen: list[tuple] = []
    monkeypatch.setattr(muse_model.logger, "info", lambda *args: seen.append(args))
    model = _model(0, {0: _table()})
    muse_model.MuseSpark1xModel._check_expert_placement(model)
    # 3 MoE layers (MegaMoE counts, the dense one does not), 1 of them placed.
    assert seen[-1][1:] == (1, 3)


def test_layers_without_an_expert_map_are_not_checked(monkeypatch) -> None:
    """Without EP (flag on, layers fall back to linear with a warning) every rank holds
    a slice of every expert: that is no partition, and it is no error. The group is
    not even asked (it would fail here: there is none)."""
    monkeypatch.setattr(
        muse_model,
        "get_ep_group",
        lambda: SimpleNamespace(world_size=8, cpu_group=None),
    )
    seen: list[tuple] = []
    monkeypatch.setattr(muse_model.logger, "info", lambda *args: seen.append(args))
    moe = object.__new__(muse_model.MuseSpark1xMoE)
    moe.use_mega_moe = False
    moe.num_experts = 16
    moe.expert_placement = None
    routed = _Routed(_manager(1, 0, 16))
    assert routed.expert_map is None
    moe.experts = SimpleNamespace(routed_experts=routed)
    model = SimpleNamespace(
        start_layer=0, end_layer=1, layers=[SimpleNamespace(feed_forward=moe)]
    )
    muse_model.MuseSpark1xModel._check_expert_placement(model)
    assert seen[-1][1:] == (0, 1)


def test_the_agreement_check_names_what_is_wrong(tmp_path) -> None:
    """The pure part: a partition is accepted, a double or missing expert is not, and
    the table-took counts must be all or nothing (single-process group of size 1 is a
    trivial partition only when every expert is held)."""
    path = tmp_path / "store"
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{path}", world_size=1, rank=0
    )
    try:
        group = torch.distributed.group.WORLD
        all_held = torch.ones(2, 4, dtype=torch.int32)
        check_placement_agreement([5, 9], all_held, [True, False], group)
        # The model is built under a default device that is not the CPU (the GPU): the
        # reduced tensors still are (a "meta" default device fails any other choice).
        with torch.device("meta"):
            check_placement_agreement([5, 9], all_held, [True, False], group)
        missing = all_held.clone()
        missing[1, 2] = 0
        with pytest.raises(RuntimeError, match=r"layer 9: 0 experts .* 1 by none"):
            check_placement_agreement([5, 9], missing, [True, False], group)
        # An expert held twice with none missing counts on its own (unequal ranks).
        crowded = all_held.clone()
        crowded[0, 0] = 2
        with pytest.raises(
            RuntimeError,
            match=r"layer 5: 1 experts are held by more than one .*, 0 by none",
        ):
            check_placement_agreement([5, 9], crowded, [True, False], group)
        with pytest.raises(ValueError, match="one row per layer"):
            check_placement_agreement([5], all_held, [True], group)
    finally:
        torch.distributed.destroy_process_group()
