# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The static expert placement table of MuseSpark and its install (CPU only).

The table says which EP rank owns every expert of every layer. It is applied as the
layer's ``expert_map`` before the weights load: these tests check the file and its
refusals, the map derived from it (a permutation, the numbering of ``new_id``), how a
table is found for a model, and the install on a real ``ExpertMapManager``. The layers
that cannot take a placement and the model's hook are in
``test_muse_spark_expert_placement_hook.py``; what the experts compute under such a map
is checked on the GPU in ``tests/kernels/moe/test_muse_spark_sm120_expert_placement.py``
and the decode and prefetch tests.
"""

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.expert_map_manager import (
    ExpertMapManager,
    determine_expert_map,
)
from vllm.models.muse_spark.nvidia.ops.expert_placement import (
    ExpertPlacementTable,
    canonical_new_ids,
    install_expert_placement,
    load_matching_table,
    resolve_snapshot,
    table_path_for,
)

SNAPSHOT = "fc6908f53c6882d4ec48fca5ba144e41d21e1f92"
REPO = Path(__file__).resolve().parents[3]
SHIPPED = table_path_for(SNAPSHOT)
BUILDER_DIR = REPO / "benchmarks" / "kernels" / "muse_spark_expert_placement"
LAYERS, EXPERTS, EP, PER_RANK = 62, 256, 8, 32


# ------------------------------------------------------------------------------
# The shipped table
# ------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def shipped() -> ExpertPlacementTable:
    return ExpertPlacementTable.load(SHIPPED)


def test_shipped_table_is_the_packaged_one_of_the_model_snapshot(shipped) -> None:
    assert SHIPPED.is_file() and SHIPPED.name == "expert_placement_fc6908f53c.json"
    assert shipped.snapshot == SNAPSHOT
    assert (shipped.num_hidden_layers, shipped.num_experts, shipped.top_k) == (
        LAYERS,
        EXPERTS,
        8,
    )
    assert (shipped.ep_size, shipped.experts_per_rank) == (EP, PER_RANK)
    assert shipped.base_placement == "linear"
    assert sorted(shipped.layers) == list(range(LAYERS))


def test_shipped_table_is_a_balanced_assignment_in_every_layer(shipped) -> None:
    for layer in range(LAYERS):
        rank_of = shipped.rank_of(layer)
        assert len(rank_of) == EXPERTS
        assert all(rank_of.count(r) == PER_RANK for r in range(EP)), layer
    # It is a placement, not the linear one, in every layer (224 of 256 experts leave
    # their linear rank on average).
    moved = [shipped.num_moved(layer) for layer in range(LAYERS)]
    assert min(moved) > 150 and sum(moved) / LAYERS > 200, moved


def test_shipped_table_carries_its_provenance(shipped) -> None:
    data = json.loads(SHIPPED.read_text())
    prov = data["provenance"]
    assert prov["base_commit"] == "a89b873288" and prov["host_commit"] == "b207e54"
    assert prov["trace"]["steps_jsonl_sha256"]
    assert prov["training"]["variant"] == "comb"
    assert set(prov["training"]["domains"]) == {
        "gsm",
        "chat",
        "code",
        "docs",
        "zh",
        "bio",
    }
    assert prov["cost_models"]["prefill"] and prov["cost_models"]["c64"]
    assert prov["cross_validation"] and prov["final_eval"]
    assert data["model"]["repo"] == "unsloth/Muse-Spark-Open-816B-A42B-NVFP4"
    assert data["model"]["moe_layers"] == [0, 61]


def test_shipped_table_was_made_by_the_builder_in_the_tree() -> None:
    """The four scripts the file names are the ones in benchmarks/kernels."""
    prov = json.loads(SHIPPED.read_text())["provenance"]
    hashes = prov["built_by_sha256"]
    assert sorted(hashes) == ["r2d_agg.py", "r2d_cv.py", "r2d_lib.py", "r2d_table.py"]
    for name, digest in hashes.items():
        path = BUILDER_DIR / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name


def test_shipped_table_new_ids_are_the_canonical_numbering(shipped) -> None:
    """``new_id`` of the file is the id the experts get in the permuted linear layout:
    rank * 32 + ascending slot. Both the file and the loader's own numbering agree."""
    data = json.loads(SHIPPED.read_text())
    for layer in range(LAYERS):
        entry = data["layers"][str(layer)]
        assert entry["rank_of"] == list(shipped.rank_of(layer))
        assert entry["new_id"] == list(shipped.new_ids(layer))
        assert sorted(entry["new_id"]) == list(range(EXPERTS))


def test_expert_maps_of_all_ranks_are_a_permutation_round_trip(shipped) -> None:
    """The per-rank maps partition the experts; slot s of rank r holds the s-th
    smallest expert of the rank; global -> (rank, slot) -> new_id -> back is the
    identity."""
    for layer in range(LAYERS):
        maps = torch.stack([shipped.expert_map(layer, r) for r in range(EP)])
        assert maps.dtype == torch.int32 and maps.shape == (EP, EXPERTS)
        owner = (maps >= 0).to(torch.int64)
        # Every expert is local to exactly one rank.
        assert (owner.sum(0) == 1).all()
        rank = owner.argmax(0)
        assert rank.tolist() == list(shipped.rank_of(layer))
        slots = maps.max(0).values
        new_id = rank * PER_RANK + slots
        assert new_id.tolist() == list(shipped.new_ids(layer))
        # new_id is a permutation; its inverse sends every new id back to the expert.
        inverse = torch.empty(EXPERTS, dtype=torch.int64)
        inverse[new_id] = torch.arange(EXPERTS)
        assert (inverse[new_id] == torch.arange(EXPERTS)).all()
        for r in range(EP):
            local = shipped.local_experts(layer, r)
            assert len(local) == PER_RANK and list(local) == sorted(local)
            # Slot i of the map is the i-th expert of the rank: ascending ids.
            assert maps[r][list(local)].tolist() == list(range(PER_RANK))
            assert (maps[r] >= 0).sum().item() == PER_RANK


def test_the_linear_table_gives_the_linear_expert_map() -> None:
    """A table that is the linear placement yields what ``determine_expert_map``
    computes: the install path reproduces the stock map where nothing moves."""
    rank_of = tuple(e // PER_RANK for e in range(EXPERTS))
    table = ExpertPlacementTable(
        snapshot=SNAPSHOT,
        num_hidden_layers=1,
        num_experts=EXPERTS,
        top_k=8,
        ep_size=EP,
        experts_per_rank=PER_RANK,
        base_placement="linear",
        layers={0: rank_of},
    )
    for rank in range(EP):
        _, stock, _ = determine_expert_map(EP, rank, EXPERTS, "linear")
        assert torch.equal(table.expert_map(0, rank), stock)
    assert table.num_moved(0) == 0
    assert table.new_ids(0) == tuple(range(EXPERTS))


# ------------------------------------------------------------------------------
# A small valid table and the files that must be refused
# ------------------------------------------------------------------------------
def _small_json(num_layers: int = 3, experts: int = 16, ep_size: int = 4) -> dict:
    per_rank = experts // ep_size
    layers = {}
    for layer in range(num_layers):
        # A deterministic scatter: expert e goes to rank (e * 5 + layer) % ep_size
        # when that rank has room, which makes a different split per layer.
        order = sorted(range(experts), key=lambda e: ((e * 7 + layer * 3) % experts, e))
        rank_of = [0] * experts
        for position, expert in enumerate(order):
            rank_of[expert] = position // per_rank
        layers[str(layer)] = dict(
            rank_of=rank_of, new_id=canonical_new_ids(rank_of, ep_size)
        )
    return dict(
        version=1,
        model=dict(
            repo="x/y",
            snapshot=SNAPSHOT,
            num_hidden_layers=num_layers,
            num_experts=experts,
            top_k=2,
            moe_layers=[],
        ),
        layout=dict(
            ep_size=ep_size, experts_per_rank=per_rank, base_placement="linear"
        ),
        provenance={},
        layers=layers,
    )


def _small_table() -> ExpertPlacementTable:
    return ExpertPlacementTable.from_json(_small_json(), source="small.json")


def test_small_table_parses() -> None:
    table = _small_table()
    assert (table.num_hidden_layers, table.num_experts, table.ep_size) == (3, 16, 4)
    for layer in range(3):
        assert sorted(table.new_ids(layer)) == list(range(16))


def _mutations():
    def version(d):
        d["version"] = 2

    def product(d):
        d["layout"]["experts_per_rank"] = 5

    def missing_layer(d):
        del d["layers"]["1"]

    def extra_layer(d):
        d["layers"]["3"] = copy.deepcopy(d["layers"]["0"])

    # new_id is optional: the cases of rank_of drop it, so that only the checks of
    # rank_of itself can refuse the file.
    def short_rank_of(d):
        del d["layers"]["0"]["new_id"]
        d["layers"]["0"]["rank_of"] = d["layers"]["0"]["rank_of"][:-1]

    def rank_out_of_range(d):
        del d["layers"]["0"]["new_id"]
        d["layers"]["0"]["rank_of"][0] = 4

    def negative_rank(d):
        del d["layers"]["0"]["new_id"]
        d["layers"]["0"]["rank_of"][0] = -1

    def unbalanced(d):
        del d["layers"]["0"]["new_id"]
        ro = d["layers"]["0"]["rank_of"]
        ro[ro.index(0)] = 1  # rank 1 holds 5 experts, rank 0 holds 3

    def non_integer(d):
        del d["layers"]["0"]["new_id"]
        d["layers"]["0"]["rank_of"][0] = True

    def float_rank(d):
        del d["layers"]["0"]["new_id"]
        d["layers"]["0"]["rank_of"][0] = 1.0

    def wrong_new_id(d):
        n = d["layers"]["2"]["new_id"]
        n[0], n[1] = n[1], n[0]

    def new_id_not_a_permutation(d):
        d["layers"]["2"]["new_id"][0] = d["layers"]["2"]["new_id"][1]

    return {
        "version": version,
        "experts_per_rank times ep_size is not the experts": product,
        "a layer is missing": missing_layer,
        "a layer beyond num_hidden_layers": extra_layer,
        "rank_of is short": short_rank_of,
        "a rank beyond ep_size": rank_out_of_range,
        "a negative rank": negative_rank,
        "ranks hold unequal numbers of experts": unbalanced,
        "a boolean rank": non_integer,
        "a float rank": float_rank,
        "new_id is not the canonical numbering": wrong_new_id,
        "new_id is not a permutation": new_id_not_a_permutation,
    }


@pytest.mark.parametrize("name", list(_mutations()))
def test_malformed_tables_are_refused(name: str) -> None:
    data = _small_json()
    _mutations()[name](data)
    with pytest.raises(ValueError, match="expert placement table"):
        ExpertPlacementTable.from_json(data, source="bad.json")


def test_new_id_is_optional_but_checked_when_present() -> None:
    data = _small_json()
    for entry in data["layers"].values():
        del entry["new_id"]
    table = ExpertPlacementTable.from_json(data)
    assert table.new_ids(0) == tuple(canonical_new_ids(table.rank_of(0), 4))


# ------------------------------------------------------------------------------
# Which model a table describes
# ------------------------------------------------------------------------------
def _fits(**overrides) -> dict:
    fields = dict(
        snapshot=SNAPSHOT,
        num_hidden_layers=3,
        num_experts=16,
        top_k=2,
        ep_size=4,
        ep_rank=1,
        local_num_experts=4,
        placement="linear",
    )
    fields.update(overrides)
    return fields


def test_table_fits_the_model_it_was_made_for() -> None:
    table = _small_table()
    assert table.mismatch(**_fits()) is None
    # A snapshot that cannot be read does not mismatch: the caller decides.
    assert table.mismatch(**_fits(snapshot=None)) is None


@pytest.mark.parametrize(
    ("override", "word"),
    [
        (dict(snapshot="0123456789abcdef"), "snapshot"),
        (dict(num_hidden_layers=4), "layers"),
        (dict(num_experts=32), "experts"),
        (dict(top_k=8), "top-"),
        (dict(ep_size=2), "EP"),
        (dict(placement="round_robin"), "round_robin"),
        (dict(placement="custom"), "custom"),
        (dict(local_num_experts=8), "local experts"),
        (dict(ep_rank=4), "EP rank"),
        (dict(ep_rank=-1), "EP rank"),
    ],
)
def test_table_refuses_every_field_of_the_model_that_differs(override, word) -> None:
    why = _small_table().mismatch(**_fits(**override))
    assert why is not None and word in why


# ------------------------------------------------------------------------------
# Finding the table of a model
# ------------------------------------------------------------------------------
def _snapshot_dir(tmp_path: Path, snapshot: str = SNAPSHOT) -> Path:
    path = tmp_path / "models--unsloth--Muse" / "snapshots" / snapshot
    path.mkdir(parents=True)
    return path


def _model_fields(**overrides) -> dict:
    fields = dict(
        revision=None,
        num_hidden_layers=LAYERS,
        num_experts=EXPERTS,
        top_k=8,
        ep_size=EP,
        ep_rank=3,
        local_num_experts=PER_RANK,
        placement="linear",
    )
    fields.update(overrides)
    return fields


def test_snapshot_comes_from_a_directory_or_the_hub_cache(
    tmp_path, monkeypatch
) -> None:
    assert resolve_snapshot(str(_snapshot_dir(tmp_path))) == SNAPSHOT
    other = tmp_path / "plain"
    other.mkdir()
    assert resolve_snapshot(str(other)) is None
    cached = tmp_path / "hub" / "models--a--b" / "snapshots" / "abc123" / "config.json"
    cached.parent.mkdir(parents=True)
    cached.write_text("{}")
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub, "try_to_load_from_cache", lambda *a, **k: str(cached)
    )
    assert resolve_snapshot("a/b") == "abc123"
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    assert resolve_snapshot("a/b") is None


def test_packaged_table_is_found_for_the_snapshot_of_the_model(tmp_path) -> None:
    table, why = load_matching_table(
        model=str(_snapshot_dir(tmp_path)), **_model_fields()
    )
    assert why == "" and table is not None
    assert table.snapshot == SNAPSHOT and table.source == str(SHIPPED)


def test_no_table_for_another_snapshot(tmp_path) -> None:
    model = _snapshot_dir(tmp_path, "1234567890abcdef1234567890abcdef12345678")
    table, why = load_matching_table(model=str(model), **_model_fields())
    assert table is None and "1234567890" in why


def test_no_table_where_the_snapshot_cannot_be_read(tmp_path) -> None:
    table, why = load_matching_table(model=str(tmp_path), **_model_fields())
    assert table is None and "commit hash" in why


@pytest.mark.parametrize(
    ("override", "word"),
    [
        (dict(ep_size=4), "EP"),
        (dict(ep_rank=8), "EP rank"),
        (dict(local_num_experts=64), "local experts"),
        (dict(placement="round_robin"), "round_robin"),
        (dict(num_hidden_layers=61), "layers"),
        (dict(top_k=4), "top-"),
    ],
)
def test_packaged_table_is_not_used_for_another_layout(
    tmp_path, override, word
) -> None:
    table, why = load_matching_table(
        model=str(_snapshot_dir(tmp_path)), **_model_fields(**override)
    )
    assert table is None and word in why


def test_explicit_table_is_used_whatever_it_is_named_and_trusts_the_snapshot(
    tmp_path,
) -> None:
    path = tmp_path / "mine.json"
    path.write_text(json.dumps(_small_json()))
    plain = tmp_path / "plain"
    plain.mkdir()
    fields = _model_fields(
        num_hidden_layers=3, num_experts=16, top_k=2, ep_size=4, local_num_experts=4
    )
    table, why = load_matching_table(model=str(plain), path=str(path), **fields)
    assert why == "" and table is not None and table.source == str(path)
    # With a readable snapshot it has to be the table's.
    wrong = _snapshot_dir(tmp_path, "1234567890abcdef1234567890abcdef12345678")
    table, why = load_matching_table(model=str(wrong), path=str(path), **fields)
    assert table is None and "1234567890" in why


def test_unreadable_or_invalid_tables_give_the_reason(tmp_path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    fields = _model_fields()
    table, why = load_matching_table(
        model=str(plain), path=str(tmp_path / "missing.json"), **fields
    )
    assert table is None and "cannot be read" in why
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    table, why = load_matching_table(model=str(plain), path=str(broken), **fields)
    assert table is None and "cannot be read" in why
    data = _small_json()
    data["layers"]["0"]["rank_of"][0] = 9
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(data))
    table, why = load_matching_table(model=str(plain), path=str(invalid), **fields)
    assert table is None and "cannot be read" in why and "ValueError" in why


# ------------------------------------------------------------------------------
# Installing a placement on a layer's expert map
# ------------------------------------------------------------------------------
def _parallel_config(ep_size: int, ep_rank: int, *, eplb: bool = False):
    return FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=ep_rank,
        sp_size=1,
        use_ep=ep_size > 1,
        all2all_backend="allgather_reducescatter",
        enable_eplb=eplb,
    )


def _manager(ep_size: int, ep_rank: int, experts: int, strategy: str = "linear"):
    return ExpertMapManager(
        max_num_batched_tokens=64,
        top_k=2,
        global_num_experts=experts,
        num_redundant_experts=0,
        num_expert_group=2 if strategy == "round_robin" else None,
        moe_parallel_config=_parallel_config(ep_size, ep_rank),
        placement_strategy=strategy,
        enable_eplb=False,
    )


class _Routed:
    """What ``install_expert_placement`` needs of a layer's routed experts."""

    def __init__(self, manager: ExpertMapManager):
        self.expert_map_manager = manager
        self.updates = 0

    @property
    def expert_map(self) -> torch.Tensor | None:
        return self.expert_map_manager.expert_map

    def update_expert_map_info(self) -> None:
        self.updates += 1


@pytest.mark.parametrize("rank", range(4))
def test_install_puts_the_experts_of_the_table_into_ascending_slots(rank) -> None:
    table = _small_table()
    routed = _Routed(_manager(4, rank, 16))
    manager = routed.expert_map_manager
    assert manager.placement_strategy == "linear"
    linear = manager.expert_map.clone()
    local = install_expert_placement(routed, table, layer_idx=1)
    assert local == table.local_experts(1, rank) and list(local) == sorted(local)
    assert routed.updates == 1
    assert manager.placement_strategy == "custom"
    assert torch.equal(manager.expert_map, table.expert_map(1, rank))
    assert manager.get_local_expert_ids() == list(local)
    # The loader resolves every expert through the map: slot i holds the i-th expert.
    for slot, expert in enumerate(local):
        assert manager.map_global_to_local(expert) == slot
        assert manager.is_local_expert(expert)
    remote = [e for e in range(16) if e not in local]
    assert all(manager.map_global_to_local(e) == -1 for e in remote)
    # It really moved experts relative to the linear map the layer started with.
    assert not torch.equal(linear, manager.expert_map)


def test_install_keeps_the_map_on_its_device_and_dtype() -> None:
    routed = _Routed(_manager(4, 2, 16))
    before = routed.expert_map_manager.expert_map
    install_expert_placement(routed, _small_table(), layer_idx=0)
    after = routed.expert_map_manager.expert_map
    assert after.dtype == before.dtype == torch.int32
    assert after.device == before.device and after.shape == before.shape


def test_every_rank_of_a_layer_installs_a_disjoint_cover() -> None:
    table = _small_table()
    owners = torch.zeros(16, dtype=torch.int64)
    for rank in range(4):
        routed = _Routed(_manager(4, rank, 16))
        install_expert_placement(routed, table, layer_idx=2)
        owners += (routed.expert_map_manager.expert_map >= 0).to(torch.int64)
    assert (owners == 1).all()


def test_install_refuses_a_layer_that_does_not_hold_the_tables_layout() -> None:
    table = _small_table()
    # EP off: there is no map to replace.
    routed = _Routed(_manager(1, 0, 16))
    with pytest.raises(ValueError, match="expert parallelism is off"):
        install_expert_placement(routed, table, 0)
    # Another EP size or expert count.
    for ep_size, experts in ((2, 16), (4, 32)):
        routed = _Routed(_manager(ep_size, 0, experts))
        with pytest.raises(ValueError, match="EP"):
            install_expert_placement(routed, table, 0)
    # Another starting placement than the table permutes.
    routed = _Routed(_manager(4, 1, 16, strategy="round_robin"))
    assert routed.expert_map_manager.placement_strategy == "round_robin"
    with pytest.raises(ValueError, match="round_robin"):
        install_expert_placement(routed, table, 0)
    # A layer the table does not have.
    routed = _Routed(_manager(4, 1, 16))
    with pytest.raises(ValueError, match="no entry for layer 7"):
        install_expert_placement(routed, table, 7)
    # None of them touched a map.
    assert routed.updates == 0
    _, stock, _ = determine_expert_map(4, 1, 16, "linear")
    assert torch.equal(routed.expert_map_manager.expert_map, stock)


def test_install_refuses_a_rank_that_does_not_own_the_linear_block() -> None:
    routed = _Routed(_manager(4, 1, 16))
    manager = routed.expert_map_manager
    # Move one expert by hand: the layer no longer starts from the linear map.
    shifted = manager.expert_map.clone()
    shifted[4], shifted[8] = -1, 0
    manager._expert_map = shifted
    with pytest.raises(ValueError, match="linear block"):
        install_expert_placement(routed, _small_table(), 0)
    assert routed.updates == 0


def test_installed_map_cannot_be_recomputed() -> None:
    """EPLB and elastic EP recompute the map; with the weights loaded for a placement
    that would silently desynchronize the routes from the weights."""
    routed = _Routed(_manager(4, 0, 16))
    manager = routed.expert_map_manager
    install_expert_placement(routed, _small_table(), 0)
    with pytest.raises(RuntimeError, match="install_expert_map"):
        manager.update(_parallel_config(4, 0), global_num_experts=16)
    assert torch.equal(manager.expert_map, _small_table().expert_map(0, 0))


def test_manager_update_still_recomputes_a_stock_map() -> None:
    manager = _manager(4, 0, 16)
    manager.update(_parallel_config(2, 1), global_num_experts=16)
    _, stock, _ = determine_expert_map(2, 1, 16, "linear")
    assert torch.equal(manager.expert_map, stock)
    assert manager.placement_strategy == "linear"


def _scatter(manager: ExpertMapManager, rank_slots: dict[int, int]) -> torch.Tensor:
    expert_map = torch.full_like(manager.expert_map, -1)
    for expert, slot in rank_slots.items():
        expert_map[expert] = slot
    return expert_map


def test_install_expert_map_checks_the_map() -> None:
    manager = _manager(4, 0, 16)
    good = _scatter(manager, {1: 2, 5: 0, 9: 3, 14: 1})
    for bad, word in (
        (good.to(torch.int64), "int32"),
        (good[:-1], "(15,)"),
        (_scatter(manager, {1: 0, 5: 1, 9: 2}), "each of 0 .. 3 once"),
        (_scatter(manager, {1: 0, 5: 1, 9: 2, 14: 2}), "each of 0 .. 3 once"),
        (_scatter(manager, {1: 0, 5: 1, 9: 2, 14: 4}), "each of 0 .. 3 once"),
        (_scatter(manager, {1: 0, 5: 1, 9: 2, 14: 3, 15: 3}), "each of 0 .. 3 once"),
    ):
        with pytest.raises(ValueError, match=re.escape(word)):
            manager.install_expert_map(bad)
    stock = manager.expert_map.clone()
    assert torch.equal(manager.expert_map, stock)
    manager.install_expert_map(good)
    assert manager.map_global_to_local(9) == 3 and manager.map_global_to_local(0) == -1
    assert manager.placement_strategy == "custom"
    with pytest.raises(ValueError, match="no map to replace"):
        _manager(1, 0, 16).install_expert_map(good)


def _aiter_manager(shared: int, monkeypatch) -> ExpertMapManager:
    from vllm.model_executor.layers.fused_moe import expert_map_manager

    monkeypatch.setattr(
        expert_map_manager, "init_aiter_topK_meta_data", lambda **kwargs: None
    )
    return ExpertMapManager(
        max_num_batched_tokens=64,
        top_k=2,
        global_num_experts=16,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=_parallel_config(4, 0),
        placement_strategy="linear",
        enable_eplb=False,
        num_fused_shared_experts=shared,
        rocm_aiter_enabled=True,
    )


@pytest.mark.parametrize("shared", [0, 1])
def test_install_expert_map_keeps_the_aiter_mask_and_shared_entries(
    shared, monkeypatch
) -> None:
    """The mask of the AITER path says which experts the rank holds, like the map: it
    follows the installed map (sentinel entry and fused shared experts untouched)."""
    manager = _aiter_manager(shared, monkeypatch)
    assert manager.expert_mask.shape == (17 + shared,)
    assert manager.expert_mask[:16].tolist() == [1] * 4 + [0] * 12
    new = manager.expert_map.clone()
    new[:16] = -1
    for expert, slot in {1: 2, 5: 0, 9: 3, 14: 1}.items():
        new[expert] = slot
    manager.install_expert_map(new)
    mask = manager.expert_mask
    assert mask.shape == (17 + shared,) and mask.dtype == torch.int32
    assert mask[:16].tolist() == [int(e in (1, 5, 9, 14)) for e in range(16)]
    assert mask[16:].tolist() == [1] * shared + [0]
    assert torch.equal(manager.expert_map[16:], new[16:])
    assert manager.placement_strategy == "custom"
    if shared:
        # The fused shared expert entry is not the placement's to move.
        moved = new.clone()
        moved[16] = 7
        with pytest.raises(ValueError, match="fused shared experts"):
            manager.install_expert_map(moved)
        # ... and the slots check does not count it as a routed expert.
        crowded = new.clone()
        crowded[3] = 4
        with pytest.raises(ValueError, match="each of 0 .. 3 once"):
            manager.install_expert_map(crowded)
