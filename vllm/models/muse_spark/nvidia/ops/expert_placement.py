# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static expert placement of the Muse Spark MoE layers over the EP ranks (SM120).

With EP8 and the linear placement rank ``r`` owns the experts ``[32 r, 32 r + 32)`` of
every layer. How many routes land on a rank is then up to the router, and per layer
the slowest rank sets the time: the expert kernels of a prefill chunk spend 28 ms
(sum over the layers of max minus mean over the ranks) waiting for the rank with the
most routes, and a decode step waits for the rank with the most distinct experts.
``data/expert_placement_<snapshot>.json`` assigns the 256 experts of every layer to the
ranks differently (still 32 per rank), chosen from routing statistics of natural text
(built by ``benchmarks/kernels/muse_spark_expert_placement/``; the file carries its
provenance, the cost models and the held-out gains).

A placement changes nothing but where an expert lives. The router, its bias and the ids
it produces stay as they are; what moves is the layer's ``expert_map`` (global expert id
-> local slot on this rank, or -1), which every Muse expert path reads to find the
routes of its rank: the muse decode kernels (small and large batch), their expert
prefetch, the CUTLASS grouped GEMMs of the prefill (local rows and the stock sort), the
sequence-parallel and packed-latent prefill, the b12x decode engine, the route-major
finalizers and the Triton experts of the BF16 layers. The map is installed before the
weights load, and ``RoutedExperts.weight_loader`` resolves the expert id of every
per-expert tensor through it, so the weights, block scales and global scales of expert
``e`` land in the slot the map gives it; the tensors derived from them (the folded and
exact alphas, the decode kernel's ``prm``, the quant config) are computed from the
loaded, local tensors. The slots of a rank are numbered by ascending expert id, which is
the numbering of the table's ``new_id``. The outputs differ from the linear placement
by rounding only: every rank sums a different set of experts in fp32.

The layers whose experts read their layout from the rank number (the monolithic TRT-LLM
kernels, MegaMoE) are not eligible; see ``muse_moe_runner.expert_placement_blocker``.
"""

import functools
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

TABLE_VERSION = 1

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def canonical_new_ids(rank_of: tuple[int, ...] | list[int], ep_size: int) -> list[int]:
    """The id of every expert in the permuted linear layout: ``rank * n + slot`` with
    the slots of a rank numbered by ascending expert id."""
    per_rank = len(rank_of) // ep_size
    next_slot = [0] * ep_size
    new_ids = []
    for rank in rank_of:
        new_ids.append(rank * per_rank + next_slot[rank])
        next_slot[rank] += 1
    return new_ids


@dataclass(frozen=True, slots=True)
class ExpertPlacementTable:
    """For every layer, the EP rank that owns each expert."""

    snapshot: str
    num_hidden_layers: int
    num_experts: int
    top_k: int
    ep_size: int
    experts_per_rank: int
    # The placement the table is a permutation of (the one the map it replaces has).
    base_placement: str
    # layer -> [expert id] -> owning EP rank.
    layers: Mapping[int, tuple[int, ...]]
    source: str = ""

    @classmethod
    def from_json(
        cls, data: Mapping[str, Any], source: str = ""
    ) -> "ExpertPlacementTable":
        """Parse and validate the content of a table file."""
        where = "expert placement table" + (f" {source}" if source else "")
        if data.get("version") != TABLE_VERSION:
            raise ValueError(
                f"{where}: version {data.get('version')!r}, expected {TABLE_VERSION}"
            )
        model, layout = data["model"], data["layout"]
        num_experts = int(model["num_experts"])
        num_layers = int(model["num_hidden_layers"])
        ep_size = int(layout["ep_size"])
        per_rank = int(layout["experts_per_rank"])
        if ep_size < 1 or per_rank < 1 or per_rank * ep_size != num_experts:
            raise ValueError(f"{where}: experts_per_rank * ep_size != num_experts")
        entries = data["layers"]
        if set(entries) != {str(i) for i in range(num_layers)}:
            raise ValueError(
                f"{where}: the layers must be exactly 0 .. {num_layers - 1}"
            )
        layers: dict[int, tuple[int, ...]] = {}
        for key, entry in entries.items():
            rank_of = entry["rank_of"]
            if len(rank_of) != num_experts or any(
                type(r) is not int or not 0 <= r < ep_size for r in rank_of
            ):
                raise ValueError(
                    f"{where}: layer {key}: rank_of must be {num_experts} ranks"
                )
            counts = [0] * ep_size
            for r in rank_of:
                counts[r] += 1
            if counts != [per_rank] * ep_size:
                raise ValueError(
                    f"{where}: layer {key}: not {per_rank} experts per rank"
                )
            new_id = entry.get("new_id")
            canonical = canonical_new_ids(rank_of, ep_size)
            if new_id is not None and list(new_id) != canonical:
                raise ValueError(
                    f"{where}: layer {key}: new_id is not rank * {per_rank} + the "
                    "ascending slot of the expert on its rank"
                )
            layers[int(key)] = tuple(rank_of)
        return cls(
            snapshot=str(model["snapshot"]),
            num_hidden_layers=num_layers,
            num_experts=num_experts,
            top_k=int(model["top_k"]),
            ep_size=ep_size,
            experts_per_rank=per_rank,
            base_placement=str(layout["base_placement"]),
            layers=layers,
            source=source,
        )

    @classmethod
    def load(cls, path: str | os.PathLike) -> "ExpertPlacementTable":
        with open(path) as f:
            return cls.from_json(json.load(f), source=str(path))

    def mismatch(
        self,
        *,
        snapshot: str | None,
        num_hidden_layers: int,
        num_experts: int,
        top_k: int,
        ep_size: int,
        ep_rank: int,
        local_num_experts: int,
        placement: str,
    ) -> str | None:
        """Why this table does not describe the model, None if it does.

        A ``snapshot`` of None (the local copy of the model has no commit hash that
        could be read) never mismatches: the caller decides whether to trust the table.
        A table only moves experts between ranks, so on another checkpoint it is a
        valid layout that merely balances nothing in particular.
        """
        if snapshot is not None and snapshot != self.snapshot:
            return (
                f"the table is for snapshot {self.snapshot[:10]}, the model is "
                f"{snapshot[:10]}"
            )
        if (self.num_hidden_layers, self.num_experts, self.top_k) != (
            num_hidden_layers,
            num_experts,
            top_k,
        ):
            return (
                f"the table is for {self.num_hidden_layers} layers, {self.num_experts} "
                f"experts and top-{self.top_k}, the model has {num_hidden_layers}, "
                f"{num_experts} and top-{top_k}"
            )
        if placement != self.base_placement or ep_size != self.ep_size:
            return (
                f"the table is for EP{self.ep_size} from the {self.base_placement} "
                f"placement, the model runs EP{ep_size} with the {placement} placement"
            )
        if local_num_experts != self.experts_per_rank or not 0 <= ep_rank < ep_size:
            return (
                f"the table has {self.experts_per_rank} local experts per rank, the "
                f"model {local_num_experts} on EP rank {ep_rank}"
            )
        return None

    def rank_of(self, layer_idx: int) -> tuple[int, ...]:
        return self.layers[layer_idx]

    def new_ids(self, layer_idx: int) -> tuple[int, ...]:
        """The id of every expert of a layer in the permuted linear layout."""
        return tuple(canonical_new_ids(self.layers[layer_idx], self.ep_size))

    def local_experts(self, layer_idx: int, ep_rank: int) -> tuple[int, ...]:
        """The experts of a layer that ``ep_rank`` owns, by ascending id: the expert of
        local slot ``i`` is the ``i``-th."""
        return tuple(e for e, r in enumerate(self.layers[layer_idx]) if r == ep_rank)

    def expert_map(
        self, layer_idx: int, ep_rank: int, device: torch.device | str = "cpu"
    ) -> torch.Tensor:
        """The global-to-local expert map (int32, -1: not on this rank) of a layer."""
        expert_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
        local = self.local_experts(layer_idx, ep_rank)
        expert_map[list(local)] = torch.arange(len(local), dtype=torch.int32)
        return expert_map.to(device)

    def num_moved(self, layer_idx: int) -> int:
        """Experts of a layer that are not on the rank the linear placement gives."""
        rank_of = self.layers[layer_idx]
        return sum(r != e // self.experts_per_rank for e, r in enumerate(rank_of))


def resolve_snapshot(model: str, revision: str | None = None) -> str | None:
    """The commit hash of the local copy of ``model``.

    The ``snapshots/<hash>`` component of its directory, or of its ``config.json`` in
    the Hugging Face cache for a repo id; None if there is none to read (a copy outside
    the cache, not downloaded).
    """
    candidates: list[str] = []
    if os.path.isdir(model):
        candidates += [os.path.abspath(model), os.path.realpath(model)]
    else:
        try:
            from huggingface_hub import try_to_load_from_cache

            found = try_to_load_from_cache(model, "config.json", revision=revision)
        except Exception:
            found = None
        if isinstance(found, str):
            candidates.append(found)
    for path in candidates:
        parts = Path(path).parts
        if "snapshots" in parts:
            i = parts.index("snapshots")
            if i + 1 < len(parts):
                return parts[i + 1]
    return None


def table_path_for(snapshot: str) -> Path:
    return DATA_DIR / f"expert_placement_{snapshot[:10]}.json"


@functools.lru_cache(maxsize=4)
def _load_table_file(path: str) -> ExpertPlacementTable:
    # Every MoE layer asks for the table; the file is read once per process.
    return ExpertPlacementTable.load(path)


def load_matching_table(
    *,
    model: str,
    revision: str | None,
    num_hidden_layers: int,
    num_experts: int,
    top_k: int,
    ep_size: int,
    ep_rank: int,
    local_num_experts: int,
    placement: str,
    path: str | None = None,
) -> tuple[ExpertPlacementTable | None, str]:
    """``(table, why)``: the table of the model when one exists and fits it, else None
    and the reason it is not used.

    Without ``path`` the packaged table of the model snapshot is looked up
    (``data/expert_placement_<snapshot>.json``), which needs the snapshot to be
    readable. An explicit ``path`` (``VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT_TABLE``) is
    used whatever it is named, and is trusted for the snapshot when none can be read
    from the model.
    """
    snapshot = resolve_snapshot(model, revision)
    try:
        if path is not None:
            table = _load_table_file(os.path.abspath(path))
        elif snapshot is None:
            return None, (
                f"the commit hash of the local copy of {model!r} cannot be read "
                "(no snapshots/<hash> in its path or in the Hugging Face cache)"
            )
        else:
            candidate = table_path_for(snapshot)
            if not candidate.is_file():
                return None, f"no table for snapshot {snapshot[:10]} ({candidate.name})"
            table = _load_table_file(str(candidate))
    except (OSError, ValueError, KeyError, TypeError) as e:
        return None, f"the table cannot be read ({type(e).__name__}: {e})"
    why = table.mismatch(
        snapshot=snapshot,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
        ep_rank=ep_rank,
        local_num_experts=local_num_experts,
        placement=placement,
    )
    if why is not None:
        return None, why
    if snapshot is None:
        logger.warning_once(
            "MuseSpark: the snapshot of %r is not known; the expert placement table %s "
            "is used on the layout alone.",
            model,
            table.source,
        )
    return table, ""


def install_expert_placement(
    routed_experts: Any, table: ExpertPlacementTable, layer_idx: int
) -> tuple[int, ...]:
    """Give a layer the placement of ``table``: its ``expert_map`` is replaced, and
    the weights that load afterwards go where the map says.

    Call it before the weights load. Returns the experts of this rank (global ids, by
    local slot). Raises ValueError, leaving the layer on its own map, where the layer
    does not hold the layout the table permutes (EP size, local experts, the linear
    map as its starting point) or has no entry in the table.
    """
    manager = routed_experts.expert_map_manager
    if not manager.use_ep or manager.expert_map is None:
        raise ValueError("expert parallelism is off")
    if (manager.ep_size, manager.local_num_experts, manager.global_num_experts) != (
        table.ep_size,
        table.experts_per_rank,
        table.num_experts,
    ):
        raise ValueError(
            f"the layer runs EP{manager.ep_size} with {manager.local_num_experts} "
            f"local of {manager.global_num_experts} experts, the table "
            f"EP{table.ep_size} with {table.experts_per_rank} of {table.num_experts}"
        )
    if manager.placement_strategy != table.base_placement:
        raise ValueError(
            f"the layer has the {manager.placement_strategy} expert placement, the "
            f"table permutes the {table.base_placement} one"
        )
    if layer_idx not in table.layers:
        raise ValueError(f"the table has no entry for layer {layer_idx}")
    ep_rank = manager.ep_rank
    base = manager.get_local_expert_ids()
    first = ep_rank * table.experts_per_rank
    if base != list(range(first, first + table.experts_per_rank)):
        raise ValueError(
            f"EP rank {ep_rank} does not own the linear block of experts {first} .. "
            f"{first + table.experts_per_rank - 1}"
        )
    # The layer's own buffer is where the kernels read the map: that is the device.
    device = routed_experts.expert_map.device
    manager.install_expert_map(table.expert_map(layer_idx, ep_rank, device))
    routed_experts.update_expert_map_info()
    return table.local_experts(layer_idx, ep_rank)


def check_placement_agreement(
    layers: Sequence[int],
    owned: torch.Tensor,
    placed: Sequence[bool],
    group: Any,
) -> None:
    """Raise unless every rank of ``group`` took the same placement decision in every
    layer and, per layer, the ranks hold each expert exactly once.

    The decision to take the placement is made by every rank on its own (its own read of
    the table, its own look-up of the snapshot), and the tokens are replicated, so the
    all-reduced output is right only if the experts of the ranks are a partition of the
    layer's experts. A rank that kept a layer linear while the others took the table (a
    failed read, another table version on another node) holds some experts twice and
    others not at all, and nothing fails until the output is wrong. Call it on every
    rank of ``group`` with the same ``layers`` (a collective: one CPU all-reduce).

    ``owned`` is int32 ``[len(layers), num_experts]`` on the CPU, 1 where this rank
    holds the expert; ``placed[i]`` tells whether this rank installed a placement table
    in ``layers[i]``.
    """
    n = len(layers)
    if owned.shape[0] != n or len(placed) != n:
        raise ValueError("check_placement_agreement: one row per layer expected")
    # The CPU group reduces CPU tensors; a model is built under a default device that is
    # the GPU, which a bare ``torch.tensor`` would follow.
    cpu = torch.device("cpu")
    stats = torch.cat(
        [
            owned.to(device=cpu, dtype=torch.int32),
            torch.tensor(placed, dtype=torch.int32, device=cpu).view(n, 1),
        ],
        dim=1,
    )
    torch.distributed.all_reduce(stats, group=group)
    world_size = torch.distributed.get_world_size(group)
    held, took = stats[:, :-1], stats[:, -1]
    problems = []
    for i, layer in enumerate(layers):
        if 0 < int(took[i]) < world_size:
            problems.append(
                f"layer {layer}: {int(took[i])} of {world_size} EP ranks took the "
                "placement table"
            )
        twice = int((held[i] > 1).sum())
        never = int((held[i] == 0).sum())
        if twice or never:
            problems.append(
                f"layer {layer}: {twice} experts are held by more than one EP rank, "
                f"{never} by none"
            )
    if problems:
        raise RuntimeError(
            "The EP ranks disagree on the static expert placement "
            "(VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT=1), so the MoE output would be "
            "wrong: "
            + "; ".join(problems[:8])
            + (f"; and {len(problems) - 8} more" if len(problems) > 8 else "")
            + ". Check that every rank reads the same table (the file, the model "
            "snapshot) and that none failed to read it."
        )
