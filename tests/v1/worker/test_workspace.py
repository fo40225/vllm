# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.worker.workspace as workspace
from vllm.config import VllmConfig
from vllm.v1.worker.gpu_worker import _num_workspace_lanes, _num_workspace_ubatches


class _SpecConfig:
    def __init__(self, dspark: bool) -> None:
        self._dspark = dspark

    def use_dspark(self) -> bool:
        return self._dspark


class _VllmConfig:
    def __init__(self, spec_config: _SpecConfig | None) -> None:
        self.speculative_config = spec_config


@pytest.mark.parametrize(
    ("use_v2_model_runner", "spec_config", "expected"),
    [
        (True, _SpecConfig(True), 2),
        (False, _SpecConfig(True), 1),
        (True, _SpecConfig(False), 1),
        (True, None, 1),
    ],
)
def test_workspace_lane_count_is_dspark_only(
    use_v2_model_runner: bool,
    spec_config: _SpecConfig | None,
    expected: int,
) -> None:
    config = cast(VllmConfig, _VllmConfig(spec_config))
    assert _num_workspace_lanes(config, use_v2_model_runner) == expected


def test_workspace_lanes_do_not_alias_and_restore_context(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    assert manager._current_workspaces == [None, None, None, None]

    (target,) = manager.get_simultaneous(((512,), torch.uint8))
    with workspace.use_workspace_lane(1):
        (draft,) = manager.get_simultaneous(((256,), torch.uint8))
        (draft_reused,) = manager.get_simultaneous(((8,), torch.uint8))
    (target_reused,) = manager.get_simultaneous(((8,), torch.uint8))

    assert manager._current_workspaces[0].numel() == 512
    assert manager._current_workspaces[1].numel() == 256
    assert manager._current_workspaces[2:] == [None, None]
    assert target.data_ptr() != draft.data_ptr()
    assert draft.data_ptr() == draft_reused.data_ptr()
    assert target.data_ptr() == target_reused.data_ptr()


def test_workspace_lanes_compose_with_ubatches(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )

    pointers = set()
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                (buffer,) = manager.get_simultaneous(((16,), torch.uint8))
                pointers.add(buffer.data_ptr())

    assert len(pointers) == 4


def test_workspace_lock_blocks_growth_and_unlock_restores(monkeypatch) -> None:
    """Once locked, oversized requests fail loudly instead of reallocating the
    buffer that captured CUDA graphs point at; unlock restores growth."""
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    (buf,) = manager.get_simultaneous(((256,), torch.uint8))
    manager.lock()
    assert manager.is_locked()

    # Requests within the reserved size still reuse the same buffer.
    (same,) = manager.get_simultaneous(((256,), torch.uint8))
    (smaller,) = manager.get_simultaneous(((8,), torch.uint8))
    assert same.data_ptr() == buf.data_ptr()
    assert smaller.data_ptr() == buf.data_ptr()

    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((512,), torch.uint8))

    manager.unlock()
    (grown,) = manager.get_simultaneous(((512,), torch.uint8))
    assert grown.numel() == 512


def test_workspace_lane_validation(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=1)

    with (
        pytest.raises(ValueError, match="non-negative"),
        workspace.use_workspace_lane(-1),
    ):
        pass

    with (
        workspace.use_workspace_lane(1),
        pytest.raises(RuntimeError, match="is not configured"),
    ):
        manager.get_simultaneous(((1,), torch.uint8))

    with pytest.raises(ValueError, match="at least one"):
        workspace.WorkspaceManager(torch.device("cpu"), num_lanes=0)


def test_persistent_resources_are_lazy_and_isolated(monkeypatch) -> None:
    """Only the requesting slot allocates; each ubatch/lane owns its resource."""
    active_ubatch = 0
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch)
    manager = workspace.WorkspaceManager(
        torch.device("cpu"), num_ubatches=2, num_lanes=2
    )
    factory = Mock(side_effect=lambda: {"buffer": torch.zeros(8)})
    resources = []
    for ubatch in range(2):
        active_ubatch = ubatch
        for lane in range(2):
            with workspace.use_workspace_lane(lane):
                resource = manager.get_persistent_resource("scratch", factory)
                resources.append(resource)
                resource["buffer"].fill_(len(resources))
                assert manager.get_persistent_resource("scratch", factory) is resource
                assert factory.call_count == len(resources)

    assert len({r["buffer"].data_ptr() for r in resources}) == 4
    for i, resource in enumerate(resources, 1):
        assert torch.all(resource["buffer"] == i)


def test_persistent_resource_lock_applies_to_each_slot(monkeypatch) -> None:
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_lanes=2)
    factory = Mock(side_effect=object)
    first = manager.get_persistent_resource("scratch", factory)
    manager.lock()
    assert manager.get_persistent_resource("scratch", factory) is first
    with pytest.raises(AssertionError, match="was not allocated during warmup"):
        manager.get_persistent_resource("new", factory)
    with workspace.use_workspace_lane(1):
        with pytest.raises(AssertionError, match="was not allocated during warmup"):
            manager.get_persistent_resource("scratch", factory)
        assert factory.call_count == 1
        manager.unlock()
        assert manager.get_persistent_resource("scratch", factory) is not first
    assert manager.get_persistent_resource("scratch", factory) is first


def test_persistent_tensor_preserves_contents_and_rejects_changes() -> None:
    """Initialization happens once and transient scratch cannot overwrite locks."""
    manager = workspace.WorkspaceManager(torch.device("cpu"))
    first = manager.get_persistent("locks", (8,), torch.int32, zero_init=True)
    assert torch.count_nonzero(first) == 0
    first.fill_(1)
    (scratch,) = manager.get_simultaneous(((1024,), torch.uint8))
    scratch.zero_()
    assert manager.get_persistent("locks", (8,), torch.int32, zero_init=True) is first
    assert torch.all(first == 1)
    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (16,), torch.int32)
    with pytest.raises(ValueError, match="requested shape"):
        manager.get_persistent("locks", (8,), torch.int64)
    manager.lock()
    assert manager.get_persistent("locks", (8,), torch.int32) is first
    with pytest.raises(AssertionError, match="was not allocated during warmup"):
        manager.get_persistent("new", (8,), torch.int32)


@pytest.mark.parametrize(
    ("tp_collective_overlap", "use_ubatching", "num_ubatches", "expected"),
    [
        # Microbatches that only overlap TP collectives share one scratch slot.
        (True, True, 2, 1),
        (True, True, 3, 1),
        # Expert all-to-all microbatching hands off with scratch live: a slot each.
        (False, True, 2, 2),
        (False, True, 4, 4),
        (False, False, 2, 1),
    ],
)
def test_workspace_slots_follow_the_kind_of_microbatching(
    tp_collective_overlap: bool,
    use_ubatching: bool,
    num_ubatches: int,
    expected: int,
) -> None:
    parallel_config = SimpleNamespace(
        ubatch_overlap_tp_collectives=tp_collective_overlap,
        use_ubatching=use_ubatching,
        num_ubatches=num_ubatches,
    )
    config = cast(VllmConfig, SimpleNamespace(parallel_config=parallel_config))
    assert _num_workspace_ubatches(config) == expected


def test_one_workspace_slot_serves_every_microbatch(monkeypatch) -> None:
    active_ubatch = [0]
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: active_ubatch[0])
    manager = workspace.WorkspaceManager(torch.device("cpu"), num_ubatches=1)
    assert manager.shares_ubatch_slots

    for ubatch_id in range(3):
        active_ubatch[0] = ubatch_id
        (buffer,) = manager.get_simultaneous(((256 * (ubatch_id + 1),), torch.uint8))
        assert buffer.numel() == 256 * (ubatch_id + 1)
    # One slot, grown to the largest request of any microbatch.
    assert manager._current_workspaces[0].numel() == 768  # type: ignore[union-attr]
    assert len(manager._current_workspaces) == 1

    factory = Mock(side_effect=lambda: {"buffer": torch.zeros(8)})
    resources = []
    for ubatch_id in range(2):
        active_ubatch[0] = ubatch_id
        resources.append(manager.get_persistent_resource("scratch", factory))
    assert resources[0] is resources[1] and factory.call_count == 1

    two = workspace.WorkspaceManager(torch.device("cpu"), num_ubatches=2)
    assert not two.shares_ubatch_slots


class _FakeExperts:
    """The part of a fused-experts class the modular kernel reads before apply."""

    def __init__(self) -> None:
        self.a2_scale = None
        self.seen: dict[str, tuple[int, int]] = {}

    def moe_problem_size(self, a1q, w1, w2, topk_ids):
        return 4, a1q.shape[0], 8, 16, 2

    def workspace_dtype(self, out_dtype):
        return out_dtype

    def workspace_shapes(self, M, N, K, top_k, *args):
        return (M * top_k, max(N, K)), (M * top_k, N), (M, K)

    def finalize_weight_and_reduce_impl(self):
        return None

    def apply(self, output, workspace13, workspace2, **kwargs):
        def span(t):
            return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()

        self.seen = {
            "output": span(output),
            "workspace13": span(workspace13),
            "workspace2": span(workspace2),
        }
        return None


class _FakePrepareFinalize:
    def allocate_fused_expert_output(self, shape, dtype, device, impl):
        return None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("shared_slot", [True, False])
@pytest.mark.parametrize("output_alias", [True, False])
def test_microbatched_expert_output_leaves_a_shared_scratch_slot(
    monkeypatch, shared_slot: bool, output_alias: bool
) -> None:
    """Microbatches that share one scratch slot hand off while the sum of the
    expert output is in flight, so that output may not live in the scratch."""
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk

    device = torch.device("cuda")
    manager = workspace.WorkspaceManager(device, num_ubatches=1 if shared_slot else 2)
    monkeypatch.setattr(workspace, "_manager", manager)
    monkeypatch.setattr(workspace, "dbo_current_ubatch_id", lambda: 0)
    monkeypatch.setattr(mk, "dbo_enabled", lambda: True)

    kernel = mk.FusedMoEKernelModularImpl.__new__(mk.FusedMoEKernelModularImpl)
    experts = _FakeExperts()
    kernel.fused_experts = experts  # type: ignore[assignment]
    kernel.prepare_finalize = _FakePrepareFinalize()  # type: ignore[assignment]

    num_tokens, hidden = 32, 16
    a1q = torch.zeros(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    out_alias = (
        torch.empty(num_tokens, hidden, dtype=torch.bfloat16, device=device)
        if output_alias
        else None
    )
    fused_out = kernel._fused_experts(
        in_dtype=torch.bfloat16,
        a1q=a1q,
        a1q_scale=None,
        w1=a1q,
        w2=a1q,
        topk_weights=a1q,
        topk_ids=a1q,
        activation=None,
        global_num_experts=4,
        local_num_experts=4,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
        output_alias=out_alias,
    )
    start, stop = experts.seen["output"]
    ws_start, ws_stop = (
        manager._current_workspaces[0].data_ptr(),
        manager._current_workspaces[0].data_ptr()
        + manager._current_workspaces[0].numel(),
    )
    inside_scratch = ws_start <= start and stop <= ws_stop
    assert fused_out.data_ptr() == start
    if output_alias:
        assert not inside_scratch and fused_out is out_alias
    elif shared_slot:
        assert not inside_scratch
    else:
        # A slot of its own per microbatch: the output may stay in the scratch.
        assert inside_scratch
