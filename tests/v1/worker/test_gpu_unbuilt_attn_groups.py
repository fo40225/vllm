# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The target runner builds no attention metadata for draft-only attention groups.

A DFlash-family speculator builds the metadata of its draft layers itself, from its own
attention groups; the groups the target runner made for them (a draft layer that
differs from the target's in softmax scale or head count gets a group and a metadata
builder of its own) cost a metadata build per group and step for nothing (CPU only).
"""

from types import SimpleNamespace

import pytest

import vllm.v1.worker.gpu.model_runner as gpu_model_runner
from vllm.v1.worker.gpu.attn_utils import drop_unbuilt_attn_groups
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

pytestmark = pytest.mark.cpu_test


def _group(*names: str) -> SimpleNamespace:
    return SimpleNamespace(layer_names=list(names))


def test_groups_of_draft_layers_only_are_dropped():
    target0, target1 = _group("t0", "t1"), _group("t2")
    mixed = _group("t3", "d0")  # shares the target's group: a target layer is in it
    draft_only = _group("d1", "d2")
    attn_groups = [[target0, draft_only], [target1, mixed], [draft_only], []]

    kept = drop_unbuilt_attn_groups(attn_groups, {"d0", "d1", "d2"})

    assert kept == [[target0], [target1, mixed], [], []]
    # The input is left as it was (the caller rebinds).
    assert attn_groups[0] == [target0, draft_only]


def test_an_empty_name_set_keeps_everything():
    groups = [[_group("a")], [_group("b"), _group("c")]]
    assert drop_unbuilt_attn_groups(groups, set()) == groups


@pytest.mark.parametrize(
    "speculator",
    [DFlashSpeculator, DFlash2Speculator, DSparkSpeculator, LiLiCorrSpeculator],
)
def test_the_dflash_family_builds_its_own_metadata(speculator):
    assert speculator.builds_own_attn_metadata is True


def test_other_speculators_keep_the_target_groups_of_their_layers():
    assert BaseSpeculator.builds_own_attn_metadata is False


def _zeroer_groups(monkeypatch, speculator) -> list:
    seen: list = []

    def zeroer(device, attn_groups_iter, **kwargs):
        seen.extend(attn_groups_iter)

    monkeypatch.setattr(gpu_model_runner, "KVBlockZeroer", zeroer)
    runner = SimpleNamespace(
        device="cpu",
        attn_groups=[[_group("t0")], [], [_group("t1")]],
        speculator=speculator,
        kernel_block_sizes=[16, 16, 16],
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_cache_config=SimpleNamespace(num_blocks=8),
    )
    GPUModelRunner._init_kv_zero_meta(runner)
    return seen


def test_the_kv_zeroer_still_sees_the_draft_layers(monkeypatch):
    """The draft layers' groups left the target's list: their KV blocks are still
    zeroed through the speculator's own groups."""
    draft = [[_group("d0")], [_group("d1")], []]
    speculator = SimpleNamespace(builds_own_attn_metadata=True, attn_groups=draft)

    seen = _zeroer_groups(monkeypatch, speculator)

    names = sorted(name for g in seen for name in g.layer_names)
    assert names == ["d0", "d1", "t0", "t1"]


@pytest.mark.parametrize(
    "speculator",
    [
        None,
        SimpleNamespace(builds_own_attn_metadata=False, attn_groups=[[_group("d")]]),
    ],
)
def test_the_kv_zeroer_is_unchanged_for_other_speculators(monkeypatch, speculator):
    seen = _zeroer_groups(monkeypatch, speculator)
    assert sorted(name for g in seen for name in g.layer_names) == ["t0", "t1"]
