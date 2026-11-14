# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU-group agreements of Muse Spark reduce CPU tensors (CPU only).

A model is built under a default device that is the GPU. A bare ``torch.tensor`` made
there is a CUDA tensor, and the gloo CPU group then touches the GPU from worker threads
whose current device is 0: every TP rank but 0 got a CUDA context on GPU 0 (554 MiB
each, 3.8 GiB on rank 0, which sets the KV pool). The ``meta`` default device stands in
for the GPU here: a tensor that follows it is not a CPU tensor.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.models.muse_spark.nvidia.model as model_ops
import vllm.models.muse_spark.nvidia.ops.gemm_rs as gemm_rs
import vllm.models.muse_spark.nvidia.ops.mnnvl_fusions as mnnvl_fusions

pytestmark = pytest.mark.cpu_test


class _Recorder:
    """Stands in for ``all_reduce`` of the CPU group and records the tensor it gets."""

    def __init__(self):
        self.devices: list[str] = []

    def __call__(self, tensor, *args, **kwargs):
        self.devices.append(tensor.device.type)


@pytest.fixture
def recorder(monkeypatch):
    rec = _Recorder()
    group = SimpleNamespace(world_size=8, cpu_group="cpu-group")
    monkeypatch.setattr(torch.distributed, "all_reduce", rec)
    monkeypatch.setattr(mnnvl_fusions, "get_tp_group", lambda: group)
    monkeypatch.setattr(model_ops, "get_tp_group", lambda: group)
    monkeypatch.setattr(gemm_rs, "get_tp_group", lambda: group)
    return rec


def test_default_device_context_is_visible_to_the_check():
    with torch.device("meta"):
        assert torch.tensor(1).device.type == "meta"


def test_mnnvl_agreement_reduces_a_cpu_tensor(recorder):
    with torch.device("meta"):
        assert mnnvl_fusions._all_tp_ranks(True) is True
        assert mnnvl_fusions._all_tp_ranks(False) is False
    assert recorder.devices == ["cpu", "cpu"]


def test_verdict_agreement_reduces_a_cpu_tensor(recorder):
    with torch.device("meta"):
        assert model_ops._agreed_by_tp_group(True)
    assert recorder.devices == ["cpu"]


def test_scale_agreement_reduces_a_cpu_tensor(recorder):
    with torch.device("meta"):
        # The recorder leaves the tensor as built: [present, value, -value].
        assert model_ops._agreed_scale_by_tp_group(0.5) is True
        assert model_ops._agreed_scale_by_tp_group(None) is False
    assert recorder.devices == ["cpu", "cpu"]


def test_gemm_rs_agreement_reduces_a_cpu_tensor(recorder, monkeypatch):
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8, pipeline_parallel_size=1, use_ubatching=False
        ),
        lora_config=None,
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
    )
    platform = SimpleNamespace(
        is_cuda=lambda: True, is_device_capability=lambda capability: capability == 103
    )
    monkeypatch.setattr(gemm_rs, "current_platform", platform)
    monkeypatch.setattr(
        gemm_rs.MuseSparkGemmRs, "supports_projection", staticmethod(lambda p: False)
    )
    with torch.device("meta"):
        # The recorder leaves the flag at 0: no projection qualifies, so no workspace.
        assert gemm_rs.maybe_create_gemm_rs(vllm_config, [object()]) is None
    assert recorder.devices == ["cpu"]
