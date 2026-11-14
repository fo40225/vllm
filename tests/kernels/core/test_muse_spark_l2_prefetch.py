# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for the decode weight L2 prefetch (``_muse_C::l2_prefetch``).

The prefetch is a cache hint: it must leave every tensor and every GEMM result
bit-for-bit as they were, also when it runs on a side stream inside a CUDA graph.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import vllm.models.muse_spark.nvidia.model as model_ops
from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.models.muse_spark.nvidia.ops.l2_prefetch import (
    AFTER_EXPERTS_CTAS,
    L2_BUDGET_FRACTION,
    L2Prefetcher,
    l2_budget_bytes,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

_unusable = muse_c.preflight()
if _unusable is not None:
    pytest.skip(
        f"The _muse_C extension cannot be built: {_unusable}", allow_module_level=True
    )

DEVICE = torch.device("cuda")
# The per-rank (TP8) QKV+gate and O projection weights of the model.
QKV_SHAPE = (2304, 8192)
O_SHAPE = (8192, 1024)


def _prefetch(tensors, num_ctas=8, chunk_kib=64, evict_last=False) -> None:
    muse_c.load()
    torch.ops._muse_C.l2_prefetch(tensors, num_ctas, chunk_kib, evict_last)


def _weights() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    w_qkv = torch.randn(QKV_SHAPE, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    w_o = torch.randn(O_SHAPE, device=DEVICE, dtype=torch.bfloat16, generator=gen)
    return w_qkv, w_o


@pytest.mark.parametrize("evict_last", [False, True])
@pytest.mark.parametrize("num_ctas,chunk_kib", [(1, 64), (8, 64), (188, 16), (4, 1024)])
def test_prefetch_changes_nothing(num_ctas, chunk_kib, evict_last):
    w_qkv, w_o = _weights()
    x = torch.randn(8, QKV_SHAPE[1], device=DEVICE, dtype=torch.bfloat16)
    ref_w = [w_qkv.clone(), w_o.clone()]
    ref_y = F.linear(x, w_qkv)
    _prefetch([w_qkv, w_o], num_ctas, chunk_kib, evict_last)
    torch.accelerator.synchronize()
    assert torch.equal(w_qkv, ref_w[0]) and torch.equal(w_o, ref_w[1])
    assert torch.equal(F.linear(x, w_qkv), ref_y)


@pytest.mark.parametrize("nbytes", [16, 4096, 65536 + 16, 1_000_000 + 4, 3 * 65536])
def test_ragged_sizes(nbytes):
    # A size that is no multiple of the chunk or of 16 B must neither fault nor
    # touch the bytes next to the tensor.
    buf = torch.arange(nbytes + 4096, device=DEVICE, dtype=torch.int64).to(torch.uint8)
    ref = buf.clone()
    _prefetch([buf[:nbytes], buf[: nbytes // 2 + 16]], 3, 16)
    torch.accelerator.synchronize()
    assert torch.equal(buf, ref)


def test_rejects_unaligned_and_empty():
    buf = torch.zeros(4096, device=DEVICE, dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        _prefetch([buf[1:]])
    with pytest.raises(RuntimeError):
        _prefetch([])


@pytest.mark.parametrize("evict_last", [False, True])
def test_graph_side_stream(evict_last):
    """A fork after the first GEMM and a join at the end replay bit-exactly."""
    w_qkv, w_o = _weights()
    x = torch.randn(4, QKV_SHAPE[1], device=DEVICE, dtype=torch.bfloat16)
    prefetcher = L2Prefetcher(evict_last=evict_last)
    ref = F.linear(F.linear(x, w_qkv)[:, :1024], w_o)

    # Outside a capture the prefetcher does nothing.
    prefetcher.launch([w_qkv, w_o])
    prefetcher.join()

    # Warm up on a side stream (cuBLAS workspaces), then capture.
    warm = torch.cuda.Stream()
    with torch.cuda.stream(warm):
        F.linear(x, w_qkv)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = F.linear(x, w_qkv)
        prefetcher.launch([w_qkv, w_o])
        out = F.linear(y[:, :1024], w_o)
        prefetcher.join()
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("num_ctas", [None, 1, AFTER_EXPERTS_CTAS])
def test_graph_launch_with_its_own_cta_count(num_ctas):
    """The launch behind the experts takes its own CTA count; the default stays."""
    w_qkv, w_o = _weights()
    x = torch.randn(4, QKV_SHAPE[1], device=DEVICE, dtype=torch.bfloat16)
    prefetcher = L2Prefetcher()
    ref = F.linear(F.linear(x, w_qkv)[:, :1024], w_o)
    warm = torch.cuda.Stream()
    with torch.cuda.stream(warm):
        F.linear(x, w_qkv)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = F.linear(x, w_qkv)
        prefetcher.launch([w_qkv, w_o], num_ctas=num_ctas)
        out = F.linear(y[:, :1024], w_o)
        prefetcher.join()
    for _ in range(3):
        graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(out, ref)


MIB = 1 << 20


def _bf16(rows: int, cols: int) -> torch.Tensor:
    return torch.empty(rows, cols, device=DEVICE, dtype=torch.bfloat16)


def test_budget_is_a_share_of_the_device_l2():
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    assert l2_budget_bytes() == int(l2_bytes * L2_BUDGET_FRACTION)
    assert 0 < l2_budget_bytes() < l2_bytes


def test_select_keeps_what_fits_the_budget_in_order():
    prefetcher = L2Prefetcher(budget_bytes=10 * MIB)
    tensors = [
        _bf16(3 * MIB, 1),
        _bf16(5 * MIB // 2, 1),
        _bf16(MIB, 1),
        _bf16(MIB // 2, 1),
    ]
    # 6 MiB, 5 MiB (11 > 10: skipped), 2 MiB (8), 1 MiB (9); nothing is reordered.
    assert prefetcher.select(tensors) == [tensors[0], tensors[2], tensors[3]]
    assert prefetcher.select([_bf16(11 * MIB // 2, 1)]) == []


def test_select_skips_unusable_tensors_and_caps_the_count():
    prefetcher = L2Prefetcher(budget_bytes=1 << 30)
    good = [_bf16(64, 64) for _ in range(6)]
    odd = torch.zeros(4096 + 1, device=DEVICE, dtype=torch.uint8)[1:]  # unaligned
    strided = _bf16(64, 64).t()
    tensors = [odd, strided, torch.empty(0, device=DEVICE), torch.zeros(64), *good]
    assert prefetcher.select(tensors) == good[:4]


def test_the_tp8_decode_set_fits_the_default_budget_and_a_smaller_tp_is_trimmed():
    l2_bytes = 128 * MIB
    budget = int(l2_bytes * L2_BUDGET_FRACTION)
    prefetcher = L2Prefetcher(budget_bytes=budget)
    # After the experts of a layer (unpooler shard, QKV+gate, O), per rank.
    tp8 = [_bf16(1024, 4096), _bf16(*QKV_SHAPE), _bf16(*O_SHAPE)]
    assert prefetcher.select(tp8) == tp8
    # TP4 doubles the attention shards: QKV and O would not fit together.
    tp4 = [_bf16(2048, 4096), _bf16(2 * QKV_SHAPE[0], QKV_SHAPE[1]), _bf16(8192, 2048)]
    assert prefetcher.select(tp4) == tp4[:2]


class _Experts(model_ops.MuseMoERunner):
    supports_after_gemm2_hook = True

    def set_after_gemm2_hook(self, hook) -> bool:
        self.hooks.append(hook)
        return True


def _model_with_layers(hookable: bool) -> model_ops.MuseSpark1xModel:
    """A model with just the state ``_configure_l2_prefetch`` reads: three layers
    with MoE blocks, whose experts do or do not have the hook call site."""
    layers = []
    for _ in range(3):
        moe = object.__new__(model_ops.MuseSpark1xMoE)
        torch.nn.Module.__init__(moe)
        experts = object.__new__(_Experts)
        torch.nn.Module.__init__(experts)
        moe.experts = experts
        moe.experts.hooks = []
        moe.experts.supports_after_gemm2_hook = hookable
        layers.append(SimpleNamespace(feed_forward=moe, self_attn=SimpleNamespace()))
    model = object.__new__(model_ops.MuseSpark1xModel)
    model.layers = layers
    model.start_layer, model.end_layer = 0, 3
    model._l2_prefetcher = None
    return model


def test_a_failed_extension_build_leaves_the_prefetch_off(monkeypatch):
    """The preflight passes but the build fails (or failed earlier): the model must
    still be constructed, without the prefetch and without hooks."""
    monkeypatch.setattr(muse_c, "preflight", lambda: None)

    def cannot_build(*args, **kwargs):
        raise RuntimeError("nvcc: out of luck")

    monkeypatch.setattr(model_ops, "L2Prefetcher", cannot_build)
    model = _model_with_layers(hookable=True)
    model._configure_l2_prefetch()
    assert model._l2_prefetcher is None
    assert all(not layer.feed_forward.experts.hooks for layer in model.layers)


def test_a_model_without_hook_sites_builds_no_prefetcher(monkeypatch):
    monkeypatch.setattr(muse_c, "preflight", lambda: None)
    monkeypatch.setattr(
        model_ops, "L2Prefetcher", lambda: pytest.fail("built for nothing")
    )
    model = _model_with_layers(hookable=False)
    model._configure_l2_prefetch()
    assert model._l2_prefetcher is None


def test_the_prefetch_is_hooked_into_all_but_the_last_layer(monkeypatch):
    monkeypatch.setattr(muse_c, "preflight", lambda: None)
    model = _model_with_layers(hookable=True)
    model._configure_l2_prefetch()
    assert isinstance(model._l2_prefetcher, L2Prefetcher)
    assert [len(layer.feed_forward.experts.hooks) for layer in model.layers] == [
        1,
        1,
        0,
    ]
