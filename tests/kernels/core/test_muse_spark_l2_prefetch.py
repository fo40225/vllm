# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for the decode weight L2 prefetch (``_muse_C::l2_prefetch``).

The prefetch is a cache hint: it must leave every tensor and every GEMM result
bit-for-bit as they were, also when it runs on a side stream inside a CUDA graph.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.models.muse_spark.nvidia.ops.l2_prefetch import L2Prefetcher
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
