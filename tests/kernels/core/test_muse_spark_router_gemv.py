# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for the decode router GEMV (``_muse_C::router_gemv``).

The kernel sums in the fixed fp32 order documented in ``muse_spark_router_gemv.cu``. The
tests check that order bit for bit against a torch emulation, that a token row gets the
same bits at every batch size, and that the logits are as close to the exact product as
the cuBLAS ones.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

_unusable = muse_c.preflight()
if _unusable is not None:
    pytest.skip(
        f"The _muse_C extension cannot be built: {_unusable}", allow_module_level=True
    )

DEVICE = torch.device("cuda")
K = 8192
N = 256
WPR = 4  # warps per row, both geometries


def _router_gemv(x: torch.Tensor, w: torch.Tensor, pdl: bool = False) -> torch.Tensor:
    muse_c.load()
    out = torch.empty(x.shape[0], w.shape[0], device=DEVICE, dtype=torch.float32)
    torch.ops._muse_C.router_gemv(x, w, out, pdl)
    return out


def _emulate(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The fixed order of the kernel in torch fp32 (IEEE adds; the fma of a product that
    is not exact is emulated in float64, which holds the product exactly)."""
    T = x.shape[0]
    lpl = K // 4 // WPR // 32
    xf = x.double().view(T, 1, WPR, lpl, 32, 4)
    wf = w.double().view(1, w.shape[0], WPR, lpl, 32, 4)
    p = xf * wf  # exact in float64 (24 x 24 bit significands)
    acc = [
        torch.zeros(T, w.shape[0], WPR, 32, dtype=torch.float32, device=x.device)
        for _ in range(4)
    ]
    for j in range(lpl):
        for e in range(4):
            # fma(w, x, acc) = round_fp32(w * x + acc): float64 holds w * x exactly
            # and adds acc exactly (up to the double rounding of the sum, which needs
            # 53+ bits; a double rounding is possible but not for the data of these
            # tests)
            acc[e] = (p[:, :, :, j, :, e] + acc[e].double()).float()
    v = (acc[0] + acc[1]) + (acc[2] + acc[3])
    for half in (16, 8, 4, 2, 1):
        v = v[..., :half] + v[..., half : 2 * half]
    v = v[..., 0]
    s = v[..., 0]
    for k in range(1, WPR):
        s = s + v[..., k]
    return s


def _data(T: int, seed: int = 0, kind: str = "randn"):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(T, K, device=DEVICE, generator=g)
    w = torch.randn(N, K, device=DEVICE, generator=g) * 0.02
    if kind == "mixed":
        e = torch.randint(-5, 6, (K,), device=DEVICE, generator=g).float()
        x = x * torch.exp2(e)
    return x, w


@pytest.mark.parametrize("T", range(1, 9))
@pytest.mark.parametrize("kind", ["randn", "mixed"])
def test_order_is_the_documented_one(T, kind):
    x, w = _data(T, T, kind)
    got = _router_gemv(x, w)
    ref = _emulate(x, w)
    # float64 emulation of fma: equal bits except in the (practically absent)
    # double-rounding case
    assert torch.equal(got, ref)


@pytest.mark.parametrize("T", range(2, 9))
def test_a_row_does_not_depend_on_the_batch(T):
    x, w = _data(T, 100 + T)
    got = _router_gemv(x, w)
    for t in range(T):
        assert torch.equal(_router_gemv(x[t : t + 1].contiguous(), w)[0], got[t])


@pytest.mark.parametrize("T", [1, 2, 4, 8])
def test_as_accurate_as_cublas(T):
    x, w = _data(T, 200 + T, "mixed")
    exact = x.double() @ w.double().t()
    mine = (_router_gemv(x, w).double() - exact).abs()
    cublas = (F.linear(x, w).double() - exact).abs()
    scale = exact.abs().clamp_min(1e-3)
    assert (mine / scale).max() <= 8 * (cublas / scale).max() + 1e-6
    assert mine.mean() <= 2 * cublas.mean() + 1e-9


def test_programmatic_launch_and_graph():
    x, w = _data(4, 7)
    ref = _router_gemv(x, w)
    out = torch.empty(4, N, device=DEVICE)
    torch.ops._muse_C.router_gemv(x, w, out, True)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._muse_C.router_gemv(x, w, out, True)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_rejects_what_it_cannot_serve():
    x, w = _data(2, 1)
    out = torch.empty(2, N, device=DEVICE)
    with pytest.raises(RuntimeError):
        torch.ops._muse_C.router_gemv(x.bfloat16(), w, out, False)
    with pytest.raises(RuntimeError):
        torch.ops._muse_C.router_gemv(
            torch.randn(9, K, device=DEVICE), w, torch.empty(9, N, device=DEVICE), False
        )
    with pytest.raises(RuntimeError):
        torch.ops._muse_C.router_gemv(x[:, :4096].contiguous(), w, out, False)


class _RouterStub(torch.nn.Module):
    """Stands in for the GateLinear: the cuBLAS path, counted."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return F.linear(x.float(), self.weight), None


def _moe_stub(ready=True, engine=True, n=N):
    import vllm.models.muse_spark.nvidia.model as model_ops

    _, w = _data(1, 3)
    w = w[:n].contiguous() if n <= N else torch.cat([w, w[: n - N]]).contiguous()
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.router = _RouterStub(w)
    moe.n_logical_experts = n
    moe.router_gemv_ready = ready
    moe.experts = SimpleNamespace(has_muse_decode_engine=engine)
    return moe, w


@pytest.mark.parametrize("T", [1, 3, 8])
def test_model_uses_the_kernel_for_fp32_decode_batches(T):
    moe, w = _moe_stub()
    x, _ = _data(T, 11)
    got = moe._router_logits(x)
    assert moe.router.calls == 0
    assert torch.equal(got, _router_gemv(x, w))


@pytest.mark.parametrize(
    "case", ["not ready", "bf16 input", "too many tokens", "strided"]
)
def test_model_keeps_cublas_where_the_kernel_does_not_serve(case):
    moe, w = _moe_stub(ready=case != "not ready")
    T = 9 if case == "too many tokens" else 4
    x, _ = _data(T, 12)
    if case == "bf16 input":
        x = x.bfloat16()
    if case == "strided":
        x = torch.randn(T, 2 * K, device=DEVICE)[:, ::2]
        assert not x.is_contiguous()
    got = moe._router_logits(x)
    assert moe.router.calls == 1
    assert got.shape == (T, N)


@pytest.mark.parametrize(
    "case",
    [
        "no fp32 input in a capture",
        "eager forward",
        "layer without the muse decode kernel",
        "fp32 input",
        "too many tokens",
        "not ready",
    ],
)
def test_model_says_when_a_decode_graph_misses_the_kernel(monkeypatch, case):
    """With ROUTER_GEMV=1 a decode graph that gets no fp32 router input (the boundary
    writes it only with the expert prefetch) routes on cuBLAS; that is logged. The
    cases that are no news (an eager forward, a BF16 layer, a prefill batch) are not."""
    import vllm.models.muse_spark.nvidia.model as model_ops

    messages = []
    monkeypatch.setattr(
        model_ops,
        "logger",
        SimpleNamespace(warning_once=lambda msg, *a: messages.append(msg)),
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: case != "eager forward"
    )
    moe, w = _moe_stub(
        ready=case != "not ready", engine=case != "layer without the muse decode kernel"
    )
    T = 9 if case == "too many tokens" else 4
    x, _ = _data(T, 13)
    if case != "fp32 input":
        x = x.bfloat16()
    moe._router_logits(x)
    if case == "fp32 input":
        assert moe.router.calls == 0
    else:
        assert moe.router.calls == 1
    if case == "no fp32 input in a capture":
        assert len(messages) == 1
        assert "EXPERT_PREFETCH" in messages[0] and "ROUTER_GEMV" in messages[0]
    else:
        assert messages == []


def test_model_keeps_cublas_for_an_empty_batch():
    """The kernel needs at least one row; an empty fp32 batch is cuBLAS's, as before."""
    moe, _ = _moe_stub()
    got = moe._router_logits(torch.empty(0, K, device=DEVICE))
    assert moe.router.calls == 1
    assert got.shape == (0, N)


@pytest.mark.parametrize("n", [250, 254, 258])
def test_model_keeps_cublas_for_an_expert_count_the_kernel_does_not_cover(n):
    """The CTAs own 2 or 4 rows of the weight: other expert counts raise in the kernel,
    so the model must not offer them."""
    moe, w = _moe_stub(n=n)
    got = moe._router_logits(_data(3, 14)[0])
    assert moe.router.calls == 1
    assert got.shape == (3, n)


def test_model_uses_the_kernel_for_the_widest_expert_count_it_covers():
    moe, w = _moe_stub(n=260)
    x, _ = _data(2, 15)
    got = moe._router_logits(x)
    assert moe.router.calls == 0
    assert torch.equal(got, _router_gemv(x, w))


# ---- _configure_router_gemv: which layers get the kernel -----------------------------


def _model_stub(prefetch=(True, True, True), moe=True):
    """A model with one layer per entry of `prefetch`: an MoE layer that queues the
    expert prefetch (True) or not (False), or a layer without an MoE."""
    import vllm.models.muse_spark.nvidia.model as model_ops

    layers = []
    for queues in prefetch:
        if moe:
            layer_moe = object.__new__(model_ops.MuseSpark1xMoE)
            torch.nn.Module.__init__(layer_moe)
            layer_moe.router_gemv_ready = False
            layer_moe.expert_prefetch = object() if queues else None
        else:
            layer_moe = torch.nn.Identity()
        layers.append(SimpleNamespace(feed_forward=layer_moe))
    model = object.__new__(model_ops.MuseSpark1xModel)
    model.layers = layers
    model.start_layer, model.end_layer = 0, len(layers)
    return model


@pytest.fixture
def configure(monkeypatch):
    """Runs `_configure_router_gemv` with the extension calls and the log recorded."""
    import vllm.models.muse_spark.nvidia.model as model_ops

    rec = SimpleNamespace(loads=0, infos=[], warnings=[], load_error=None)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_ROUTER_GEMV", "1")
    monkeypatch.setattr(model_ops, "_on_sm120", lambda: True)
    monkeypatch.setattr(muse_c, "preflight", lambda: None)

    def fake_load():
        rec.loads += 1
        if rec.load_error is not None:
            raise rec.load_error

    monkeypatch.setattr(muse_c, "load", fake_load)
    monkeypatch.setattr(
        model_ops,
        "logger",
        SimpleNamespace(
            info_once=lambda msg, *a: rec.infos.append(msg % a),
            warning_once=lambda msg, *a: rec.warnings.append(msg % a),
        ),
    )
    return rec


def _ready(model):
    return [layer.feed_forward.router_gemv_ready for layer in model.layers]


def test_configure_marks_the_layers_that_queue_the_expert_prefetch(configure):
    # a BF16 layer 0 and a last layer without the prefetch keep cuBLAS
    model = _model_stub(prefetch=(False, True, True, False))
    model._configure_router_gemv()
    assert _ready(model) == [False, True, True, False]
    assert configure.loads == 1
    assert len(configure.infos) == 1 and "in 2 layers" in configure.infos[0]


def test_configure_is_off_when_the_flag_is_off(configure, monkeypatch):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_ROUTER_GEMV", "0")
    model = _model_stub()
    model._configure_router_gemv()
    assert _ready(model) == [False] * 3
    assert configure.loads == 0 and configure.infos == []


def test_configure_is_off_outside_sm120(configure, monkeypatch):
    import vllm.models.muse_spark.nvidia.model as model_ops

    monkeypatch.setattr(model_ops, "_on_sm120", lambda: False)
    model = _model_stub()
    model._configure_router_gemv()
    assert _ready(model) == [False] * 3
    assert configure.loads == 0


def test_configure_is_off_without_moe_layers(configure):
    model = _model_stub(moe=False)
    model._configure_router_gemv()
    assert configure.loads == 0 and configure.infos == []


def test_configure_is_off_without_the_expert_prefetch(configure):
    """No layer queues the prefetch (EXPERT_PREFETCH=0): the boundary writes no FP32
    router input, so the flag is logged off and no extension is built for nothing."""
    model = _model_stub(prefetch=(False, False, False))
    model._configure_router_gemv()
    assert _ready(model) == [False] * 3
    assert configure.loads == 0
    assert len(configure.infos) == 1 and "EXPERT_PREFETCH" in configure.infos[0]
    assert "active" not in configure.infos[0]


def test_configure_is_off_when_the_extension_cannot_be_used(configure, monkeypatch):
    monkeypatch.setattr(muse_c, "preflight", lambda: "no nvcc")
    model = _model_stub()
    model._configure_router_gemv()
    assert _ready(model) == [False] * 3
    assert configure.loads == 0
    assert configure.infos == ["The router GEMV is off: no nvcc"]


def test_configure_is_off_when_the_build_fails(configure):
    configure.load_error = RuntimeError("nvcc failed")
    model = _model_stub()
    model._configure_router_gemv()
    assert _ready(model) == [False] * 3
    assert configure.loads == 1
    assert len(configure.warnings) == 1 and "nvcc failed" in configure.warnings[0]
    assert configure.infos == []


def test_the_model_configures_the_router_after_the_expert_prefetch():
    """The kernel needs the layers that queue the prefetch: the constructor must
    configure it afterwards."""
    import inspect

    import vllm.models.muse_spark.nvidia.model as model_ops

    src = inspect.getsource(model_ops.MuseSpark1xModel.__init__)
    assert src.index("self._configure_expert_prefetch()") < src.index(
        "self._configure_router_gemv()"
    )


# ---- the programmatic path: weights early, activations behind the wait --------------


@triton.jit
def _slow_writer(dst, src, n, BLOCK: tl.constexpr):
    """Copies `src` to `dst` after a pause, and lets its programmatic dependent start
    first thing: the producer of an activation that is not ready when the dependent
    becomes resident."""
    tl.extra.cuda.gdc_launch_dependents()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(src + offs, mask=mask, other=0.0)
    bits = v.to(tl.int32, bitcast=True)
    bits = tl.inline_asm_elementwise(
        "{ nanosleep.u32 200000; mov.b32 $0, $1; }",
        "=r,r",
        [bits],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )
    tl.store(dst + offs, bits.to(tl.float32, bitcast=True), mask=mask)


@triton.jit
def _copy_without_wait(dst, src, n, BLOCK: tl.constexpr):
    """A programmatic dependent that does not call griddepcontrol.wait: the control
    that shows the slow writer to be early enough to expose a missing wait."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask, other=0.0), mask=mask)


def _slow_write(dst: torch.Tensor, src: torch.Tensor) -> None:
    n = src.numel()
    _slow_writer[(triton.cdiv(n, 1024),)](dst.view(-1), src.view(-1), n, BLOCK=1024)


def test_programmatic_launch_reads_the_activations_behind_the_wait():
    """With pdl=True the kernel becomes resident while its producer is still writing x;
    it loads the weights then and the activations only behind griddepcontrol.wait."""
    x, w = _data(4, 21)
    ref = _router_gemv(x, w)
    grid = (triton.cdiv(x.numel(), 1024),)
    # Compile both Triton kernels first: a compile between the two launches would let
    # the writer finish before its dependent is queued.
    stale = torch.zeros_like(x)
    seen = torch.empty_like(x)
    _slow_write(stale, x)
    _copy_without_wait[grid](
        seen.view(-1), stale.view(-1), x.numel(), BLOCK=1024, launch_pdl=True
    )
    torch.cuda.synchronize()
    # control: a dependent without the wait reads the activations before they exist
    stale.zero_()
    _slow_write(stale, x)
    _copy_without_wait[grid](
        seen.view(-1), stale.view(-1), x.numel(), BLOCK=1024, launch_pdl=True
    )
    torch.cuda.synchronize()
    if torch.equal(seen, x):
        pytest.skip("a programmatic dependent does not start early enough here")
    assert torch.equal(stale, x)
    for _ in range(3):
        stale.zero_()
        out = torch.empty(4, N, device=DEVICE)
        _slow_write(stale, x)
        torch.ops._muse_C.router_gemv(stale, w, out, True)
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
