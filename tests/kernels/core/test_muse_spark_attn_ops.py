# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark fused attention ops (vllm-project/vllm#62132).

Every test runs against each implementation that exists on the machine: the
stock ``torch.ops._C`` ops (when the build has them) and the JIT-built
``torch.ops._muse_C`` ops (SM12x with nvcc and ninja, or a finished build).
"""

import types

import pytest
import torch

import vllm._custom_ops as ops
from vllm.models.muse_spark.common.ops import has_compiled_op, resolve_compiled_op
from vllm.platforms import current_platform

HEAD_DIM = 64
FUSED_OP = "fused_muse_spark_qknorm_rope_kv_insert"
SOFT_CAP_OP = "muse_spark_mul_tanh_soft_cap"

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike() or not torch.cuda.is_available(),
    reason="Muse Spark kernels require a CUDA or ROCm GPU",
)


def cdiv(a, b):
    return (a + b - 1) // b


def _load_namespace(namespace):
    """The op namespace ``namespace``, or a skip when it is not available."""
    if namespace == "_C":
        if not has_compiled_op(SOFT_CAP_OP):
            pytest.skip("torch.ops._C lacks the Muse Spark ops")
        return torch.ops._C
    if not (
        current_platform.is_cuda() and current_platform.is_device_capability_family(120)
    ):
        pytest.skip("_muse_C is built for SM12x")
    from vllm.models.muse_spark.nvidia.ops import muse_c

    if (reason := muse_c.preflight()) is not None:
        pytest.skip(f"_muse_C is unavailable: {reason}")
    muse_c.load()
    return torch.ops._muse_C


@pytest.fixture(params=["_C", "_muse_C"])
def namespace(request):
    return _load_namespace(request.param)


@pytest.fixture
def fused_insert(namespace):
    """The fused op with the keyword defaults of the vllm._custom_ops wrapper."""
    op = getattr(namespace, FUSED_OP)

    def call(
        qkv,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        eps,
        cos_sin_cache=None,
        rotary_dim=0,
        slot_mapping=None,
        kv_cache=None,
        block_size=0,
        kv_cache_dtype="auto",
        k_scale=None,
        v_scale=None,
        q_fp8_out=None,
        q_scale=None,
        q_out=None,
    ):
        extra = ()
        if q_out is not None:
            # Only the _muse_C op has the dense Q output.
            if not any(a.name == "q_out" for a in op.default._schema.arguments):
                pytest.skip("this op has no q_out")
            extra = (q_out,)
        op(
            qkv,
            positions,
            num_heads,
            num_kv_heads,
            head_dim,
            eps,
            cos_sin_cache,
            rotary_dim,
            slot_mapping,
            kv_cache,
            block_size,
            kv_cache_dtype,
            k_scale,
            v_scale,
            q_fp8_out,
            q_scale,
            *extra,
        )

    return call


def make_cos_sin_cache(max_pos, rotary_dim, base, device, dtype=torch.float32):
    """The table ``RotaryEmbedding`` builds: cos||sin, fp32 for Muse-Spark."""
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def qk_norm_rope_ref(x, positions, cos_sin_cache, rotary_dim, eps):
    """fp32 reference for one q or k branch, rounded once; no table means NoPE."""
    out = x.float()
    out = out * torch.rsqrt(out.pow(2).mean(dim=-1, keepdim=True) + eps)
    if cos_sin_cache is not None:
        half = rotary_dim // 2
        cs = cos_sin_cache[positions].float()
        cos = cs[..., :half].unsqueeze(1)
        sin = cs[..., half:].unsqueeze(1)
        # GPT-J interleaved: the rotated pair is (2i, 2i+1).
        x0 = out[..., :rotary_dim:2]
        x1 = out[..., 1:rotary_dim:2]
        out = out.clone()
        out[..., :rotary_dim:2] = x0 * cos - x1 * sin
        out[..., 1:rotary_dim:2] = x0 * sin + x1 * cos
    return out.to(x.dtype)


def assert_fp8_cache_close(got, expected, exact_fraction=0.999):
    """Compare raw cache codes in the platform's FP8 encoding."""
    diff = (got.int() - expected.int()).abs()
    within = diff <= 1
    assert bool(within.all()), (
        f"fp8 cache differs by more than one code in {int((~within).sum())} "
        f"of {diff.numel()} elements"
    )
    exact = float((diff == 0).float().mean())
    assert exact >= exact_fraction, (
        f"only {exact:.5f} of the fp8 cache is bit-identical to "
        f"reshape_and_cache_flash, expected at least {exact_fraction}"
    )


def make_case(
    num_tokens, nq, nkv, dtype, device, with_gate=True, seed=0, head_dim=HEAD_DIM
):
    torch.manual_seed(seed)
    qsz, kvsz = nq * head_dim, nkv * head_dim
    gate = qsz if with_gate else 0
    qkv = torch.randn(num_tokens, qsz + 2 * kvsz + gate, dtype=dtype, device=device)
    return qkv, qsz, kvsz, gate


def split_qkv(qkv, qsz, kvsz, gate):
    return qkv.split([qsz, kvsz, kvsz, gate], dim=-1)


def check_qk_norm_rope(
    fused_insert,
    num_tokens,
    nq,
    nkv,
    dtype,
    use_rope,
    rotary_dim=HEAD_DIM,
    with_gate=True,
    rope_dtype=torch.float32,
    head_dim=HEAD_DIM,
):
    device, eps, max_pos = "cuda", 1e-5, 4096

    qkv, qsz, kvsz, gate = make_case(
        num_tokens, nq, nkv, dtype, device, with_gate, head_dim=head_dim
    )
    qkv_orig = qkv.clone()
    cos_sin = (
        make_cos_sin_cache(max_pos, rotary_dim, 500_000.0, device, rope_dtype)
        if use_rope
        else None
    )
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    fused_insert(
        qkv,
        positions,
        nq,
        nkv,
        head_dim,
        eps,
        cos_sin_cache=cos_sin,
        rotary_dim=rotary_dim if use_rope else 0,
    )

    q_out, k_out, v_out, og_out = split_qkv(qkv, qsz, kvsz, gate)
    q_in, k_in, v_in, og_in = split_qkv(qkv_orig, qsz, kvsz, gate)

    q_ref = qk_norm_rope_ref(
        q_in.view(num_tokens, nq, head_dim), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, qsz)
    k_ref = qk_norm_rope_ref(
        k_in.view(num_tokens, nkv, head_dim), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, kvsz)

    # rsqrtf and lane summation order cost a couple of fp32 ulp, well inside
    # one ulp of the stored dtype.
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)
    torch.testing.assert_close(og_out, og_in, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_rope", [True, False])
def test_qk_norm_rope(fused_insert, dtype, use_rope):
    check_qk_norm_rope(fused_insert, 17, 32, 4, dtype, use_rope)


@pytest.mark.parametrize(
    "num_tokens,variant",
    [
        pytest.param(64, {"rotary_dim": 32}, id="partial_rotary_dim"),
        # A [q | k | v] row (no o-gate) is addressed by its own stride.
        pytest.param(33, {"with_gate": False}, id="no_gate_row"),
        pytest.param(128, {"rope_dtype": torch.bfloat16}, id="bf16_cos_sin_cache"),
    ],
)
def test_qk_norm_rope_layout_variants(fused_insert, num_tokens, variant):
    check_qk_norm_rope(fused_insert, num_tokens, 32, 4, torch.bfloat16, True, **variant)


# Cover narrow/wide head groups for TP4/TP8 and the one-chunk-per-token case.
@pytest.mark.parametrize(
    "num_tokens,nq,nkv,kv_cache_dtype,use_rope",
    [
        pytest.param(1, 32, 4, "auto", True, id="tp4-decode"),
        pytest.param(1, 16, 2, "fp8", False, id="tp8-decode-fp8-nope"),
        pytest.param(1, 16, 2, "fp8", True, id="tp8-decode-fp8-rope"),
        pytest.param(8, 16, 2, "fp8", True, id="tp8-batch8-fp8-rope"),
        pytest.param(1025, 32, 4, "auto", True, id="tp4-group2"),
        pytest.param(1025, 32, 4, "fp8", True, id="tp4-group2-fp8"),
        pytest.param(1537, 32, 4, "auto", True, id="tp4-group4"),
        pytest.param(1537, 32, 4, "auto", False, id="tp4-nope"),
        pytest.param(3073, 16, 2, "auto", True, id="tp8-group4"),
        pytest.param(3073, 16, 2, "fp8", False, id="tp8-fp8-nope"),
        pytest.param(3073, 16, 2, "fp8_e5m2", True, id="tp8-fp8-e5m2-rope"),
        pytest.param(40000, 2, 1, "auto", True, id="single-chunk-per-token"),
    ],
)
@pytest.mark.parametrize(
    "scales,storage",
    [
        pytest.param((0.37, 1.83), torch.uint8, id="scales-uint8"),
        # The production run: unit scales, and the cache viewed as e4m3fn.
        pytest.param((1.0, 1.0), torch.float8_e4m3fn, id="unit-scales-e4m3fn"),
    ],
)
def test_kv_insert(
    fused_insert, num_tokens, nq, nkv, kv_cache_dtype, use_rope, scales, storage
):
    check_kv_insert(
        fused_insert, num_tokens, nq, nkv, kv_cache_dtype, use_rope, scales, storage
    )


def check_kv_insert(
    fused_insert,
    num_tokens,
    nq,
    nkv,
    kv_cache_dtype,
    use_rope,
    scales,
    storage,
    head_dim=HEAD_DIM,
    rotary_dim=None,
):
    device, dtype, eps = "cuda", torch.bfloat16, 1e-5
    rotary_dim = head_dim if rotary_dim is None else rotary_dim
    max_pos, block_size = 4096, 16
    is_fp8 = kv_cache_dtype != "auto"
    if is_fp8 and kv_cache_dtype == "fp8_e5m2" and storage != torch.uint8:
        pytest.skip("e5m2 codes are stored as raw bytes")
    if not is_fp8 and storage != torch.uint8:
        pytest.skip("the storage dtype only matters for an fp8 cache")

    qkv, qsz, kvsz, gate = make_case(
        num_tokens, nq, nkv, dtype, device, seed=1, head_dim=head_dim
    )
    qkv_orig = qkv.clone()
    cos_sin = (
        make_cos_sin_cache(max_pos, rotary_dim, 500_000.0, device) if use_rope else None
    )
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )

    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    cache_dtype = storage if is_fp8 else dtype
    kv_cache = torch.zeros(
        num_blocks, nkv, block_size, 2 * head_dim, dtype=cache_dtype, device=device
    )
    slot_mapping = torch.randperm(
        num_blocks * block_size, dtype=torch.int64, device=device
    )[:num_tokens]
    if num_tokens > 1:
        # -1 marks a token the scheduler did not place; the kernel must skip it
        # and leave the cache alone.
        slot_mapping[::3] = -1

    # Non-unit scales, so a kernel that ignored them would fail loudly.
    k_scale = torch.tensor(scales[0], dtype=torch.float32, device=device)
    v_scale = torch.tensor(scales[1], dtype=torch.float32, device=device)

    fused_insert(
        qkv,
        positions,
        nq,
        nkv,
        head_dim,
        eps,
        cos_sin_cache=cos_sin,
        rotary_dim=rotary_dim if use_rope else 0,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=k_scale if is_fp8 else None,
        v_scale=v_scale if is_fp8 else None,
    )

    q_out, k_out, v_out, og_out = split_qkv(qkv, qsz, kvsz, gate)
    q_in, k_in, v_in, og_in = split_qkv(qkv_orig, qsz, kvsz, gate)

    q_ref = qk_norm_rope_ref(
        q_in.view(num_tokens, nq, head_dim), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, qsz)
    k_ref = qk_norm_rope_ref(
        k_in.view(num_tokens, nkv, head_dim), positions, cos_sin, rotary_dim, eps
    ).view(num_tokens, kvsz)
    torch.testing.assert_close(q_out, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(v_out, v_in, rtol=0, atol=0)
    torch.testing.assert_close(og_out, og_in, rtol=0, atol=0)

    expected = torch.zeros_like(kv_cache)
    exp_k, exp_v = expected.transpose(1, 2).split(head_dim, dim=-1)
    ops.reshape_and_cache_flash(
        k_out.view(num_tokens, nkv, head_dim),
        v_out.view(num_tokens, nkv, head_dim),
        exp_k,
        exp_v,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
    if not is_fp8:
        torch.testing.assert_close(kv_cache, expected, rtol=0, atol=0)
    else:
        assert_fp8_cache_close(kv_cache.view(torch.uint8), expected.view(torch.uint8))


def full_wave_tokens(nq, nkv, head_dim, insert_kv):
    """Fewest tokens whose one-head-per-group grid fills a wave of resident blocks.

    The launcher widens its head groups from there on. This mirrors its wave
    (256-thread blocks, as many per SM as the thread limit allows, at most the 6
    that its register cap guarantees) so the tests straddle the switch on any GPU.
    """
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    resident = max(1, min(props.max_threads_per_multi_processor // 256, 6))
    wave = props.multi_processor_count * resident
    slots = nq + nkv + (nkv if insert_kv else 0)
    # Smallest n with ceil(n * slots * lanes / 256) >= wave, 8 elements per lane.
    return ((wave - 1) * 256) // (slots * (head_dim // 8)) + 1


# (nq, nkv, head_dim, rotary_dim): head layouts beyond this model's TP4/TP8 ones,
# including an odd slot count (only one head per group can divide it).
HEAD_CONFIGS = [
    pytest.param(8, 1, 64, 64, id="q8-kv1-hd64"),
    pytest.param(12, 4, 64, 32, id="q12-kv4-hd64-partial-rope"),
    pytest.param(7, 1, 64, 64, id="q7-kv1-hd64-odd-slots"),
    pytest.param(64, 8, 64, 64, id="q64-kv8-hd64"),
    pytest.param(8, 1, 128, 128, id="q8-kv1-hd128"),
    pytest.param(32, 8, 128, 64, id="q32-kv8-hd128-partial-rope"),
    pytest.param(12, 4, 256, 128, id="q12-kv4-hd256"),
]


@pytest.mark.parametrize("nq,nkv,head_dim,rotary_dim", HEAD_CONFIGS)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize("use_rope", [True, False])
def test_head_configurations(
    fused_insert, nq, nkv, head_dim, rotary_dim, kv_cache_dtype, use_rope
):
    """Every head layout is right below, at and far above the group widening point."""
    wave_tokens = full_wave_tokens(nq, nkv, head_dim, insert_kv=True)
    for num_tokens in (5, wave_tokens - 1, wave_tokens, 2 * wave_tokens + 1):
        check_kv_insert(
            fused_insert,
            num_tokens,
            nq,
            nkv,
            kv_cache_dtype,
            use_rope,
            (0.37, 1.83),
            torch.uint8,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
        )
    # No cache bound: one slot fewer per K head, so a different widening point.
    check_qk_norm_rope(
        fused_insert,
        2 * full_wave_tokens(nq, nkv, head_dim, insert_kv=False) + 1,
        nq,
        nkv,
        torch.bfloat16,
        use_rope,
        rotary_dim=rotary_dim,
        head_dim=head_dim,
    )


@pytest.mark.parametrize(
    "nq,nkv,head_dim,rotary_dim",
    [
        pytest.param(16, 2, 64, 64, id="tp8"),
        pytest.param(32, 4, 64, 64, id="tp4"),
        pytest.param(7, 1, 64, 64, id="odd-slots"),
        pytest.param(12, 4, 256, 128, id="hd256"),
    ],
)
@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
def test_group_width_does_not_change_the_bits(
    fused_insert, nq, nkv, head_dim, rotary_dim, kv_cache_dtype
):
    """A token's output cannot depend on how many tokens share the launch.

    The batch size picks the head group width, so the first n tokens of a large
    batch (wide groups) must match a launch of just those n (narrow groups).
    """
    device, dtype, block_size, max_pos = "cuda", torch.bfloat16, 16, 4096
    is_fp8 = kv_cache_dtype != "auto"
    wave_tokens = full_wave_tokens(nq, nkv, head_dim, insert_kv=True)
    total = 2 * wave_tokens + 1
    qkv, _, _, _ = make_case(total, nq, nkv, dtype, device, seed=3, head_dim=head_dim)
    positions = torch.randint(0, max_pos, (total,), dtype=torch.int64, device=device)
    cos_sin = make_cos_sin_cache(max_pos, rotary_dim, 500_000.0, device)
    num_blocks = (total + block_size - 1) // block_size + 1
    slots = torch.randperm(num_blocks * block_size, dtype=torch.int64, device=device)
    slots = slots[:total].contiguous()
    slots[::3] = -1
    k_scale = torch.tensor(0.37, dtype=torch.float32, device=device)
    v_scale = torch.tensor(1.83, dtype=torch.float32, device=device)

    def run(n, placed):
        """Launch on the first n tokens; only the first `placed` write the cache."""
        x = qkv[:n].clone()
        cache = torch.zeros(
            num_blocks,
            nkv,
            block_size,
            2 * head_dim,
            dtype=torch.uint8 if is_fp8 else dtype,
            device=device,
        )
        slot_mapping = slots[:n].clone()
        slot_mapping[placed:] = -1
        fused_insert(
            x,
            positions[:n],
            nq,
            nkv,
            head_dim,
            1e-5,
            cos_sin_cache=cos_sin,
            rotary_dim=rotary_dim,
            slot_mapping=slot_mapping,
            kv_cache=cache,
            block_size=block_size,
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale if is_fp8 else None,
            v_scale=v_scale if is_fp8 else None,
        )
        return x, cache.view(torch.uint8) if is_fp8 else cache.view(torch.int16)

    for n in (1, 5, wave_tokens - 1, wave_tokens, total - 1):
        x_prefix, cache_prefix = run(n, n)
        x_full, cache_full = run(total, n)
        assert torch.equal(x_full[:n], x_prefix), n
        assert torch.equal(cache_full, cache_prefix), n


# The fused Q output requires E4M3FN; scaled_fp8_quant uses the platform dtype.
@pytest.mark.skipif(
    current_platform.is_rocm() and current_platform.fp8_dtype() != torch.float8_e4m3fn,
    reason="Fused E4M3FN Q output cannot be compared with ROCm FNUZ quantization",
)
@pytest.mark.parametrize(
    "head_dim,num_tokens,rotary_fraction,use_cuda_graph",
    [
        pytest.param(64, 17, 0, False, id="head64-nope"),
        pytest.param(128, 17, 0.5, False, id="head128-partial-rope"),
        pytest.param(256, 17, 0, False, id="head256-nope"),
        pytest.param(256, 3073, 0.5, False, id="head256-partial-rope-large"),
        pytest.param(64, 3073, 0.5, True, id="graph-rope"),
        pytest.param(64, 3073, 0, True, id="graph-nope"),
    ],
)
def test_fused_query_quant_matches_separate_quant(
    fused_insert, head_dim, num_tokens, rotary_fraction, use_cuda_graph
):
    """Preserve the rounding boundary, scales, padded Q, and KV/gate writes."""
    torch.manual_seed(42)
    nq, nkv, block_size = 16, 2, 16
    qsz, kvsz = nq * head_dim, nkv * head_dim
    qkv = torch.randn(num_tokens, 2 * (qsz + kvsz), device="cuda", dtype=torch.bfloat16)
    original = qkv.clone()
    reference = qkv.clone()
    q_fp8 = torch.empty(num_tokens, qsz, device="cuda", dtype=torch.float8_e4m3fn)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    rotary_dim = int(head_dim * rotary_fraction)
    cos_sin = (
        make_cos_sin_cache(num_tokens, rotary_dim, 500_000.0, "cuda")
        if rotary_dim
        else None
    )
    cache = torch.zeros(
        (num_tokens + block_size - 1) // block_size,
        nkv,
        block_size,
        2 * head_dim,
        device="cuda",
        dtype=torch.uint8,
    )
    expected_cache = torch.zeros_like(cache)
    slots = positions.clone()
    slots[::3] = -1
    # Small enough that part of Q saturates the e4m3 range.
    scale = torch.tensor(0.001, device="cuda", dtype=torch.float32)
    kwargs = dict(
        cos_sin_cache=cos_sin,
        rotary_dim=rotary_dim,
        slot_mapping=slots,
        block_size=block_size,
        kv_cache_dtype="fp8",
        k_scale=torch.tensor(0.37, device="cuda"),
        v_scale=torch.tensor(1.83, device="cuda"),
    )
    fused_insert(
        reference,
        positions,
        nq,
        nkv,
        head_dim,
        1e-5,
        kv_cache=expected_cache,
        **kwargs,
    )
    expected_q, _ = ops.scaled_fp8_quant(reference[:, :qsz], scale=scale)

    def run():
        qkv.copy_(original)
        fused_insert(
            qkv,
            positions,
            nq,
            nkv,
            head_dim,
            1e-5,
            kv_cache=cache,
            q_fp8_out=q_fp8,
            q_scale=scale,
            **kwargs,
        )

    run()
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        qkv.zero_()
        cache.zero_()
        q_fp8.zero_()
        graph.replay()
    torch.testing.assert_close(
        q_fp8.view(torch.uint8), expected_q.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    torch.testing.assert_close(qkv[:, :qsz], original[:, :qsz], rtol=0, atol=0)
    torch.testing.assert_close(qkv[:, qsz:], reference[:, qsz:], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,nq,nkv,head_dim,rotary_dim,kv_cache_dtype",
    [
        pytest.param(1, 16, 2, 64, 64, "fp8", id="tp8-1"),
        pytest.param(7, 16, 2, 64, 64, "fp8", id="tp8-7"),
        pytest.param(9, 16, 2, 64, 64, "fp8", id="tp8-9"),
        pytest.param(64, 16, 2, 64, 64, "fp8", id="tp8-64"),
        pytest.param(128, 16, 2, 64, 64, "fp8", id="tp8-128"),
        pytest.param(64, 16, 2, 64, 0, "fp8", id="tp8-64-nope"),
        pytest.param(64, 16, 2, 64, 64, "auto", id="tp8-64-bf16-cache"),
        pytest.param(77, 32, 4, 64, 64, "fp8", id="tp4-77"),
        pytest.param(33, 16, 2, 128, 64, "fp8", id="head128-33"),
        pytest.param(3073, 16, 2, 64, 64, "fp8", id="prefill-3073"),
    ],
)
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_dense_q_output_is_bitwise_the_in_place_q(
    fused_insert,
    dtype,
    num_tokens,
    nq,
    nkv,
    head_dim,
    rotary_dim,
    kv_cache_dtype,
    use_cuda_graph,
):
    """q_out holds the bits of the in-place Q; qkv keeps its Q columns."""
    device, eps, block_size, max_pos = "cuda", 1e-5, 16, 4096
    torch.manual_seed(7)
    qsz, kvsz = nq * head_dim, nkv * head_dim
    qkv = torch.randn(num_tokens, 2 * (qsz + kvsz), dtype=dtype, device=device)
    original = qkv.clone()
    reference = qkv.clone()
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.int64, device=device
    )
    cos_sin = (
        make_cos_sin_cache(max_pos, rotary_dim, 500_000.0, device)
        if rotary_dim
        else None
    )
    is_fp8 = kv_cache_dtype != "auto"
    num_blocks = (num_tokens + block_size - 1) // block_size + 1
    cache_dtype = torch.uint8 if is_fp8 else dtype
    cache = torch.zeros(
        num_blocks, nkv, block_size, 2 * head_dim, dtype=cache_dtype, device=device
    )
    expected_cache = torch.zeros_like(cache)
    slots = torch.randperm(num_blocks * block_size, dtype=torch.int64, device=device)[
        :num_tokens
    ]
    if num_tokens > 1:
        slots[::3] = -1  # a token the scheduler did not place
    kwargs = dict(
        cos_sin_cache=cos_sin,
        rotary_dim=rotary_dim,
        slot_mapping=slots,
        block_size=block_size,
        kv_cache_dtype=kv_cache_dtype,
        k_scale=torch.tensor(0.37, device=device) if is_fp8 else None,
        v_scale=torch.tensor(1.83, device=device) if is_fp8 else None,
    )
    fused_insert(
        reference, positions, nq, nkv, head_dim, eps, kv_cache=expected_cache, **kwargs
    )

    q_out = torch.full((num_tokens, qsz), float("nan"), dtype=dtype, device=device)

    def run():
        qkv.copy_(original)
        fused_insert(
            qkv,
            positions,
            nq,
            nkv,
            head_dim,
            eps,
            kv_cache=cache,
            q_out=q_out,
            **kwargs,
        )

    run()
    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        qkv.zero_()
        cache.zero_()
        q_out.fill_(float("nan"))
        graph.replay()
    assert torch.equal(q_out.view(torch.int16), reference[:, :qsz].view(torch.int16))
    # The view of qkv is untouched where Q was; K, V, the gate and the cache are the
    # ones of the in-place run.
    assert torch.equal(qkv[:, :qsz], original[:, :qsz])
    assert torch.equal(qkv[:, qsz:], reference[:, qsz:])
    assert torch.equal(cache, expected_cache)


def test_dense_q_output_is_what_flashinfer_needs(fused_insert):
    """The in-place Q is a strided view for more than one row; q_out is dense."""
    num_tokens, nq, nkv = 64, 16, 2
    qkv, qsz, _, _ = make_case(num_tokens, nq, nkv, torch.bfloat16, "cuda")
    assert not qkv[:, :qsz].view(-1, nq, HEAD_DIM).is_contiguous()
    assert qkv[:1, :qsz].view(-1, nq, HEAD_DIM).is_contiguous()
    q_out = torch.empty(num_tokens, qsz, dtype=torch.bfloat16, device="cuda")
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    fused_insert(qkv, positions, nq, nkv, HEAD_DIM, 1e-5, q_out=q_out)
    assert q_out.view(-1, nq, HEAD_DIM).is_contiguous()
    assert q_out.data_ptr() % 16 == 0


def test_dense_q_output_rejects_bad_arguments(fused_insert):
    num_tokens, nq, nkv = 8, 16, 2
    qkv, qsz, _, _ = make_case(num_tokens, nq, nkv, torch.bfloat16, "cuda")
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)

    def call(q_out, **kwargs):
        fused_insert(qkv, positions, nq, nkv, HEAD_DIM, 1e-5, q_out=q_out, **kwargs)

    with pytest.raises(RuntimeError, match="q_out"):
        call(torch.empty(num_tokens, qsz - 64, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(RuntimeError, match="q_out"):
        call(torch.empty(num_tokens + 1, qsz, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(RuntimeError, match="q_out"):
        call(torch.empty(num_tokens, qsz, dtype=torch.float16, device="cuda"))
    with pytest.raises(RuntimeError, match="q_out"):
        # Not dense.
        wide = torch.empty(num_tokens, 2 * qsz, dtype=torch.bfloat16, device="cuda")
        call(wide[:, :qsz])
    with pytest.raises(RuntimeError, match="q_out"):
        call(
            torch.empty(num_tokens, qsz, dtype=torch.bfloat16, device="cuda"),
            q_fp8_out=torch.empty(
                num_tokens, qsz, dtype=torch.float8_e4m3fn, device="cuda"
            ),
            q_scale=torch.ones(1, device="cuda"),
        )


def _wrapper_layer(backend_name):
    """The parts of an attention layer that the python wrapper of the insert reads."""
    return types.SimpleNamespace(
        attn_backend=types.SimpleNamespace(get_name=lambda: backend_name),
        kv_cache_dtype="fp8",
        _k_scale=torch.tensor(0.37, device="cuda"),
        _v_scale=torch.tensor(1.83, device="cuda"),
        _q_scale=torch.ones(1, device="cuda"),
    )


@pytest.mark.parametrize("num_tokens", [1, 64, 3073])
def test_wrapper_returns_the_dense_q_of_the_in_place_run(
    monkeypatch, namespace, num_tokens
):
    from vllm.models.muse_spark.common.ops import fused_qk_norm_rope_kv as wrapper

    op = getattr(namespace, FUSED_OP)
    if not wrapper._op_has_dense_q_output(op):
        pytest.skip("this op has no q_out")
    nq, nkv, block_size = 16, 2, 16
    qkv, qsz, _, _ = make_case(num_tokens, nq, nkv, torch.bfloat16, "cuda", seed=3)
    positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    slots = positions.clone()
    slots[::3] = -1
    cache = torch.zeros(
        cdiv(num_tokens, block_size),
        nkv,
        block_size,
        2 * HEAD_DIM,
        device="cuda",
        dtype=torch.uint8,
    )
    layer = _wrapper_layer("FLASHINFER")
    monkeypatch.setattr(
        wrapper, "get_attention_context", lambda name: (None, layer, cache, slots)
    )
    monkeypatch.setattr(wrapper, "resolve_compiled_op", lambda name: op)
    cos_sin = make_cos_sin_cache(num_tokens, HEAD_DIM, 500_000.0, "cuda")

    def run(dense_q):
        x, c = qkv.clone(), torch.zeros_like(cache)
        monkeypatch.setattr(
            wrapper, "get_attention_context", lambda name: (None, layer, c, slots)
        )
        q = wrapper.fused_qk_norm_rope_kv_insert(
            x, positions, cos_sin, "layer", nq, nkv, HEAD_DIM, HEAD_DIM, 1e-5, dense_q
        )
        return x, c, q

    x_ref, cache_ref, q_ref = run(False)
    assert q_ref is None
    x_dense, cache_dense, q_dense = run(True)
    assert q_dense is not None and q_dense.shape == (num_tokens, qsz)
    assert q_dense.view(-1, nq, HEAD_DIM).is_contiguous()
    assert torch.equal(q_dense.view(torch.int16), x_ref[:, :qsz].view(torch.int16))
    assert torch.equal(x_dense[:, :qsz], qkv[:, :qsz])
    assert torch.equal(x_dense[:, qsz:], x_ref[:, qsz:])
    assert torch.equal(cache_dense, cache_ref)


def test_supports_dense_q_needs_flashinfer_and_the_muse_op(monkeypatch, namespace):
    from vllm.models.muse_spark.common.ops import fused_qk_norm_rope_kv as wrapper

    op = getattr(namespace, FUSED_OP)
    monkeypatch.setattr(wrapper, "resolve_compiled_op", lambda name: op)
    has_q_out = wrapper._op_has_dense_q_output(op)
    assert wrapper.supports_dense_q(_wrapper_layer("FLASHINFER")) is has_q_out
    assert not wrapper.supports_dense_q(_wrapper_layer("FLASH_ATTN"))

    stock = types.SimpleNamespace(
        default=types.SimpleNamespace(
            _schema=types.SimpleNamespace(
                arguments=[types.SimpleNamespace(name="qkv")]
            )
        )
    )
    monkeypatch.setattr(wrapper, "resolve_compiled_op", lambda name: stock)
    assert not wrapper.supports_dense_q(_wrapper_layer("FLASHINFER"))
    monkeypatch.setattr(wrapper, "resolve_compiled_op", lambda name: None)
    assert not wrapper.supports_dense_q(_wrapper_layer("FLASHINFER"))


def test_fused_insert_rejects_a_misshaped_cache(fused_insert):
    qkv, _, _, _ = make_case(4, 16, 2, torch.bfloat16, "cuda")
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    # NHD layout [nb, bs, nkv, 2*hd] with bs != nkv.
    cache = torch.zeros(2, 16, 2, 2 * HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="kv_cache"):
        fused_insert(
            qkv,
            positions,
            16,
            2,
            HEAD_DIM,
            1e-5,
            slot_mapping=positions.clone(),
            kv_cache=cache,
            block_size=16,
        )


def soft_cap_ref(x, multiplier, temp):
    return temp * torch.tanh(x.float() * multiplier / temp)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [(1, 202048), (4, 202048), (3, 1001), (32, 8), (257, 50)],
)
def test_mul_tanh_soft_cap(namespace, dtype, shape):
    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 30).to(dtype)
    out = torch.empty_like(x, dtype=torch.float32)
    getattr(namespace, SOFT_CAP_OP)(out, x, 0.5, 20.0)
    torch.testing.assert_close(out, soft_cap_ref(x, 0.5, 20.0), rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mul_tanh_soft_cap_misaligned_input(namespace, dtype):
    """A contiguous view that starts mid-vector takes the scalar prefix path."""
    torch.manual_seed(0)
    flat = (torch.randn(2 * 8000 + 5, device="cuda") * 30).to(dtype)
    for offset in (1, 2, 3):
        x = flat[offset : offset + 2 * 8000].view(2, 8000)
        out = torch.empty_like(x, dtype=torch.float32)
        getattr(namespace, SOFT_CAP_OP)(out, x, 1.0, 20.0)
        torch.testing.assert_close(
            out, soft_cap_ref(x, 1.0, 20.0), rtol=1e-6, atol=1e-5
        )


def test_mul_tanh_soft_cap_rejects_bad_arguments(namespace):
    op = getattr(namespace, SOFT_CAP_OP)
    x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="float32"):
        op(torch.empty_like(x), x, 1.0, 20.0)
    with pytest.raises(RuntimeError, match="nonzero"):
        op(torch.empty_like(x, dtype=torch.float32), x, 1.0, 0.0)
    with pytest.raises(RuntimeError, match="contiguous"):
        op(torch.empty(2, 16, device="cuda"), x.t().contiguous().t(), 1.0, 20.0)


# torch.compile and fake tensors


def test_ops_trace_under_torch_compile(namespace):
    """The mutating ops are traceable: fake tensors, functionalization, replay."""
    soft_cap = getattr(namespace, SOFT_CAP_OP)
    fused = getattr(namespace, FUSED_OP)
    nq, nkv, block_size, tokens = 16, 2, 16, 8

    def fn(logits, qkv, positions, cos_sin, slots, cache, k_scale, v_scale):
        out = torch.empty_like(logits, dtype=torch.float32)
        soft_cap(out, logits, 1.0, 20.0)
        fused(
            qkv,
            positions,
            nq,
            nkv,
            HEAD_DIM,
            1e-5,
            cos_sin,
            HEAD_DIM,
            slots,
            cache,
            block_size,
            "fp8",
            k_scale,
            v_scale,
            None,
            None,
        )
        return out

    torch.manual_seed(0)
    logits = torch.randn(2, 4096, device="cuda", dtype=torch.bfloat16)
    qkv, _, _, _ = make_case(tokens, nq, nkv, torch.bfloat16, "cuda")
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cos_sin = make_cos_sin_cache(64, HEAD_DIM, 500_000.0, "cuda")
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    scales = (torch.tensor(1.0, device="cuda"), torch.tensor(1.0, device="cuda"))

    def run(f):
        qkv_run = qkv.clone()
        cache = torch.zeros(
            1, nkv, block_size, 2 * HEAD_DIM, dtype=torch.uint8, device="cuda"
        )
        out = f(logits, qkv_run, positions, cos_sin, slots, cache, *scales)
        return out, qkv_run, cache

    eager = run(fn)
    compiled = run(torch.compile(fn, backend="aot_eager", fullgraph=True))
    for got, expected in zip(compiled, eager):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


# Dispatch


@pytest.fixture
def fresh_resolution():
    resolve_compiled_op.cache_clear()
    yield
    resolve_compiled_op.cache_clear()


@pytest.mark.parametrize("op_name", [SOFT_CAP_OP, FUSED_OP])
def test_resolve_compiled_op_is_cached_and_can_be_disabled(
    fresh_resolution, monkeypatch, op_name
):
    first = resolve_compiled_op(op_name)
    assert resolve_compiled_op(op_name) is first
    if first is None:
        pytest.skip("no fused kernel on this machine")

    resolve_compiled_op.cache_clear()
    monkeypatch.setenv("VLLM_MUSE_SPARK_FUSED_KERNELS", "0")
    assert resolve_compiled_op(op_name) is None


def test_mul_tanh_soft_cap_module_matches_the_torch_formula(
    fresh_resolution, monkeypatch
):
    from vllm.models.muse_spark.common.layers import MulTanhSoftCap

    fused = MulTanhSoftCap(1.0, 20.0)
    monkeypatch.setenv("VLLM_MUSE_SPARK_FUSED_KERNELS", "0")
    resolve_compiled_op.cache_clear()
    plain = MulTanhSoftCap(1.0, 20.0)
    assert plain._fused_op is None
    if fused._fused_op is None:
        pytest.skip("no fused kernel on this machine")

    torch.manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        x = (torch.randn(3, 5000, device="cuda") * 30).to(dtype)
        torch.testing.assert_close(fused(x), plain(x), rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize("op_name", [SOFT_CAP_OP, FUSED_OP])
def test_resolve_compiled_op_prefers_the_jit_variant_on_sm120(
    fresh_resolution, monkeypatch, op_name
):
    """SM120 never routes to the upstream torch.ops._C kernel of the same name."""
    import vllm.models.muse_spark.common.ops as muse_ops

    def jit_variant(*args):
        raise AssertionError("must not be called")

    monkeypatch.setattr(muse_ops, "_is_sm120", lambda: True)
    monkeypatch.setattr(muse_ops, "has_compiled_op", lambda name: True)
    monkeypatch.setattr(muse_ops, "_jit_op", lambda name: jit_variant)
    assert resolve_compiled_op(op_name) is jit_variant

    # The JIT extension being unavailable means the unfused fallback, not _C.
    resolve_compiled_op.cache_clear()
    monkeypatch.setattr(muse_ops, "_jit_op", lambda name: None)
    assert resolve_compiled_op(op_name) is None

    # Other devices without the compiled op take the fallback.
    resolve_compiled_op.cache_clear()
    monkeypatch.setattr(muse_ops, "_is_sm120", lambda: False)
    monkeypatch.setattr(muse_ops, "has_compiled_op", lambda name: False)
    assert resolve_compiled_op(op_name) is None
