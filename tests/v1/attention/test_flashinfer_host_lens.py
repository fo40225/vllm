# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer metadata builder: how the host gets the sequence lengths.

VLLM_MUSE_SPARK_SM120_FI_HOST_LENS (level 0 = as before, 1 = fetch the exact lengths
only for rows that read them on the host, 2 = host upper bound when only prefill rows
read them). The decision is `host_seq_lens_plan`; the glue in `build()` is exercised on
a stub builder (no kernel runs) that counts the `.cpu()` calls on `seq_lens`.
"""

import itertools
import types
from unittest import mock

import pytest
import torch

pytest.importorskip("flashinfer")

from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402
from vllm.v1.attention.backends import flashinfer as fi  # noqa: E402
from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    FlashInferDecodeKernel,
    FlashInferMetadataBuilder,
    host_bound_supported,
    host_seq_lens_plan,
    prefill_rows_all_long,
)


def _old_rule(use_dcp, all_uses_trtllm, num_spec, has_ub):
    """The decision before the flag (needs, use_upper_bound)."""
    needs = use_dcp or not all_uses_trtllm
    return needs, needs and num_spec == 0 and has_ub


def _all_uses_trtllm(causal, num_prefills, num_decodes, prefill_trtllm, decode_trtllm):
    return causal and (
        (num_prefills == 0 or prefill_trtllm) and (num_decodes == 0 or decode_trtllm)
    )


GRID = list(
    itertools.product(
        (False, True),  # use_dcp
        (False, True),  # causal
        (0, 3),  # num_prefills
        (0, 5),  # num_decodes
        (False, True),  # prefill_use_trtllm
        (False, True),  # decode_with_flashinfer_trtllm_api
        (0, 7),  # num_speculative_tokens
        (False, True),  # has_upper_bound
        (False, True),  # prefill_rows_exact
    )
)


def _plan(level, g):
    use_dcp, causal, npf, nd, ptrt, dtrt, nspec, has_ub, pre_exact = g
    return host_seq_lens_plan(
        level,
        use_dcp=use_dcp,
        all_uses_trtllm=_all_uses_trtllm(causal, npf, nd, ptrt, dtrt),
        num_prefills=npf,
        num_decodes=nd,
        prefill_use_trtllm=ptrt,
        decode_with_flashinfer_trtllm_api=dtrt,
        num_speculative_tokens=nspec,
        has_upper_bound=has_ub,
        prefill_rows_exact=pre_exact,
    )


@pytest.mark.parametrize("g", GRID)
def test_level_0_is_the_old_rule(g):
    use_dcp, causal, npf, nd, ptrt, dtrt, nspec, has_ub, pre_exact = g
    all_trtllm = _all_uses_trtllm(causal, npf, nd, ptrt, dtrt)
    assert _plan(0, g) == _old_rule(use_dcp, all_trtllm, nspec, has_ub)


@pytest.mark.parametrize("g", GRID)
def test_level_1_fetches_only_what_a_host_reader_needs(g):
    use_dcp, causal, npf, nd, ptrt, dtrt, nspec, has_ub, pre_exact = g
    needs, use_ub = _plan(1, g)
    host_readers = use_dcp or (npf > 0 and not ptrt) or (nd > 0 and not dtrt)
    assert needs == host_readers
    # The upper bound is used exactly when it was used before (no speculation).
    assert use_ub == (needs and nspec == 0 and has_ub)
    if causal:
        # No behaviour change for causal batches: same need as the old rule.
        old_needs, _ = _old_rule(
            use_dcp, _all_uses_trtllm(causal, npf, nd, ptrt, dtrt), nspec, has_ub
        )
        assert needs == old_needs
    else:
        assert not needs or _old_rule(use_dcp, False, nspec, has_ub)[0]


@pytest.mark.parametrize("g", GRID)
def test_level_2_upper_bound_only_when_decode_rows_do_not_read_host_lengths(g):
    use_dcp, causal, npf, nd, ptrt, dtrt, nspec, has_ub, pre_exact = g
    needs1, ub1 = _plan(1, g)
    needs2, ub2 = _plan(2, g)
    assert needs2 == needs1
    if nspec == 0:
        assert ub2 == ub1
        return
    host_decode_rows = nd > 0 and not dtrt
    assert ub2 == (
        needs2 and has_ub and not use_dcp and not host_decode_rows and pre_exact
    )
    # Never worse than level 1: an upper bound is only added, never taken away.
    assert (not ub1) or ub2


def test_named_scenarios():
    # Non-causal draft batch, XQA decode rows only: nothing reads the host lengths.
    draft = dict(
        use_dcp=False, all_uses_trtllm=False, num_prefills=0, num_decodes=8,
        prefill_use_trtllm=False, decode_with_flashinfer_trtllm_api=True,
        num_speculative_tokens=7, has_upper_bound=True, prefill_rows_exact=True,
    )  # fmt: skip
    assert host_seq_lens_plan(0, **draft) == (True, False)
    assert host_seq_lens_plan(1, **draft) == (False, False)
    assert host_seq_lens_plan(2, **draft) == (False, False)
    # Target prefill step (native prefill, no decode row): the upper bound of the
    # prefilling rows is exact.
    prefill = dict(draft, all_uses_trtllm=False, num_prefills=1, num_decodes=0)
    assert host_seq_lens_plan(0, **prefill) == (True, False)
    assert host_seq_lens_plan(1, **prefill) == (True, False)
    assert host_seq_lens_plan(2, **prefill) == (True, True)
    # Mixed step, decode rows on XQA: the decode rows' bound is only over-counted,
    # and they read the device lengths.
    mixed = dict(prefill, num_decodes=4)
    assert host_seq_lens_plan(2, **mixed) == (True, True)
    # Mixed step whose decode rows take the native FlashInfer decode: they need the
    # exact lengths, so the fetch stays.
    native_decode = dict(mixed, decode_with_flashinfer_trtllm_api=False)
    assert host_seq_lens_plan(2, **native_decode) == (True, False)
    # DCP keeps the fetch (the lengths are converted per rank).
    assert host_seq_lens_plan(2, **dict(prefill, use_dcp=True)) == (True, False)
    # No upper bound available: fetch.
    assert host_seq_lens_plan(2, **dict(prefill, has_upper_bound=False)) == (
        True,
        False,
    )
    # Non-causal native (non-XQA) draft: every row counts as a prefill row, although
    # they are decode rows with drafts (the bound over-counts them): fetch always.
    native_draft = dict(
        draft, num_prefills=8, num_decodes=0, decode_with_flashinfer_trtllm_api=False,
        prefill_rows_exact=False,
    )  # fmt: skip
    for level in (0, 1, 2):
        assert host_seq_lens_plan(level, **native_draft) == (True, False)
    # A split that requires uniform query lengths may call a decode row a prefill row.
    assert host_seq_lens_plan(2, **dict(prefill, prefill_rows_exact=False)) == (
        True,
        False,
    )


class _CountingLens(torch.Tensor):
    """CPU int32 tensor that counts `.cpu()` calls (the DtoH of the real run)."""

    calls = 0

    def cpu(self, *args, **kwargs):
        type(self).calls += 1
        return torch.Tensor.cpu(self.as_subclass(torch.Tensor), *args, **kwargs)


class _FakeFence:
    """Records the order of the fence calls and the plan() calls of a build."""

    def __init__(self, log):
        self.log = log

    def wait(self, wrapper):
        self.log.append(("wait", id(wrapper)))

    def record(self, wrapper):
        self.log.append(("record", id(wrapper)))


def _stub_builder(level, *, use_xqa, num_spec, bound_ok=True, fence_log=None):
    plan_calls = []
    kv_calls = []
    wrapper = mock.MagicMock(spec=fi.BatchPrefillWithPagedKVCacheWrapper)
    wrapper.plan.side_effect = lambda **kw: (
        plan_calls.append(kw),
        fence_log.append(("plan", id(wrapper))) if fence_log is not None else None,
    )
    stub = types.SimpleNamespace(
        _host_lens_level=level,
        _host_bound_supported=bound_ok,
        _plan_fence=_FakeFence(fence_log) if fence_log is not None else None,
        _num_speculative_tokens=num_spec,
        attention_config=types.SimpleNamespace(use_trtllm_attention=None),
        model_config=types.SimpleNamespace(dtype=torch.bfloat16),
        cache_dtype="auto",
        kv_cache_dtype=torch.bfloat16,
        dcp_kv_cache_interleave_size=1,
        dcp_rank=0,
        dcp_world_size=1,
        use_dcp=False,
        decode_fixed_split_size=-1,
        prefill_fixed_split_size=-1,
        disable_split_kv=False,
        enable_cuda_graph=False,
        _decode_cudagraph_max_bs=0,
        flashinfer_trtllm_api_decode_kernel=(
            FlashInferDecodeKernel.XQA if use_xqa else None
        ),
        global_hyperparameters=types.SimpleNamespace(
            has_same_all_params=True, has_same_window_lefts=True
        ),
        has_sinks=False,
        head_dim=64,
        kv_cache_layout="NHD",
        logits_soft_cap=None,
        num_kv_heads=2,
        num_qo_heads=16,
        nvfp4_trtllm=False,
        page_size=16,
        paged_kv_indptr=types.SimpleNamespace(cpu=torch.zeros(64, dtype=torch.int32)),
        paged_kv_last_page_len=types.SimpleNamespace(
            cpu=torch.zeros(64, dtype=torch.int32)
        ),
        q_data_type_decode=torch.bfloat16,
        q_data_type_prefill=torch.bfloat16,
        reorder_batch_threshold=8,
        sm_scale=0.125,
        use_trtllm_decode_attention=use_xqa,
        use_trtllm_gen_varlen_decode=False,
        use_xqa=use_xqa,
        window_left=-1,
    )
    stub._get_prefill_wrapper = lambda causal=True: wrapper
    stub._get_decode_mask = lambda *a, **k: None
    for name in (
        "_compute_decode_query_lens",
        "_prefill_rows_exact",
        "_wait_plan",
        "_mark_plan",
    ):
        setattr(
            stub, name, types.MethodType(getattr(FlashInferMetadataBuilder, name), stub)
        )

    def kv_meta(num_blocks_np, seq_lens_np, block_table_tensor, num_reqs, page_size):
        kv_calls.append((num_blocks_np.copy(), seq_lens_np.copy(), num_reqs))
        indptr = stub.paged_kv_indptr.cpu
        indptr[0] = 0
        indptr[1 : num_reqs + 1] = torch.from_numpy(num_blocks_np).cumsum(0)
        last = stub.paged_kv_last_page_len.cpu
        last[:num_reqs] = torch.from_numpy((seq_lens_np - 1) % page_size + 1)
        return torch.zeros(1, dtype=torch.int32)

    stub._compute_flashinfer_kv_metadata = kv_meta
    return stub, plan_calls, kv_calls


def _common(query_lens, seq_lens, *, causal):
    n = len(query_lens)
    qsl = torch.zeros(n + 1, dtype=torch.int32)
    qsl[1:] = torch.tensor(query_lens, dtype=torch.int32).cumsum(0)
    lens = torch.tensor(seq_lens, dtype=torch.int32).as_subclass(_CountingLens)
    return CommonAttentionMetadata(
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        seq_lens=lens,
        seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=n,
        num_actual_tokens=int(sum(query_lens)),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros(n, 64, dtype=torch.int32),
        slot_mapping=torch.zeros(int(sum(query_lens)), dtype=torch.int64),
        causal=causal,
    )


def _build(stub, cam):
    _CountingLens.calls = 0
    # SM120: the prefill never takes the trtllm-gen path (the platform query is mocked
    # so the test needs no device).
    with (
        mock.patch.object(fi, "PIN_MEMORY", False),
        mock.patch.object(fi, "use_trtllm_attention", return_value=False),
    ):
        md = FlashInferMetadataBuilder.build(stub, 0, cam)
    return md, _CountingLens.calls


@pytest.mark.parametrize("level", [0, 1, 2])
def test_draft_decode_build_fetches_only_at_level_0(level):
    """The non-causal draft: XQA decode rows read the device lengths."""
    stub, _, _ = _stub_builder(level, use_xqa=True, num_spec=7)
    cam = _common([8] * 4, [100, 200, 300, 400], causal=False)
    md, cpu_calls = _build(stub, cam)
    assert cpu_calls == (1 if level == 0 else 0)
    assert md.num_decodes == 4 and md.num_prefills == 0
    # The metadata does not depend on the level.
    ref_stub, _, _ = _stub_builder(0, use_xqa=True, num_spec=7)
    ref, _ = _build(ref_stub, _common([8] * 4, [100, 200, 300, 400], causal=False))
    assert md.decode.q_len_per_req == ref.decode.q_len_per_req == 8
    assert torch.equal(md.decode.seq_lens, ref.decode.seq_lens)
    assert torch.equal(md.decode.block_tables, ref.decode.block_tables)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_target_prefill_build_uses_the_upper_bound_at_level_2(level):
    """A prefill chunk with the drafter: native prefill plan, no decode row."""
    q, s = [4096, 4096], [8192 + 4096, 3000 + 4096]
    stub, plans, kvs = _stub_builder(level, use_xqa=True, num_spec=7)
    md, cpu_calls = _build(stub, _common(q, s, causal=True))
    ref_stub, ref_plans, ref_kvs = _stub_builder(0, use_xqa=True, num_spec=7)
    _build(ref_stub, _common(q, s, causal=True))
    assert cpu_calls == (0 if level == 2 else 1)
    assert md.num_prefills == 2 and md.num_decodes == 0
    # The plan inputs are identical for every level (the upper bound is exact here).
    assert len(plans) == len(ref_plans) == 1
    for key in ("qo_indptr", "paged_kv_indptr", "paged_kv_last_page_len", "seq_lens"):
        assert torch.equal(plans[0][key], ref_plans[0][key]), key
    assert (kvs[0][0] == ref_kvs[0][0]).all() and (kvs[0][1] == ref_kvs[0][1]).all()


@pytest.mark.parametrize("level", [0, 2])
def test_mixed_step_with_xqa_decode_rows(level):
    """Decode rows (XQA, drafts in flight: the host bound over-counts them) in front
    of a prefilling row: the prefill row's lengths are exact at every level."""
    q, s = [8, 8, 4096], [500, 900, 5000 + 4096]
    cam = _common(q, s, causal=True)
    # The host bound over-counts the two decode rows (rejected drafts of the last step).
    cam.seq_lens_cpu_upper_bound = torch.tensor(
        [507, 907, 5000 + 4096], dtype=torch.int32
    )
    stub, plans, kvs = _stub_builder(level, use_xqa=True, num_spec=7)
    md, cpu_calls = _build(stub, cam)
    ref_stub, ref_plans, ref_kvs = _stub_builder(0, use_xqa=True, num_spec=7)
    _build(ref_stub, _common(q, s, causal=True))
    assert cpu_calls == (0 if level == 2 else 1)
    assert md.num_decodes == 2 and md.num_prefills == 1
    # Only the prefill row is planned; its lengths and page count equal the exact ones
    # (its page-index offset shifts with the decode rows' bound, on both sides).
    assert kvs[0][0][2] == ref_kvs[0][0][2]
    assert plans[0]["seq_lens"].tolist() == ref_plans[0]["seq_lens"].tolist() == [9096]
    assert torch.equal(
        plans[0]["paged_kv_last_page_len"], ref_plans[0]["paged_kv_last_page_len"]
    )
    # The decode rows read the device lengths (identical object content).
    assert torch.equal(md.decode.seq_lens, torch.tensor(s[:2], dtype=torch.int32))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_native_noncausal_draft_keeps_the_fetch(level):
    """A non-causal draft whose decode rows run the native wrapper (no XQA): every row
    counts as a prefill row, but they are decode rows with drafts, and the host bound
    over-counts them. The exact lengths are fetched at every level."""
    q, s = [8, 8, 8], [100, 200, 300]
    cam = _common(q, s, causal=False)
    cam.seq_lens_cpu_upper_bound = torch.tensor([107, 207, 307], dtype=torch.int32)
    stub, plans, _ = _stub_builder(level, use_xqa=False, num_spec=7)
    md, cpu_calls = _build(stub, cam)
    assert cpu_calls == 1
    assert md.num_prefills == 3 and md.num_decodes == 0
    assert plans[0]["seq_lens"].tolist() == s


@pytest.mark.parametrize("level", [0, 1, 2])
def test_short_row_behind_a_long_row_keeps_the_fetch(level):
    """A decode row (drafts, over-counted bound) behind a long first row is counted
    as a prefill row by the split; the host bound must not be used for it."""
    q, s = [4096, 8, 4096], [4096, 600, 3000 + 4096]
    cam = _common(q, s, causal=True)
    cam.seq_lens_cpu_upper_bound = torch.tensor([4096, 607, 7096], dtype=torch.int32)
    stub, plans, _ = _stub_builder(level, use_xqa=True, num_spec=7)
    md, cpu_calls = _build(stub, cam)
    assert md.num_decodes == 0 and md.num_prefills == 3
    assert cpu_calls == 1
    assert plans[0]["seq_lens"].tolist() == s  # the exact lengths, not the bound


@pytest.mark.parametrize("level", [0, 1, 2])
def test_unsupported_launch_keeps_the_fetch(level):
    """Adaptive verification / prefill context parallelism: no host bound."""
    stub, _, _ = _stub_builder(level, use_xqa=True, num_spec=7, bound_ok=False)
    _, cpu_calls = _build(stub, _common([4096, 4096], [8192, 7096], causal=True))
    assert cpu_calls == 1


def test_prefill_rows_all_long():
    qsl = torch.tensor([0, 8, 16, 4112, 8208], dtype=torch.int32)
    assert prefill_rows_all_long(qsl, 2, 4, 8)
    assert not prefill_rows_all_long(qsl, 0, 4, 8)  # rows 0 and 1 are short
    assert not prefill_rows_all_long(qsl, 1, 4, 8)
    assert prefill_rows_all_long(qsl, 2, 4, 4095)
    assert not prefill_rows_all_long(qsl, 2, 4, 4096)  # a row of exactly 4096


def _config(adaptive=None, pcp=1):
    spec = (
        None
        if adaptive is None
        else types.SimpleNamespace(enable_adaptive_verification=adaptive)
    )
    return types.SimpleNamespace(
        speculative_config=spec,
        parallel_config=types.SimpleNamespace(prefill_context_parallel_size=pcp),
    )


def test_host_bound_supported():
    assert host_bound_supported(_config())  # no speculation
    assert host_bound_supported(_config(adaptive=False))
    assert not host_bound_supported(_config(adaptive=True))
    assert not host_bound_supported(_config(adaptive=False, pcp=2))
    assert not host_bound_supported(_config(pcp=4))


@pytest.mark.parametrize("level", [0, 1, 2])
def test_plan_is_fenced_on_the_wrapper(level):
    """Wait before and record after every plan() of a build (levels 1 and 2 only)."""
    log = []
    stub, plans, _ = _stub_builder(
        level, use_xqa=True, num_spec=7, fence_log=log if level else None
    )
    _build(stub, _common([4096, 4096], [8192, 7096], causal=True))
    assert len(plans) == 1
    if level == 0:
        assert stub._plan_fence is None
        return
    kinds = [k for k, _ in log]
    assert kinds == ["wait", "plan", "record"]
    assert len({w for _, w in log}) == 1  # one wrapper
