# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""min_p and top_p rows verified from the top-K logits of every vocabulary shard.

The reference is the full path: the gathered logits through the soft cap,
`apply_temperature`, `apply_min_p` and `apply_top_k_top_p` (the Qrita kernel: top-k,
then top-p over the top-k set), then `rejection_sample`.

* min_p is a threshold on the temperature-scaled logits relative to the row maximum:
  the candidate kernel takes the same threshold with the same ``tl.log`` and matches the
  full path bit for bit (the ladder test puts logits on both sides of the threshold, ulp
  by ulp).
* top_p over the top-k set is the nucleus of the probabilities of the kept tokens. The
  full path computes it with an iterative pivot search in fp32 whose result for tokens
  that tie at the nucleus boundary depends on how an exp / log round trip rounds, and
  that round trip depends on the exact bits of a sum whose order depends on the whole
  row (the kernel's own outlier buffer). The candidate kernel reproduces the pivot rule
  (minimal prefix of the sorted tokens, the lowest ids of a tie group that straddles the
  boundary, the round-trip test of the duplicate rule) with the same ``exp`` / ``log``
  on the candidates' own sum. The tests below use rows whose logits are large (the
  round trip is then exact: the capped logits of Muse Spark in a decode step are up to
  10 and more), which hold with no differing token over the pinned inputs, and the
  test of the boundary ties constructs the tie groups of every size and position.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k verify tests", allow_module_level=True)

from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.spec_decode.topk_rejection import _target_row, topk_verify

from .test_gpu_topk_verify import (
    DEVICE,
    EXACT_IDS,
    EXACT_K,
    EXACT_STEPS,
    EXACT_TEMPERATURE,
    EXACT_TOP_K,
    EXACT_VOCAB,
    MULTIPLIER,
    SOFT_CAP,
    TP,
    VOCAB,
    Batch,
    K,
    as_lists,
    block_verification_law,
    chi_square,
    compare,
    exact_case,
    gather_candidates,
    soft_cap,
)

# --------------------------------------------------------------------------------
# The kept set of the candidate kernel (the production ``_target_row``) against the full
# path's, row by row.
# --------------------------------------------------------------------------------


@triton.jit
def _kept_kernel(
    keep_ptr,
    ids_ptr,
    words_ptr,
    capped_ptr,
    temp_ptr,
    top_k_ptr,
    min_p_ptr,
    top_p_ptr,
    num_rows,
    shard_size,
    TP: tl.constexpr,
    K: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    HAS_MIN_P: tl.constexpr,
    HAS_TOP_P: tl.constexpr,
):
    row = tl.program_id(0)
    temp = tl.load(temp_ptr + row)
    top_k = tl.minimum(tl.load(top_k_ptr + row), TP * K)
    min_p = 0.0
    if HAS_MIN_P:
        min_p = tl.load(min_p_ptr + row).to(tl.float32)
    top_p = 1.0
    if HAS_TOP_P:
        top_p = tl.load(top_p_ptr + row).to(tl.float32)
    j = tl.arange(0, NUM_CANDIDATES)
    _, token_id, keep = _target_row(
        words_ptr,
        capped_ptr,
        j,
        row,
        num_rows,
        shard_size,
        temp,
        top_k,
        min_p,
        top_p,
        TP,
        K,
        ROW_WORDS,
        HAS_MIN_P,
        HAS_TOP_P,
    )
    tl.store(keep_ptr + row * NUM_CANDIDATES + j, keep.to(tl.int8))
    tl.store(ids_ptr + row * NUM_CANDIDATES + j, token_id)


def candidate_kept(raw, temp, top_k, min_p=None, top_p=None, k=K) -> torch.Tensor:
    """The tokens the candidate kernel keeps, as a bool mask over the vocabulary."""
    rows, vocab = raw.shape
    topk = gather_candidates(raw, k=k)
    nc = triton.next_power_of_2(TP * k)
    keep = torch.empty(rows, nc, dtype=torch.int8, device=DEVICE)
    ids = torch.empty(rows, nc, dtype=torch.int32, device=DEVICE)
    _kept_kernel[(rows,)](
        keep,
        ids,
        topk.words.view(torch.int16),
        topk.capped,
        temp,
        top_k,
        top_k if min_p is None else min_p,
        top_k if top_p is None else top_p,
        rows,
        topk.shard_size,
        TP=TP,
        K=k,
        ROW_WORDS=2 * k,
        NUM_CANDIDATES=nc,
        HAS_MIN_P=min_p is not None,
        HAS_TOP_P=top_p is not None,
        num_warps=4,
    )
    mask = torch.zeros(rows, vocab + 1, dtype=torch.bool, device=DEVICE)
    where = torch.where(keep != 0, ids.long(), torch.full_like(ids.long(), vocab))
    mask.scatter_(1, where, True)
    return mask[:, :-1]


def full_path_kept(raw, temp, top_k, min_p=None, top_p=None) -> torch.Tensor:
    """What the full path keeps of the same rows (soft cap, temperature, min_p, top-k +
    top-p), as a bool mask over the vocabulary."""
    rows = raw.shape[0]
    logits = soft_cap(raw).contiguous()
    rows_idx = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    apply_temperature(logits, rows_idx, temp)
    if min_p is not None:
        apply_min_p(logits, rows_idx, min_p)
    return torch.isfinite(apply_top_k_top_p(logits, top_k.to(torch.int32), top_p))


def per_row(value, rows, dtype=torch.float32):
    return torch.full((rows,), value, dtype=dtype, device=DEVICE)


def head_rows(kind: str, rows: int, seed: int, offset: float = 50.0) -> torch.Tensor:
    """bf16 raw logits of rows that look like a decode step of a large model: a head of
    tokens with large logits (ties are frequent: bf16 has 8 bits) over a noisy floor.
    The capped logits of the head are 8 .. 13 at offset 50 .. 80."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if kind == "gauss":
        x = torch.randn(rows, VOCAB, generator=g, device=DEVICE) * 3.0 - 3.0
        ids = torch.randint(0, VOCAB, (rows, 160), generator=g, device=DEVICE)
        head = offset + 3.0 * torch.randn(rows, 160, generator=g, device=DEVICE)
        x.scatter_(1, ids, head)
    elif kind == "flat":
        x = torch.randn(rows, VOCAB, generator=g, device=DEVICE) * 0.5
        ids = torch.randint(0, VOCAB, (rows, 96), generator=g, device=DEVICE)
        x.scatter_(
            1, ids, offset + 0.4 * torch.randn(rows, 96, generator=g, device=DEVICE)
        )
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


def straddling_rows(raw, temp, top_p) -> torch.Tensor:
    """Rows whose nucleus boundary falls inside a group of tied tokens, from the top 64
    (an independent float64 computation)."""
    capped = soft_cap(raw)
    if temp != 1.0:
        capped = capped / temp
    top = capped.topk(K, dim=-1).values.double()
    prob = torch.exp(top - top[:, :1])
    prob = prob / prob.sum(-1, keepdim=True)
    before = prob.cumsum(-1) - prob
    n = (before < top_p).sum(-1).clamp(min=1)
    last = top.gather(1, (n - 1)[:, None])[:, 0]
    nxt = top.gather(1, n.clamp(max=K - 1)[:, None])[:, 0]
    return (n < K) & (last == nxt)


# --------------------------------------------------------------------------------
# min_p.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("min_p", [0.02, 0.1, 0.5, 0.9])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
def test_min_p_kept_set_matches_the_full_path(temperature: float, min_p: float):
    rows = 512
    for seed, kind in ((1, "gauss"), (2, "flat")):
        raw = head_rows(kind, rows, seed, offset=40.0)
        temp, top_k = per_row(temperature, rows), per_row(K, rows, torch.int32)
        mp = per_row(min_p, rows)
        expected = full_path_kept(raw, temp, top_k, mp)
        got = candidate_kept(raw, temp, top_k, mp)
        assert bool((expected == got).all()), int((expected != got).any(-1).sum())
        # The threshold does cut: the rows do not all keep their whole top-k.
        if kind == "gauss" and min_p >= 0.5:
            assert float(expected.sum(-1).float().mean()) < 0.8 * K


def test_min_p_threshold_ladder_matches_ulp_by_ulp():
    """Logits on both sides of the threshold, down to the last bit. A candidate whose
    scaled logit is exactly the threshold stays (the full path drops ``logits < thr``);
    one ulp below it goes. The min_p values are chosen so that the threshold lands
    within a few ulps of a token of the ladder: the two kernels must agree on every
    one, so their ``max + log(min_p)`` have the same bits."""
    ladder = torch.arange(1, 41, device=DEVICE) * 0.5
    raw_row = torch.full((VOCAB,), -20.0, device=DEVICE)
    raw_row[0] = 60.0
    raw_row[1:41] = 60.0 - ladder
    raw_row = raw_row.to(torch.bfloat16)
    capped = soft_cap(raw_row.unsqueeze(0)).squeeze(0)
    top = capped[0].item()
    base = torch.exp(capped[1:41] - top)  # min_p that puts the threshold at token i
    steps = torch.arange(-40, 41, device=DEVICE)
    min_p = (base[:, None] * (1.0 + steps[None, :] * 2.0**-23)).reshape(-1).float()
    rows = min_p.shape[0]
    raw = raw_row.unsqueeze(0).repeat(rows, 1)
    temp, top_k = per_row(1.0, rows), per_row(K, rows, torch.int32)
    expected_all, got_all = [], []
    for i in range(0, rows, 512):
        sl = slice(i, i + 512)
        expected_all.append(full_path_kept(raw[sl], temp[sl], top_k[sl], min_p[sl]))
        got_all.append(candidate_kept(raw[sl], temp[sl], top_k[sl], min_p[sl]))
    expected, got = torch.cat(expected_all), torch.cat(got_all)
    assert bool((expected == got).all()), int((expected != got).any(-1).sum())
    # Non-vacuous: along the sweep of one token the kept count steps by one.
    counts = expected.sum(-1).view(40, 81)
    assert int((counts[:, 0] != counts[:, -1]).sum()) >= 20


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
@pytest.mark.parametrize("num_steps", [1, 3, 7])
def test_min_p_verification_matches_full_path(temperature: float, num_steps: int):
    n = 160
    rng = np.random.default_rng(num_steps)
    batch = Batch(
        rng.integers(0, num_steps + 1, n).tolist(),
        [temperature] * n,
        rng.choice([5, 17, 64], n).tolist(),
        num_steps,
        seed=num_steps + 40,
        min_p=rng.choice([0.0, 0.02, 0.1, 0.5], n).tolist(),
    )
    expected, got, bad = compare(batch)
    assert not bad, [(expected[r], got[r]) for r in bad]


# --------------------------------------------------------------------------------
# top_p.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("top_p", [0.5, 0.8, 0.9, 0.95])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("kind", ["gauss", "flat"])
def test_top_p_kept_set_matches_the_full_path(
    kind: str, temperature: float, top_p: float
):
    """Rows with large logits: the same kept set as Qrita on every row, the rows whose
    nucleus boundary splits a group of tied tokens included (they are most of them)."""
    total = straddled = 0
    for seed in range(2):
        rows = 1024
        raw = head_rows(kind, rows, 100 + seed, offset=60.0)
        temp, top_k = per_row(temperature, rows), per_row(K, rows, torch.int32)
        tp = per_row(top_p, rows)
        expected = full_path_kept(raw, temp, top_k, None, tp)
        got = candidate_kept(raw, temp, top_k, None, tp)
        assert bool((expected == got).all()), int((expected != got).any(-1).sum())
        total += rows
        straddled += int(straddling_rows(raw, temperature, top_p).sum())
    # The test exercises the tie rule: a good number of rows split a tie group.
    assert straddled >= 20, (straddled, total)


def test_top_p_on_a_tie_group_that_splits_the_nucleus():
    """Constructed tie groups: a head of distinct tokens, then a group of ``size``
    tokens with one logit whose probability mass crosses top_p. Qrita keeps the lowest
    ids of the group that the mass needs; so does the kernel. The group sits inside one
    shard, at the shard edges, and across shards."""
    shard = VOCAB // TP
    rows = []
    meta = []
    for size in (2, 3, 6, 11):
        for placement in range(4):
            for need in (1, size // 2 + 1, size - 1):
                if need < 1 or need >= size:
                    continue
                row = torch.full((VOCAB,), -30.0, device=DEVICE)
                head_logits = torch.tensor([62.0, 59.0, 57.0], device=DEVICE)
                row[torch.tensor([10, 20, 30], device=DEVICE)] = head_logits
                if placement == 0:
                    ids = torch.arange(100, 100 + size)
                elif placement == 1:
                    ids = shard - 2 + torch.arange(size)
                elif placement == 2:
                    ids = torch.arange(size) * (VOCAB // size) + 5
                else:
                    ids = VOCAB - 1 - torch.arange(size) * 3
                row[ids.to(DEVICE)] = 50.0
                rows.append(row.to(torch.bfloat16))
                # top_p: the head plus ``need`` members of the group are needed.
                cap = soft_cap(row.to(torch.bfloat16).unsqueeze(0))[0]
                e = torch.exp(cap.topk(K).values.double() - cap.max().double())
                prob = e / e.sum()
                cum = prob.cumsum(0)
                meta.append(float(cum[2 + need] - 0.4 * prob[3]))
    raw = torch.stack(rows)
    n = raw.shape[0]
    temp, top_k = per_row(1.0, n), per_row(K, n, torch.int32)
    top_p = torch.tensor(meta, dtype=torch.float32, device=DEVICE)
    expected = full_path_kept(raw, temp, top_k, None, top_p)
    got = candidate_kept(raw, temp, top_k, None, top_p)
    assert bool((expected == got).all()), torch.nonzero(
        (expected != got).any(-1)
    ).tolist()
    # The group is cut: some rows keep a part of it.
    kept = expected.sum(-1)
    assert int(((kept > 3) & (kept < 3 + 11)).sum()) > 0


@pytest.mark.parametrize("top_p", [0.5, 0.9, 0.95])
@pytest.mark.parametrize("temperature", [0.7, 1.0])
@pytest.mark.parametrize("num_steps", [1, 3, 7])
def test_top_p_verification_matches_full_path(
    temperature: float, num_steps: int, top_p: float
):
    n = 160
    rng = np.random.default_rng(num_steps + 5)
    logits = head_rows(
        "gauss", int(n * (num_steps + 1)), seed=70 + num_steps, offset=60.0
    )
    num_draft = rng.integers(0, num_steps + 1, n).tolist()
    rows = sum(d + 1 for d in num_draft)
    batch = Batch(
        num_draft,
        [temperature] * n,
        rng.choice([5, 17, 64], n).tolist(),
        num_steps,
        seed=num_steps + 50,
        logits=logits[:rows],
        peaky=False,
        top_p=rng.choice([1.0, top_p], n).tolist(),
    )
    expected, got, bad = compare(batch)
    assert not bad, [(expected[r], got[r]) for r in bad]
    # The nucleus matters: without it the draws differ (on many requests for a small
    # top_p, on a few for 0.95: the nucleus only cuts the far tail).
    batch.use_top_p = False
    ignored = as_lists(*batch.candidate_path())
    differ = sum(1 for a, b in zip(expected, ignored) if a != b)
    assert differ >= (10 if top_p <= 0.5 else 1), differ


def test_mixed_batch_of_sampling_parameters_matches_full_path():
    """Greedy, plain, min_p and top_p requests in one batch: a request without min_p or
    top_p is not touched by the kernel variant that serves its neighbours."""
    num_steps = 7
    n = 256
    rng = np.random.default_rng(8)
    kind = rng.integers(0, 4, n)  # 0 greedy, 1 plain, 2 min_p, 3 top_p
    temperature = [0.0 if k == 0 else float(rng.choice([0.7, 1.0])) for k in kind]
    logits = head_rows("gauss", n * 8, seed=9, offset=60.0)
    num_draft = rng.choice([0, 7, 7, 7, 3, 1], n).tolist()
    rows = sum(d + 1 for d in num_draft)
    batch = Batch(
        num_draft,
        temperature,
        [
            VOCAB if (k == 0 and i % 2) else int(rng.choice([17, 64]))
            for i, k in enumerate(kind)
        ],
        num_steps,
        seed=11,
        logits=logits[:rows],
        peaky=False,
        accept_prefix=rng.integers(0, 8, n).tolist(),
        placeholders=0.05,
        min_p=[float(rng.choice([0.05, 0.3])) if k == 2 else 0.0 for k in kind],
        top_p=[float(rng.choice([0.8, 0.95])) if k == 3 else 1.0 for k in kind],
    )
    assert batch.use_min_p and batch.use_top_p
    expected, got, bad = compare(batch)
    assert not bad, [(expected[r], got[r]) for r in bad]


# --------------------------------------------------------------------------------
# The law of a top-p verification: exact against the analytic nucleus distribution.
# --------------------------------------------------------------------------------


def first_token_counts(top_p, use_top_p: bool, n: int = 24000):
    num_steps = 1
    vocab = 4096
    g = torch.Generator(device=DEVICE).manual_seed(31)
    # One target row with a flat head: 20 tokens of logits within 1.5 of each other.
    row = torch.randn(2, vocab, generator=g, device=DEVICE) * 0.5 - 10.0
    head = torch.arange(20, device=DEVICE) * 150
    row[:, head] = 50.0 + 1.5 * torch.rand(2, 20, generator=g, device=DEVICE)
    logits = row.to(torch.bfloat16).repeat(n, 1)
    top_k = 16
    batch = Batch(
        [1] * n,
        [1.0] * n,
        [top_k] * n,
        num_steps,
        seed=32,
        logits=logits,
        peaky=False,
        outside=0.0,
        draft_spread=0.8,
        vocab=vocab,
        top_p=[top_p] * n,
    )
    batch.use_top_p = use_top_p
    sampled, _ = batch.candidate_path()
    first = sampled[:, 0].cpu()
    capped = soft_cap(batch.logits[0].float())
    top = capped.topk(top_k)
    values = top.values.cpu().double().numpy()
    ids = top.indices.cpu().numpy()
    probs = np.exp(values - values.max())
    probs /= probs.sum()
    # The nucleus: the tokens by logit, ties by id, while the mass before them is < p.
    order = np.lexsort((ids, -values))
    before = np.cumsum(probs[order]) - probs[order]
    n_nuc = int((before < top_p).sum())
    law = np.zeros_like(probs)
    law[order[:n_nuc]] = probs[order[:n_nuc]]
    law /= law.sum()
    counts = torch.stack([(first == int(t)).sum() for t in ids]).double().numpy()
    return counts, law * n, int(counts.sum()), n, n_nuc


@pytest.mark.parametrize("top_p", [0.6, 0.9])
def test_first_token_has_the_nucleus_distribution(top_p: float):
    """A chi-square test of the first token of a one-step verification against the
    nucleus of the target (renormalised over the kept tokens), and the power check: the
    kernel without the nucleus (top-k only) does not pass it."""
    counts, expected, total, n, n_nuc = first_token_counts(top_p, True)
    assert 3 <= n_nuc < 16, n_nuc
    assert total == n, "a token outside the top-k set was drawn"
    live = expected > 0
    assert int((counts[~live] > 0).sum()) == 0, "a token outside the nucleus was drawn"
    chi2 = float(((counts[live] - expected[live]) ** 2 / expected[live]).sum())
    dof = int(live.sum()) - 1
    # The 99.9th percentile of chi-square with up to 15 degrees of freedom is 37.7.
    assert chi2 < 37.7 and dof <= 15, (chi2, dof)
    counts0, *_ = first_token_counts(top_p, False)
    assert int((counts0[~live] > 0).sum()) > 0, "top-k only draws outside the nucleus"


# --------------------------------------------------------------------------------
# The exact law of a three-step verification with a min_p / top_p target.
# --------------------------------------------------------------------------------

NUCLEUS_P = 0.8
MIN_P_VALUE = 0.3
# A seed of ``exact_case`` whose rows have six distinct logits (no tie rule involved)
# and on which both filters cut the top-3 of several rows.
EXACT_FILTER_SEED = 28


def filtered_exact_case(kind: str):
    """``exact_case`` with the sampling parameter applied to the target rows: the top-3
    of each row at temperature 0.7, then the nucleus (top_p 0.8) or the min_p threshold
    (0.3) of the *scaled logits*, renormalised. The logits of a row are distinct, so
    the kept set has no tie rule to fall back on."""
    z, P, score, Q = exact_case(EXACT_FILTER_SEED)
    capped = SOFT_CAP * np.tanh(z * MULTIPLIER / SOFT_CAP) / EXACT_TEMPERATURE
    out = np.zeros_like(P)
    for i in range(P.shape[0]):
        assert len(set(z[i].tolist())) == 6, "tied logits in the exact case"
        order = np.argsort(-capped[i], kind="stable")[:EXACT_TOP_K]
        keep = order
        if kind == "top_p":
            probs = P[i][order]
            before = np.cumsum(probs) - probs
            keep = order[before < NUCLEUS_P]
        elif kind == "min_p":
            threshold = capped[i].max() + np.log(MIN_P_VALUE)
            keep = order[capped[i][order] >= threshold]
        e = np.exp(capped[i][keep] - capped[i][keep].max())
        out[i][keep] = e / e.sum()
    return z, out, score, Q


def run_exact_law(P_kept: np.ndarray, z, score, Q, kind: str, n: int):
    """The outcomes of ``n`` verifications by the kernel with the sampling parameter."""
    rows_per = EXACT_STEPS + 1
    logits = torch.full((rows_per, EXACT_VOCAB), -20.0, device=DEVICE)
    logits[:, EXACT_IDS] = torch.tensor(z, dtype=torch.float32, device=DEVICE)
    base = gather_candidates(logits.to(torch.bfloat16), tp=TP, k=EXACT_K)
    width = base.words.shape[-1]
    words = (
        base.words.view(TP, rows_per, width).repeat(1, n, 1).reshape(-1, width)
    ).contiguous()
    topk = type(base)(
        words=words,
        capped=soft_cap(words),
        num_rows=rows_per * n,
        k=EXACT_K,
        tp_size=TP,
        shard_size=EXACT_VOCAB // TP,
    )
    rng = np.random.default_rng(2024)
    ids = np.asarray(EXACT_IDS)
    draws = np.stack(
        [rng.choice(6, size=n, p=Q[i]) for i in range(EXACT_STEPS)], axis=1
    )
    draft_sampled = np.zeros((n, rows_per), dtype=np.int64)
    draft_sampled[:, 1:] = ids[draws]
    draft_logits = torch.full((EXACT_STEPS, EXACT_VOCAB), float("-inf"), device=DEVICE)
    draft_logits[:, EXACT_IDS] = torch.from_numpy(score).to(DEVICE)
    draft_logits = draft_logits.expand(n, EXACT_STEPS, EXACT_VOCAB)
    state = torch.arange(n, device=DEVICE, dtype=torch.int64)
    gen = torch.Generator(device=DEVICE).manual_seed(77)
    sampled, num_sampled = topk_verify(
        topk,
        torch.from_numpy(draft_sampled.reshape(-1)).to(DEVICE),
        torch.arange(n * rows_per, device=DEVICE, dtype=torch.int64) + 1000,
        torch.arange(n + 1, device=DEVICE, dtype=torch.int32) * rows_per,
        state,
        state.repeat_interleave(rows_per),
        torch.arange(rows_per, device=DEVICE, dtype=torch.int32).repeat(n),
        torch.full((n,), EXACT_TEMPERATURE, device=DEVICE),
        torch.full((n,), EXACT_TOP_K, device=DEVICE, dtype=torch.int32),
        torch.randint(-(1 << 62), 1 << 62, (n,), generator=gen, device=DEVICE),
        EXACT_STEPS,
        draft_logits=draft_logits,
        has_sampled=True,
        min_p=torch.full((n,), MIN_P_VALUE, device=DEVICE) if kind == "min_p" else None,
        top_p=torch.full((n,), NUCLEUS_P, device=DEVICE) if kind == "top_p" else None,
    )
    local = {t: i for i, t in enumerate(EXACT_IDS)}
    counts: dict[tuple, int] = {}
    for tokens in as_lists(sampled, num_sampled):
        outcome = tuple(local[t] for t in tokens)
        counts[outcome] = counts.get(outcome, 0) + 1
    return counts


@pytest.mark.parametrize("kind", ["top_p", "min_p"])
def test_the_outcome_has_the_exact_block_law_with_a_filtered_target(kind: str):
    """Verifications of three drafts drawn from Q against the enumerated law of the
    block method whose target rows are the top-3 of the row with the nucleus (or the
    min_p threshold) applied and renormalised: what the kernel verifies is the
    distribution the full path samples from. The test checks its own power: the law of
    the target without the filter (top-3 only) is far from the observed counts."""
    n = 40000
    z, kept, score, Q = filtered_exact_case(kind)
    _, plain, _, _ = exact_case(EXACT_FILTER_SEED)
    assert (kept > 0).sum() < (plain > 0).sum(), "the filter keeps every top-3 token"
    law = block_verification_law(kept, Q)
    assert abs(sum(law.values()) - 1.0) < 1e-9
    counts = run_exact_law(kept, z, score, Q, kind, n)
    assert set(counts) <= set(law), "an outcome the filtered target cannot produce"
    stat, dof = chi_square(counts, law, n)
    limit = dof + 4.0 * (2.0 * dof) ** 0.5
    assert dof > 20, dof
    assert stat < limit, (stat, dof, limit)
    # Power: the same counts against the law of the unfiltered top-3 target.
    stat0, dof0 = chi_square(counts, block_verification_law(plain, Q), n)
    assert stat0 > dof0 + 8.0 * (2.0 * dof0) ** 0.5, (stat0, dof0)
