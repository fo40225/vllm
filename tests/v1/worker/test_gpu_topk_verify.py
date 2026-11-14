# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifying drafts from the top-K logits of every vocab-parallel rank.

The reference is the full verification path: the gathered logits through the soft cap,
`apply_temperature`, `apply_top_k_top_p` and `rejection_sample` (block method, with
draft logits). ``topk_verify`` has to return, for every request, the accepted tokens
and the resampled token of the full path:

* an all-greedy batch bit for bit, for every number of draft tokens and every accepted
  length;
* a batch with sampled rows (temperature > 0, top_k <= K) with the same draws (the
  uniform of the acceptance test and the Gumbel noise are keyed by seed, position and
  token id in both), up to fp32 rounding of the sums over the vocabulary, which differ
  in order (the pinned inputs below show no differing request at all); plus chi-square
  tests against the analytic law of the block method: the first token of a one-step
  verification, and the exact law of the whole outcome of a three-step verification at
  temperature 0.7 with draft mass outside the target's top-k support.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k verify tests", allow_module_level=True)

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p
from vllm.v1.worker.gpu.sample.topk_logits import TopKLogits, local_topk_candidates
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample
from vllm.v1.worker.gpu.spec_decode.topk_rejection import topk_verify

DEVICE = "cuda"
VOCAB = 202048
TP = 8
K = 64
# The soft cap of Muse Spark: 20 * tanh(x * 0.1768 / 20).
MULTIPLIER = 0.17677669529663687
SOFT_CAP = 20.0


def soft_cap(x: torch.Tensor) -> torch.Tensor:
    x = x.float() * MULTIPLIER
    return SOFT_CAP * torch.tanh(x / SOFT_CAP)


def make_logits(rows: int, vocab: int = VOCAB, seed: int = 0) -> torch.Tensor:
    """bf16 logits whose top entries tie often (bf16 is coarser than their gaps)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g, device=DEVICE) * 3.0
    return logits.to(torch.bfloat16)


def make_peaky_logits(rows: int, vocab: int = VOCAB, seed: int = 0) -> torch.Tensor:
    """Logits like a confident model: 2-4 tokens of a row dominate the rest, so that a
    draft of those tokens is accepted for several steps in a row."""
    logits = make_logits(rows, vocab, seed).float()
    g = torch.Generator(device=DEVICE).manual_seed(seed + 99)
    ids = torch.randint(0, vocab, (rows, 4), generator=g, device=DEVICE)
    boost = 30.0 + 15.0 * torch.rand(rows, 4, generator=g, device=DEVICE)
    # The 4th token dominates only on half of the rows.
    boost[:, 3] *= (torch.rand(rows, generator=g, device=DEVICE) < 0.5).float()
    keep = torch.rand(rows, generator=g, device=DEVICE) < 0.85
    logits.scatter_(1, ids, torch.where(keep[:, None], boost, logits.gather(1, ids)))
    return logits.to(torch.bfloat16)


def gather_candidates(
    logits: torch.Tensor, tp: int = TP, k: int = K, cap=soft_cap
) -> TopKLogits:
    """What the model does on `tp` ranks, on one device: a shard per rank, the
    all-gather (dim 0, rank-major) as a concatenation."""
    rows, vocab = logits.shape
    shard = vocab // tp
    words = torch.cat(
        [
            local_topk_candidates(logits[:, r * shard : (r + 1) * shard], k)
            for r in range(tp)
        ]
    ).view(torch.bfloat16)
    return TopKLogits(
        words=words,
        capped=cap(words),
        num_rows=rows,
        k=k,
        tp_size=tp,
        shard_size=shard,
    )


class Batch:
    """A speculative verify batch: per request a number of draft tokens (its rows are
    the draft tokens plus one), a temperature, a top_k, and drafts that are partly
    right (the argmax or a token of the target's top candidates) and partly not."""

    def __init__(
        self,
        num_draft: list[int],
        temperature: list[float],
        top_k: list[int],
        num_steps: int,
        seed: int = 0,
        logits: torch.Tensor | None = None,
        accept_prefix: list[int] | None = None,
        placeholders: float = 0.0,
        outside: float = 0.15,
        draft_spread: float = 0.3,
        peaky: bool = True,
        share_drafts: bool = False,
        vocab: int = VOCAB,
        min_p: list[float] | None = None,
        top_p: list[float] | None = None,
    ):
        n = len(num_draft)
        self.num_reqs = n
        self.num_steps = num_steps
        self.vocab = vocab
        rows_per = [d + 1 for d in num_draft]
        cu = np.concatenate([[0], np.cumsum(rows_per)]).astype(np.int32)
        rows = int(cu[-1])
        self.rows = rows
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        rng = np.random.default_rng(seed)
        if logits is None:
            make = make_peaky_logits if peaky else make_logits
            logits = make(rows, vocab, seed=seed + 1)
        self.logits = logits
        capped = soft_cap(self.logits)
        max_reqs = n + 3
        state = rng.permutation(max_reqs)[:n].astype(np.int32)
        self.idx_mapping = torch.from_numpy(state).to(DEVICE)
        self.cu_num_logits = torch.from_numpy(cu).to(DEVICE)
        self.expanded_idx_mapping = self.idx_mapping.repeat_interleave(
            torch.tensor(rows_per, device=DEVICE)
        )
        self.expanded_local_pos = torch.cat(
            [torch.arange(r, dtype=torch.int32, device=DEVICE) for r in rows_per]
        )
        temp = torch.zeros(max_reqs, dtype=torch.float32, device=DEVICE)
        tk = torch.full((max_reqs,), vocab, dtype=torch.int32, device=DEVICE)
        temp[self.idx_mapping.long()] = torch.tensor(
            temperature, dtype=torch.float32, device=DEVICE
        )
        tk[self.idx_mapping.long()] = torch.tensor(
            top_k, dtype=torch.int32, device=DEVICE
        )
        self.temperature, self.top_k = temp, tk
        mp = torch.zeros(max_reqs, dtype=torch.float32, device=DEVICE)
        tp = torch.ones(max_reqs, dtype=torch.float32, device=DEVICE)
        if min_p is not None:
            mp[self.idx_mapping.long()] = torch.tensor(
                min_p, dtype=torch.float32, device=DEVICE
            )
        if top_p is not None:
            tp[self.idx_mapping.long()] = torch.tensor(
                top_p, dtype=torch.float32, device=DEVICE
            )
        self.min_p, self.top_p = mp, tp
        self.use_min_p = min_p is not None and any(m != 0.0 for m in min_p)
        self.use_top_p = top_p is not None and any(t != 1.0 for t in top_p)
        self.seeds = torch.randint(
            -(1 << 62), 1 << 62, (max_reqs,), generator=g, device=DEVICE
        )
        base_pos = rng.integers(1, 100000, n)
        self.pos = torch.from_numpy(
            np.concatenate([base_pos[r] + np.arange(rows_per[r]) for r in range(n)])
        ).to(DEVICE)
        self.has_sampled = any(t != 0.0 for t in temperature)

        # The drafts. A draft token of row i is draft_sampled[start + 1 + i]; the slot
        # start + 0 is the input token of the step.
        argmax = capped.argmax(dim=-1).cpu().numpy()
        top_ids = capped.topk(16, dim=-1).indices.cpu().numpy()
        capped_cpu = (
            capped.gather(1, torch.from_numpy(top_ids).to(DEVICE)).cpu().numpy()
        )
        draft_sampled = np.zeros(rows, dtype=np.int64)
        draft_logits = torch.full(
            (max_reqs, num_steps, vocab), float("-inf"), device=DEVICE
        )
        writes = []
        for r in range(n):
            start = int(cu[r])
            t = temperature[r]
            draft_sampled[start] = rng.integers(0, vocab)
            for i in range(num_draft[r]):
                row = start + i
                # The draft distribution of this step lives on the 5 best tokens of the
                # target and 3 more of its top 16 (as the candidates of the DFlash2
                # selector), scored like the target plus noise.
                cand = list(top_ids[row][:5]) + list(
                    rng.choice(top_ids[row][5:], 3, replace=False)
                )
                score = np.array(
                    [capped_cpu[row][list(top_ids[row]).index(c)] for c in cand]
                ) + rng.normal(0.0, draft_spread, len(cand))
                if accept_prefix is not None and t == 0.0 and i < accept_prefix[r]:
                    token = int(argmax[row])
                elif rng.random() < outside:
                    token = int(rng.integers(0, vocab))
                    cand.append(token)
                    score = np.append(score, rng.normal(0.0, 1.0))
                else:
                    q = np.exp((score - score.max()) / (t if t > 0.0 else 1.0))
                    token = int(cand[rng.choice(len(cand), p=q / q.sum())])
                if rng.random() < placeholders:
                    token = -1
                draft_sampled[start + 1 + i] = token
                writes.append((int(state[r]), i, cand, score))
        self.draft_sampled = torch.from_numpy(draft_sampled).to(DEVICE)
        for s_, i, cand, score in writes:
            ids = torch.tensor([int(c) for c in cand], device=DEVICE)
            draft_logits[s_, i, ids] = torch.tensor(
                score, dtype=torch.float32, device=DEVICE
            )
        self.draft_logits = draft_logits
        if share_drafts:
            # Every request proposes what request 0 proposes (the same draft tokens and
            # draft distribution): only the seeds and positions differ.
            rows_per = num_draft[0] + 1
            assert len(set(num_draft)) == 1
            self.draft_sampled = self.draft_sampled[:rows_per].repeat(n)
            first_state = int(state[0])
            self.draft_logits[:] = draft_logits[first_state]

    def full_path(self, use_fp64: bool = False):
        """The full path: `RejectionSampler._verify` on the gathered logits."""
        logits = soft_cap(self.logits).contiguous()
        apply_temperature(logits, self.expanded_idx_mapping, self.temperature)
        if self.use_min_p:
            apply_min_p(logits, self.expanded_idx_mapping, self.min_p)
        # As `SamplingStates.get_top_k_top_p`: a tensor only when a request needs it.
        top_k_rows = self.top_k[self.expanded_idx_mapping.long()]
        top_p_rows = self.top_p[self.expanded_idx_mapping.long()]
        do_top_k = bool((top_k_rows != self.vocab).any())
        if do_top_k or self.use_top_p:
            logits = apply_top_k_top_p(
                logits,
                top_k_rows if do_top_k else None,
                top_p_rows if self.use_top_p else None,
            )
        return rejection_sample(
            logits,
            self.draft_logits,
            self.draft_sampled,
            self.cu_num_logits,
            self.pos,
            self.idx_mapping,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.temperature,
            self.seeds,
            self.num_steps,
            None,
            use_fp64=use_fp64,
            use_block_verification=True,
        )

    def candidate_path(self, k: int = K, use_fp64: bool = False):
        topk = gather_candidates(self.logits, k=k)
        return topk_verify(
            topk,
            self.draft_sampled,
            self.pos,
            self.cu_num_logits,
            self.idx_mapping,
            self.expanded_idx_mapping,
            self.expanded_local_pos,
            self.temperature,
            self.top_k,
            self.seeds,
            self.num_steps,
            draft_logits=self.draft_logits,
            has_sampled=self.has_sampled,
            use_fp64=use_fp64,
            min_p=self.min_p if self.use_min_p else None,
            top_p=self.top_p if self.use_top_p else None,
        )


def as_lists(sampled: torch.Tensor, num_sampled: torch.Tensor):
    sampled = sampled.cpu().numpy()
    num_sampled = num_sampled.cpu().numpy()
    return [sampled[r, : num_sampled[r]].tolist() for r in range(len(num_sampled))]


def compare(batch: Batch, **kwargs):
    expected = as_lists(*batch.full_path(**kwargs))
    got = as_lists(*batch.candidate_path(**kwargs))
    bad = [r for r in range(batch.num_reqs) if expected[r] != got[r]]
    return expected, got, bad


# --------------------------------------------------------------------------------
# Greedy batches: the full path's tokens, bit for bit.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("num_steps", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("k", [1, 64])
def test_greedy_matches_full_path(num_steps: int, k: int):
    """Every number of draft tokens and every accepted length 0 .. num_steps."""
    rng = np.random.default_rng(num_steps)
    n = 96
    num_draft = rng.integers(0, num_steps + 1, n).tolist()
    # Accepted prefix lengths cover 0 .. num_steps (more than the drafts: all accepted).
    accept = rng.integers(0, num_steps + 1, n).tolist()
    batch = Batch(
        num_draft,
        [0.0] * n,
        [VOCAB] * n,
        num_steps,
        seed=num_steps,
        accept_prefix=accept,
    )
    expected = as_lists(*batch.full_path())
    assert as_lists(*batch.candidate_path(k=k)) == expected
    # Every accepted length shows up, the bonus case (all accepted) included.
    seen = {len(x) - 1 for x in expected}
    assert 0 in seen and num_steps in seen, seen


def test_greedy_mixed_with_placeholders_and_zero_drafts():
    num_steps = 7
    rng = np.random.default_rng(1)
    n = 128
    num_draft = rng.choice([0, 7, 7, 3, 1], n).tolist()
    batch = Batch(
        num_draft,
        [0.0] * n,
        [VOCAB] * n,
        num_steps,
        seed=2,
        accept_prefix=rng.integers(0, 8, n).tolist(),
        placeholders=0.1,
    )
    assert as_lists(*batch.full_path()) == as_lists(*batch.candidate_path(k=1))
    assert as_lists(*batch.full_path()) == as_lists(*batch.candidate_path(k=K))


def test_greedy_ties_match_full_path():
    """Ties on the argmax (inside a shard, across shards, at the shard edges) and in the
    bonus row."""
    num_steps = 3
    n = 12
    shard = VOCAB // TP
    num_draft = [3] * n
    logits = make_logits(4 * n, seed=7) - 4.0
    top = 30.0
    for r in range(n):
        base = 4 * r
        for i in range(4):
            row = logits[base + i]
            kind = (r + i) % 6
            if kind == 0:
                row[[100, 200]] = top
            elif kind == 1:
                row[[3 * shard - 1, 3 * shard]] = top
            elif kind == 2:
                row[
                    5 * shard
                    + 5
                    + torch.arange(0, 70 * 211, 211, device=DEVICE) % (3 * shard)
                ] = top
            elif kind == 3:
                row[:] = 1.5
            elif kind == 4:
                row[[11, 12, 13]] = torch.tensor(
                    [-0.0, 0.0, -0.0], device=DEVICE, dtype=torch.bfloat16
                )
                row[14:] = -2.0
            else:
                row[[0, VOCAB - 1]] = top
    batch = Batch(
        num_draft,
        [0.0] * n,
        [VOCAB] * n,
        num_steps,
        seed=8,
        logits=logits,
        accept_prefix=[(r % 4) for r in range(n)],
    )
    for k in (1, K):
        assert as_lists(*batch.full_path()) == as_lists(*batch.candidate_path(k=k))


# --------------------------------------------------------------------------------
# Sampled rows.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
@pytest.mark.parametrize("top_k", [1, 5, 64])
@pytest.mark.parametrize("num_steps", [1, 3, 7])
def test_sampled_matches_full_path(temperature: float, top_k: int, num_steps: int):
    n = 160
    rng = np.random.default_rng(top_k * 7 + num_steps)
    batch = Batch(
        rng.integers(0, num_steps + 1, n).tolist(),
        [temperature] * n,
        [top_k] * n,
        num_steps,
        seed=top_k + num_steps,
    )
    expected, got, bad = compare(batch)
    # The draws are those of the full path: a request would differ only where its
    # uniform or Gumbel scores sit within fp32 rounding of a threshold (about 1e-7 per
    # draw; the inputs are pinned and none does).
    assert not bad, [(expected[r], got[r]) for r in bad]


@pytest.mark.parametrize("use_fp64", [False, True])
def test_mixed_batch_matches_full_path(use_fp64: bool):
    """Greedy and sampled requests of several temperatures and top_k in one batch, with
    zero-draft rows (a prefill row or a request without drafts), -1 placeholders and
    drafts outside the top-k set."""
    num_steps = 7
    n = 256
    rng = np.random.default_rng(3)
    temperature = rng.choice([0.0, 0.0, 0.5, 1.0, 1.7], n).tolist()
    top_k = rng.choice([1, 2, 17, 63, 64], n).tolist()
    top_k = [
        VOCAB if t == 0.0 and i % 2 else k
        for i, (t, k) in enumerate(zip(temperature, top_k))
    ]
    batch = Batch(
        rng.choice([0, 7, 7, 7, 3, 1], n).tolist(),
        temperature,
        top_k,
        num_steps,
        seed=4,
        accept_prefix=rng.integers(0, 8, n).tolist(),
        placeholders=0.08,
        outside=0.2,
    )
    expected, got, bad = compare(batch, use_fp64=use_fp64)
    greedy_rows = [r for r in range(n) if temperature[r] == 0.0]
    assert not set(bad) & set(greedy_rows), "a greedy request differs"
    assert not bad, [(expected[r], got[r]) for r in bad]


def test_sampled_ties_at_the_top_k_boundary_match_full_path():
    """Ties across the top_k-th logit (bf16 logits tie often): the kept set is the
    lowest token ids among the tied, as `apply_top_k_top_p`, for the target of the
    acceptance test and for the resampled token."""
    num_steps = 3
    shard = VOCAB // TP
    n = 80
    g = torch.Generator(device=DEVICE).manual_seed(5)
    rows = n * (num_steps + 1)
    logits = make_logits(rows, seed=31) - 8.0
    spread = torch.arange(5, VOCAB, 2000, device=DEVICE)
    top = torch.linspace(20, 30, 40, device=DEVICE).to(torch.bfloat16)
    for row in range(rows):
        kind = row % 5
        if kind == 0:
            logits[row, torch.randperm(VOCAB, generator=g, device=DEVICE)[:40]] = top
            logits[row, spread[:100]] = 8.0
        elif kind == 1:
            logits[row, [10, shard + 3, 5 * shard]] = torch.tensor(
                [25.0, 26.0, 27.0], device=DEVICE
            ).to(torch.bfloat16)
            logits[row, spread[:50]] = 12.0
        elif kind == 2:
            logits[row, spread[3:73]] = 15.0
        elif kind == 3:
            logits[row, [2 * shard - 1, 2 * shard]] = 30.0
        else:
            edges = torch.cat(
                [torch.arange(TP) * shard, torch.arange(1, TP + 1) * shard - 1]
            )
            logits[row, edges.to(DEVICE)] = 18.0
    rng = np.random.default_rng(6)
    batch = Batch(
        [num_steps] * n,
        [1.0] * n,
        rng.choice([1, 5, 64], n).tolist(),
        num_steps,
        seed=7,
        logits=logits,
        peaky=False,
        outside=0.0,
    )
    expected, got, bad = compare(batch)
    assert not bad, [(expected[r], got[r]) for r in bad]


@pytest.mark.parametrize("tp,k", [(8, 64), (4, 32), (2, 64), (8, 128)])
def test_sampled_other_shapes(tp: int, k: int):
    vocab = tp * 16000
    n = 64
    rng = np.random.default_rng(tp * k)
    batch = Batch(
        rng.integers(0, 4, n).tolist(),
        [1.0] * n,
        [min(k, 64)] * n,
        3,
        seed=tp,
        vocab=vocab,
    )
    exp = as_lists(*batch.full_path())
    topk = gather_candidates(batch.logits, tp=tp, k=k)
    got = as_lists(
        *topk_verify(
            topk,
            batch.draft_sampled,
            batch.pos,
            batch.cu_num_logits,
            batch.idx_mapping,
            batch.expanded_idx_mapping,
            batch.expanded_local_pos,
            batch.temperature,
            batch.top_k,
            batch.seeds,
            3,
            draft_logits=batch.draft_logits,
            has_sampled=True,
        )
    )
    bad = [r for r in range(n) if exp[r] != got[r]]
    assert not bad, [(exp[r], got[r]) for r in bad]


def test_is_deterministic():
    n = 64
    batch = Batch([7] * n, [1.0] * n, [64] * n, 7, seed=9)
    first = as_lists(*batch.candidate_path())
    assert first == as_lists(*batch.candidate_path())


def test_draw_does_not_depend_on_the_request_slot_order():
    """The draws are keyed by the request seed and the position, not by the slot."""
    n = 48
    batch = Batch([7] * n, [1.0] * n, [64] * n, 7, seed=10)
    first = as_lists(*batch.candidate_path())
    perm = torch.randperm(n, device=DEVICE)
    # The same requests, listed in another order (their rows are permuted with them).
    rows_per = (batch.cu_num_logits[1:] - batch.cu_num_logits[:-1]).cpu().tolist()
    starts = batch.cu_num_logits[:-1].cpu().tolist()
    order = perm.cpu().tolist()
    row_order = torch.tensor(
        [starts[r] + i for r in order for i in range(rows_per[r])], device=DEVICE
    )
    shuffled = Batch.__new__(Batch)
    shuffled.__dict__.update(batch.__dict__)
    shuffled.logits = batch.logits[row_order]
    shuffled.draft_sampled = batch.draft_sampled[row_order]
    shuffled.pos = batch.pos[row_order]
    shuffled.idx_mapping = batch.idx_mapping[perm]
    shuffled.expanded_idx_mapping = batch.expanded_idx_mapping[row_order]
    shuffled.expanded_local_pos = batch.expanded_local_pos[row_order]
    again = as_lists(*shuffled.candidate_path())
    assert [again[i] for i in range(n)] == [first[r] for r in order]


def test_first_token_distribution_matches_the_block_verification_distribution():
    """A chi-square test of the first sampled token of a one-step verification (the
    draft token if accepted, else the residual draw) against the analytic distribution
    of token t: for a single draft token x the block method reduces to the speculative
    sampling of Leviathan et al., whose output is exactly the target distribution p."""
    num_steps = 1
    n = 24000
    vocab = 4096
    g = torch.Generator(device=DEVICE).manual_seed(11)
    row_logits = torch.randn(2, vocab, generator=g, device=DEVICE) * 1.2
    # One target row (and a bonus row) replicated for every request; top_k 16.
    top_k_val = 16
    logits = row_logits.to(torch.bfloat16).repeat(n, 1)
    batch = Batch(
        [1] * n,
        [1.0] * n,
        [top_k_val] * n,
        num_steps,
        seed=12,
        logits=logits,
        peaky=False,
        outside=0.0,
        draft_spread=0.8,
        vocab=vocab,
    )
    # Distinct seeds and positions per request.
    topk = gather_candidates(batch.logits, k=64)
    sampled, num_sampled = topk_verify(
        topk,
        batch.draft_sampled,
        batch.pos,
        batch.cu_num_logits,
        batch.idx_mapping,
        batch.expanded_idx_mapping,
        batch.expanded_local_pos,
        batch.temperature,
        batch.top_k,
        batch.seeds,
        num_steps,
        draft_logits=batch.draft_logits,
        has_sampled=True,
    )
    # The first token of every request: a draw from p over the top-16 of row 0.
    first = sampled[:, 0].cpu()
    capped = soft_cap(batch.logits[0].float())
    top = capped.topk(top_k_val).indices.cpu()
    p = torch.softmax(capped[top.to(DEVICE)], dim=0).cpu().double()
    counts = torch.stack([(first == t).sum() for t in top]).double()
    assert int(counts.sum()) == n, "a token outside the top-k set was drawn"
    # The draft distribution differs per request (random draft candidates), but the
    # output of the verification is p for each of them: pool all requests.
    expected = p * n
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # 15 degrees of freedom: the 99.9th percentile is 37.7.
    assert chi2 < 37.7, (chi2, counts.tolist(), expected.tolist())


def test_outputs_have_the_distribution_of_the_full_path():
    """Fixed draft proposals, many seeds: the joint distribution of (accepted length,
    tokens) of the candidate path against the full path (a two-sample chi-square over
    the observed outcomes) and the first-token marginal against the target. The two
    paths use the same draws, so the test also bounds how far they ever drift."""
    num_steps = 3
    n = 20000
    vocab = 4096
    g = torch.Generator(device=DEVICE).manual_seed(21)
    row = torch.randn(num_steps + 1, vocab, generator=g, device=DEVICE) * 1.2
    # Make rows peaky so that several steps are accepted with decent probability.
    for i in range(num_steps + 1):
        row[i, torch.tensor([7, 70, 700], device=DEVICE)] += torch.tensor(
            [9.0, 8.5, 8.0], device=DEVICE
        )
    logits = row.to(torch.bfloat16).repeat(n, 1)
    batch = Batch(
        [num_steps] * n,
        [1.0] * n,
        [8] * n,
        num_steps,
        seed=22,
        logits=logits,
        peaky=False,
        outside=0.0,
        draft_spread=0.4,
        share_drafts=True,
        vocab=vocab,
    )
    # The tp / shard layout of a small vocabulary.
    topk = gather_candidates(batch.logits, k=64)
    full = as_lists(*batch.full_path())
    got = as_lists(
        *topk_verify(
            topk,
            batch.draft_sampled,
            batch.pos,
            batch.cu_num_logits,
            batch.idx_mapping,
            batch.expanded_idx_mapping,
            batch.expanded_local_pos,
            batch.temperature,
            batch.top_k,
            batch.seeds,
            num_steps,
            draft_logits=batch.draft_logits,
            has_sampled=True,
        )
    )
    from collections import Counter

    cf, cg = Counter(map(tuple, full)), Counter(map(tuple, got))
    assert len(cf) > 10, "the outcomes did not spread"
    mismatches = sum(1 for a, b in zip(full, got) if a != b)
    # Same draws: the paths differ only where a threshold sits within fp32 rounding of
    # the other path's sum (about 1e-7 per draw; 0 of 20000 on the pinned inputs).
    assert mismatches <= 2, mismatches
    # Two-sample chi-square over the outcomes seen at least 5 times in either.
    keys = [k for k in set(cf) | set(cg) if cf[k] + cg[k] >= 10]
    chi2 = sum((cf[k] - cg[k]) ** 2 / (cf[k] + cg[k]) for k in keys)
    assert chi2 < len(keys) + 4 * (2 * len(keys)) ** 0.5, (chi2, len(keys))
    # The first token (a block of several steps does not give the target distribution
    # at its first position, only jointly): its counts against the full path's.
    top = torch.tensor([7, 70, 700], device=DEVICE).cpu()
    first = torch.tensor([x[0] for x in got])
    firstf = torch.tensor([x[0] for x in full])
    counts = torch.stack([(first == t).sum() for t in top]).double()
    countsf = torch.stack([(firstf == t).sum() for t in top]).double()
    assert float(counts.sum()) > 0.5 * n
    chi2 = float(((counts - countsf) ** 2 / (counts + countsf).clamp(min=1)).sum())
    assert chi2 < 20.0, (chi2, counts.tolist(), countsf.tolist())


# --------------------------------------------------------------------------------
# The exact law of a multi-step verification (temperature 0.7, draft mass outside the
# target's top-k support).
# --------------------------------------------------------------------------------

EXACT_VOCAB = 1024
EXACT_K = 8
EXACT_STEPS = 3
EXACT_TEMPERATURE = 0.7
EXACT_TOP_K = 3
# Six tokens carry all the mass (in six different shards); every other logit is -20.
EXACT_IDS = [3, 130, 301, 520, 777, 1000]


def exact_case(seed: int = 12):
    """Target probabilities P [steps + 1, 6] (top-3 of each row, temperature 0.7) and
    draft probabilities Q [steps, 6] (temperature 0.7) over the six tokens, with the
    logits that make them. Q has mass on tokens the target's top-k mask removes."""
    r = np.random.default_rng(seed)
    z = np.round(r.normal(0, 4.0, (EXACT_STEPS + 1, 6)) * 4) / 4  # exact in bf16
    capped = SOFT_CAP * np.tanh(z * MULTIPLIER / SOFT_CAP) / EXACT_TEMPERATURE
    P = np.zeros_like(capped)
    for i in range(EXACT_STEPS + 1):
        top = np.argsort(-capped[i], kind="stable")[:EXACT_TOP_K]
        e = np.exp(capped[i][top] - capped[i][top].max())
        P[i][top] = e / e.sum()
    score = r.normal(0, 1.0, (EXACT_STEPS, 6)).astype(np.float32)
    Q = np.exp(score.astype(np.float64) / EXACT_TEMPERATURE)
    return z, P, score, Q / Q.sum(axis=1, keepdims=True)


def block_verification_law(P: np.ndarray, Q: np.ndarray, mutant: str | None = None):
    """The exact law of the output of the block method (accepted prefix, then the
    resampled or bonus token) for drafts x_i ~ Q_i and target rows P_i, by
    enumeration. ``mutant`` breaks one piece of it (to show the test would notice)."""
    import itertools
    import math

    g, v = Q.shape
    law: dict[tuple, float] = {}
    for x in itertools.product(range(v), repeat=g):
        w = float(np.prod([Q[i][x[i]] for i in range(g)]))
        if w == 0.0:
            continue
        log_p, ps, hs = 0.0, [], []
        for i in range(g):
            lp = math.log(P[i][x[i]]) if P[i][x[i]] > 0 else -math.inf
            log_p = min(log_p + lp - math.log(Q[i][x[i]]), 0.0)
            ps.append(math.exp(log_p))
        for i in range(g):
            if i < g - 1:
                residual = float(np.maximum(ps[i] * P[i + 1] - Q[i + 1], 0.0).sum())
                denom = residual + 1.0 - ps[i]
                h = residual / denom if denom > 0 else 1.0
            else:
                h = ps[i]
            hs.append(1.0 if mutant == "h_one" else h)
        for tau in range(g + 1):
            if tau == 0:
                p_tau = float(np.prod([1 - h for h in hs]))
            else:
                p_tau = hs[tau - 1] * float(np.prod([1 - h for h in hs[tau:]]))
            if p_tau == 0.0:
                continue
            if tau == g:
                resid = P[g].copy()
            elif mutant == "resample_target":
                resid = P[tau].copy()
            else:
                scale = 1.0 if tau == 0 or mutant == "no_p_tau" else ps[tau - 1]
                resid = np.maximum(scale * P[tau] - Q[tau], 0.0)
            if resid.sum() <= 0.0:
                continue
            resid = resid / resid.sum()
            for y in range(v):
                if resid[y] > 0:
                    out = tuple(x[:tau]) + (y,)
                    law[out] = law.get(out, 0.0) + w * p_tau * float(resid[y])
    return law


def chi_square(counts: dict, law: dict, n: int):
    """Pearson statistic over the outcomes expected at least 5 times (the rest pooled
    in one cell) and its degrees of freedom."""
    stat, cells, rest_obs, rest_exp = 0.0, 0, 0.0, 0.0
    for outcome in set(counts) | set(law):
        expected = n * law.get(outcome, 0.0)
        observed = counts.get(outcome, 0)
        if expected >= 5.0:
            stat += (observed - expected) ** 2 / expected
            cells += 1
        else:
            rest_obs += observed
            rest_exp += expected
    if rest_exp > 0:
        stat += (rest_obs - rest_exp) ** 2 / rest_exp
        cells += 1
    return stat, cells - 1


def test_the_outcome_has_the_exact_law_of_the_block_method_over_three_steps():
    """Verifications of three drafts drawn from Q (a quarter to two thirds of the mass
    of Q sits on tokens outside the target's top-3), at temperature 0.7: the joint law
    of (accepted prefix, resampled token) against the enumerated law of the block
    method. It sees what the one-step test cannot: the cumulative probability ratio
    p_tau of the residual, the residual mass of the next step in the acceptance
    probability, the temperature on the draft side, and draft tokens outside the
    support. The test checks its own power: the laws of three broken variants are far
    from the correct one at this n."""
    n = 40000
    z, P, score, Q = exact_case()
    law = block_verification_law(P, Q)
    assert abs(sum(law.values()) - 1.0) < 1e-9
    outside = [float(Q[i][P[i] == 0].sum()) for i in range(EXACT_STEPS)]
    assert max(outside) > 0.5 and min(outside) > 0.1, outside

    # Row logits: the six tokens carry z, all the others -20.
    logits = torch.full((EXACT_STEPS + 1, EXACT_VOCAB), -20.0, device=DEVICE)
    logits[:, EXACT_IDS] = torch.tensor(z, dtype=torch.float32, device=DEVICE)
    logits = logits.to(torch.bfloat16)
    base = gather_candidates(logits, tp=TP, k=EXACT_K)
    rows_per = EXACT_STEPS + 1
    width = base.words.shape[-1]
    words = (
        base.words.view(TP, rows_per, width).repeat(1, n, 1).reshape(-1, width)
    ).contiguous()
    topk = TopKLogits(
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
    draft_sampled[:, 1:] = ids[draws]  # slot 0 is the input token of the step
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
    )
    local = {t: i for i, t in enumerate(EXACT_IDS)}
    counts: dict[tuple, int] = {}
    for tokens in as_lists(sampled, num_sampled):
        outcome = tuple(local[t] for t in tokens)
        counts[outcome] = counts.get(outcome, 0) + 1
    assert set(counts) <= set(law), "an outcome the block method cannot produce"
    stat, dof = chi_square(counts, law, n)
    limit = dof + 4.0 * (2.0 * dof) ** 0.5
    assert dof > 50, dof
    assert stat < limit, (stat, dof, limit)

    # Power: the expected statistic of a broken variant (its law as the counts).
    for mutant in ("no_p_tau", "resample_target", "h_one"):
        broken = block_verification_law(P, Q, mutant)
        shifted, _ = chi_square({o: n * p for o, p in broken.items()}, law, n)
        assert shifted > 2.0 * limit, (mutant, shifted, limit)
