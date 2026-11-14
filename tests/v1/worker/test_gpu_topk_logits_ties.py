# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ties at the top_k boundary, for both routes of the full sampler.

bf16 logits tie constantly (about 44 % of the real decode rows of Muse Spark have an
equal 64th and 65th logit), so what a sampler does with the tokens tied with the
top_k-th logit is part of its distribution:

* the Gumbel-max route (a step with a greedy row or an explicit seed) keeps exactly
  top_k tokens, the lowest ids among the ties (tests/v1/worker/test_gpu_topk_logits.py);
* FlashInfer's top-k sampler (every other step of top_k-only sampling) keeps EVERY
  token tied with the top_k-th logit: a token is accepted when fewer than top_k tokens
  are strictly larger.

The candidate path of a "threshold" step has to give the second distribution on its
own: the draw over the kept set, whatever the size of the tie group, whether the ties
sit in one shard or in all of them, and beyond the candidates each rank lists (the
tokens tied with a rank's K-th logit that did not make its list). The tests build
those rows (and the all-equal row, whose kept set is the whole vocabulary):

1. the real FlashInfer kernel draws with the threshold distribution and not with the
   exact-k one (the premise of the rule), for every constructed row;
2. the candidate path draws, position by position, the token of the Gumbel-max draw
   over the threshold set of the full logits (the same noise per seed, position and
   token id): equal draws mean equal kept sets and equal weights;
3. chi-square of the candidate path's draws against the analytic probabilities
   exp(logit / T) / Z over the threshold set, with no draw outside the set.
"""

import math

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k logits tests", allow_module_level=True)

from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.topk_logits import (
    TIE_EXTRA_WORDS,
    TieDraw,
    TopKLogits,
    fill_tie_closure,
    local_topk_candidates,
    topk_merge_sample,
)

DEVICE = "cuda"
TP = 8
K = 64
MULTIPLIER = 0.17677669529663687
SOFT_CAP = 20.0
SMALL = 4096  # a vocabulary of 8 shards of 512 tokens: draws are cheap to count


def soft_cap(x: torch.Tensor) -> torch.Tensor:
    x = x.float() * MULTIPLIER
    return SOFT_CAP * torch.tanh(x / SOFT_CAP)


# --------------------------------------------------------------------------------
# Rows with constructed ties.
# --------------------------------------------------------------------------------

TIE_VALUE = 4.0  # exact in bf16; above the base logits, below the large ones


def planted_row(
    vocab: int,
    large: int,
    ties_in_shard: dict[int, int],
    seed: int = 0,
    tie_value: float = TIE_VALUE,
) -> torch.Tensor:
    """Base logits in [-1, 1], `large` distinct values above `tie_value` (spread over
    the shards) and `ties_in_shard[s]` tokens equal to `tie_value` in shard s."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    shard = vocab // TP
    row = torch.rand(vocab, generator=g, device=DEVICE) * 2.0 - 1.0
    order = torch.randperm(vocab, generator=g, device=DEVICE)
    taken = 0
    for i in range(large):
        row[order[taken]] = 8.0 + 0.25 * i  # distinct and exact in bf16
        taken += 1
    for s, n in ties_in_shard.items():
        # tokens of shard s that are not large already
        free = (row[s * shard : (s + 1) * shard] < 5.0).nonzero().flatten() + s * shard
        row[free[torch.randperm(len(free), generator=g, device=DEVICE)[:n]]] = tie_value
    return row.to(torch.bfloat16)


def build_rows(vocab: int) -> dict[str, torch.Tensor]:
    rows = {
        # More tokens tied with the 64th than any rank lists: one shard holds them all.
        "unlisted_one_shard": planted_row(
            vocab, large=20, ties_in_shard={3: 100}, seed=1
        ),
        # Unlisted ties in every shard (each holds more ties than the rest of its list).
        "unlisted_every_shard": planted_row(
            vocab, large=10, ties_in_shard={s: 80 for s in range(TP)}, seed=2
        ),
        # Ties across shards that every rank lists whole.
        "listed_across_shards": planted_row(
            vocab, large=30, ties_in_shard={s: 5 for s in range(TP)}, seed=3
        ),
        # The boundary tie of two tokens on two shards: exact-k would keep one of them.
        "two_way_boundary": planted_row(
            vocab, large=63, ties_in_shard={1: 1, 6: 1}, seed=4
        ),
        # Ties that the lists end on: shard 5 has 56 ties, shard 2 one more than K - 8.
        "list_ends_on_ties": planted_row(
            vocab, large=8, ties_in_shard={5: 56, 2: 57}, seed=5
        ),
        # No tie group at all.
        "no_ties": planted_row(vocab, large=70, ties_in_shard={}, seed=6),
    }
    # +0.0 and -0.0 are one value: a boundary of mixed zeros.
    g = torch.Generator(device=DEVICE).manual_seed(7)
    zero = -(torch.rand(vocab, generator=g, device=DEVICE) * 0.9 + 0.05)
    order = torch.randperm(vocab, generator=g, device=DEVICE)
    zero[order[:30]] = 1.0 + torch.arange(30, device=DEVICE) * 0.125
    zero[order[30:70]] = 0.0
    zero[order[70:110]] = -0.0
    rows["signed_zero_boundary"] = zero.to(torch.bfloat16)
    # Every token the same: the kept set is the whole vocabulary.
    rows["all_equal"] = torch.full((vocab,), 1.5, device=DEVICE).to(torch.bfloat16)
    # A natural row of bf16 logits (the real ones tie often at the 64th place).
    g = torch.Generator(device=DEVICE).manual_seed(8)
    rows["natural"] = (torch.randn(vocab, generator=g, device=DEVICE) * 3.0).to(
        torch.bfloat16
    )
    return rows


# --------------------------------------------------------------------------------
# The two ways to sample, on the full logits and from the candidates.
# --------------------------------------------------------------------------------


class Draws:
    """`rows` requests that each see the same logits row: one temperature, one top_k,
    a seed and a position per request."""

    def __init__(self, rows: int, temperature: float, top_k: int, seed: int):
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        self.rows = rows
        self.temperature = torch.full((rows,), temperature, device=DEVICE)
        self.top_k = torch.full((rows,), top_k, dtype=torch.int32, device=DEVICE)
        self.seeds = torch.randint(
            -(1 << 62), 1 << 62, (rows,), generator=g, device=DEVICE
        )
        self.pos = torch.randint(1, 100000, (rows,), generator=g, device=DEVICE)
        self.idx = torch.arange(rows, dtype=torch.int64, device=DEVICE)

    def candidates(self, row: torch.Tensor, use_fp64: bool = False) -> TopKLogits:
        """What the model does on 8 ranks, on one device: a shard per rank, the
        all-gather (dim 0, rank-major) as a concatenation, the soft cap."""
        logits = row.view(1, -1).expand(self.rows, -1).contiguous()
        shard = logits.shape[1] // TP
        ties = TieDraw(self.idx, self.seeds, self.pos, use_fp64)
        parts = []
        for r in range(TP):
            local = logits[:, r * shard : (r + 1) * shard]
            words = local_topk_candidates(local, K, extra_words=TIE_EXTRA_WORDS)
            fill_tie_closure(local, words, K, ties, token_base=r * shard)
            parts.append(words)
        words = torch.cat(parts).view(torch.bfloat16)
        return TopKLogits(
            words=words,
            capped=soft_cap(words),
            num_rows=self.rows,
            k=K,
            tp_size=TP,
            shard_size=shard,
            threshold=True,
        )

    def merge(self, topk: TopKLogits, use_fp64: bool = False) -> torch.Tensor:
        return topk_merge_sample(
            topk, self.idx, self.temperature, self.top_k, self.seeds, self.pos, use_fp64
        )

    def reference(self, row: torch.Tensor, use_fp64: bool = False) -> torch.Tensor:
        """The Gumbel-max draw over the threshold set of the full logits: the
        full-path function and noise, on the logits of the tokens >= the top_k-th."""
        capped = soft_cap(row).view(1, -1).expand(self.rows, -1).contiguous()
        kth = torch.topk(capped[0], int(self.top_k[0])).values[-1]
        masked = torch.where(capped >= kth, capped, torch.full_like(capped, -math.inf))
        apply_temperature(masked, self.idx, self.temperature)
        return gumbel_sample(
            masked,
            self.idx,
            self.temperature,
            self.seeds,
            self.pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=use_fp64,
        )


def threshold_set(row: torch.Tensor, top_k: int) -> torch.Tensor:
    """Token ids with a logit >= the top_k-th largest (FlashInfer's kept set)."""
    capped = soft_cap(row)
    kth = torch.topk(capped, top_k).values[-1]
    return (capped >= kth).nonzero().flatten()


def exact_k_set(row: torch.Tensor, top_k: int) -> torch.Tensor:
    """The top_k largest, the lowest ids among the ties at the boundary."""
    capped = soft_cap(row)
    order = torch.sort(capped, descending=True, stable=True).indices[:top_k]
    return torch.sort(order).values


def probabilities(row: torch.Tensor, ids: torch.Tensor, temperature: float):
    capped = soft_cap(row)[ids].double() / temperature
    return torch.softmax(capped, dim=0)


def z_score(chi2: float, dof: int) -> float:
    """Wilson-Hilferty normal approximation of a chi-square statistic."""
    return ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))


def chi_square_z(counts: torch.Tensor, ids: torch.Tensor, probs: torch.Tensor) -> float:
    """z of the chi-square of the counts of `ids` against `probs` (bins with fewer than
    10 expected draws merged)."""
    n = int(counts.sum())
    observed = counts[ids].double()
    expected = probs * n
    big = expected >= 10
    obs = torch.cat([observed[big], observed[~big].sum().view(1)])
    exp = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = exp > 0
    chi2 = float(((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum())
    dof = int(keep.sum()) - 1
    return z_score(chi2, dof) if dof >= 1 else 0.0


ROW_NAMES = list(build_rows(SMALL))


@pytest.fixture(
    scope="module", params=[SMALL, 202048], ids=["vocab4096", "vocab202048"]
)
def rows(request) -> dict[str, torch.Tensor]:
    return build_rows(request.param)


def test_the_constructed_rows_have_the_ties_they_are_for(rows):
    """The tie groups are what the names say (the premise of the other tests)."""
    vocab = rows["natural"].numel()
    shard = vocab // TP
    sizes = {name: len(threshold_set(row, K)) for name, row in rows.items()}
    assert sizes["unlisted_one_shard"] == 120
    assert sizes["unlisted_every_shard"] == 10 + 8 * 80
    assert sizes["listed_across_shards"] == 70
    assert sizes["two_way_boundary"] == 65
    assert sizes["list_ends_on_ties"] == 8 + 56 + 57
    assert sizes["no_ties"] == 64
    assert sizes["signed_zero_boundary"] == 30 + 80
    assert sizes["all_equal"] == vocab
    assert sizes["natural"] >= K
    # The tie group of a shard really exceeds what the rank lists.
    row = rows["unlisted_one_shard"]
    in_shard3 = threshold_set(row, K)
    assert int(((in_shard3 >= 3 * shard) & (in_shard3 < 4 * shard)).sum()) > K


# --------------------------------------------------------------------------------
# 1. The premise: the real FlashInfer kernel keeps every tie of the top_k-th logit.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,temperature",
    [
        ("unlisted_one_shard", 1.0),
        ("unlisted_every_shard", 0.8),
        ("listed_across_shards", 1.0),
        ("two_way_boundary", 1.0),
        ("list_ends_on_ties", 1.3),
        ("signed_zero_boundary", 1.0),
    ],
)
def test_flashinfer_keeps_every_token_tied_with_the_kth_logit(name, temperature):
    from vllm.v1.sample.ops.topk_topp_sampler import (
        flashinfer_sample,
        flashinfer_sampler_supported,
    )

    if not flashinfer_sampler_supported():
        pytest.skip("FlashInfer sampler not supported here")
    top_k = 64
    row = build_rows(SMALL)[name]
    keep = threshold_set(row, top_k)
    exact = exact_k_set(row, top_k)
    assert len(keep) > len(exact)
    batch, rounds = 20000, 10
    logits = (soft_cap(row) / temperature).view(1, -1).expand(batch, -1).contiguous()
    kk = torch.full((batch,), top_k, dtype=torch.int32, device=DEVICE)
    counts = torch.zeros(SMALL, dtype=torch.int64, device=DEVICE)
    for _ in range(rounds):
        counts += torch.bincount(flashinfer_sample(logits, kk, None), minlength=SMALL)
    n = batch * rounds
    assert int(counts[keep].sum()) == n, "FlashInfer drew outside the threshold set"
    z = chi_square_z(counts, keep, probabilities(row, keep, temperature))
    assert abs(z) < 5.0, z
    # ... and it does draw the tied tokens that exact-k would drop.
    dropped = torch.tensor(
        sorted(set(keep.tolist()) - set(exact.tolist())), device=DEVICE
    )
    assert int(counts[dropped].sum()) > 0


# --------------------------------------------------------------------------------
# 2. The candidate path draws the Gumbel-max token of the threshold set, position by
#    position.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ROW_NAMES)
@pytest.mark.parametrize("top_k", [1, 20, 64])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.7])
def test_draws_equal_the_gumbel_draw_over_the_threshold_set(
    rows, name, top_k, temperature
):
    row = rows[name]
    draws = Draws(96, temperature, top_k, seed=10 * ROW_NAMES.index(name) + top_k)
    got = draws.merge(draws.candidates(row))
    assert torch.equal(got, draws.reference(row))
    assert bool(torch.isin(got, threshold_set(row, top_k)).all())


@pytest.mark.parametrize(
    "name", ["unlisted_one_shard", "unlisted_every_shard", "all_equal"]
)
def test_draws_equal_the_reference_in_fp64(rows, name):
    row = rows[name]
    draws = Draws(96, 1.0, 64, seed=11)
    got = draws.merge(draws.candidates(row, use_fp64=True), use_fp64=True)
    assert torch.equal(got, draws.reference(row, use_fp64=True))


def test_the_unlisted_ties_are_drawn():
    """The draws reach tied tokens that no rank lists, with the weight of the others."""
    row = build_rows(SMALL)["unlisted_one_shard"]
    shard = SMALL // TP
    draws = Draws(8000, 1.0, 64, seed=3)
    topk = draws.candidates(row)
    local_ids = topk.words.view(torch.int16).view(TP, 8000, -1)[3, 0, K : 2 * K]
    listed = {int(i) + 3 * shard for i in (local_ids.to(torch.int64) & 0xFFFF)}
    got = draws.merge(topk)
    in_shard = [t for t in got.tolist() if shard * 3 <= t < shard * 4]
    unlisted = [t for t in in_shard if t not in listed]
    assert len(unlisted) > 0
    assert torch.equal(got, draws.reference(row))


# --------------------------------------------------------------------------------
# 3. Chi-square against the analytic distribution over the threshold set.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,top_k,temperature",
    [
        ("unlisted_one_shard", 64, 1.0),
        ("unlisted_every_shard", 64, 0.8),
        ("unlisted_every_shard", 20, 1.0),
        ("listed_across_shards", 64, 1.0),
        ("two_way_boundary", 64, 1.0),
        ("list_ends_on_ties", 64, 1.3),
        ("signed_zero_boundary", 64, 1.0),
        ("all_equal", 64, 1.0),
        ("natural", 64, 1.0),
        ("natural", 5, 0.8),
    ],
)
def test_draws_follow_the_threshold_distribution(name, top_k, temperature):
    row = build_rows(SMALL)[name]
    keep = threshold_set(row, top_k)
    probs = probabilities(row, keep, temperature)
    batch, rounds = 50000, 6
    counts = torch.zeros(SMALL, dtype=torch.int64, device=DEVICE)
    for i in range(rounds):
        draws = Draws(batch, temperature, top_k, seed=500 + i)
        counts += torch.bincount(draws.merge(draws.candidates(row)), minlength=SMALL)
    assert int(counts.sum()) == batch * rounds
    assert int(counts[keep].sum()) == batch * rounds, "drew a token outside the set"
    assert abs(chi_square_z(counts, keep, probs)) < 5.0
    # exact-k would be a different distribution wherever there are ties beyond k
    extra = len(keep) - top_k
    if extra > 0:
        exact = set(exact_k_set(row, top_k).tolist())
        dropped = torch.tensor(sorted(set(keep.tolist()) - exact), device=DEVICE)
        assert int(counts[dropped].sum()) > 0


# --------------------------------------------------------------------------------
# The kernels of the tie closure.
# --------------------------------------------------------------------------------


def test_the_tie_words_name_the_unlisted_tie_of_the_largest_noise():
    row = build_rows(SMALL)["unlisted_one_shard"]
    shard = SMALL // TP
    draws = Draws(5, 1.0, 64, seed=21)
    topk = draws.candidates(row)
    words = topk.words.view(torch.int16).view(TP, 5, 2 * K + TIE_EXTRA_WORDS)
    # Rank 3 has unlisted ties (100 tokens at the value, its list holds fewer). The
    # other ranks may name one too (ties at their own K-th logit), which no one counts:
    # only a rank whose K-th logit is the global top_k-th one does.
    assert bool((words[3, :, 2 * K + 1] == 1).all())
    # The padding words are zero (finite bf16 on the wire).
    assert bool((words[:, :, 2 * K + 2 :] == 0).all())
    idx = words[3, :, 2 * K].to(torch.int64) & 0xFFFF
    local = row[3 * shard : 4 * shard].float()
    assert bool((local[idx] == TIE_VALUE).all())
    # The named token is not listed.
    listed = words[3, :, K : 2 * K].to(torch.int64) & 0xFFFF
    assert not bool((listed == idx.view(-1, 1)).any())


def test_the_payload_rows_are_whole_16_byte_units():
    # The host plugin's all-gather needs shard bytes % 16 == 0: 2 * 64 + 8 words of 2 B.
    assert (2 * K) * 2 % 16 == 0 and (2 * K + TIE_EXTRA_WORDS) * 2 % 16 == 0


@pytest.mark.parametrize("rows_n", [1, 7, 64, 128, 256])
def test_the_threshold_path_replays_from_a_cuda_graph(rows_n):
    row = build_rows(SMALL)["unlisted_one_shard"]
    other = build_rows(SMALL)["natural"]
    draws = Draws(rows_n, 1.0, 64, seed=rows_n)
    shard = SMALL // TP
    static = row.view(1, -1).expand(rows_n, -1).contiguous()
    ties = TieDraw(draws.idx, draws.seeds, draws.pos)

    def run() -> torch.Tensor:
        parts = []
        for r in range(TP):
            local = static[:, r * shard : (r + 1) * shard]
            words = local_topk_candidates(local, K, extra_words=TIE_EXTRA_WORDS)
            fill_tie_closure(local, words, K, ties, token_base=r * shard)
            parts.append(words)
        words = torch.cat(parts).view(torch.bfloat16)
        topk = TopKLogits(words, soft_cap(words), rows_n, K, TP, shard, threshold=True)
        return draws.merge(topk)

    run()  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for fresh in (row, other):
        static.copy_(fresh.view(1, -1).expand(rows_n, -1))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, draws.reference(fresh))
