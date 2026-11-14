# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling from the top-K logits of every vocab-parallel rank.

The reference is the full path of the V2 sampler: the gathered logits through the
soft cap, `apply_temperature`, `apply_top_k_top_p` and `gumbel_sample`. The candidate
path has to return the same token for every row: the argmax with the lowest index
among equal logits for a greedy row, and for a row with temperature > 0 and
top_k <= K the same Gumbel-max draw (the noise is keyed by seed, position and token id
in both). A chi-square test checks the draw against the analytic distribution.
"""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k logits tests", allow_module_level=True)

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
from vllm.v1.worker.gpu.sample.topk_logits import (
    TopKLogits,
    local_topk_candidates,
    topk_merge_sample,
)

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
    """bf16 logits whose top entries tie often (bf16 spacing is wider than their gaps)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g, device=DEVICE) * 3.0
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


class Requests:
    """Per-request sampling state, one request per row."""

    def __init__(self, temperature, top_k, seed=1234, rows=None):
        rows = len(temperature) if rows is None else rows
        g = torch.Generator(device=DEVICE).manual_seed(seed)
        self.rows = rows
        self.temperature = torch.tensor(temperature, dtype=torch.float32, device=DEVICE)
        self.top_k = torch.tensor(top_k, dtype=torch.int32, device=DEVICE)
        self.seeds = torch.randint(
            -(1 << 62), 1 << 62, (rows,), generator=g, device=DEVICE
        )
        self.pos = torch.randint(1, 100000, (rows,), generator=g, device=DEVICE)
        self.idx = torch.arange(rows, dtype=torch.int64, device=DEVICE)

    def merge(self, topk: TopKLogits, use_fp64: bool = False) -> torch.Tensor:
        return topk_merge_sample(
            topk,
            self.idx,
            self.temperature,
            self.top_k,
            self.seeds,
            self.pos,
            use_fp64=use_fp64,
        )

    def full_path(self, logits: torch.Tensor, use_fp64: bool = False) -> torch.Tensor:
        """The V2 sampler's route on the gathered logits."""
        capped = soft_cap(logits).contiguous()
        apply_temperature(capped, self.idx, self.temperature)
        if bool((self.top_k != VOCAB).any()):
            capped = apply_top_k_top_p(capped, self.top_k, None)
        return gumbel_sample(
            capped,
            self.idx,
            self.temperature,
            self.seeds,
            self.pos,
            apply_temperature=False,
            is_drafting=False,
            use_fp64=use_fp64,
        )


# --------------------------------------------------------------------------------
# The local kernel.
# --------------------------------------------------------------------------------


def reference_topk(row: torch.Tensor, k: int, num_valid: int | None = None):
    """Indices of the k largest values, the lowest index first among equals."""
    row = row[: num_valid or row.numel()].float()
    order = torch.sort(row, descending=True, stable=True).indices[:k]
    return torch.sort(order).values


def check_local_topk(logits: torch.Tensor, k: int, num_valid: int | None = None):
    words = local_topk_candidates(logits, k, num_valid)
    assert words.shape == (logits.shape[0], 2 * k) and words.dtype == torch.int16
    values = words[:, :k].view(torch.bfloat16)
    idx = words[:, k:].to(torch.int64) & 0xFFFF
    for r in range(logits.shape[0]):
        expected = reference_topk(logits[r], k, num_valid)
        # The set is the reference's; the order within a row is not specified.
        order = torch.argsort(idx[r])
        assert torch.equal(idx[r][order], expected.to(idx.dtype)), r
        assert torch.equal(
            values[r][order].view(torch.int16), logits[r, idx[r][order]].view(torch.int16)
        ), r


@pytest.mark.parametrize("cols", [25256, 50512, 4096, 1000])
@pytest.mark.parametrize("k", [1, 8, 64, 128])
def test_local_topk_matches_reference(cols: int, k: int):
    check_local_topk(make_logits(24, cols, seed=cols + k), k)


def test_local_topk_ties_take_the_lowest_indices():
    cols = 25256
    logits = torch.full((8, cols), -1.0, device=DEVICE, dtype=torch.bfloat16)
    # Row 0: all equal. Row 1: a 65-way tie on the maximum, K = 64.
    logits[1, torch.arange(0, 65 * 7, 7)] = 5.0
    # Row 2: a tie at the K-th value only, 100 candidates for the last 20 places.
    logits[2, :44] = 9.0
    logits[2, 1000:1100] = 4.0
    # Row 3: the maximum on the last columns only.
    logits[3, -3:] = 2.0
    # Row 4: zeros of both signs on the maximum.
    logits[4, 5] = -0.0
    logits[4, 9] = 0.0
    logits[4, 17] = -0.0
    # Row 5: infinities.
    logits[5, 3] = float("inf")
    logits[5, 4] = float("-inf")
    logits[5, 8] = float("inf")
    # Row 6: an 84-way tie on the maximum, spread over the chunks of the first pass.
    logits[6, torch.arange(10, cols, 300)] = 5.0
    # Row 7: 40 distinct large values, then a 100-way tie for the last 24 places that
    # reaches over all the chunks.
    logits[7, torch.arange(0, cols, 631)[:40]] = torch.linspace(
        20, 30, 40, device=DEVICE
    ).to(torch.bfloat16)
    logits[7, torch.arange(5, cols, 250)[:100]] = 8.0
    check_local_topk(logits, 64)
    check_local_topk(logits, 1)


def test_local_topk_ignores_padding():
    logits = make_logits(8, 1024)
    logits[:, 1000:] = 100.0
    check_local_topk(logits, 64, num_valid=1000)


def test_local_topk_is_deterministic():
    logits = make_logits(64, 25256)
    first = local_topk_candidates(logits, K)
    assert torch.equal(first, local_topk_candidates(logits, K))


# --------------------------------------------------------------------------------
# Greedy rows: the token of the full path, bit for bit.
# --------------------------------------------------------------------------------


def greedy(rows: int) -> Requests:
    return Requests([0.0] * rows, [VOCAB] * rows)


@pytest.mark.parametrize("rows", [1, 7, 64, 128, 255, 256])
def test_greedy_matches_full_path(rows: int):
    logits = make_logits(rows, seed=rows)
    req = greedy(rows)
    assert torch.equal(
        req.merge(gather_candidates(logits)), req.full_path(logits)
    )


def test_greedy_ties_match_full_path():
    rows = 10
    logits = make_logits(rows, seed=3) - 4.0
    shard = VOCAB // TP
    top = 30.0
    # 0: two-way tie inside a shard. 1: tie on the boundary of ranks 2 and 3. 2: a 70-way
    # tie across ranks, the lowest index on rank 5. 3: everything equal. 4: zeros of both
    # signs on the maximum. 5: the maximum on the very last token. 6: the maximum on the
    # very first token and the last one. 7: a tie of ranks 0 and 7. 8, 9: heavy ties: a
    # quantized row, and a quantized row with the maximum on the last shard only.
    logits[0, [100, 200]] = top
    logits[1, [3 * shard - 1, 3 * shard]] = top
    ids = 5 * shard + 5 + torch.arange(0, 70 * 211, 211, device=DEVICE) % (3 * shard)
    logits[2, ids] = top
    logits[3] = 1.5
    logits[4] = -2.0
    logits[4, [11, 12, 13]] = torch.tensor(
        [-0.0, 0.0, -0.0], device=DEVICE, dtype=torch.bfloat16
    )
    logits[5, VOCAB - 1] = top
    logits[6, [0, VOCAB - 1]] = top
    logits[7, [shard - 1, 7 * shard + 17]] = top
    logits[8] = (torch.randint(-3, 4, (VOCAB,), device=DEVICE) * 0.5).to(
        torch.bfloat16
    )
    logits[9] = logits[8]
    logits[9, 7 * shard :] += 1.0
    req = greedy(rows)
    expected = req.full_path(logits)
    assert torch.equal(req.merge(gather_candidates(logits)), expected)
    assert expected[0] == 100 and expected[1] == 3 * shard - 1
    assert expected[3] == 0 and expected[4] == 11 and expected[6] == 0
    assert expected[7] == shard - 1


@pytest.mark.parametrize("tp,k", [(8, 1), (8, 16), (4, 32), (2, 64), (8, 128)])
def test_greedy_matches_full_path_for_other_shapes(tp: int, k: int):
    logits = make_logits(33, vocab=tp * 16000, seed=tp * k)
    req = greedy(33)
    assert torch.equal(
        req.merge(gather_candidates(logits, tp=tp, k=k)), req.full_path(logits)
    )


# --------------------------------------------------------------------------------
# Rows with temperature > 0 and top_k <= K: the draw of the full path.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("temperature", [0.7, 1.0, 1.3])
@pytest.mark.parametrize("top_k", [1, 5, 64])
@pytest.mark.parametrize("use_fp64", [False, True])
def test_top_k_draw_matches_full_path(temperature: float, top_k: int, use_fp64: bool):
    rows = 192
    logits = make_logits(rows, seed=top_k)
    req = Requests([temperature] * rows, [top_k] * rows, seed=top_k + 1)
    expected = req.full_path(logits, use_fp64=use_fp64)
    got = req.merge(gather_candidates(logits), use_fp64=use_fp64)
    assert torch.equal(got, expected)


def test_top_k_boundary_ties_match_full_path():
    """Constructed ties at the top_k boundary: the same tokens survive the mask as in
    the full path (the lowest ids among the tied), seen through draws of 128 seeds."""
    shard = VOCAB // TP
    g = torch.Generator(device=DEVICE).manual_seed(5)
    base = make_logits(5, seed=31) - 8.0
    top = torch.linspace(20, 30, 40, device=DEVICE).to(torch.bfloat16)
    spread = torch.arange(5, VOCAB, 2000, device=DEVICE)
    # 0: 40 distinct large logits, then a 100-way tie for the last 24 places of top_k 64,
    # spread over all the ranks.
    base[0, torch.randperm(VOCAB, generator=g, device=DEVICE)[:40]] = top
    base[0, spread[:100]] = 8.0
    # 1: 3 distinct large logits, then a 50-way tie for the last 2 places of top_k 5.
    base[1, [10, shard + 3, 5 * shard]] = torch.tensor(
        [25.0, 26.0, 27.0], device=DEVICE
    ).to(torch.bfloat16)
    base[1, spread[:50]] = 12.0
    # 2: a 70-way tie on the maximum, top_k 64: the 64 lowest ids.
    base[2, spread[3:73]] = 15.0
    # 3: a tie of the last token of a rank and the first of the next, top_k 1.
    base[3, [2 * shard - 1, 2 * shard]] = 30.0
    # 4: ties on the edges of every rank, top_k 64.
    edges = torch.cat([torch.arange(TP) * shard, torch.arange(1, TP + 1) * shard - 1])
    base[4, edges.to(DEVICE)] = 18.0
    top_k = torch.tensor([64, 5, 64, 1, 64]).repeat_interleave(128).tolist()
    logits = base.repeat_interleave(128, dim=0)
    rows = logits.shape[0]
    req = Requests([1.0] * rows, top_k, seed=41)
    assert torch.equal(req.merge(gather_candidates(logits)), req.full_path(logits))
    # The draws did see the boundary: ties below the cut are drawn in row 0.
    drawn = req.full_path(logits)[:128]
    assert bool(torch.isin(drawn, spread[:100].to(drawn.dtype)).any())


def test_mixed_batch_matches_full_path():
    """Greedy rows and rows of several temperatures and top_k in one batch."""
    rows = 256
    g = np.random.default_rng(5)
    temperature = g.choice([0.0, 0.0, 0.5, 1.0, 1.7], rows).tolist()
    top_k = g.choice([1, 2, 17, 63, 64], rows).tolist()
    # Half of the greedy rows have no top_k, half have one (which a greedy row ignores).
    top_k = [
        VOCAB if t == 0.0 and i % 2 else k
        for i, (t, k) in enumerate(zip(temperature, top_k))
    ]
    logits = make_logits(rows, seed=11)
    req = Requests(temperature, top_k, seed=12)
    assert torch.equal(req.merge(gather_candidates(logits)), req.full_path(logits))


def test_draw_does_not_depend_on_the_request_slot_order():
    rows = 64
    logits = make_logits(rows, seed=21)
    req = Requests([1.0] * rows, [64] * rows, seed=22)
    topk = gather_candidates(logits)
    expected = req.merge(topk)
    # The same requests behind a reversed slot mapping.
    perm = torch.randperm(rows, device=DEVICE)
    req.idx = perm
    req.temperature = req.temperature[perm.argsort()]
    req.top_k = req.top_k[perm.argsort()]
    req.seeds = req.seeds[perm.argsort()]
    assert torch.equal(req.merge(topk), expected)


def repeat_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """The rank-major candidates of one row as those of `rows` identical rows."""
    return x.view(TP, 1, -1).expand(TP, rows, -1).reshape(TP * rows, -1)


def z_score(chi2: float, dof: int) -> float:
    """Wilson-Hilferty normal approximation of a chi-square statistic."""
    return ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))


@pytest.mark.parametrize("temperature,top_k", [(1.0, 64), (0.8, 20), (1.0, 3)])
def test_draw_follows_the_top_k_distribution(temperature: float, top_k: int):
    """Chi-square of 400k draws of one row against softmax over its top_k tokens."""
    logits = make_logits(1, seed=77)
    topk = gather_candidates(logits)
    capped = soft_cap(logits)[0]
    values, ids = torch.topk(capped / temperature, top_k)
    probs = torch.softmax(values.double(), dim=0)
    batch, rounds = 50000, 8
    counts = torch.zeros(VOCAB, dtype=torch.int64, device=DEVICE)
    for i in range(rounds):
        req = Requests([temperature] * batch, [top_k] * batch, seed=100 + i)
        wide = TopKLogits(
            words=repeat_rows(topk.words, batch),
            capped=repeat_rows(topk.capped, batch),
            num_rows=batch,
            k=K,
            tp_size=TP,
            shard_size=topk.shard_size,
        )
        counts += torch.bincount(req.merge(wide), minlength=VOCAB)
    n = batch * rounds
    assert int(counts.sum()) == n
    assert int(counts[ids].sum()) == n, "a token outside the top_k set was drawn"
    observed = counts[ids].double()
    expected = probs * n
    # Merge the bins with fewer than 10 expected draws into one.
    big = expected >= 10
    obs = torch.cat([observed[big], observed[~big].sum().view(1)])
    exp = torch.cat([expected[big], expected[~big].sum().view(1)])
    keep = exp > 0
    chi2 = float(((obs[keep] - exp[keep]) ** 2 / exp[keep]).sum())
    dof = int(keep.sum()) - 1
    if dof >= 1:
        # Far looser than any plausible chance: a 5-sigma band.
        assert abs(z_score(chi2, dof)) < 5.0, (chi2, dof)


# --------------------------------------------------------------------------------
# CUDA graphs: no host sync, no data-dependent shape.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 7, 64, 128, 255, 256])
def test_path_replays_from_a_cuda_graph(rows: int):
    logits = make_logits(rows, seed=rows)
    half, odd = rows // 2, rows % 2
    req = Requests([0.0, 1.0] * half + [0.0] * odd, [VOCAB, 64] * half + [VOCAB] * odd)
    shard = VOCAB // TP
    static = logits.clone()

    def run() -> torch.Tensor:
        words = torch.cat(
            [
                local_topk_candidates(static[:, r * shard : (r + 1) * shard], K)
                for r in range(TP)
            ]
        ).view(torch.bfloat16)
        topk = TopKLogits(words, soft_cap(words), rows, K, TP, shard)
        return req.merge(topk)

    run()  # compile
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for seed in (1, 2):
        fresh = make_logits(rows, seed=1000 + seed)
        static.copy_(fresh)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, req.full_path(fresh))
