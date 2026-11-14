# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sampling from the top-K logits of every vocab-parallel rank.

The full path gathers the whole ``[rows, vocab]`` logits over the tensor-parallel
group (50 KB per row and rank at a 25k shard) only to take an argmax or a top-k
Gumbel-max draw from it. These kernels exchange the K largest logits of each shard
instead:

1. ``local_topk_candidates`` keeps, per row, the K largest bf16 logits of the local
   shard (the lowest indices among equal values) as ``2 * K`` 16-bit words: the K
   logits and their shard-local indices.
2. The model gathers the words over the TP group and soft-caps them with the very op
   of its full path (an elementwise op, so every candidate gets the bits the full
   path gives its logit).
3. ``topk_merge_sample`` draws the token of each row from its ``TP * K`` candidates:
   argmax with the lowest token id among equal logits for a greedy row, and for a
   row with temperature > 0 the top-k mask followed by the Gumbel-max draw of
   ``gumbel_sample``, with the noise keyed by the token id exactly as there.

Candidates are ordered by their bf16 logit, then by their token id, which is the order
of the soft-capped fp32 values (the cap is strictly increasing on the bf16 range of the
logits: for the cap of 20 and the multiplier of 0.177 of Muse Spark, up to |logit| of
about 900) with the lowest id first among equal ones, as in the full path. So a greedy
row returns the token ``gumbel_sample`` returns for the full logits, and a row with
top_k <= K keeps the same ``top_k`` tokens as ``apply_top_k_top_p`` (the lowest ids win
a tie at the boundary: every shard keeps the K lowest-indexed of its ties, and the
top_k <= K that the merge needs are among them), draws the same noise for them and so
returns the same token.
"""

from dataclasses import dataclass

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax

# The widest shard row (a column is a 16-bit index of the candidate word).
MAX_SHARD_COLS = 65535
# The widest candidate list of a row (TP * K) the merge kernel holds in registers.
MAX_CANDIDATES = 2048
_INT32_MAX = tl.constexpr(0x7FFFFFFF)


@dataclass(slots=True)
class TopKLogits:
    """The gathered candidates of a batch, rank-major.

    ``words`` is the all-gather (dim 0) of the per-rank ``[rows, 2 * k]`` words of
    ``local_topk_candidates``: row ``r * rows + t`` holds the k logits (bf16) of rank
    ``r`` for row ``t``, then the k shard-local indices (int16 bits). ``capped`` is the
    soft cap of ``words`` as fp32 (only its first k columns of a row are logits).
    """

    words: torch.Tensor  # [tp * rows, 2 * k], bf16 carrier of int16 / bf16 bits
    capped: torch.Tensor  # [tp * rows, 2 * k], fp32
    num_rows: int
    k: int
    tp_size: int
    shard_size: int  # token ids per rank: id = rank * shard_size + local index


@triton.jit
def _bf16_key(raw_bits, mask):
    """An order-preserving key in [0, 65535] of bf16 bits (-1 where masked out).

    -0.0 and +0.0 share a key, as they compare equal.
    """
    bits = raw_bits.to(tl.int32) & 0xFFFF
    bits = tl.where(bits == 0x8000, 0, bits)
    key = tl.where(bits >= 0x8000, 0xFFFF - bits, bits + 0x8000)
    return tl.where(mask, key, -1)


@triton.jit
def _kth_largest_key(key, k):
    """The largest t with at least k keys >= t (0 if fewer than k keys are >= 0)."""
    lo = 0
    hi = 65536
    for _ in tl.static_range(16):
        mid = (lo + hi) >> 1
        enough = tl.sum((key >= mid).to(tl.int32), axis=0) >= k
        lo = tl.where(enough, mid, lo)
        hi = tl.where(enough, hi, mid)
    return lo


@triton.jit
def _select_top(key, k):
    """The k largest keys of a vector: every key above the k-th, then, in vector
    order, as many of the keys equal to it as fill the k places.

    Returns the selection mask and the place of each selected element in a dense
    output: the elements above the k-th key first, then the equal ones, each group in
    vector order. Two keys are equal when they are, so a vector that lists equal keys
    by ascending index keeps the lowest indices.
    """
    kth = _kth_largest_key(key, k)
    above = key > kth
    equal = key == kth
    num_above = tl.sum(above.to(tl.int32), axis=0)
    # One scan counts both sets: the keys above in the low half, the equal in the high.
    count = tl.cumsum(above.to(tl.int32) + (equal.to(tl.int32) << 16), axis=0)
    equal_rank = count >> 16
    selected = above | (equal & (equal_rank <= k - num_above))
    place = tl.where(above, (count & 0xFFFF) - 1, num_above + equal_rank - 1)
    return selected, place


@triton.jit(do_not_specialize=["num_valid"])
def _chunk_topk_kernel(
    logits_ptr,
    logits_stride,
    chunks_ptr,
    num_valid,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    CHUNK: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    ALIGNED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    if ALIGNED:
        logits_stride = tl.multiple_of(logits_stride, 8)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < num_valid
    raw_bits = tl.load(
        logits_ptr + row * logits_stride + offs, mask=mask, other=0.0
    ).to(tl.int16, bitcast=True)
    selected, place = _select_top(_bf16_key(raw_bits, mask), K)
    # A candidate word: the logit's bf16 bits, then the column.
    words = (raw_bits.to(tl.int32) << 16) | (offs & 0xFFFF)
    out = chunks_ptr + (row * NUM_CHUNKS + chunk) * K
    tl.store(out + place, words, mask=selected)
    # A chunk with fewer than K columns leaves places to fill: with all ones (a NaN
    # of the lowest key, below any logit).
    unused = tl.arange(0, K_POW2)
    num_selected = tl.sum(selected.to(tl.int32), axis=0)
    tl.store(out + unused, -1, mask=(unused >= num_selected) & (unused < K))


@triton.jit
def _merge_chunks_kernel(
    chunks_ptr,
    out_ptr,
    out_stride,
    K: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    NUM_CHUNK_WORDS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, NUM_CANDIDATES)
    mask = j < NUM_CHUNK_WORDS
    words = tl.load(chunks_ptr + row * NUM_CHUNK_WORDS + j, mask=mask, other=-1)
    raw_bits = (words >> 16).to(tl.int16)
    selected, place = _select_top(_bf16_key(raw_bits, mask), K)
    tl.store(out_ptr + row * out_stride + place, raw_bits, mask=selected)
    tl.store(
        out_ptr + row * out_stride + K + place,
        (words & 0xFFFF).to(tl.int16),
        mask=selected,
    )


# Warps of the program that draws the token of a row from its candidates.
MERGE_NUM_WARPS = 4

# Columns of one program of the first pass: a row is cut into chunks, each reduced to
# its K largest, and the K * chunks candidates of a row to K by a second pass.
CHUNK_COLS = 2048


def local_topk_candidates(
    logits: torch.Tensor,
    k: int,
    num_valid: int | None = None,
) -> torch.Tensor:
    """The K largest logits of every row, with their column indices.

    Args:
        logits: ``[rows, cols]`` bf16, contiguous rows.
        k: candidates per row, at most ``num_valid``.
        num_valid: columns that are vocabulary (the rest is shard padding).

    Returns:
        ``[rows, 2 * k]`` int16: the k logits as bf16 bits, then their column indices.
        The k are the largest of the row, ties broken towards the lowest column; their
        order within the row is fixed but unspecified. No host sync, no data-dependent
        shape: safe to capture in a CUDA graph.
    """
    assert logits.dtype == torch.bfloat16 and logits.dim() == 2
    assert logits.stride(1) == 1
    rows, cols = logits.shape
    if num_valid is None:
        num_valid = cols
    assert 1 <= k <= num_valid <= cols <= MAX_SHARD_COLS
    out = torch.empty((rows, 2 * k), dtype=torch.int16, device=logits.device)
    if rows == 0:
        return out
    num_chunks = triton.cdiv(num_valid, CHUNK_COLS)
    chunks = torch.empty(
        (rows, num_chunks * k), dtype=torch.int32, device=logits.device
    )
    _chunk_topk_kernel[(rows, num_chunks)](
        logits,
        logits.stride(0),
        chunks,
        num_valid,
        K=k,
        K_POW2=triton.next_power_of_2(k),
        CHUNK=CHUNK_COLS,
        NUM_CHUNKS=num_chunks,
        ALIGNED=logits.stride(0) % 8 == 0 and logits.data_ptr() % 16 == 0,
        num_warps=4,
    )
    _merge_chunks_kernel[(rows,)](
        chunks,
        out,
        out.stride(0),
        K=k,
        NUM_CANDIDATES=triton.next_power_of_2(num_chunks * k),
        NUM_CHUNK_WORDS=num_chunks * k,
        num_warps=8,
    )
    return out


@triton.jit
def _load_candidates(
    words_ptr,
    capped_ptr,
    j,
    row,
    num_rows,
    shard_size,
    TP: tl.constexpr,
    K: tl.constexpr,
):
    """Candidates ``j`` of ``row``: soft-capped value, token id, bf16 order key."""
    rank = j // K
    col = j - rank * K
    in_range = j < TP * K
    base = (rank * num_rows + row).to(tl.int64) * (2 * K)
    value = tl.load(capped_ptr + base + col, mask=in_range, other=float("-inf"))
    raw_bits = tl.load(words_ptr + base + col, mask=in_range, other=0)
    local = tl.load(words_ptr + base + K + col, mask=in_range, other=0)
    token_id = rank * shard_size + (local.to(tl.int32) & 0xFFFF)
    # A NaN logit never wins, as an absent candidate.
    value = tl.where(value != value, float("-inf"), value)
    token_id = tl.where(in_range, token_id, _INT32_MAX)
    return value, token_id, in_range, _bf16_key(raw_bits, in_range)


@triton.jit(do_not_specialize=["num_rows"])
def _topk_merge_sample_kernel(
    sampled_ptr,
    words_ptr,
    capped_ptr,
    expanded_idx_mapping_ptr,
    temperature_ptr,
    top_k_ptr,
    seeds_ptr,
    pos_ptr,
    num_rows,
    shard_size,
    TP: tl.constexpr,
    K: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temperature_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )

    j = tl.arange(0, NUM_CANDIDATES)
    value, token_id, in_range, key = _load_candidates(
        words_ptr, capped_ptr, j, row, num_rows, shard_size, TP, K
    )
    if temp == 0.0:
        # The lowest token id among the largest values, as the argmax of the full path.
        best = tl.max(value, axis=0)
        at_best = in_range & (value == best)
        token = tl.min(tl.where(at_best, token_id, _INT32_MAX), axis=0)
    else:
        # The temperature as the full path divides by it: not at all for 1.0.
        value = tl.where(temp != 1.0, value / temp, value)
        # The top_k largest by the bf16 order of the logits, which is that of the
        # capped, scaled values. Each rank lists equal logits by ascending token id,
        # and the ranks by ascending id: the first of equal candidates has the lowest
        # id.
        top_k = tl.minimum(tl.load(top_k_ptr + req_state_idx), TP * K)
        keep, _ = _select_top(key, top_k)
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + row)
        _, winner = gumbel_noised_argmax(
            value,
            token_id,
            keep,
            seed,
            pos,
            temp,
            IS_DRAFTING=False,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
        token = tl.sum(tl.where(j == winner, token_id, 0), axis=0)
    tl.store(sampled_ptr + row, token.to(tl.int64))


def topk_merge_sample(
    topk: TopKLogits,
    expanded_idx_mapping: torch.Tensor,  # [rows]
    temperature: torch.Tensor,  # [max_num_reqs]
    top_k: torch.Tensor,  # [max_num_reqs]
    seeds: torch.Tensor,  # [max_num_reqs]
    pos: torch.Tensor,  # [rows]
    use_fp64: bool = False,
) -> torch.Tensor:
    """Sample one token per row from the gathered candidates.

    Every row of temperature 0 is greedy; the others must have ``top_k <= topk.k``
    (the caller checks: a larger top_k would be silently cut to the candidates).
    Returns ``[rows]`` int64 token ids. No host sync: safe to capture in a CUDA graph.
    """
    rows, k, tp = topk.num_rows, topk.k, topk.tp_size
    num_candidates = triton.next_power_of_2(tp * k)
    assert tp * k <= MAX_CANDIDATES
    assert topk.words.is_contiguous() and topk.capped.is_contiguous()
    assert topk.words.shape == (tp * rows, 2 * k) == tuple(topk.capped.shape)
    sampled = torch.empty(rows, dtype=torch.int64, device=topk.words.device)
    if rows == 0:
        return sampled
    _topk_merge_sample_kernel[(rows,)](
        sampled,
        topk.words.view(torch.int16),
        topk.capped,
        expanded_idx_mapping.contiguous(),
        temperature,
        top_k,
        seeds,
        pos.contiguous(),
        rows,
        topk.shard_size,
        TP=tp,
        K=k,
        NUM_CANDIDATES=num_candidates,
        USE_FP64=use_fp64,
        num_warps=MERGE_NUM_WARPS,
    )
    return sampled
