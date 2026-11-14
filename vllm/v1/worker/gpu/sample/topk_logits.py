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
   ``gumbel_sample``, with the noise keyed by the token id exactly as there. The mask
   keeps exactly top_k tokens, or every token tied with the top_k-th (see below).

Candidates are ordered by their bf16 logit, then by their token id, which is the order
of the soft-capped fp32 values with the lowest id first among equal ones, as in the full
path: the cap is non-decreasing over the whole bf16 range and strictly increasing for
|logit| < 816 (measured on the CUDA op of Muse Spark, cap 20 and multiplier 0.177: 816
and 820 are the first bf16 values that share a capped value; the cap reaches exactly
20.0 from 1020). So a greedy row returns the token ``gumbel_sample`` returns for the
full logits, unless more than K tokens of one shard have |logit| >= 816 (their capped
values can tie where their bf16 values do not): Muse logits are tens, not hundreds.

Ties at the top_k boundary. bf16 logits tie constantly (about 44 % of the real decode
rows of Muse Spark have an equal 64th and 65th logit), and the two samplers of the full
path treat a tie across the top_k-th place differently:

* the Gumbel-max route (a step with a greedy row or an explicit seed) keeps exactly
  ``top_k`` tokens, the lowest ids among the ones tied at the boundary
  (``apply_top_k_top_p``, Qrita). The candidate path does the same, with the same noise
  per (seed, position, token id): it returns the very token of the full path;
* FlashInfer's top-k sampler (any other step of top_k-only sampling) keeps *every*
  token tied with the top_k-th logit: a token is accepted when fewer than top_k tokens
  are strictly larger. The candidate path of such a step (``TopKLogits.threshold``)
  draws over the same set, with weights exp(logit / T), so its distribution is
  FlashInfer's (the random streams differ: unseeded draws of one distribution).

The second rule needs the whole tie group, which a rank's K candidates may not hold: a
rank lists the lowest indices among the tokens tied with its K-th logit, and a group
can have any size (all of the vocabulary, for equal logits). So the candidates close the
ties: a token tied with the K-th logit of a rank but not listed is *unlisted*; it is
in the kept set only if that K-th logit is the global top_k-th one (the candidates of
every rank have at most top_k - 1 tokens above it); all tokens tied with one logit have
one weight; and Gumbel-max over a group is the Gumbel-max over its token of the largest
noise, which only the owning rank can find (the noise is a function of seed, position
and token id, all known to it). So every rank sends, in the same gather, its best
unlisted tie (``fill_tie_closure``, 8 more words per row), and the merge counts it as
one more candidate of the weight of the group exactly when its rank's K-th logit equals
the top_k-th one. The kept set, the weights and so the distribution are those of the
full path for any tie group, without a second collective or a host sync, and
``tests/v1/worker/test_gpu_topk_logits_ties.py`` checks it on constructed ties (in
one shard, in every shard, beyond the lists, signed zeros, all logits equal) against the
real FlashInfer kernel, draw by draw against the Gumbel-max over the threshold set of
the full logits, and by chi-square. (FlashInfer's probabilities tie exactly where the
logits do while their gaps, at least 0.011 in the capped logit, stay above fp32
rounding after the division by the temperature, which is at most 2.)
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noise, gumbel_noised_argmax

# The widest shard row of the local kernel (a column is a 16-bit index of the candidate
# word).
MAX_SHARD_COLS = 65535
# The widest shard the gathered carrier takes: the index words travel as bf16 bits, and
# the indices 0 .. 0x7F7F are finite bf16 values (0x7F80 .. 0x7FFF and 0xFF80 .. 0xFFFF
# are Inf / NaN patterns, which a backend that converts or quiets NaNs would corrupt).
MAX_GATHER_SHARD_COLS = 0x7F80
# The widest candidate list of a row (TP * K) the merge kernel holds in registers.
MAX_CANDIDATES = 2048
# Words after the 2 * K candidate words of a row in the "threshold" payload: the best
# unlisted tie of the rank (shard-local index, presence flag) and padding, so that the
# row stays a multiple of 16 bytes. See ``fill_tie_closure``.
TIE_EXTRA_WORDS = 8
_INT32_MAX = tl.constexpr(0x7FFFFFFF)


@dataclass(slots=True)
class TopKLogits:
    """The gathered candidates of a batch, rank-major.

    ``words`` is the all-gather (dim 0) of the per-rank ``[rows, 2 * k (+ extra)]``
    words of ``local_topk_candidates``: row ``r * rows + t`` holds the k logits (bf16)
    of rank ``r`` for row ``t``, then the k shard-local indices (int16 bits), then, in a
    ``threshold`` batch, ``TIE_EXTRA_WORDS`` words of ``fill_tie_closure``. ``capped``
    is the soft cap of ``words`` as fp32 (only the first k columns of a row are
    logits).
    """

    # [tp * rows, 2 * k (+ extra)], bf16 carrier of int16 / bf16 bits
    words: torch.Tensor
    capped: torch.Tensor  # same shape, fp32
    num_rows: int
    k: int
    tp_size: int
    shard_size: int  # token ids per rank: id = rank * shard_size + local index
    # Sampling rule at the top_k boundary: False keeps exactly top_k tokens (the lowest
    # ids among the ones tied at the boundary), True keeps every token tied with the
    # top_k-th logit. The payload of the second carries the closure of the ties.
    threshold: bool = False

    @property
    def extra_words(self) -> int:
        return TIE_EXTRA_WORDS if self.threshold else 0


@dataclass(slots=True)
class TieDraw:
    """What the ranks need to draw among the ties they could not list: the Gumbel noise
    of a token is a function of the request's seed, the position and the token id.

    The seed of a request without one is a host numpy draw: identical on every TP rank
    because every worker seeds numpy alike (``set_random_seed``) and admits the
    requests in the same order, which the Gumbel-max route of the full path relies on
    as well."""

    expanded_idx_mapping: torch.Tensor  # [rows] request state index of every row
    seeds: torch.Tensor  # [max_num_reqs]
    pos: torch.Tensor  # [rows]
    use_fp64: bool = False


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

# Columns of one program of the tie closure.
TIE_BLOCK_COLS = 2048

# Columns of one program of the first pass: a row is cut into chunks, each reduced to
# its K largest, and the K * chunks candidates of a row to K by a second pass.
CHUNK_COLS = 2048


def local_topk_candidates(
    logits: torch.Tensor,
    k: int,
    num_valid: int | None = None,
    extra_words: int = 0,
) -> torch.Tensor:
    """The K largest logits of every row, with their column indices.

    Args:
        logits: ``[rows, cols]`` bf16, contiguous rows.
        k: candidates per row, at most ``num_valid``.
        num_valid: columns that are vocabulary (the rest is shard padding).
        extra_words: columns appended to every row for the caller (uninitialised).

    Returns:
        ``[rows, 2 * k + extra_words]`` int16: the k logits as bf16 bits, then their
        column indices. The k are the largest of the row, ties broken towards the lowest
        column; their order within the row is fixed but unspecified. No host sync, no
        data-dependent shape: safe to capture in a CUDA graph.
    """
    assert logits.dtype == torch.bfloat16 and logits.dim() == 2
    assert logits.stride(1) == 1
    rows, cols = logits.shape
    if num_valid is None:
        num_valid = cols
    assert 1 <= k <= num_valid <= cols <= MAX_SHARD_COLS
    out = torch.empty(
        (rows, 2 * k + extra_words), dtype=torch.int16, device=logits.device
    )
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


@triton.jit(do_not_specialize=["num_valid", "token_base"])
def _tie_partial_kernel(
    logits_ptr,
    logits_stride,
    words_ptr,
    words_stride,
    expanded_idx_mapping_ptr,
    seeds_ptr,
    pos_ptr,
    noise_ptr,
    arg_ptr,
    num_valid,
    token_base,
    K: tl.constexpr,
    K_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    USE_FP64: tl.constexpr,
):
    """The unlisted tie of the largest noise in one block of columns of one row."""
    row = tl.program_id(0).to(tl.int64)
    block = tl.program_id(1)
    j = tl.arange(0, K_POW2)
    in_list = j < K
    base = words_ptr + row * words_stride
    # The K-th key of the rank (the lowest of the list) and the highest index among the
    # listed tokens that have it: the listed ties are the lowest indices of the ties.
    list_key = _bf16_key(tl.load(base + j, mask=in_list, other=0), in_list)
    kappa = tl.min(tl.where(in_list, list_key, 65536), axis=0)
    list_idx = tl.load(base + K + j, mask=in_list, other=0).to(tl.int32) & 0xFFFF
    last_listed = tl.max(tl.where(in_list & (list_key == kappa), list_idx, -1), axis=0)

    offs = block * BLOCK + tl.arange(0, BLOCK)
    in_cols = offs < num_valid
    bits = tl.load(logits_ptr + row * logits_stride + offs, mask=in_cols, other=0.0).to(
        tl.int16, bitcast=True
    )
    unlisted = (_bf16_key(bits, in_cols) == kappa) & (offs > last_listed)
    if USE_FP64:
        best_noise = tl.full((), float("-inf"), tl.float64)
    else:
        best_noise = tl.full((), float("-inf"), tl.float32)
    best_idx = tl.full((), -1, tl.int32)
    if tl.max(unlisted.to(tl.int32), axis=0) > 0:
        req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
        is_valid_req = req_state_idx >= 0
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + row)
        noise = gumbel_noise(seed, pos, token_base + offs, USE_FP64)
        noise = tl.where(unlisted, noise, float("-inf"))
        best_noise, arg = tl.max(noise, axis=0, return_indices=True)
        best_idx = block * BLOCK + arg
    tl.store(noise_ptr + row * NUM_BLOCKS + block, best_noise)
    tl.store(arg_ptr + row * NUM_BLOCKS + block, best_idx)


@triton.jit
def _tie_reduce_kernel(
    noise_ptr,
    arg_ptr,
    words_ptr,
    words_stride,
    K: tl.constexpr,
    EXTRA: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCKS_POW2: tl.constexpr,
):
    """Word 0 and 1 of the extra words: the best of the per-block unlisted ties."""
    row = tl.program_id(0).to(tl.int64)
    b = tl.arange(0, BLOCKS_POW2)
    in_blocks = b < NUM_BLOCKS
    noise = tl.load(
        noise_ptr + row * NUM_BLOCKS + b, mask=in_blocks, other=float("-inf")
    )
    args = tl.load(arg_ptr + row * NUM_BLOCKS + b, mask=in_blocks, other=-1)
    # The first block of the largest noise holds the lowest index among equal noise.
    _, best = tl.max(noise, axis=0, return_indices=True)
    idx = tl.sum(tl.where(b == best, args, 0), axis=0)
    present = tl.max(args, axis=0) >= 0
    e = tl.arange(0, EXTRA)
    extra = tl.where(
        e == 0, tl.where(present, idx, 0), tl.where(e == 1, present.to(tl.int32), 0)
    )
    tl.store(words_ptr + row * words_stride + 2 * K + e, extra.to(tl.int16))


def fill_tie_closure(
    logits: torch.Tensor,
    words: torch.Tensor,
    k: int,
    ties: TieDraw,
    token_base: int,
    num_valid: int | None = None,
) -> None:
    """Writes the ``TIE_EXTRA_WORDS`` words that close the ties of a rank's candidates.

    The rank lists the k largest of its shard, the lowest indices among equal values. A
    token tied with the rank's k-th logit that did not make the list is *unlisted*: a
    sampler that keeps every token tied with the top_k-th logit (FlashInfer's rule)
    must still be able to draw it, however many there are. Its Gumbel noise is known to
    this rank alone (a function of seed, position and token id), and the tokens tied
    with one logit have one weight, so the whole group is represented, exactly, by the
    unlisted token with the largest noise: its index (word 0) and whether there is one
    (word 1). ``topk_merge_sample`` counts it as a candidate whenever the rank's k-th
    logit is the global top_k-th one (the only case in which the unlisted ties are in
    the kept set).

    Args:
        logits: the ``[rows, cols]`` shard the candidates were taken from.
        words: ``[rows, 2 * k + TIE_EXTRA_WORDS]`` int16 from ``local_topk_candidates``.
        token_base: token id of column 0 of the shard (rank * shard size).
    """
    rows, cols = logits.shape
    if num_valid is None:
        num_valid = cols
    assert words.shape == (rows, 2 * k + TIE_EXTRA_WORDS) and words.dtype == torch.int16
    assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
    if rows == 0:
        return
    block = TIE_BLOCK_COLS
    num_blocks = triton.cdiv(num_valid, block)
    scratch_dtype = torch.float64 if ties.use_fp64 else torch.float32
    best_noise = torch.empty(
        (rows, num_blocks), dtype=scratch_dtype, device=logits.device
    )
    best_idx = torch.empty((rows, num_blocks), dtype=torch.int32, device=logits.device)
    _tie_partial_kernel[(rows, num_blocks)](
        logits,
        logits.stride(0),
        words,
        words.stride(0),
        ties.expanded_idx_mapping.contiguous(),
        ties.seeds,
        ties.pos.contiguous(),
        best_noise,
        best_idx,
        num_valid,
        token_base,
        K=k,
        K_POW2=triton.next_power_of_2(k),
        BLOCK=block,
        NUM_BLOCKS=num_blocks,
        USE_FP64=ties.use_fp64,
        num_warps=4,
    )
    _tie_reduce_kernel[(rows,)](
        best_noise,
        best_idx,
        words,
        words.stride(0),
        K=k,
        EXTRA=TIE_EXTRA_WORDS,
        NUM_BLOCKS=num_blocks,
        BLOCKS_POW2=triton.next_power_of_2(num_blocks),
        num_warps=1,
    )


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
    ROW_WORDS: tl.constexpr,
):
    """Candidates ``j`` of ``row``: soft-capped value, token id, bf16 order key."""
    rank = j // K
    col = j - rank * K
    in_range = j < TP * K
    base = (rank * num_rows + row).to(tl.int64) * ROW_WORDS
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
    TP_POW2: tl.constexpr,
    K: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
    THRESHOLD: tl.constexpr,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    is_valid_req = req_state_idx >= 0
    temp = tl.load(temperature_ptr + req_state_idx, mask=is_valid_req, other=0.0).to(
        tl.float32
    )

    j = tl.arange(0, NUM_CANDIDATES)
    value, token_id, in_range, key = _load_candidates(
        words_ptr, capped_ptr, j, row, num_rows, shard_size, TP, K, ROW_WORDS
    )
    if temp == 0.0:
        # The lowest token id among the largest values, as the argmax of the full path.
        best = tl.max(value, axis=0)
        at_best = in_range & (value == best)
        token = tl.min(tl.where(at_best, token_id, _INT32_MAX), axis=0)
    else:
        # The temperature as the full path divides by it: not at all for 1.0.
        value = tl.where(temp != 1.0, value / temp, value)
        top_k = tl.minimum(tl.load(top_k_ptr + req_state_idx), TP * K)
        if THRESHOLD:
            # Every token with at least the top_k-th largest logit (bf16 order, which
            # is that of the capped, scaled values).
            threshold = _kth_largest_key(key, top_k)
            keep = in_range & (key >= threshold)
        else:
            # The top_k largest by the bf16 order of the logits, which is that of the
            # capped, scaled values. Each rank lists equal logits by ascending token
            # id, and the ranks by ascending id: the first of equal candidates has the
            # lowest id.
            keep, _ = _select_top(key, top_k)
        seed = tl.load(seeds_ptr + req_state_idx, mask=is_valid_req, other=0)
        pos = tl.load(pos_ptr + row)
        score, winner = gumbel_noised_argmax(
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
        if THRESHOLD:
            # The ties a rank could not list: its k-th logit is the global top_k-th one
            # only if they are in the kept set, and then the unlisted token of the
            # largest noise (word 0 of the extra words, if word 1) stands for them all.
            r = tl.arange(0, TP_POW2)
            in_tp = r < TP
            base = (r * num_rows + row).to(tl.int64) * ROW_WORDS
            kth_bits = tl.load(words_ptr + base + (K - 1), mask=in_tp, other=0)
            relevant = (
                in_tp
                & (tl.load(words_ptr + base + 2 * K + 1, mask=in_tp, other=0) != 0)
                & (_bf16_key(kth_bits, in_tp) == threshold)
            )
            tie_value = tl.load(
                capped_ptr + base + (K - 1), mask=in_tp, other=float("-inf")
            )
            tie_value = tl.where(temp != 1.0, tie_value / temp, tie_value)
            tie_local = tl.load(words_ptr + base + 2 * K, mask=in_tp, other=0)
            tie_id = r * shard_size + (tie_local.to(tl.int32) & 0xFFFF)
            tie_score, tie_winner = gumbel_noised_argmax(
                tie_value,
                tie_id,
                relevant,
                seed,
                pos,
                temp,
                IS_DRAFTING=False,
                USE_FP64=USE_FP64,
                APPLY_TEMPERATURE=False,
            )
            tie_token = tl.sum(tl.where(r == tie_winner, tie_id, 0), axis=0)
            wins = (tie_score > score) | ((tie_score == score) & (tie_token < token))
            token = tl.where(wins, tie_token, token)
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
    (the caller checks: a larger top_k would be silently cut to the candidates). A
    ``topk.threshold`` batch has no greedy row. Returns ``[rows]`` int64 token ids. No
    host sync: safe to capture in a CUDA graph.
    """
    rows, k, tp = topk.num_rows, topk.k, topk.tp_size
    num_candidates = triton.next_power_of_2(tp * k)
    assert tp * k <= MAX_CANDIDATES
    row_words = 2 * k + topk.extra_words
    assert topk.words.is_contiguous() and topk.capped.is_contiguous()
    assert topk.words.shape == (tp * rows, row_words) == tuple(topk.capped.shape)
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
        TP_POW2=triton.next_power_of_2(tp),
        K=k,
        ROW_WORDS=row_words,
        NUM_CANDIDATES=num_candidates,
        USE_FP64=use_fp64,
        THRESHOLD=topk.threshold,
        num_warps=MERGE_NUM_WARPS,
    )
    return sampled


# Level 4 of VLLM_MUSE_SPARK_SM120_VERIFY_TOPK_LOGITS verifies a top_p request from the
# candidates only when its temperature, top_p and top_k (host state) are within these
# bounds (``Sampler.topp_candidate_class``); the other top_p requests keep the full
# path.
#
# The candidate kernel's kept set differs from the full path's top-k + top-p kernel
# (Qrita) at a nucleus boundary that splits a group of tied tokens: Qrita trims the
# group by an fp32 count and a round-trip test that depend on the bits of a sum over
# the whole row, which the candidates do not have. The two then differ by members of
# the group, in either direction, deterministically and identically on all ranks. No
# class has a zero rate. Audit on 43,668 real decode rows (two independent captures,
# 22,776 and 20,892, in contexts of temperature 0.6-1.5, flat rows included), kept sets
# compared row by row against Qrita: a differing row in 7e-5 (top_k 64) to 1.2e-4
# (top_k 40) of the rows at the corner temperature 1, top_p 0.95, fewer below it. Above
# it the rate grows: top_p 0.97 3e-4, temperature 1.25 at top_p 0.95 4e-4, temperature
# 1.5 1.2e-3, top_p 0.99 1.5e-3, temperature 2 3e-3.
#
# Size of the difference inside the class (19 diagnosed cases): 17 of them are 1-3
# tokens of one tie group, 0.1-0.5 % of the row's probability each, a total variation
# of 0.0035-0.0093 for the row. The other 2 are an exact tie: at a low temperature two
# exactly tied top tokens carry the whole mass and top_p equals the mass of one of them
# (temperature 0.1, top_p 0.5, top_k 64). The candidates keep both (the exact float64
# nucleus) and Qrita keeps one, decided by the bits of its sum: the total variation of
# that row is 0.5. That is 2 of 43,668 rows (4.6e-5) at that cell, an expected total
# variation of about 2e-5 per row there against about 7e-7 at the temperature 1 / top_p
# 0.95 corner. No host state sees the case: the bounds below do not exclude it.
#
# The bounds are the largest audited class with a worst-cell rate of about 1e-4 per
# row. The audited grid is temperature 0.05-2, top_p 0.5-0.99, top_k 10-64 at K = 64;
# top_p below 0.5, temperature below 0.05 and (with a raised TOPK_LOGITS_K) top_k above
# 64 are inside the class but were not audited. The lower bound on top_k is a margin:
# real rows of top_k 10 to 32 differ as rarely, but synthetic rows of almost equal
# logits (tie groups of tens of tokens across the k-th place) differ on 0.3-0.6 % of the
# rows at top_k 10 and 20, and on none at 40 and 64.
TOPP_CANDIDATE_MAX_TEMPERATURE = 1.0
TOPP_CANDIDATE_MAX_TOP_P = 0.95
TOPP_CANDIDATE_MIN_TOP_K = 40


@dataclass(slots=True)
class VerifyPlan:
    """How a speculative step verifies its drafts when requests may take different
    routes (``Sampler.topk_verify_split_plan``): every field is host state, identical
    on every TP rank.

    ``reason`` is set when no request can verify from the candidates: the whole step
    takes the full path. Otherwise ``candidate`` lists the batch positions of the
    requests that verify from the ``k`` candidates per rank (None: all of them) and
    ``full`` those whose logits are gathered whole (None: none); ``full_reasons``
    counts those requests by reason. ``has_sampled``, ``min_p`` and ``top_p`` say
    whether a candidate request needs the sampled branch, a min_p threshold, a top-p
    nucleus (they select the compiled variant of the verification kernel).
    """

    reason: str | None
    k: int = 0
    candidate: np.ndarray | None = None
    full: np.ndarray | None = None
    full_reasons: dict[str, int] = field(default_factory=dict)
    has_sampled: bool = False
    min_p: bool = False
    top_p: bool = False


class RouteStats:
    """Host-side counts of the steps that sampled from the top-K logits, and of the
    steps that gathered the logits instead, by the reason.

    ``record`` counts one step and returns a line to log when one is due: for the first
    step, for a reason seen for the first time, and then at most once per
    ``interval_s``. Plain Python integers: no device sync, and every TP rank counts the
    same steps.
    """

    def __init__(
        self,
        k: int,
        interval_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
        label: str = "sampled steps",
        threshold_note: bool = True,
        action: str = "drew from",
    ) -> None:
        self.k = k
        self.label = label
        self.action = action
        self.threshold_note = threshold_note
        self.taken = 0
        self.taken_threshold = 0  # of the taken steps, those that keep every tie
        self.fallbacks: dict[str, int] = {}
        # Steps taken with some requests on the full logits (per-request routing),
        # those requests and the requests of those steps, and why.
        self.split_steps = 0
        self.split_full_reqs = 0
        self.split_reqs = 0
        self.split_reasons: dict[str, int] = {}
        self._interval_s = interval_s
        self._clock = clock
        self._next_log = 0.0

    @property
    def steps(self) -> int:
        return self.taken + sum(self.fallbacks.values())

    def summary(self) -> str:
        steps = self.steps
        share = 100.0 * self.taken / steps
        line = (
            f"{self.taken} of {steps} {self.label} ({share:.1f}%) {self.action} the "
            f"top-{self.k} logits of every TP rank"
        )
        if self.threshold_note:
            line += (
                f" ({self.taken_threshold} of them on FlashInfer's rule at the top_k "
                "boundary, the others on exactly top_k tokens)"
            )
        if self.split_steps:
            reasons = ", ".join(
                f"{r}: {n}" for r, n in sorted(self.split_reasons.items())
            )
            line += (
                f"; {self.split_steps} of them with {self.split_full_reqs} of "
                f"{self.split_reqs} requests on the full logits ({reasons})"
            )
        if self.fallbacks:
            reasons = ", ".join(f"{r}: {n}" for r, n in sorted(self.fallbacks.items()))
            line += f"; the others gathered the logits ({reasons})"
        return line

    def record(
        self,
        reason: str | None,
        threshold: bool = False,
        split: dict[str, int] | None = None,
        num_reqs: int = 0,
    ) -> str | None:
        """Count a step that gathered the logits for ``reason``, or (None) did not.

        ``split`` (reason -> requests): the step verified from the candidates but the
        logits of these requests were gathered whole, ``num_reqs`` requests in all.
        """
        new_reason = False
        if reason is None:
            self.taken += 1
            self.taken_threshold += threshold
            if split:
                self.split_steps += 1
                self.split_full_reqs += sum(split.values())
                self.split_reqs += num_reqs
                for r, n in split.items():
                    new_reason |= r not in self.split_reasons
                    self.split_reasons[r] = self.split_reasons.get(r, 0) + n
        else:
            new_reason = reason not in self.fallbacks
            self.fallbacks[reason] = self.fallbacks.get(reason, 0) + 1
        now = self._clock()
        if self.steps == 1 or new_reason or now >= self._next_log:
            self._next_log = now + self._interval_s
            return self.summary()
        return None
