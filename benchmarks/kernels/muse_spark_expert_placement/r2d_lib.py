"""Cost models, data access, annealer (numba) and evaluators of the R2D static expert placement analysis.

A placement of one layer is rank_of[256] (EP rank of every original expert id, 32 experts per rank); the linear production
placement is rank_of[e] = e // 32.

Cost models (all in ms per layer):
  prefill (NVFP4 layers 1..60): the CUTLASS grouped GEMM1 + GEMM2 of a rank take  base + 11.94 us x sum_e ceil(routes_e / 128)
      (fit within each (step, layer, microbatch) group of the natpre profile traces x routing of the same prompts, R2 within groups
      0.84, reproduces the measured sum of (max - mean) over ranks to +-5 %); BF16 layers 0 / 61 (Triton fused_moe): 42.6 us per tile.
      A microbatch = 4096 consecutive prefill rows (the production split of an 8192-token chunk), so a chunk = 2 microbatches.
  decode c1: muse kernel time on the slowest rank by its number k of local experts (S1-5 / R0 table): 2.8 / 26.5 / 42.4 / 62.0 us,
      then +20.2 us per expert; BF16 layers: KAPPA1_BF us per local expert of the slowest rank.
  decode c64 / c128: 3.0 us + 19.0 (c64) / 20.3 (c128) us per distinct local expert of the slowest rank (S1-6 L3 kbench real-trace
      rank times: T=64 340.6 us at 17.8 distinct and 419.6 us at 21.8; T=128 472.7 us at 24.0 and 546.1 us at 26.7); BF16 layers:
      KAPPA64_BF us per distinct expert.
"""
import json
import os

import numba as nb
import numpy as np

NL, NE, NR, PR = 62, 256, 8, 32
LIN = np.repeat(np.arange(NR), PR).astype(np.int64)
BF16_LAYERS = (0, 61)
NV_LAYERS = np.arange(1, 61)
TILE = 128
US_TILE = 11.94
US_TILE_BF = 42.62
T_C1_US = np.array([2.8, 26.5, 42.4, 62.0] + [62.0 + 20.2 * i for i in range(1, 6)])  # k = 0..8
US64_FIXED, US64_SLOPE, US128_SLOPE = 3.0, 19.0, 20.3
KAPPA1_BF, KAPPA64_BF = 31.5, 58.0  # us per (distinct) local expert of the slowest rank, BF16 layers: profile_s16r2_real BF16 buckets 0.16 ms (c1, E[k_max] 2.54) and 2.57 ms (c64 real, E[d_max] 22.1) for both layers
CHUNK_MBS = 2  # microbatches per 8192-token chunk
# step times that define "equal relative weight" (production, profile_s16r2_real): c1 8.623 ms, c64 real plateau 41.98 ms,
# natural 8192 chunk 721.7 ms
BASE_MS = dict(prefill=721.7, c1=8.623, c64=41.977)


def g1_table(layer):
    if layer in BF16_LAYERS:
        return (KAPPA1_BF * np.arange(9) / 1000.0).astype(np.float32)
    return (T_C1_US / 1000.0).astype(np.float32)


def g64_table(layer, slope=None):
    d = np.arange(PR + 1)
    if layer in BF16_LAYERS:
        return (KAPPA64_BF * d / 1000.0).astype(np.float32)
    return ((US64_FIXED + (US64_SLOPE if slope is None else slope) * d) / 1000.0).astype(np.float32)


def tile_ms(layer):
    """ms per 128-row tile, per chunk (two microbatches share one placement: the per-microbatch term is multiplied by CHUNK_MBS)."""
    return (US_TILE_BF if layer in BF16_LAYERS else US_TILE) / 1000.0


def load_domain(agg_dir, dom):
    z = np.load(os.path.join(agg_dir, dom + ".npz"))
    d = {k: z[k] for k in z.files if k != "meta"}
    d["meta"] = json.loads(str(z["meta"]))
    return d


def tiles(cnt):
    """cnt [..., 256] -> ceil(cnt / TILE) as float32."""
    return np.ceil(cnt.astype(np.float32) / TILE)


def onehot(rank_of):
    o = np.zeros((NE, NR), dtype=np.float32)
    o[np.arange(NE), rank_of] = 1.0
    return o


# ------------------------------------------------------------------------------------------------ batches of decode rows
def shuffled_presence(dec_ids_layers, T, nb, rng, layers=None):
    """Random batches of T decode rows (with replacement, one row set for all layers) -> presence [nb, L, 256] uint8."""
    n = dec_ids_layers.shape[1]
    idx = rng.integers(0, n, size=(nb, T))
    L = dec_ids_layers.shape[0] if layers is None else len(layers)
    out = np.zeros((nb, L, NE), dtype=np.uint8)
    rows = np.repeat(np.arange(nb), T * 8)
    for j in range(L):
        l = j if layers is None else layers[j]
        ids = dec_ids_layers[l][idx].reshape(nb, -1)
        out[rows, j, ids.ravel()] = 1
    return out


# ------------------------------------------------------------------------------------------------ evaluators (numpy)
def eval_prefill(cnt_w, rank_of, layer):
    """cnt_w [W, 256] routes per expert of W microbatches -> dict(ms_chunk: expected max_r t per chunk (model), gap_chunk: expected
    (max - mean) per chunk, mm_win: mean over windows of max/mean routes, loads [W, 8])."""
    if cnt_w.shape[0] == 0:
        return None
    oh = onehot(rank_of)
    loads = cnt_w.astype(np.float32) @ oh
    t = tiles(cnt_w) @ oh * tile_ms(layer)  # [W, 8] ms per microbatch tile cost
    mx, mean = t.max(1), t.mean(1)
    return dict(ms_max=float(mx.mean() * CHUNK_MBS), gap=float((mx - mean).mean() * CHUNK_MBS),
                mm_win=float((loads.max(1) / loads.mean(1)).mean()), loads_sum=loads.sum(0))


def eval_c1(ids, rank_of, layer):
    """ids [N, 8] (one decode token per row) -> dict(emax, ms)"""
    r = rank_of[ids.astype(np.int64)]  # [N, 8]
    k = (r[:, :, None] == np.arange(NR)[None, None, :]).sum(1)  # [N, 8 ranks]
    kmax = k.max(1)
    g = g1_table(layer)
    return dict(emax=float(kmax.mean()), ms=float(g[kmax].mean()))


def eval_batches(pres, rank_of, layer, slope=None):
    """pres [B, 256] presence of experts in B decode batches -> dict(dmax, ms) (distinct experts on the slowest rank)"""
    if pres.shape[0] == 0:
        return None
    d = pres.astype(np.float32) @ onehot(rank_of)
    dmax = d.max(1)
    g = g64_table(layer, slope)
    return dict(dmax=float(dmax.mean()), dmean=float(d.mean()), ms=float(g[dmax.astype(np.int64)].mean()))


# ------------------------------------------------------------------------------------------------ annealer
@nb.njit(nogil=True, cache=False)
def _prefill_obj(Sp, swp):
    nw = Sp.shape[0]
    tot = 0.0
    for w in range(nw):
        m = Sp[w, 0]
        for r in range(1, 8):
            if Sp[w, r] > m:
                m = Sp[w, r]
        tot += swp[w] * m
    return tot


@nb.njit(nogil=True, cache=False)
def _anneal(rank0, WpT, swp, lam_p, toks, tok_off, tok_idx, sw1, g1, lam1, PT, swb, g64, lam64, iters, T0, T1, seed,
            greedy_iters):
    np.random.seed(seed)
    nw = WpT.shape[1]
    N = toks.shape[0]
    nbt = PT.shape[1]
    rank_of = rank0.copy()
    members = np.zeros((8, 32), dtype=np.int64)
    cnt = np.zeros(8, dtype=np.int64)
    for e in range(256):
        r = rank_of[e]
        members[r, cnt[r]] = e
        cnt[r] += 1
    # prefill state
    Sp = np.zeros((nw, 8))
    for e in range(256):
        for w in range(nw):
            Sp[w, rank_of[e]] += WpT[e, w]
    # c1 state
    K = np.zeros((N, 8), dtype=np.int16)
    for t in range(N):
        for j in range(8):
            K[t, rank_of[toks[t, j]]] += 1
    stamp = np.zeros(N, dtype=np.int64)
    tmp_t = np.zeros(N, dtype=np.int64)
    tmp_old = np.zeros(N)
    # c64 state
    D = np.zeros((nbt, 8), dtype=np.int16)
    for e in range(256):
        for b in range(nbt):
            D[b, rank_of[e]] += PT[e, b]
    cost_p = _prefill_obj(Sp, swp)
    cost_1 = 0.0
    for t in range(N):
        m = 0
        for r in range(8):
            if K[t, r] > m:
                m = K[t, r]
        cost_1 += sw1[t] * g1[m]
    cost_64 = 0.0
    for b in range(nbt):
        m = 0
        for r in range(8):
            if D[b, r] > m:
                m = D[b, r]
        cost_64 += swb[b] * g64[m]
    J = lam_p * cost_p + lam1 * cost_1 + lam64 * cost_64
    best = rank_of.copy()
    bestJ = J
    best_terms = np.array([cost_p, cost_1, cost_64])
    stampc = 0
    Sp_new = np.zeros((nw, 8))
    total = iters + greedy_iters
    ratio = (T1 / T0) ** (1.0 / max(iters, 1))
    temp = T0
    for it in range(total):
        greedy = it >= iters
        if not greedy:
            temp *= ratio
        else:
            temp = 0.0
        ra = np.random.randint(8)
        rb = np.random.randint(7)
        if rb >= ra:
            rb += 1
        ia = np.random.randint(32)
        ib = np.random.randint(32)
        a = members[ra, ia]
        b = members[rb, ib]
        # --- prefill delta
        dp = 0.0
        if lam_p != 0.0:
            tot_new = 0.0
            for w in range(nw):
                d = WpT[b, w] - WpT[a, w]
                m = -1.0
                for r in range(8):
                    v = Sp[w, r]
                    if r == ra:
                        v += d
                    elif r == rb:
                        v -= d
                    Sp_new[w, r] = v
                    if v > m:
                        m = v
                tot_new += swp[w] * m
            dp = tot_new - cost_p
        # --- c1 delta
        d1 = 0.0
        n_aff = 0
        if lam1 != 0.0:
            stampc += 1
            for i in range(tok_off[a], tok_off[a + 1]):
                t = tok_idx[i]
                if stamp[t] != stampc:
                    stamp[t] = stampc
                    tmp_t[n_aff] = t
                    n_aff += 1
            for i in range(tok_off[b], tok_off[b + 1]):
                t = tok_idx[i]
                if stamp[t] != stampc:
                    stamp[t] = stampc
                    tmp_t[n_aff] = t
                    n_aff += 1
            for q in range(n_aff):
                t = tmp_t[q]
                m = 0
                for r in range(8):
                    if K[t, r] > m:
                        m = K[t, r]
                tmp_old[q] = sw1[t] * g1[m]
            for i in range(tok_off[a], tok_off[a + 1]):
                t = tok_idx[i]
                K[t, ra] -= 1
                K[t, rb] += 1
            for i in range(tok_off[b], tok_off[b + 1]):
                t = tok_idx[i]
                K[t, rb] -= 1
                K[t, ra] += 1
            for q in range(n_aff):
                t = tmp_t[q]
                m = 0
                for r in range(8):
                    if K[t, r] > m:
                        m = K[t, r]
                d1 += sw1[t] * g1[m] - tmp_old[q]
        # --- c64 delta (D kept, changes applied tentatively)
        d64 = 0.0
        if lam64 != 0.0:
            for bb in range(nbt):
                dd = np.int16(PT[b, bb]) - np.int16(PT[a, bb])
                if dd != 0:
                    m0 = 0
                    for r in range(8):
                        if D[bb, r] > m0:
                            m0 = D[bb, r]
                    D[bb, ra] += dd
                    D[bb, rb] -= dd
                    m1 = 0
                    for r in range(8):
                        if D[bb, r] > m1:
                            m1 = D[bb, r]
                    D[bb, ra] -= dd
                    D[bb, rb] += dd
                    if m1 != m0:
                        d64 += swb[bb] * (g64[m1] - g64[m0])
        dJ = lam_p * dp + lam1 * d1 + lam64 * d64
        accept = dJ <= 0.0 or (temp > 0.0 and np.random.random() < np.exp(-dJ / temp))
        if accept:
            if lam_p != 0.0:
                for w in range(nw):
                    Sp[w, ra] = Sp_new[w, ra]
                    Sp[w, rb] = Sp_new[w, rb]
                cost_p += dp
            cost_1 += d1
            if lam64 != 0.0:
                for bb in range(nbt):
                    dd = np.int16(PT[b, bb]) - np.int16(PT[a, bb])
                    D[bb, ra] += dd
                    D[bb, rb] -= dd
                cost_64 += d64
            J += dJ
            members[ra, ia] = b
            members[rb, ib] = a
            rank_of[a] = rb
            rank_of[b] = ra
            if J < bestJ - 1e-12:
                bestJ = J
                best[:] = rank_of
                best_terms[0] = cost_p
                best_terms[1] = cost_1
                best_terms[2] = cost_64
        else:
            # revert the tentative K changes of the c1 term
            if lam1 != 0.0:
                for i in range(tok_off[a], tok_off[a + 1]):
                    t = tok_idx[i]
                    K[t, ra] += 1
                    K[t, rb] -= 1
                for i in range(tok_off[b], tok_off[b + 1]):
                    t = tok_idx[i]
                    K[t, rb] += 1
                    K[t, ra] -= 1
    return best, bestJ, best_terms


def posting(toks):
    """CSR of tokens containing each expert: toks [N, 8] -> (off [257], idx [N*8])"""
    N = toks.shape[0]
    flat = toks.astype(np.int64).ravel()
    order = np.argsort(flat, kind="stable")
    idx = (order // 8).astype(np.int32)
    off = np.zeros(NE + 1, dtype=np.int32)
    np.add.at(off, flat + 1, 1)
    return np.cumsum(off).astype(np.int32), idx


def lpt_assign(pop):
    """Greedy number partitioning: the heaviest expert to the lightest rank that still has room (32 per rank)."""
    rank = np.empty(NE, dtype=np.int64)
    load = np.zeros(NR)
    room = np.full(NR, PR)
    for e in np.argsort(-pop, kind="stable"):
        cand = np.nonzero(room > 0)[0]
        r = cand[np.argmin(load[cand])]
        rank[e] = r
        load[r] += pop[e]
        room[r] -= 1
    return rank


def optimize_layer(layer, train, weights, iters, seed, start="lpt", greedy_iters=20000, max_tok=15000, rng_seed=0):
    """train: dict dom -> dict(pre_cnt [W,256] of this layer, dec [N,8] ids, pres [B,256] batches (real + shuffled))
    weights: dict(lam_p, lam1, lam64) in 1/ms. Domain-balanced sample weights. Returns (rank_of, terms, J0)."""
    doms = [d for d in train]
    nd = len(doms)
    rng = np.random.default_rng(rng_seed + layer)
    Wp, swp, toks, sw1, pres, swb = [], [], [], [], [], []
    tm = tile_ms(layer) * CHUNK_MBS
    for d in doms:
        t = train[d]
        if t["pre_cnt"].shape[0]:
            Wp.append(tiles(t["pre_cnt"]) * tm)
            swp.append(np.full(t["pre_cnt"].shape[0], 1.0 / nd / t["pre_cnt"].shape[0], dtype=np.float32))
        if t["dec"].shape[0]:
            x = t["dec"]
            if x.shape[0] > max_tok:
                x = x[rng.choice(x.shape[0], max_tok, replace=False)]
            toks.append(x)
            sw1.append(np.full(x.shape[0], 1.0 / nd / x.shape[0], dtype=np.float32))
        if t["pres"].shape[0]:
            pres.append(t["pres"])
            swb.append(np.full(t["pres"].shape[0], 1.0 / nd / t["pres"].shape[0], dtype=np.float32))
    Wp = np.concatenate(Wp) if Wp else np.zeros((0, NE), np.float32)
    swp = np.concatenate(swp) if swp else np.zeros(0, np.float32)
    toks = np.concatenate(toks).astype(np.uint8) if toks else np.zeros((0, 8), np.uint8)
    sw1 = np.concatenate(sw1) if sw1 else np.zeros(0, np.float32)
    pres = np.concatenate(pres) if pres else np.zeros((0, NE), np.uint8)
    swb = np.concatenate(swb) if swb else np.zeros(0, np.float32)
    off, idx = posting(toks) if len(toks) else (np.zeros(NE + 1, np.int32), np.zeros(0, np.int32))
    lam_p, lam1, lam64 = weights["lam_p"], weights["lam1"], weights["lam64"]
    if start == "lpt":
        pop = lam_p * (swp[:, None] * Wp).sum(0)
        if lam1 != 0 and len(toks):
            pop = pop + lam1 * np.bincount(toks.ravel(), minlength=NE) * sw1.mean() * g1_table(layer)[3]
        if lam64 != 0 and len(pres):
            pop = pop + lam64 * (swb[:, None] * pres).sum(0) * g64_table(layer)[1]
        rank0 = lpt_assign(pop)
    elif start == "linear":
        rank0 = LIN.copy()
    else:
        rank0 = np.random.default_rng(seed).permutation(LIN)
    WpT = np.ascontiguousarray(Wp.T.astype(np.float32))
    PT = np.ascontiguousarray(pres.T)
    g1, g64 = g1_table(layer), g64_table(layer)
    # temperature scale: mean |delta| of random swaps from the start
    rng2 = np.random.default_rng(seed + 99)
    deltas = []
    base = _objective(layer, rank0, WpT, swp, toks, sw1, PT, swb, weights)
    for _ in range(60):
        a, b = rng2.integers(NE, size=2)
        if rank0[a] == rank0[b]:
            continue
        r2 = rank0.copy()
        r2[a], r2[b] = rank0[b], rank0[a]
        deltas.append(abs(_objective(layer, r2, WpT, swp, toks, sw1, PT, swb, weights) - base))
    T0 = float(np.mean(deltas)) * 0.5 if deltas else 1e-3
    best, bestJ, terms = _anneal(rank0.astype(np.int64), WpT, swp, lam_p, toks, off, idx, sw1, g1, lam1, PT, swb, g64, lam64,
                                 int(iters), T0, T0 * 0.01, int(seed), int(greedy_iters))
    return best, terms, bestJ


def _objective(layer, rank_of, WpT, swp, toks, sw1, PT, swb, weights):
    """Plain numpy objective (check of the annealer, and the temperature scale)."""
    tot = 0.0
    if weights["lam_p"] != 0 and WpT.shape[1]:
        S = np.zeros((WpT.shape[1], NR))
        for r in range(NR):
            S[:, r] = WpT[rank_of == r].sum(0)
        tot += weights["lam_p"] * float((swp * S.max(1)).sum())
    if weights["lam1"] != 0 and len(toks):
        r = rank_of[toks.astype(np.int64)]
        k = (r[:, :, None] == np.arange(NR)[None, None, :]).sum(1).max(1)
        tot += weights["lam1"] * float((sw1 * g1_table(layer)[k]).sum())
    if weights["lam64"] != 0 and PT.shape[1]:
        D = np.zeros((PT.shape[1], NR))
        for r in range(NR):
            D[:, r] = PT[rank_of == r].sum(0)
        tot += weights["lam64"] * float((swb * g64_table(layer)[D.max(1).astype(np.int64)]).sum())
    return tot
