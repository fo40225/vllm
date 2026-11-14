"""Leave-one-domain-out cross-validation of the static expert placement (R2D step 2).

usage: r2d_cv.py <agg_dir> <out_dir> --variants comb,pre,dec,lpt [--folds docs,zh,...|all] [--iters N] [--threads N]
       [--final] [--wscale S]

For every held-out domain d of the natural set (gsm, chat, code, docs, zh, bio) a placement is trained on the other five and
evaluated on d (prefill windows, c1 decode rows, c64 real + shuffled batches, c128 shuffled batches); --final also trains on all six
and evaluates on the eval-only sets (vdocs, natpre, mix, bioseq, random). Metrics are totals over the layers (NVFP4 layers 1..60 and
all 62 layers); `lin` = production placement, `rand` = mean of three random placements (null).
"""
import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2d_lib as L  # noqa: E402

NAT = ["gsm", "chat", "code", "docs", "zh", "bio"]
EXTRA = ["bioseq", "vdocs", "natpre", "mix", "random"]


def build(agg, dom, seed=0):
    """All arrays of one domain for all 62 layers."""
    rng = np.random.default_rng(seed + zlib.crc32(dom.encode()) % 1000)
    z = L.load_domain(agg, dom)
    out = dict(name=dom, pre=z["pre_cnt"], dec=z["dec_ids"], bpres=z["b_pres"], half=z["half_cnt"])
    n = z["dec_ids"].shape[1]
    out["n_dec"] = n
    if n >= 200:
        out["s64"] = L.shuffled_presence(z["dec_ids"], 64, 600, rng)
        out["s128"] = L.shuffled_presence(z["dec_ids"], 128, 300, rng)
    else:
        out["s64"] = out["s128"] = np.zeros((0, L.NL, L.NE), np.uint8)
    return out


def layer_train(doms, layer):
    t = {}
    for d in doms:
        pres = [d["bpres"][:, layer, :]] if d["bpres"].shape[0] else []
        if d["s64"].shape[0]:
            pres.append(d["s64"][:, layer, :])
        t[d["name"]] = dict(pre_cnt=d["pre"][layer], dec=d["dec"][layer],
                            pres=np.concatenate(pres) if pres else np.zeros((0, L.NE), np.uint8))
    return t


def train_table(doms, weights, iters, threads, start="lpt", seed=1, max_tok=15000):
    ranks = np.zeros((L.NL, L.NE), dtype=np.int64)
    terms = np.zeros((L.NL, 3))
    t0 = time.time()

    def one(layer):
        tr = layer_train(doms, layer)
        r, tm, J = L.optimize_layer(layer, tr, weights, iters, seed + layer, start=start, max_tok=max_tok)
        return layer, r, tm

    with ThreadPoolExecutor(threads) as ex:
        for layer, r, tm in ex.map(one, range(L.NL)):
            ranks[layer] = r
            terms[layer] = tm
    print("  trained %d layers on %s in %.0fs" % (L.NL, [d["name"] for d in doms], time.time() - t0), flush=True)
    return ranks, terms


def lpt_table(doms):
    """LPT on the tile-cost popularity of the training windows (prefill only, no annealing)."""
    ranks = np.zeros((L.NL, L.NE), dtype=np.int64)
    for layer in range(L.NL):
        pop = np.zeros(L.NE)
        for d in doms:
            if d["pre"].shape[1]:
                pop += L.tiles(d["pre"][layer]).mean(0) / len(doms)
        ranks[layer] = L.lpt_assign(pop)
    return ranks


def evaluate(ranks, d):
    """Metrics of one placement table on one domain. Returns dict of per-layer arrays reduced to totals."""
    res = dict(pre_gap=np.zeros(L.NL), pre_max=np.zeros(L.NL), pre_mm=np.zeros(L.NL), c1_ms=np.zeros(L.NL), c1_emax=np.zeros(L.NL),
               c64r_ms=np.zeros(L.NL), c64r_dmax=np.zeros(L.NL), c64s_ms=np.zeros(L.NL), c64s_dmax=np.zeros(L.NL),
               c128s_ms=np.zeros(L.NL), c128s_dmax=np.zeros(L.NL), pre_win=None, c1_tok=None)
    nw = d["pre"].shape[1]
    pw = np.zeros(nw) if nw else None
    ct = np.zeros(d["dec"].shape[1]) if d["dec"].shape[1] else None
    for l in range(L.NL):
        r = ranks[l]
        if nw:
            oh = L.onehot(r)
            t = L.tiles(d["pre"][l]) @ oh * L.tile_ms(l) * L.CHUNK_MBS
            loads = d["pre"][l].astype(np.float32) @ oh
            pw += t.max(1) - t.mean(1)
            res["pre_gap"][l] = (t.max(1) - t.mean(1)).mean()
            res["pre_max"][l] = t.max(1).mean()
            res["pre_mm"][l] = (loads.max(1) / loads.mean(1)).mean()
        if d["dec"].shape[1]:
            ids = d["dec"][l]
            rk = r[ids.astype(np.int64)]
            k = (rk[:, :, None] == np.arange(L.NR)[None, None, :]).sum(1).max(1)
            g = L.g1_table(l)[k]
            res["c1_ms"][l] = g.mean()
            res["c1_emax"][l] = k.mean()
            ct += g
        for key, arr in (("c64r", d["bpres"][:, l, :]), ("c64s", d["s64"][:, l, :]), ("c128s", d["s128"][:, l, :])):
            if arr.shape[0]:
                e = L.eval_batches(arr, r, l, L.US128_SLOPE if key == "c128s" else None)
                res[key + "_ms"][l] = e["ms"]
                res[key + "_dmax"][l] = e["dmax"]
    res["pre_win"] = pw
    res["c1_tok"] = ct
    return res


def summarize(ev_lin, ev_pl, ev_rand=None):
    """Totals over layers: NVFP4 1..60 ('nv') and all 62 ('all'), with standard errors where per-sample vectors exist."""
    out = {}
    nv = slice(1, 61)
    for scope, sl in (("nv", nv), ("all", slice(0, 62))):
        o = {}
        for key, label in (("pre_gap", "prefill_gap_ms_per_chunk"), ("pre_max", "prefill_expert_ms_per_chunk"),
                           ("c1_ms", "c1_ms_per_step"), ("c64r_ms", "c64_real_ms_per_step"), ("c64s_ms", "c64_shuf_ms_per_step"),
                           ("c128s_ms", "c128_shuf_ms_per_step")):
            a, b = ev_lin[key][sl].sum(), ev_pl[key][sl].sum()
            o[label] = dict(lin=float(a), placed=float(b), gain=float(a - b))
            if ev_rand is not None:
                c = np.mean([e[key][sl].sum() for e in ev_rand])
                o[label]["rand"] = float(c)
        for key, label in (("pre_mm", "prefill_max_over_mean_load_win"), ("c1_emax", "c1_Emax_k"), ("c64r_dmax", "c64_real_Emax_distinct"),
                           ("c64s_dmax", "c64_shuf_Emax_distinct"), ("c128s_dmax", "c128_shuf_Emax_distinct")):
            o[label] = dict(lin=float(ev_lin[key][sl].mean()), placed=float(ev_pl[key][sl].mean()))
        out[scope] = o
    # standard errors over windows / tokens (sum over NVFP4 layers cannot be split per layer here; all-layer per-sample sums)
    if ev_lin["pre_win"] is not None and ev_pl["pre_win"] is not None:
        d = ev_lin["pre_win"] - ev_pl["pre_win"]
        out["prefill_gain_win_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
        out["prefill_n_windows"] = int(len(d))
    if ev_lin["c1_tok"] is not None and ev_pl["c1_tok"] is not None:
        d = ev_lin["c1_tok"] - ev_pl["c1_tok"]
        out["c1_gain_tok_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        out["c1_n_tokens"] = int(len(d))
    return out


def aggregate_load_ratio(pre, ranks_or_lin):
    """Aggregate route load max/mean over the whole domain per layer (the S1-5 'load max/mean' metric), mean over layers 1..60."""
    v = []
    for l in range(1, 61):
        c = pre[l].astype(np.float64).sum(0)
        r = LIN_RANKS[l] if ranks_or_lin is None else ranks_or_lin[l]
        loads = np.zeros(L.NR)
        np.add.at(loads, r, c)
        v.append(loads.max() / loads.mean())
    return float(np.mean(v))


LIN_RANKS = np.tile(L.LIN, (L.NL, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agg")
    ap.add_argument("out")
    ap.add_argument("--variants", default="comb")
    ap.add_argument("--folds", default="all")
    ap.add_argument("--iters", type=int, default=300000)
    ap.add_argument("--threads", type=int, default=56)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--wscale", type=float, default=1.0, help="multiplier of the decode weights (Pareto sweep)")
    ap.add_argument("--max-tok", type=int, default=15000)
    ap.add_argument("--start", default="lpt")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    doms = {d: build(a.agg, d) for d in NAT}
    print("loaded %.0fs; windows %s dec rows %s" % (time.time() - t0, {d: doms[d]["pre"].shape[1] for d in NAT}, {d: doms[d]["n_dec"] for d in NAT}), flush=True)
    extra = {}
    if a.final:
        for d in EXTRA:
            if os.path.exists(os.path.join(a.agg, d + ".npz")):
                extra[d] = build(a.agg, d)
        # the random-token control: prefill windows of the random prompts + the S1-5 decode data
        if "random" in extra and os.path.exists(os.path.join(a.agg, "random_s15.npz")):
            r15 = build(a.agg, "random_s15")
            extra["random"]["dec"], extra["random"]["bpres"] = r15["dec"], r15["bpres"]
            extra["random"]["s64"], extra["random"]["s128"], extra["random"]["n_dec"] = r15["s64"], r15["s128"], r15["n_dec"]
        elif os.path.exists(os.path.join(a.agg, "random_s15.npz")):
            extra["random"] = build(a.agg, "random_s15")
    rng = np.random.default_rng(5)
    rand_tables = [np.stack([rng.permutation(L.LIN) for _ in range(L.NL)]) for _ in range(3)]
    W = dict(lam_p=1.0 / L.BASE_MS["prefill"], lam1=a.wscale / L.BASE_MS["c1"], lam64=a.wscale / L.BASE_MS["c64"])
    variants = {
        "comb": W,
        "pre": dict(lam_p=W["lam_p"], lam1=0.0, lam64=0.0),
        "dec": dict(lam_p=0.0, lam1=W["lam1"], lam64=W["lam64"]),
    }
    folds = NAT if a.folds == "all" else ([] if a.folds == "none" else a.folds.split(","))
    report = dict(weights=W, iters=a.iters, variants=list(a.variants.split(",")), folds={}, final={})
    for var in a.variants.split(","):
        for fold in folds:
            train = [doms[d] for d in NAT if d != fold]
            if var == "lpt":
                ranks = lpt_table(train)
            else:
                ranks, _ = train_table(train, variants[var], a.iters, a.threads, start=a.start, max_tok=a.max_tok)
            np.save(os.path.join(a.out, "ranks_%s_hold_%s.npy" % (var, fold)), ranks)
            test = doms[fold]
            ev_lin, ev_pl = evaluate(LIN_RANKS, test), evaluate(ranks, test)
            ev_rand = [evaluate(t, test) for t in rand_tables]
            s = summarize(ev_lin, ev_pl, ev_rand)
            s["agg_load_ratio"] = dict(lin=aggregate_load_ratio(test["pre"], None), placed=aggregate_load_ratio(test["pre"], ranks)) if test["pre"].shape[1] else None
            report["folds"].setdefault(var, {})[fold] = s
            nv = s["nv"]
            print("[%s hold %-5s] prefill gap %.2f -> %.2f ms/chunk (gain %.2f, se %s, n=%s) | c1 %.4f -> %.4f ms (gain %.4f) | c64 real %.3f -> %.3f | c64 shuf %.3f -> %.3f" % (
                var, fold, nv["prefill_gap_ms_per_chunk"]["lin"], nv["prefill_gap_ms_per_chunk"]["placed"], nv["prefill_gap_ms_per_chunk"]["gain"],
                s.get("prefill_gain_win_se"), s.get("prefill_n_windows"), nv["c1_ms_per_step"]["lin"], nv["c1_ms_per_step"]["placed"],
                nv["c1_ms_per_step"]["gain"], nv["c64_real_ms_per_step"]["lin"], nv["c64_real_ms_per_step"]["placed"],
                nv["c64_shuf_ms_per_step"]["lin"], nv["c64_shuf_ms_per_step"]["placed"]), flush=True)
            json.dump(report, open(os.path.join(a.out, "cv_report.json"), "w"), indent=1)
        if a.final:
            train = [doms[d] for d in NAT]
            if var == "lpt":
                ranks = lpt_table(train)
            else:
                ranks, terms = train_table(train, variants[var], a.iters, a.threads, start=a.start, max_tok=a.max_tok)
            np.save(os.path.join(a.out, "ranks_%s_final.npy" % var), ranks)
            report["final"].setdefault(var, {})
            for name, d in list(extra.items()) + [(n, doms[n]) for n in NAT]:
                ev_lin, ev_pl = evaluate(LIN_RANKS, d), evaluate(ranks, d)
                ev_rand = [evaluate(t, d) for t in rand_tables]
                s = summarize(ev_lin, ev_pl, ev_rand)
                s["agg_load_ratio"] = dict(lin=aggregate_load_ratio(d["pre"], None), placed=aggregate_load_ratio(d["pre"], ranks)) if d["pre"].shape[1] else None
                s["in_sample"] = name in NAT
                report["final"][var][name] = s
                nv = s["nv"]
                print("[%s final -> %-7s%s] prefill gap %.2f -> %.2f | c1 %.4f -> %.4f | c64 real %.3f -> %.3f | c64 shuf %.3f -> %.3f" % (
                    var, name, "*" if name in NAT else " ", nv["prefill_gap_ms_per_chunk"]["lin"], nv["prefill_gap_ms_per_chunk"]["placed"],
                    nv["c1_ms_per_step"]["lin"], nv["c1_ms_per_step"]["placed"], nv["c64_real_ms_per_step"]["lin"],
                    nv["c64_real_ms_per_step"]["placed"], nv["c64_shuf_ms_per_step"]["lin"], nv["c64_shuf_ms_per_step"]["placed"]), flush=True)
            json.dump(report, open(os.path.join(a.out, "cv_report.json"), "w"), indent=1)
    print("done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
