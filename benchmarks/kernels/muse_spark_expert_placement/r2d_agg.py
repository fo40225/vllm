"""Aggregate the raw R2D routing trace into per-domain compact arrays (numpy only, CPU).

For every phase window of phases.tsv the recorded steps are split into rows (one row = one token of one request; ids [62, 8]):
  prefill rows  flag prefilling == 1 (chat-endpoint requests lose the chat-template tokens: the first n_pre and the last n_post
                prompt positions of every request, identical text in every prompt)
  decode rows   q == 1 and not prefilling
Outputs <out>/<domain>.npz:
  pre_cnt  int16 [62, W, 256]   routes per expert of W windows of WIN consecutive prefill rows (per phase stream, remainder dropped)
  pre_ph   int16 [W]            phase index of every window (names in meta)
  dec_ids  uint8 [62, N, 8]     routed experts of N decode rows (c1 samples)
  dec_req  int32 [N]            request ordinal of every decode row (for request-level folds)
  b_pres   uint8 [Bn, 62, 256]  expert presence of the pure decode steps with >= BMIN rows (real batches of the closed loop)
  b_T      int16 [Bn]           rows of every batch
  half_cnt int16 [S, 2, 62, 256] (raw prefill phases only) routes per expert of the two halves of every full 8192-token step
usage: r2d_agg.py <trace_dir> <out_dir> [phases.tsv]
"""
import glob
import json
import os
import sys

import numpy as np

WIN = 4096
BMIN = 48
NL, NE = 62, 256
d_tr, d_out = sys.argv[1], sys.argv[2]
os.makedirs(d_out, exist_ok=True)
tmpl = json.load(open(os.environ.get("R2D_TEMPLATE", "/raid/vllm-runtime/work/s1-6/R2D/corpus/template.json")))
N_PRE, N_POST = tmpl["n_pre"], tmpl["n_post"]

phases = []
for line in open(sys.argv[3] if len(sys.argv) > 3 else os.path.join(d_tr, "phases.tsv")):
    name, t0, t1, rc = line.rstrip("\n").split("\t")
    phases.append((name, float(t0), float(t1), int(rc)))
steps = [json.loads(l) for l in open(os.path.join(d_tr, "steps.jsonl"))]
by_step = {m["step"]: m for m in steps}
DOMS = ("gsm", "chat", "code", "docs", "zh", "bio", "bioseq")


def phase_role(name):
    """-> (domain, use_prefill, use_decode) or None"""
    if name in ("warm",):
        return None
    if name == "R_pre":
        return ("random", True, False)
    if name == "N_pre":
        return ("natpre", True, False)
    if name == "V_pre":
        return ("vdocs", True, False)
    if name == "mix_c64":
        return ("mix", True, True)
    for d in DOMS:
        if name == d + "_chat":
            return (d, True, True)
        if name == d + "_rep":
            return (d, False, True)
        if name in (d + "_bundle", d + "_resp"):
            return (d, True, False)
    return None


def phase_of(t):
    for i, (name, t0, t1, rc) in enumerate(phases):
        if t0 <= t <= t1:
            return i
    return -1


# request prompt lengths (rows with prefilling == 1) and ordinals
L = {}
for m in steps:
    for r, q, pf in zip(m["req"], m["q"], m["prefilling"]):
        if pf:
            L[r] = L.get(r, 0) + q
ordinal = {}
for m in steps:
    for r in m["req"]:
        ordinal.setdefault(r, len(ordinal))

# per phase accumulators
acc = {}


def A(pi):
    if pi not in acc:
        acc[pi] = dict(pre=[], dec=[], dec_req=[], b=[], bT=[], half=[])
    return acc[pi]


pos = {}
nfiles = 0
for cf in sorted(glob.glob(os.path.join(d_tr, "chunk_*.npz"))):
    z = np.load(cf)
    ids = z["ids"]
    parts = {}
    for s, l, off, n in zip(z["step"], z["layer"], z["off"], z["n"]):
        parts.setdefault(int(s), {}).setdefault(int(l), []).append((int(off), int(n)))
    nfiles += 1
    for s in sorted(parts):
        m = by_step[s]
        lay = parts[s]
        if len(lay) != NL or any(len(v) != 1 for v in lay.values()):
            continue
        T = m["num_tokens"]
        if any(lay[l][0][1] != T for l in range(NL)):
            continue
        x = np.stack([ids[lay[l][0][0]:lay[l][0][0] + T] for l in range(NL)])  # [62, T, 8] uint8
        pi = phase_of(m["t"])
        role = phase_role(phases[pi][0]) if pi >= 0 else None
        # row classification (also advances the per-request prompt position, whatever the phase)
        off = 0
        pre_idx, dec_idx, dec_rq = [], [], []
        for r, q, pf in zip(m["req"], m["q"], m["prefilling"]):
            if pf:
                p0 = pos.get(r, 0)
                pos[r] = p0 + q
                if r.startswith("chatcmpl-"):
                    keep = [j for j in range(q) if N_PRE <= p0 + j < L[r] - N_POST]
                else:
                    keep = list(range(q))
                pre_idx += [off + j for j in keep]
            elif q == 1:
                dec_idx.append(off)
                dec_rq.append(ordinal[r])
            off += q
        if role is None:
            continue
        dom, use_pre, use_dec = role
        a = A(pi)
        if use_pre and pre_idx:
            a["pre"].append(x[:, pre_idx, :])
            if dom in ("random", "natpre", "vdocs") and T == 8192:
                h = []
                for k in range(2):
                    sub = x[:, k * 4096:(k + 1) * 4096, :].astype(np.int64)
                    lay_off = (np.arange(NL)[:, None, None] * NE + sub)
                    h.append(np.bincount(lay_off.ravel(), minlength=NL * NE).reshape(NL, NE))
                a["half"].append(np.stack(h).astype(np.int16))
        if use_dec and dec_idx:
            a["dec"].append(x[:, dec_idx, :])
            a["dec_req"].append(np.asarray(dec_rq, dtype=np.int32))
            if len(dec_idx) == T and T >= BMIN:
                pres = np.zeros((NL, NE), dtype=np.uint8)
                for l in range(NL):
                    pres[l, np.unique(x[l])] = 1
                a["b"].append(pres)
                a["bT"].append(T)
print("chunk files", nfiles, "phases with data", len(acc), flush=True)

# per domain outputs
dom_of = {}
for pi in acc:
    dom_of.setdefault(phase_role(phases[pi][0])[0], []).append(pi)
names = [p[0] for p in phases]
for dom, pis in sorted(dom_of.items()):
    pre_cnt, pre_ph, dec, dec_req, b, bT, half = [], [], [], [], [], [], []
    stat = {}
    for pi in sorted(pis):
        a = acc[pi]
        if a["pre"]:
            S = np.concatenate(a["pre"], axis=1)  # [62, R, 8]
            nw = S.shape[1] // WIN
            stat[names[pi] + ":pre_rows"] = int(S.shape[1])
            for w in range(nw):
                sub = S[:, w * WIN:(w + 1) * WIN, :].astype(np.int64)
                c = np.bincount((np.arange(NL)[:, None, None] * NE + sub).ravel(), minlength=NL * NE).reshape(NL, NE)
                pre_cnt.append(c.astype(np.int16))
                pre_ph.append(pi)
            del S
        if a["dec"]:
            dec.append(np.concatenate(a["dec"], axis=1))
            dec_req.append(np.concatenate(a["dec_req"]))
            stat[names[pi] + ":dec_rows"] = int(dec[-1].shape[1])
        if a["b"]:
            b += a["b"]
            bT += a["bT"]
            stat[names[pi] + ":batches"] = len(a["b"])
        half += a["half"]
    out = dict(
        pre_cnt=np.stack(pre_cnt, axis=1) if pre_cnt else np.zeros((NL, 0, NE), np.int16),
        pre_ph=np.asarray(pre_ph, np.int16),
        dec_ids=np.concatenate(dec, axis=1) if dec else np.zeros((NL, 0, 8), np.uint8),
        dec_req=np.concatenate(dec_req) if dec_req else np.zeros(0, np.int32),
        b_pres=np.stack(b) if b else np.zeros((0, NL, NE), np.uint8),
        b_T=np.asarray(bT, np.int16),
        half_cnt=np.stack(half) if half else np.zeros((0, 2, NL, NE), np.int16),
        meta=np.asarray(json.dumps(dict(phases=names, stat=stat, win=WIN, n_pre=N_PRE, n_post=N_POST))),
    )
    np.savez_compressed(os.path.join(d_out, dom + ".npz"), **out)
    print("%-8s windows=%4d dec_rows=%7d batches=%4d halves=%2d  %s" % (
        dom, out["pre_cnt"].shape[1], out["dec_ids"].shape[1], len(b), len(half), json.dumps(stat)), flush=True)
