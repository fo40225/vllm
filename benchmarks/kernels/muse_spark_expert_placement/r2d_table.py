"""Write the versioned placement table (JSON, format of the L4 hot-expert table) and validate it.

usage: r2d_table.py build <ranks.npy> <cv_report.json> <agg_dir> <trace_dir> <out.json> [--variant comb]
       r2d_table.py check <table.json>

rank_of[l][e]  EP rank that owns the ORIGINAL expert id e in layer l (32 experts per rank)
new_id[l][e]   position of e in the permuted linear layout: rank_of[l][e] * 32 + (index of e among the experts of its rank, ascending
               original id); i.e. the id the muse kernels (local_expert_offset = ep_rank x 32) see after the remap.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

SNAPSHOT = "fc6908f53c6882d4ec48fca5ba144e41d21e1f92"
REPO = "unsloth/Muse-Spark-Open-816B-A42B-NVFP4"


def new_ids(rank_of):
    rank_of = np.asarray(rank_of)
    out = np.empty(256, dtype=np.int64)
    for r in range(8):
        e = np.nonzero(rank_of == r)[0]
        out[e] = r * 32 + np.arange(len(e))
    return out


def check(path):
    t = json.load(open(path))
    assert t["version"] == 1 and t["model"]["snapshot"] == SNAPSHOT and t["model"]["num_experts"] == 256
    assert t["layout"]["ep_size"] == 8 and t["layout"]["experts_per_rank"] == 32
    n_moved = []
    for l in range(t["model"]["num_hidden_layers"]):
        e = t["layers"][str(l)]
        ro, ni = np.asarray(e["rank_of"]), np.asarray(e["new_id"])
        assert ro.shape == (256,) and ni.shape == (256,)
        assert sorted(ni.tolist()) == list(range(256)), "layer %d: new_id is not a permutation" % l
        assert (np.bincount(ro, minlength=8) == 32).all(), "layer %d: not 32 experts per rank" % l
        assert (ni // 32 == ro).all(), "layer %d: new_id and rank_of disagree" % l
        assert (new_ids(ro) == ni).all(), "layer %d: new_id is not the canonical numbering" % l
        n_moved.append(int((ro != np.arange(256) // 32).sum()))
    print("table OK: %d layers, experts off their linear rank per layer: min %d mean %.1f max %d" % (len(n_moved), min(n_moved), np.mean(n_moved), max(n_moved)))


def build(ranks_npy, cv_json, agg_dir, trace_dir, out, variant):
    ranks = np.load(ranks_npy)
    assert ranks.shape == (62, 256)
    cv = json.load(open(cv_json))
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
    tokens = {}
    for d in ("gsm", "chat", "code", "docs", "zh", "bio", "bioseq", "vdocs", "natpre", "mix", "random"):
        p = os.path.join(agg_dir, d + ".npz")
        if os.path.exists(p):
            z = np.load(p)
            m = json.loads(str(z["meta"]))
            tokens[d] = dict(prefill_windows_4096=int(z["pre_cnt"].shape[1]), decode_rows=int(z["dec_ids"].shape[1]),
                             c64_real_batches=int(z["b_pres"].shape[0]), phase_rows=m.get("stat", {}))
    prov = dict(
        what="Static per-layer expert placement over the 8 EP ranks (32 experts per rank): objective = domain-balanced sum of "
             "lam_p x E[max over ranks of the prefill expert-kernel time per 4096-token microbatch] (x2 per 8192 chunk) + lam1 x E[c1 "
             "slowest-rank muse kernel time] + lam64 x E[c64 slowest-rank time], lam = 1 / baseline step time (equal relative weight).",
        trace=dict(source="R2D routing trace r2d_trace (eager server, branch a89b873288 + recorder, TP8 + EP8, host plugin b207e54, real "
                          "weights, PREFILL_OVERLAP off, prefix caching off)", steps_jsonl_sha256=sha(os.path.join(trace_dir, "steps.jsonl")),
                   domains=tokens, template_tokens_stripped=dict(first=141, last=3)),
        training=dict(domains=["gsm", "chat", "code", "docs", "zh", "bio"], variant=variant, weights=cv["weights"], iters=cv["iters"],
                      weighting="every domain counts equally in every term"),
        cost_models=dict(prefill="NVFP4 11.94 us per 128-row tile, BF16 layers 0/61 42.6 us per tile (fit on natpre x routing)",
                         c1="S1-5 T_C1 table (k local experts of the slowest rank) 2.8/26.5/42.4/62.0 us + 20.2 us per expert",
                         c64="3.0 + 19.0 us per distinct local expert (c128: 20.3)", bf16_decode="kappa1/kappa64 us per (distinct) expert"),
        cross_validation=cv.get("folds", {}).get(variant),
        final_eval=cv.get("final", {}).get(variant),
        base_commit="a89b873288", host_commit="b207e54", date=time.strftime("%Y-%m-%d"),
        built_by="r2d_cv.py + r2d_table.py", built_by_sha256=dict((f, sha(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)))
                                                                   for f in ("r2d_lib.py", "r2d_cv.py", "r2d_table.py", "r2d_agg.py")),
    )
    t = dict(version=1,
             model=dict(repo=REPO, snapshot=SNAPSHOT, num_hidden_layers=62, num_experts=256, top_k=8, moe_layers=[0, 61]),
             layout=dict(ep_size=8, experts_per_rank=32, base_placement="linear"),
             semantics="rank_of[e]: EP rank owning original expert e; new_id[e] = rank_of*32 + ascending slot; layers 0 and 61 are the BF16 "
                       "(Triton) MoE layers: apply the remap there only if their path allows it (table is valid for all 62 layers).",
             provenance=prov,
             layers={str(l): dict(rank_of=ranks[l].tolist(), new_id=new_ids(ranks[l]).tolist()) for l in range(62)})
    json.dump(t, open(out, "w"), separators=(",", ":"))
    print("wrote", out, os.path.getsize(out), "bytes")
    check(out)


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2])
    else:
        a = sys.argv[2:7]
        var = sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "comb"
        build(*a, variant=var)
