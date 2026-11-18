"""Compare two author-topic corpora that differ only in their construct authors: 5-fold paper CV and held-out tasks.

CV: per contrast, held-out log-likelihood (cv_abs.py rows; the absolute log P(peaks | authors)); the difference B - A in nats
per peak over the contrasts both scored, with a 95% interval from 1000 paper bootstraps, overall and per kind of
contrast (the reason column contrast_construct_authors.py would give it). Held-out tasks: per task, gain over the
average map in nats per peak (zeroshot_tasks.py), mean over the tasks both scored and the share where B is higher.

usage: python compare_authors.py CORPUS_B CV_A.npy CV_B.npy ZERO_A.json ZERO_B.json HUB_DIR OUT.json
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from contrast_construct_authors import kind  # noqa: E402


def rows(p):
    """id -> (peaks, held-out nats): the absolute log P(peaks | authors) from cv_abs.py rows (id, peaks, gain, abs,
    abs map), else the gain from task_networks cv rows; the absolute term compares corpora, the gain's reference map
    differs between them"""
    return {x[0]: (float(x[1]), float(x[3] if len(x) >= 5 else x[2])) for x in np.load(p, allow_pickle=True)}


def main():
    cb, cva, cvb, za, zb, hub_dir, outp = sys.argv[1:8]
    B = json.load(open(cb)); hub = json.load(open(f"{hub_dir}/hub.json")); assign = json.load(open(f"{hub_dir}/corpus_assign.json"))
    entries = {**hub.get("parents", {}), **hub["contrasts"]}
    A, Bv = rows(cva), rows(cvb)
    ids = sorted(set(A) & set(Bv))
    grp = np.array([B["docs"][i].get("group", i) for i in ids]); ug, gi = np.unique(grp, return_inverse=True)
    members = [np.flatnonzero(gi == j) for j in range(len(ug))]; g = np.random.default_rng(0)
    boots = [np.concatenate([members[j] for j in g.integers(0, len(ug), len(ug))]) for _ in range(1000)]
    nf = np.array([A[i][0] for i in ids]); ea = np.array([A[i][1] for i in ids]); eb = np.array([Bv[i][1] for i in ids])

    def label(i):
        a = assign.get(i)
        if a is None:
            return "not normalized"
        k = kind(a["hub"], entries); fwd = a.get("sign", 1) > 0
        if k != "condition":
            return k + ("" if fwd else " reversed (deactivation)")
        own = any(x.startswith("c:") for x in B["docs"][i]["ids"])
        return f"condition{'' if fwd else ' reversed'}, {'own constructs' if own else 'no constructs'}"
    lab = np.array([label(i) for i in ids])

    def diff(mask):
        m = np.flatnonzero(mask)
        d = (eb[m].sum() - ea[m].sum()) / nf[m].sum()
        bs = []
        for b in boots:
            bm = b[mask[b]]
            if len(bm):
                bs.append((eb[bm].sum() - ea[bm].sum()) / nf[bm].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"contrasts": int(len(m)), "A_nats_per_peak": round(float(ea[m].sum() / nf[m].sum()), 4),
                "B_minus_A": round(float(d), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4)}
    R = {"cv": {"all": diff(np.ones(len(ids), bool))}}
    for L in sorted(set(lab)):
        R["cv"][L] = diff(lab == L)
    ZA, ZB = json.load(open(za))["tasks"], json.load(open(zb))["tasks"]
    ts = sorted(set(ZA) & set(ZB))
    ga = np.array([ZA[t]["constructs"]["gain_nats_per_peak"] for t in ts]); gb = np.array([ZB[t]["constructs"]["gain_nats_per_peak"] for t in ts])
    R["heldout_tasks"] = {"tasks": len(ts), "A_mean_gain": round(float(ga.mean()), 4), "B_mean_gain": round(float(gb.mean()), 4),
                          "B_higher": int((gb > ga).sum()), "A_above_average_map": int((ga > 0).sum()), "B_above_average_map": int((gb > 0).sum())}
    print(json.dumps(R, indent=1)); json.dump(R, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
