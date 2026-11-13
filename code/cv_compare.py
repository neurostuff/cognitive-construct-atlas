"""Held-out-contrast comparison of task-networks CV runs, paired over papers.

Each CV file is the repo's `cv` output (one row per held-out contrast: id, peaks, the contrast's encoding gain in
nats over the average map, decoding scores). A run's score is total gain over total peaks (nats per peak), as in
atm_ksweep.sh. Differences
between a run and the reference are paired by contrast and resampled by paper (2,000 bootstrap draws, seed 0),
since contrasts of one paper share its peaks' coordinate space and sample.

usage: python cv_compare.py CORPUS REF.npy OTHER.npy [...]   (from pilot-results/atm; any venv with numpy)
"""
import json
import sys

import numpy as np


def load(path):
    x = np.load(path, allow_pickle=True)
    return {r[0]: (float(r[2]), int(r[1])) for r in x}


def compare(corpus, ref, others, draws=2000, seed=0):
    """[{run, score, diff, lo, hi, contrasts, papers}] for each run in `others` against `ref` (diff = run - ref,
    nats per peak, 95% paper-bootstrap interval); the first row is the reference itself"""
    paper = {d: v["group"] for d, v in json.load(open(corpus))["docs"].items()}
    R = load(ref)
    rows = [{"run": ref, "score": sum(v[0] for v in R.values()) / sum(v[1] for v in R.values()), "diff": 0.0,
             "lo": 0.0, "hi": 0.0, "contrasts": len(R), "papers": len({paper[i] for i in R})}]
    rng = np.random.default_rng(seed)
    for o in others:
        O = load(o)
        ids = sorted(set(R) & set(O))
        papers = sorted({paper[i] for i in ids})
        pidx = {p: j for j, p in enumerate(papers)}
        d = np.array([O[i][0] - R[i][0] for i in ids]); pk = np.array([R[i][1] for i in ids])
        g = np.array([pidx[paper[i]] for i in ids])
        s = np.bincount(g, weights=d, minlength=len(papers)); n = np.bincount(g, weights=pk, minlength=len(papers))
        boot = []
        for _ in range(draws):
            b = rng.integers(0, len(papers), len(papers))
            boot.append(s[b].sum() / n[b].sum())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"run": o, "score": sum(O[i][0] for i in ids) / pk.sum(), "diff": float(d.sum() / pk.sum()),
                     "lo": float(lo), "hi": float(hi), "contrasts": len(ids), "papers": len(papers)})
    return rows


def main():
    corpus, ref, others = sys.argv[1], sys.argv[2], sys.argv[3:]
    rows = compare(corpus, ref, others)
    print(f"{ref}: {rows[0]['score']:.4f} nats/peak over {rows[0]['contrasts']} contrasts")
    for r in rows[1:]:
        print(f"{r['run']}: {r['score']:.4f}; minus reference {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] "
              f"over {r['contrasts']} contrasts, {r['papers']} papers")


if __name__ == "__main__":
    main()
