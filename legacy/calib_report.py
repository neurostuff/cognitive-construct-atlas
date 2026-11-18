"""Calibration and cutoff transfer of a cross-validated claim classifier (preds/clf_<tag>.jsonl).

Scores are logit margins z (clf_cv2.py) or probabilities p (clf_cv.py, turned into logits). For each held-out fold,
Platt scaling (q = sigmoid(a z + b), fitted by Newton's method on the other five folds' out-of-fold scores) turns
scores into probabilities the fold never saw. Then:
- reliability: mean q against the observed claim rate, in bins, pooled over folds;
- cutoff transfer: per fold, the units with q >= 0.8 or 0.9, how many, and the share that are claims;
- the same after dropping units the off-target check calls "another task" (when the file has `names`);
- rank only: the most units at pooled precision >= 0.9, and the top k per paper.

usage: python calib_report.py TAG [TAG ...]   (run from cc/ or pilot-results/)
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if not (ROOT / "preds").exists():
    ROOT = ROOT.parent


def sig(x):
    return 1 / (1 + math.exp(-x)) if x > -30 else 0.0


def platt(z, y, iters=100):
    """a, b minimising the log loss of sigmoid(a z + b): Newton steps, halved until the loss falls"""
    def loss(a, b):
        out = 0.0
        for zi, yi in zip(z, y):
            t = a * zi + b
            out += math.log1p(math.exp(-abs(t))) + max(t, 0) - yi * t  # stable log(1 + e^t) - y t
        return out
    a, b = 1.0, 0.0; cur = loss(a, b)
    for _ in range(iters):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for zi, yi in zip(z, y):
            q = sig(a * zi + b); w = q * (1 - q) + 1e-9
            g0 += (q - yi) * zi; g1 += q - yi
            h00 += w * zi * zi; h01 += w * zi; h11 += w
        det = h00 * h11 - h01 * h01
        da, db = (h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det
        step = 1.0
        while step > 1e-6 and loss(a - step * da, b - step * db) > cur:
            step /= 2
        a, b = a - step * da, b - step * db
        new = loss(a, b)
        if cur - new < 1e-7:
            break
        cur = new
    return a, b


def report(tag):
    R = [json.loads(l) for l in open(ROOT / "preds" / f"clf_{tag}.jsonl")]
    for r in R:
        if "z" not in r:
            p = min(max(r["p"], 1e-6), 1 - 1e-6); r["z"] = math.log(p / (1 - p))
    for f in range(6):
        tr = [r for r in R if r["fold"] != f]
        a, b = platt([r["z"] for r in tr], [r["y"] for r in tr])
        for r in R:
            if r["fold"] == f:
                r["q"] = sig(a * r["z"] + b)
    print(f"\n=== {tag}: {len(R)} units, {sum(r['y'] for r in R)} claims")
    bins = [0, .05, .2, .4, .6, .8, .9, .95, 1.0001]; ece = 0
    print("  reliability after Platt (held-out):")
    for lo, hi in zip(bins, bins[1:]):
        b = [r for r in R if lo <= r["q"] < hi]
        if b:
            mq = sum(r["q"] for r in b) / len(b); obs = sum(r["y"] for r in b) / len(b)
            ece += len(b) / len(R) * abs(mq - obs)
            print(f"    q in [{lo:.2f},{min(hi, 1):.2f}): n {len(b):5d}  mean q {mq:.2f}  observed {obs:.2f}")
    print(f"  ECE {ece:.3f}")
    filt = [("all units", lambda r: True)]
    if "names" in R[0]:
        filt.append(("off-target dropped", lambda r: r["names"] != "other"))
    for name, ok in filt:
        for t in (0.8, 0.9):
            row, tp, n = [], 0, 0
            for f in range(6):
                k = [r for r in R if r["fold"] == f and r["q"] >= t and ok(r)]
                s = sum(r["y"] for r in k); tp += s; n += len(k)
                row.append(f"{s / len(k):.2f}({len(k)})" if k else "-(0)")
            print(f"  {name:18s} q>={t}: pooled {tp / max(n, 1):.2f} ({n}) | per fold " + " ".join(row))
    for name, ok in filt:
        S = sorted([r for r in R if ok(r)], key=lambda r: -r["z"]); tp = 0; best = 0
        for k, r in enumerate(S, 1):
            tp += r["y"]
            if tp / k >= 0.9:
                best = k
        by = defaultdict(list)
        for r in S:
            by[r["article"]].append(r)
        tops = []
        for k in (2, 4):
            sel = [x for v in by.values() for x in v[:k]]
            tops.append(f"top {k}/paper {sum(x['y'] for x in sel) / len(sel):.2f}")
        print(f"  {name:18s} rank: {best} units at precision >= 0.9; " + ", ".join(tops))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        report(t)
