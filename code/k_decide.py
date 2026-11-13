"""Which K to evaluate next, and the best K (James 2026-11-11: keep increasing K while it helps out-of-sample
prediction, and find the specific best K, to a step of 1).

For one design, from every K evaluated so far, the two out-of-sample scores:
- held-out contrasts (5-fold paper CV), in nats per peak;
- held-out tasks, as the mean per-task gain over the average map.
Search, one round at a time:
1. Bracket: while the largest K evaluated is the best on either score, go up (+4 below 32, then +8, up to KMAX).
   One step past the peak brackets it.
2. Refine: for each score, take the midpoints between its best K and its evaluated neighbours, until they are 1
   apart. A best K at the smallest evaluated K also gets the K below (down to KMIN).
3. Stop when no K is left to evaluate. For each score, every K whose 95% interval against that score's best includes
   0 is tied with it: papers are resampled for held-out contrasts (cv_compare.compare) and tasks for held-out tasks
   (2,000 draws, seed 0). The K to release is the smallest K tied with the best on both scores; if no K is tied on
   both, it is the held-out-contrast best.
Prints "NEXT K [K ...]", "STOP <release K>" or "WAIT" (fewer than two K have results), then the evidence.

usage: python k_decide.py constructs|families   (cbmr-gpu venv; from pilot-results/atm)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_compare import compare  # noqa: E402

KMIN, KMAX = 4, 64
DESIGNS = {  # K -> (cv file, held-out json, side); K=16 comes from the design runs
    "constructs": ("corpus_constructs_both_mat.json", "cv/sweep_K{K}.npy", "zeroshot_sweep_K{K}.json", "constructs",
                   {16: ("cv/K16bx.npy", "zeroshot_K16_both_mat.json")}),
    "families": ("corpus_family_mat.json", "cv/famsweep_K{K}.npy", "zeroshot_famsweep_K{K}.json", "families",
                 {16: ("cv/K16fx.npy", "zeroshot_K16_both_mat.json")}),
}


def step(K):
    return 4 if K < 32 else 8


def files(design):
    corpus, cvp, zp, side, fixed = DESIGNS[design]
    have = dict(fixed)
    for f in glob.glob(cvp.replace("{K}", "*")):
        K = int(re.search(r"K(\d+)\.npy$", f).group(1))
        have.setdefault(K, (f, zp.format(K=K)))
    return corpus, side, {K: v for K, v in have.items() if Path(v[0]).exists() and Path(v[1]).exists()}


def task_gains(path, side):
    T = json.load(open(path))["tasks"]
    return {t: v[side]["gain_nats_per_peak"] for t, v in T.items() if v.get(side, {}).get("gain_nats_per_peak") is not None}


def cv_score(path):
    a = np.load(path, allow_pickle=True)
    return sum(float(r[2]) for r in a) / sum(int(r[1]) for r in a)


def refine(Ks, best):
    """midpoints between best and its evaluated neighbours (or the K below the smallest), not yet evaluated"""
    out = set()
    lower = [K for K in Ks if K < best]; upper = [K for K in Ks if K > best]
    if lower and best - max(lower) > 1:
        out.add((best + max(lower)) // 2)
    if not lower and best > KMIN:
        out.add(max(KMIN, best - 2))
    if upper and min(upper) - best > 1:
        out.add((best + min(upper) + 1) // 2)
    return out


def main():
    design = sys.argv[1]
    corpus, side, have = files(design)
    Ks = sorted(have)
    if len(Ks) < 2:
        print(f"WAIT\n{design}: results for K {Ks} only")
        return
    cv = {K: cv_score(have[K][0]) for K in Ks}
    zs = {K: float(np.mean(list(task_gains(have[K][1], side).values()))) for K in Ks}
    bcv, bzs, top = max(Ks, key=cv.get), max(Ks, key=zs.get), Ks[-1]
    summary = (f"{design}: K " + ", ".join(f"{K}: CV {cv[K]:.4f} / tasks {zs[K]:.4f}" for K in Ks)
               + f"; best CV K {bcv}, best tasks K {bzs}")
    # 1. bracket the peaks
    if top in (bcv, bzs) and top + step(top) <= KMAX:
        print(f"NEXT {top + step(top)}"); print(summary); return
    # 2. refine around each score's best
    nxt = sorted((refine(Ks, bcv) | refine(Ks, bzs)) - set(Ks))
    if nxt:
        print("NEXT " + " ".join(map(str, nxt))); print(summary); return
    # 3. stop: ties with each best, and the K to release
    rows = compare(corpus, have[bcv][0], [have[K][0] for K in Ks if K != bcv])[1:]
    tied_cv = {bcv} | {K for K, r in zip([K for K in Ks if K != bcv], rows) if r["lo"] <= 0 <= r["hi"]}
    gb = task_gains(have[bzs][1], side); rng = np.random.default_rng(0); tied_zs = {bzs}
    for K in Ks:
        if K == bzs:
            continue
        g = task_gains(have[K][1], side); ts = sorted(set(g) & set(gb)); d = np.array([g[t] - gb[t] for t in ts])
        lo, hi = np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)], [2.5, 97.5])
        if lo <= 0 <= hi:
            tied_zs.add(K)
    both = sorted(tied_cv & tied_zs)
    release = both[0] if both else bcv
    print(f"STOP {release}"); print(summary)
    print(f"{design}: tied with the best on held-out contrasts {sorted(tied_cv)}, on held-out tasks {sorted(tied_zs)}; "
          + (f"release K {release}, the smallest tied on both" if both else f"none tied on both: release the CV best {bcv}"))


if __name__ == "__main__":
    main()
