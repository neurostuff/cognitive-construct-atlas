"""Score a seed pilot: how much do single-restart fits from different seeds agree, plain EM vs annealed EM?

For each mode (a set of fits OUT_DIR/<mode>_s<seed>), from the network maps beta:
- pairwise agreement: every pair of runs matched one-to-one (Hungarian on Pearson r of beta rows over voxels); the
  mean matched r, and the share of networks matched at r >= MATCH_R;
- most typical run (as the CBIG Yeo2015 author-topic code's ComputeMostTypicalSolution): the run with the lowest mean
  (1 - matched r) to all others; for each of its networks, recurrence = the share of the other runs with a match at
  r >= MATCH_R;
- Icasso-style clusters (Himberg et al. 2004, doi:10.1016/j.neuroimage.2004.03.027): all runs' networks pooled,
  average-linkage clustering on 1 - r cut into K clusters; per cluster, Iq = mean r inside - mean r to the rest, and
  how many distinct runs it holds;
- log-likelihood: spread, and whether the best-likelihood run is the most typical;
- shared-pattern author: per run, the share of its top network it owns (construct_map.shared_network's rule).

usage: python pilot_score.py CORPUS OUT_DIR MODE [MODE ...]   (cbmr-gpu venv; from atm)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
from construct_map import expected_counts  # noqa: E402

MATCH_R = 0.8
z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)


def matched(a, b):
    r = a @ b.T / a.shape[1]; i, j = linear_sum_assignment(-r); m = np.empty(len(a)); m[i] = r[i, j]
    return m


def main():
    corpus, out, modes = sys.argv[1], sys.argv[2], sys.argv[3:]
    C = load_corpus(corpus); bg = C["nodes"].index("background")
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    typ = {}
    for mode in modes:
        fs = sorted(glob.glob(f"{out}/{mode}_s*_beta.npy"), key=lambda f: int(re.search(r"_s(\d+)_", f).group(1)))
        pre = [f[:-9] for f in fs]; B = [z(np.load(f).astype(float)) for f in fs]
        meta = [json.load(open(f"{p}.json")) for p in pre]; ll = np.array([m["ll_tokens"] for m in meta])
        n = len(B); K = len(B[0])
        M = {(i, j): matched(B[i], B[j]) for i in range(n) for j in range(n) if i != j}
        pair_r = np.mean([M[i, j].mean() for i in range(n) for j in range(i + 1, n)])
        pair_share = np.mean([(M[i, j] >= MATCH_R).mean() for i in range(n) for j in range(i + 1, n)])
        dist = np.array([np.mean([1 - M[i, j].mean() for j in range(n) if j != i]) for i in range(n)])
        t = int(np.argmin(dist)); typ[mode] = B[t]
        rec = np.array([np.mean([M[t, j][k] >= MATCH_R for j in range(n) if j != t]) for k in range(K)])
        # Icasso-style clustering of all n*K networks
        A = np.vstack(B); S = np.clip(A @ A.T / A.shape[1], -1.0, 1.0); np.fill_diagonal(S, 1.0)  # rounding can exceed 1
        Z = linkage(squareform(1 - S, checks=False), "average"); lab = fcluster(Z, K, "maxclust")
        run = np.repeat(np.arange(n), K); iq = []
        for c in np.unique(lab):
            inn = lab == c
            sin = S[np.ix_(inn, inn)][~np.eye(inn.sum(), dtype=bool)].mean() if inn.sum() > 1 else 1.0
            sout = S[np.ix_(inn, ~inn)].mean()
            iq.append((sin - sout, len(set(run[inn])), inn.sum()))
        iq.sort(reverse=True)
        own = []
        for p in pre:
            be = np.load(f"{p}_beta.npy").astype(float); th = np.load(f"{p}_theta.npy").astype(float)
            nn = expected_counts(C, th, be); o = nn[bg] / nn.sum(0); own.append(float(o.max()))
        nm = [names[str(int(np.argmax(r)))].split(":")[0] for r in B[t] @ z(b0).T / B[t].shape[1]]
        its = [m["iterations"] for m in meta]; secs = [m["seconds"] for m in meta]
        print(f"== {mode}: {n} runs at K {K}; iterations {min(its)}-{max(its)} (median {int(np.median(its))}), "
              f"{np.median(secs):.0f} s each")
        print(f"   log-likelihood: best {ll.max():.0f}, range {ll.max() - ll.min():.0f} nats, sd {ll.std():.0f}; "
              f"best run is seed index {int(np.argmax(ll))}, most typical is {t} (its ll {ll[t] - ll.max():+.0f} vs best)")
        print(f"   pairwise: mean matched r {pair_r:.3f}; networks matched at r >= {MATCH_R}: {pair_share:.0%}")
        print(f"   most typical run's networks recurring (r >= {MATCH_R}) in >= 80% of the others: {(rec >= 0.8).sum()} "
              f"of {K}; in >= 50%: {(rec >= 0.5).sum()}")
        print("   recurrence: " + ", ".join(f"{nm[k]} {rec[k]:.2f}" for k in np.argsort(-rec)))
        print(f"   Icasso clusters (K {K}): Iq " + ", ".join(f"{q:.2f}({r_}/{n} runs, {m_})" for q, r_, m_ in iq))
        print(f"   clusters holding >= 80% of runs with Iq >= 0.5: {sum(1 for q, r_, m_ in iq if q >= 0.5 and r_ >= 0.8 * n)}")
        print(f"   shared author owns >= 0.6 of a network in {sum(o >= 0.6 for o in own)} of {n} runs "
              f"(ownership {min(own):.2f}-{max(own):.2f})")
    if len(modes) == 2:
        m = matched(typ[modes[0]], typ[modes[1]])
        print(f"most typical {modes[0]} vs most typical {modes[1]}: {(m >= MATCH_R).sum()} of {len(m)} networks match at "
              f"r >= {MATCH_R}; mean r {m.mean():.3f}")


if __name__ == "__main__":
    main()
