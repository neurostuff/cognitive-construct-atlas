"""Are a fit's networks reproduced by independent refits at the same K? (James 2026-11-11: will K=20 always split like
that?)

REF and each OTHER are fits of one corpus at one K, from disjoint seeds (each the best of its own restarts). Each
OTHER's networks are matched one-to-one to REF's (Hungarian on Pearson r of beta rows over grid voxels). Reports, per
REF network, the matched r in every OTHER (and their minimum), how many REF networks are reproduced at r >= 0.8 in
every refit, each fit's log-likelihood, and each fit's shared network: the share of its top network the shared-pattern
author owns (construct_map.shared_network's rule) and the r of that network to REF's shared network.

usage: python network_stability.py CORPUS REF_PREFIX OTHER_PREFIX [OTHER_PREFIX ...]   (cbmr-gpu venv; from atm)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
from construct_map import expected_counts  # noqa: E402

REPRODUCED_R = 0.8


def main():
    corpus, ref, others = sys.argv[1], sys.argv[2], sys.argv[3:]
    C = load_corpus(corpus); b = C["nodes"].index("background")
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))

    def load(p):
        be = np.load(f"{p}_beta.npy").astype(float); th = np.load(f"{p}_theta.npy").astype(float)
        n = expected_counts(C, th, be); own = n[b] / n.sum(0); k = int(np.argmax(own))
        return be, json.load(open(f"{p}.json"))["ll_tokens"], k, own[k]
    B, ll, sk, so = load(ref); K = len(B)
    nm = [names[str(int(np.argmax(r)))].split(":")[0] for r in z(B) @ z(b0).T / B.shape[1]]
    print(f"{ref}: K {K}, ll {ll:.0f}; shared network #{sk} (owned {so:.2f})")
    M = []
    for p in others:
        Bo, llo, sko, soo = load(p)
        r = z(B) @ z(Bo).T / B.shape[1]; i, j = linear_sum_assignment(-r); m = np.empty(K); m[i] = r[i, j]
        M.append(m)
        print(f"  {p}: ll {llo:.0f} ({llo - ll:+.0f} vs ref); shared network #{sko} owned {soo:.2f}, r to ref's shared "
              f"{r[sk, sko]:.2f}; ref networks reproduced at r >= {REPRODUCED_R}: {(m >= REPRODUCED_R).sum()} of {K}")
    M = np.array(M); lo = M.min(0)
    print(f"reproduced in every refit (min r >= {REPRODUCED_R}): {(lo >= REPRODUCED_R).sum()} of {K}")
    for k in np.argsort(-lo):
        tag = " (shared)" if k == sk else ""
        print(f"  #{k:2d} {nm[k]:30s}{tag:9s} min r {lo[k]:.2f}; " + " ".join(f"{x:.2f}" for x in M[:, k]))


if __name__ == "__main__":
    main()
