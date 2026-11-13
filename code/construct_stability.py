"""Are construct maps stable across seeds even where the networks are not? (James 2026-11-11: determine construct map
stability.)

A construct's map is its pooled mixture times the network maps (construct_map.pooled, shared network removed by
construct_map.shared_network, renormalized), so one seed splitting a region into two networks and another keeping it
whole can give the same construct map. For every fit of one mode (OUT_DIR/<mode>_s<seed>), each construct's
probability map P and enrichment map (log2 P / P(average task)) are built exactly as construct_maps_all.py builds them;
then, per construct:
- P stability: mean Pearson r of its P map between every pair of seeds, over all grid voxels;
- enrichment stability: the same for the enrichment map, over the voxels where the seed-averaged P is >= 20% of its
  maximum (the displayed region, construct_maps_all.ENRICH_MASK);
- separability: the construct's mean cross-seed r minus its largest mean cross-seed r with any other construct whose
  task set differs (constructs on the same task set have the same map by construction, METHODS section 7).
Also, as a sensitivity check, P stability without removing the shared network.

usage:
  python construct_stability.py counts CORPUS PREFIX            # cache the E-step counts (PREFIX_n.npy)
  python construct_stability.py score CORPUS OUT_DIR MODE MIN_CONTRASTS   (cbmr-gpu venv; from atm)
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
import construct_map as cm  # noqa: E402

ENRICH_MASK = 0.2


def zrows(X):
    X = X - X.mean(1, keepdims=True)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def main():
    cmd, corpus = sys.argv[1], sys.argv[2]
    C = load_corpus(corpus)
    if cmd == "counts":
        p = sys.argv[3]
        th = np.load(f"{p}_theta.npy").astype(float); be = np.load(f"{p}_beta.npy").astype(float)
        np.save(f"{p}_n.npy", cm.expected_counts(C, th, be)); print("wrote", f"{p}_n.npy"); return
    out, mode, minc = sys.argv[3], sys.argv[4], int(sys.argv[5])
    H = json.load(open(corpus)); lab = H["labels"]; nodes = C["nodes"]
    fs = sorted(glob.glob(f"{out}/{mode}_s*_n.npy"), key=lambda f: int(re.search(r"_s(\d+)_n", f).group(1)))
    # constructs, their subtree contrasts and task sets
    N0, _, children = cm.pooled_mass(C, np.load(fs[0]))
    task = np.array([cm.is_task_author(x) and not x.startswith("c:") for x in nodes])
    memo = {}

    def below(i):
        if i not in memo:
            s = {i} if task[i] else set()
            for c in children[i]:
                s |= below(c)
            memo[i] = frozenset(s)
        return memo[i]
    under = [set() for _ in nodes]
    for d, auth in enumerate(C["auth"]):
        stack, seen = [int(a) for a in auth], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); under[x].add(d); stack.extend(int(q) for q in C["parents"][x])
    cons = [i for i, x in enumerate(nodes) if x.startswith("c:") and len(under[i]) >= minc and below(i)]
    P, E, P0 = [], [], []
    for f in fs:
        pre = f[:-6]; n = np.load(f); be = np.load(f"{pre}_beta.npy").astype(float)
        N, g, _ = cm.pooled_mass(C, n)
        bar = N[cons] + cm.ALPHA_PRIOR * g; bar /= bar.sum(1, keepdims=True)
        sh = cm.shared_network(C, n)
        mix = bar.copy(); mix[:, sh] = 0; mix /= mix.sum(1, keepdims=True)
        gg = g.copy(); gg[sh] = 0; gg /= gg.sum()
        p = mix @ be; pa = gg @ be
        P.append(p); E.append(np.log2(p / pa)); P0.append(bar @ be)
    P, E, P0 = np.array(P), np.array(E), np.array(P0)   # seeds x constructs x voxels
    S, Cn, V = P.shape
    mean_p = P.mean(0)
    stab_p, stab_e, stab_p0 = np.zeros(Cn), np.zeros(Cn), np.zeros(Cn)
    iu = np.triu_indices(S, 1)
    for c in range(Cn):
        Z = zrows(P[:, c]); stab_p[c] = (Z @ Z.T)[iu].mean()
        Z0 = zrows(P0[:, c]); stab_p0[c] = (Z0 @ Z0.T)[iu].mean()
        m = mean_p[c] >= ENRICH_MASK * mean_p[c].max()
        Ze = zrows(E[:, c][:, m]); stab_e[c] = (Ze @ Ze.T)[iu].mean()
    # separability: cross-seed r with other constructs (different task sets), averaged over seed pairs
    Zs = np.array([zrows(P[s]) for s in range(S)])           # seeds x constructs x voxels
    cross = np.zeros((Cn, Cn)); npair = 0
    for a, b in zip(*iu):
        cross += Zs[a] @ Zs[b].T; npair += 1
    cross /= npair
    same = np.array([[below(cons[i]) == below(cons[j]) for j in range(Cn)] for i in range(Cn)])
    other = np.where(same, -np.inf, cross); best_other = other.max(1); sep = stab_p - best_other
    mass = np.array([N0[c].sum() for c in cons]); size = np.array([len(under[c]) for c in cons])
    L = lambda i: lab.get(nodes[cons[i]], nodes[cons[i]][2:])
    q = lambda x: f"{np.percentile(x, 10):.3f} / {np.median(x):.3f} / {np.percentile(x, 90):.3f}"
    print(f"{mode}: {S} seeds, {Cn} constructs with >= {minc} contrasts ({len(set(below(c) for c in cons))} distinct task sets)")
    print(f"  P map, mean cross-seed r (10th / median / 90th pct): {q(stab_p)}; without removing the shared network {q(stab_p0)}")
    print(f"  enrichment map (displayed region), mean cross-seed r: {q(stab_e)}")
    print(f"  constructs with P r >= 0.9: {(stab_p >= 0.9).mean():.0%}; >= 0.8: {(stab_p >= 0.8).mean():.0%}; "
          f"enrichment r >= 0.8: {(stab_e >= 0.8).mean():.0%}")
    print(f"  separable (own cross-seed r above the closest other construct's): {(sep > 0).mean():.0%}; "
          f"margin median {np.median(sep):+.3f}")
    print(f"  Spearman(P stability, credited mass) = "
          f"{np.corrcoef(np.argsort(np.argsort(stab_p)), np.argsort(np.argsort(mass)))[0, 1]:+.2f}")
    o = np.argsort(stab_p)
    print("  least stable (P r, enrichment r, contrasts): " + "; ".join(f"{L(i)} {stab_p[i]:.2f}/{stab_e[i]:.2f} ({size[i]})" for i in o[:10]))
    show = ["response inhibition", "working memory", "memory", "episodic memory", "theory of mind", "fear", "reward processing",
            "selective attention", "decision making", "language", "pain", "motor control", "attention", "cognitive control",
            "face perception", "emotion regulation"]
    idx = {L(i): i for i in range(Cn)}
    print("  panel (P r / enrichment r / closest other construct r): " + "; ".join(
        f"{s_} {stab_p[idx[s_]]:.2f}/{stab_e[idx[s_]]:.2f}/{best_other[idx[s_]]:.2f}" for s_ in show if s_ in idx))
    json.dump({"mode": mode, "seeds": S, "constructs": [{"label": L(i), "contrasts": int(size[i]), "credited_mass": float(mass[i]),
               "p_r": float(stab_p[i]), "enrichment_r": float(stab_e[i]), "p_r_with_shared": float(stab_p0[i]),
               "closest_other_r": float(best_other[i])} for i in range(Cn)]},
              open(f"{out}/construct_stability_{mode}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
