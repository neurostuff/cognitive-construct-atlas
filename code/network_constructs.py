"""How a fit's networks map onto constructs (James 2026-11-11): which constructs each network is specific to, whether
spatially similar networks load on different constructs, and why a construct's mixture can be diffuse.

All from construct_map.pooled: theta-bar per node (subtree-pooled expected peak mass, +1 per network), with the
shared-pattern network removed and renormalized, against the average task g (construct_map.reference, same removal).
- loading enrichment of network k in construct c: log2 thetabar_ck / g_k;
- constructs counted only with >= MIN_CONTRASTS contrasts below them (the +1 pseudocount pulls small ones to uniform);
- network similarity: Pearson r of beta rows over voxels; for a similar pair (a, b), the profile correlation across
  constructs of their loading enrichments, and the constructs most tilted to each (log2 of the a:b ratio over g's);
- diffuseness: KL(thetabar_c || g) in bits, ranked among constructs; for one construct (ARG 3), its subtree's tasks,
  families and contrast share, and the spread of its tasks' own mixtures (mean KL of each task to their mean).

usage: python network_constructs.py FIT_PREFIX CORPUS [CONSTRUCT_LABEL]   (cbmr-gpu venv; from pilot-results/atm)
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
from construct_map import pooled, pooled_mass, reference, shared_network  # noqa: E402

MIN_CONTRASTS = 100
SIMILAR_R = 0.3


def kl(p, q):
    return float(np.sum(p * np.log2(p / q)))


def main():
    prefix, corpus = sys.argv[1], sys.argv[2]
    focus = sys.argv[3] if len(sys.argv) > 3 else "selective attention"
    th = np.load(f"{prefix}_theta.npy").astype(float); be = np.load(f"{prefix}_beta.npy").astype(float)
    nodes = json.load(open(f"{prefix}.json"))["nodes"]; H = json.load(open(corpus)); lab = H["labels"]
    C = load_corpus(corpus); K = len(be)
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    r0 = z(be) @ z(b0).T / be.shape[1]
    name = {k: f"{names[str(int(np.argmax(r0[k])))].split(':')[0]} #{k}" for k in range(K)}
    bar, n, children = pooled(C, th, be)
    shared = shared_network(C, n); name[shared] = f"Shared pattern #{shared}"
    keep = [k for k in range(K) if k != shared]
    g = reference(C, n)[keep]; g /= g.sum()
    tb = bar[:, keep]; tb = tb / tb.sum(1, keepdims=True)
    # contrasts under each node
    under = [set() for _ in nodes]
    for d, auth in enumerate(C["auth"]):
        stack, seen = [int(a) for a in auth], set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); under[x].add(d); stack.extend(int(p) for p in C["parents"][x])
    ndoc = len(C["auth"])
    cons = [i for i, x in enumerate(nodes) if x.startswith("c:") and len(under[i]) >= MIN_CONTRASTS]
    L = np.log2(tb[cons] / g)  # constructs x networks
    L_ = lambda i: lab.get(nodes[cons[i]], nodes[cons[i]][2:])
    print(f"{prefix}: K {K}, shared network {name[shared]} removed; {len(cons)} constructs with >= {MIN_CONTRASTS} "
          f"contrasts\n")
    print("1. Each network's most enriched constructs (log2 thetabar/g; 1 = 2x the corpus share)")
    task_share = {}
    kind = np.array(["construct" if x.startswith("c:") else "mod/mat" if x.startswith("m:") else
                     "shared" if x == "background" else "task" for x in nodes])
    for j, k in enumerate(keep):
        task_share[k] = n[kind == "task", k].sum() / n[:, k].sum()
    for j in np.argsort(-g):
        k = keep[j]; o = np.argsort(-L[:, j])[:5]
        print(f"  {name[k]:34s} g {g[j]:.3f} task-owned {task_share[k]:.2f} | "
              + ", ".join(f"{L_(i)} {L[i, j]:+.2f}" for i in o)
              + f" | constructs >= 2x: {(L[:, j] >= 1).sum()}")
    print("\n   Each construct's largest network share (of its non-shared mixture): "
          + ", ".join(f"{q:.2f}" for q in np.percentile(tb[cons].max(1), [10, 50, 90])) + " (10th, 50th, 90th pct)")
    # 2. similar pairs
    zb = z(be[keep]); rr = zb @ zb.T / be.shape[1]
    pairs = sorted(((rr[a, b], a, b) for a in range(len(keep)) for b in range(a + 1, len(keep)) if rr[a, b] >= SIMILAR_R),
                   reverse=True)
    print(f"\n2. Spatially similar network pairs (r >= {SIMILAR_R}) and whether constructs tell them apart")
    for r_, a, b in pairs:
        pr = np.corrcoef(L[:, a], L[:, b])[0, 1]
        tilt = L[:, a] - L[:, b]
        ta, tb_ = np.argsort(-tilt)[:4], np.argsort(tilt)[:4]
        print(f"  {name[keep[a]]} vs {name[keep[b]]}: spatial r {r_:.2f}; loading-profile r across constructs {pr:+.2f}; "
              f"task-owned {task_share[keep[a]]:.2f} vs {task_share[keep[b]]:.2f}")
        print(f"     toward {name[keep[a]]}: " + ", ".join(f"{L_(i)} {tilt[i]:+.1f}" for i in ta))
        print(f"     toward {name[keep[b]]}: " + ", ".join(f"{L_(i)} {tilt[i]:+.1f}" for i in tb_))
    # 3. diffuseness
    D = np.array([kl(tb[c], g) for c in cons]); order = np.argsort(D)
    print(f"\n3. Diffuseness: KL(thetabar || g) in bits over {len(cons)} constructs; median {np.median(D):.3f}")
    print("   most average-like: " + ", ".join(f"{L_(i)} {D[i]:.3f} ({len(under[cons[i]])})" for i in order[:12]))
    print("   most distinctive:  " + ", ".join(f"{L_(i)} {D[i]:.3f} ({len(under[cons[i]])})" for i in order[::-1][:8]))
    size = np.array([len(under[c]) for c in cons])
    print(f"   Spearman(KL, contrasts below) = {np.corrcoef(np.argsort(np.argsort(D)), np.argsort(np.argsort(size)))[0, 1]:+.2f}")
    f = next((c for c in cons if lab.get(nodes[c], nodes[c][2:]) == focus), None)
    if f is None:
        print(f"\n{focus!r} not among the constructs counted"); return
    i = cons.index(f)
    print(f"\n4. {focus}: {len(under[f])} contrasts ({len(under[f]) / ndoc:.1%} of the corpus); KL {D[i]:.3f} bits, rank "
          f"{int(np.where(order == i)[0][0]) + 1} of {len(cons)} from the most average-like")
    print("   parents: " + ", ".join(lab.get(nodes[p], nodes[p]) for p in C["parents"][f]))
    kids = [c for c in children[f]]
    print("   child constructs: " + ", ".join(lab.get(nodes[c], nodes[c]) for c in kids if nodes[c].startswith("c:")))
    # tasks in its subtree
    def subtree(x, acc):
        for c in children[x]:
            if c not in acc:
                acc.add(c); subtree(c, acc)
        return acc
    sub = subtree(f, set())
    tasks = [t for t in sub if kind[t] == "task"]
    tc = Counter({t: len(under[t]) for t in tasks})
    fam_of = {t: [lab.get(nodes[p], nodes[p]) for p in C["parents"][t] if kind[p] == "task"] for t in tasks}
    fams = Counter(fm for t in tasks for fm in fam_of[t] for _ in [0])
    print(f"   {len(tasks)} tasks below it, in {len(fams)} families; largest tasks: "
          + ", ".join(f"{lab.get(nodes[t], nodes[t])} {c}" for t, c in tc.most_common(12)))
    print("   families: " + ", ".join(f"{fm} {c}" for fm, c in fams.most_common(12)))
    # spread of its tasks' own mixtures, vs other constructs' tasks
    def own(t):
        v = n[t, keep] + 1.0; return v / v.sum()
    def spread(x):
        ts = [t for t in subtree(x, set()) if kind[t] == "task" and len(under[t]) >= 20]
        if len(ts) < 3:
            return None, len(ts)
        M = np.array([own(t) for t in ts]); m = M.mean(0)
        return float(np.mean([kl(v, m) for v in M])), len(ts)
    sp = {c: spread(c) for c in cons}
    vals = np.array([v[0] for v in sp.values() if v[0] is not None])
    if sp[f][0] is None:
        print(f"   spread of its tasks' own mixtures: too few tasks ({sp[f][1]}); other constructs' median {np.median(vals):.3f}")
    else:
        print(f"   spread of its tasks' own mixtures (mean KL to their mean, tasks >= 20 contrasts): {sp[f][0]:.3f} bits "
              f"over {sp[f][1]} tasks; other constructs' median {np.median(vals):.3f}, "
              f"percentile {(vals < sp[f][0]).mean():.0%}")
    for t in tasks:
        print(f"   task {lab.get(nodes[t], nodes[t])}: own-mixture KL to g {kl(own(t), g):.3f} bits; constructs: "
              + ", ".join(lab.get(nodes[p], nodes[p]) for p in C["parents"][t] if kind[p] == "construct"))
    print(f"   its own largest networks: " + ", ".join(f"{name[keep[j]]} {tb[f, j]:.3f} (g {g[j]:.3f})"
                                                       for j in np.argsort(-tb[f])[:6]))
    # which constructs co-occur with it on the same tasks
    co = Counter()
    for t in tasks:
        for p in C["parents"][t]:
            if kind[p] == "construct" and p != f:
                co[lab.get(nodes[p], nodes[p])] += len(under[t])
    print("   constructs sharing its tasks (contrast-weighted): " + ", ".join(f"{c} {v}" for c, v in co.most_common(10)))

    # 5. how much data is behind each estimate
    N = pooled_mass(C, n)[0]
    ids = sorted(H["docs"]); paper = {d: H["docs"][ids[d]]["group"] for d in range(ndoc)}  # load_corpus order
    print("\n5. Data behind an estimate: contrasts below (as in the manifest), papers, and the pooled peak mass actually"
          " credited (task-author share, in contrast-equivalents), vs the prior's"
          " ALPHA_PRIOR")
    show = [focus, "response inhibition", "working memory", "episodic memory", "reward processing", "fear", "pain",
            "language", "motor control", "attention", "cognitive control", "executive function", "emotion regulation"]
    for lbl in show:
        c = next((x for x, nd in enumerate(nodes) if nd.startswith("c:") and lab.get(nd, nd[2:]) == lbl), None)
        if c is None:
            continue
        docs = under[c]; pp = {paper.get(d, d) for d in docs}
        own_tasks = [t for t in children[c] if kind[t] == "task"]
        print(f"   {lbl:20s} contrasts {len(docs):5d}  papers {len(pp):5d}  credited mass {N[c].sum():8.1f} "
              f"({N[c].sum() / max(len(docs), 1):.2f} per contrast)  direct tasks {len(own_tasks)}")


if __name__ == "__main__":
    main()
