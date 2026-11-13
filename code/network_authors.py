"""Who each network of a fit belongs to: the share of its expected peak mass (one E-step, construct_map.pooled's own
counts n, before pooling) from task authors, modality authors (m:stim:*, m:resp:*), material authors (m:mat:*) and
the shared-pattern author, with the network's top authors. Networks are named by their best-correlated original K=16
map (Pearson r of beta rows). Also, for each modality or material author, its own top networks.

usage: python network_authors.py FIT_PREFIX CORPUS   (cbmr-gpu venv; from pilot-results/atm)
"""
import json
import sys
from pathlib import Path

import numpy as np

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
from construct_map import pooled  # noqa: E402


def main():
    prefix, corpus = sys.argv[1], sys.argv[2]
    th = np.load(f"{prefix}_theta.npy").astype(float); be = np.load(f"{prefix}_beta.npy").astype(float)
    nodes = json.load(open(f"{prefix}.json"))["nodes"]; H = json.load(open(corpus)); lab = H["labels"]
    C = load_corpus(corpus)
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    r = z(be) @ z(b0).T / be.shape[1]
    name = {k: f"{names[str(int(np.argmax(r[k])))].split(':')[0]} (r {r[k].max():.2f})" for k in range(len(be))}
    _, n, _ = pooled(C, th, be)  # expected counts per node x network (own documents only)
    kind = np.array(["shared" if x == "background" else "material" if x.startswith("m:mat:") else
                     "modality" if x.startswith("m:") else "construct" if x.startswith("c:") else "task" for x in nodes])
    tot = n.sum(0)
    print(f"{prefix}: K {len(be)}; network mass by author kind (share of the network's expected peaks)")
    for k in np.argsort(-tot):
        sh = {t: n[kind == t, k].sum() / tot[k] for t in ("task", "modality", "material", "shared")}
        top = np.argsort(-n[:, k])[:4]
        flag = "  <-- mostly modality/material" if sh["modality"] + sh["material"] > 0.5 else ""
        print(f"  {name[k]:42s} mass {tot[k] / tot.sum():.3f}: task {sh['task']:.2f}, modality {sh['modality']:.2f}, "
              f"material {sh['material']:.2f}, shared {sh['shared']:.2f}; top: "
              + ", ".join(f"{lab.get(nodes[a], nodes[a])} {n[a, k] / tot[k]:.2f}" for a in top) + flag)
    print("modality and material authors: their largest networks (share of the author's own mass)")
    for a in np.where((kind == "modality") | (kind == "material"))[0]:
        if n[a].sum() == 0:
            continue
        s = n[a] / n[a].sum(); top = np.argsort(-s)[:2]
        print(f"  {nodes[a]:26s} " + "; ".join(f"{name[q]} {s[q]:.2f}" for q in top))


if __name__ == "__main__":
    main()
