"""Hemispheric laterality of construct maps: LI = (L - R) / (L + R) of P(voxel | construct) (shared-pattern network
removed, construct_map.py's map), over voxels with |x| > 4 mm; > 0 is left. Also on the enrichment map's positive
part where P >= 20% of its maximum, and the construct's top networks.

usage: python laterality.py FIT_PREFIX CORPUS "label" ["label" ...]   (cbmr-gpu venv; from pilot-results/atm)
"""
import json
import sys
from pathlib import Path

import numpy as np

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
from construct_map import pooled, reference, shared_network  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402


def main():
    prefix, corpus, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
    th = np.load(f"{prefix}_theta.npy").astype(float); be = np.load(f"{prefix}_beta.npy").astype(float)
    xyz = np.load(f"{prefix}_xyz.npy"); nodes = json.load(open(f"{prefix}.json"))["nodes"]
    H = json.load(open(corpus)); lab = H["labels"]; C = load_corpus(corpus)
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    r = z(be) @ z(b0).T / be.shape[1]; i, j = linear_sum_assignment(-r)
    name = {a: names[str(b)].split(":")[0] for a, b in zip(i, j)}
    for a in range(len(be)):  # K > 16: networks left unassigned take their best match, marked "+"
        name.setdefault(a, names[str(int(np.argmax(r[a])))].split(":")[0] + "+")
    bar, n, _ = pooled(C, th, be)
    shared = [shared_network(C, n)]
    glob = reference(C, n); glob[shared] = 0; glob /= glob.sum(); p_all = glob @ be
    L, Rt = xyz[:, 0] < -4, xyz[:, 0] > 4
    li = lambda v: (v[L].sum() - v[Rt].sum()) / (v[L].sum() + v[Rt].sum())
    print(f"{prefix}: average task LI {li(p_all):+.3f}")
    for label in labels:
        node = next((x for x in nodes if x.startswith("c:") and lab.get(x, "").lower() == label.lower()), None)
        if node is None:
            print(f"  {label}: not in this fit"); continue
        k = nodes.index(node); mix = bar[k].copy(); mix[shared] = 0; mix /= mix.sum()
        p = mix @ be; e = np.log2(p / p_all); ep = np.where(p >= 0.2 * p.max(), np.clip(e, 0, None), 0)
        top = "; ".join(f"{name[q]} {mix[q]:.2f}" for q in np.argsort(-mix)[:4])
        print(f"  {label}: LI of P {li(p):+.3f}, of enrichment {li(ep):+.3f}; {top}")


if __name__ == "__main__":
    main()
