"""Compare a fit's networks with the original K=16 networks (Pearson r of beta rows over grid voxels), and draw them.

Per original network: its two best-matching networks in the fit (a split shows as two good matches). Per network of
the fit: its best original match (r < NEW_R is called new), and the authors that dominate it
(construct_map.pooled's own counts). Writes OUT_PNG: one glass brain per network, P(voxel | network) scaled to its
maximum, titled with its best original match and r and its top author.

usage: python network_compare.py FIT_PREFIX CORPUS OUT_PNG   (cbmr-gpu venv; from pilot-results/atm)
"""
import json
import sys
from pathlib import Path

import numpy as np

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
from construct_map import pooled  # noqa: E402

NEW_R = 0.8


def main():
    prefix, corpus, out = sys.argv[1], sys.argv[2], sys.argv[3]
    th = np.load(f"{prefix}_theta.npy").astype(float); be = np.load(f"{prefix}_beta.npy").astype(float)
    xyz = np.load(f"{prefix}_xyz.npy"); nodes = json.load(open(f"{prefix}.json"))["nodes"]
    H = json.load(open(corpus)); lab = H["labels"]; C = load_corpus(corpus)
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    r = z(be) @ z(b0).T / be.shape[1]  # fit x original
    _, n, _ = pooled(C, th, be)
    tot = n.sum(0)
    print(f"{prefix} (K {len(be)}) against the original K=16 networks")
    print("original network -> best two networks of this fit (r)")
    for j in range(len(b0)):
        a = np.argsort(-r[:, j])[:2]
        split = "  <-- split" if r[a[1], j] >= 0.5 else ""
        print(f"  {names[str(j)].split(':')[0]:32s} #{a[0]} {r[a[0], j]:.2f}, #{a[1]} {r[a[1], j]:.2f}{split}")
    print(f"networks of this fit: best original match; r < {NEW_R} is new")
    titles = []
    for k in np.argsort(-tot):
        j = int(np.argmax(r[k])); top = np.argsort(-n[:, k])[:2]
        who = ", ".join(f"{lab.get(nodes[a], nodes[a])} {n[a, k] / tot[k]:.2f}" for a in top)
        tag = "NEW " if r[k, j] < NEW_R else ""
        print(f"  #{k:<2d} {tag}{names[str(j)].split(':')[0]} (r {r[k, j]:.2f}); mass {tot[k] / tot.sum():.3f}; top authors: {who}")
        titles.append((k, f"#{k} {tag}{names[str(j)].split(':')[0][:26]} r={r[k, j]:.2f}\n{lab.get(nodes[top[0]], nodes[top[0]])[:34]}"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    from nilearn import plotting
    from nimare.utils import get_template
    tmpl = get_template("mni152_2mm", mask="brain"); aff2 = tmpl.affine
    ijk = np.round((np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(aff2).T)[:, :3]).astype(int) // 2
    shape = tuple(int(x) for x in np.array(tmpl.shape) // 2 + 1); aff4 = aff2 @ np.diag([2, 2, 2, 1])
    cols = 4; rows = (len(be) + cols - 1) // cols
    fig = plt.figure(figsize=(5.2 * cols, 2.5 * rows))
    for i, (k, title) in enumerate(titles):
        ax = fig.add_subplot(rows, cols, i + 1)
        v = np.zeros(shape, np.float32); v[tuple(ijk.T)] = be[k] / be[k].max()
        plotting.plot_glass_brain(nib.Nifti1Image(v, aff4), threshold=0.3, vmin=0, vmax=1, cmap="hot", plot_abs=False,
                                  colorbar=False, axes=ax, display_mode="lzr", title=title)
    fig.suptitle(f"{prefix}: networks, P(voxel | network) scaled to max (> 0.3), by mass", y=0.995)
    fig.savefig(out, dpi=90); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
