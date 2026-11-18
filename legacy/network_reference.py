"""How well each fit's networks match reference networks, to compare fits of different K (James 2026-11-11: is there
a principled reason to prefer one K among those tied out of sample?).

For each FIT_PREFIX:
- original: the K=16 task-networks maps (K16_seed4); one-to-one assignment (Hungarian on Pearson r of beta rows over
  grid voxels); originals recovered at r >= 0.8, and the mean assigned r;
- Yeo 2011, 17 networks (resting state, cortex), as carried by the Schaefer 2018 400-parcel atlas (cached): for each Yeo network, the best Pearson
  r between its binary mask and any network of the fit, over the grid voxels inside Yeo's cortex mask;
- shared author: the share of the shared-pattern author's mass in its largest network, and the share of that network
  the author owns (one E-step, construct_map.expected_counts). A design-defined nuisance author that owns no network
  has not been separated from the rest.

usage: python network_reference.py CORPUS FIT_PREFIX [FIT_PREFIX ...]   (cbmr-gpu venv; from pilot-results/atm)
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

RECOVERED_R = 0.8


SCHAEFER = "/home/james/nilearn_data/schaefer_2018/Schaefer2018_400Parcels_17Networks_order"


def yeo17():
    """Yeo's 17 networks as a label volume (1..17), from the cached Schaefer 2018 400-parcel, 17-network atlas: each
    parcel's name carries its network ("17Networks_LH_DefaultA_PFCm_1"); returns the image and the network names"""
    import nibabel as nib
    img = nib.load(f"{SCHAEFER}_FSLMNI152_2mm.nii.gz"); P = np.asarray(img.get_fdata()).astype(int)
    net = {}
    for line in open(f"{SCHAEFER}.txt"):
        f = line.split()
        if len(f) >= 2:
            net[int(f[0])] = f[1].split("_")[2]
    names = sorted(set(net.values())); idx = {x: i + 1 for i, x in enumerate(names)}
    lut = np.zeros(P.max() + 1, int)
    for p, x in net.items():
        lut[p] = idx[x]
    return nib.Nifti1Image(lut[P].astype(np.int16), img.affine), names


def main():
    corpus, prefixes = sys.argv[1], sys.argv[2:]
    C = load_corpus(corpus)
    b0 = np.load(f"{R}/K16_seed4_beta.npy")
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    yimg, ynames = yeo17(); Y = np.asarray(yimg.get_fdata()).squeeze().astype(int); inv = np.linalg.inv(yimg.affine)
    for prefix in prefixes:
        be = np.load(f"{prefix}_beta.npy").astype(float); th = np.load(f"{prefix}_theta.npy").astype(float)
        xyz = np.load(f"{prefix}_xyz.npy"); K = len(be)
        r = z(be) @ z(b0).T / be.shape[1]; i, j = linear_sum_assignment(-r)
        rr = r[i, j]
        ijk = np.clip(np.round(np.c_[xyz, np.ones(len(xyz))] @ inv.T)[:, :3].astype(int), 0, np.array(Y.shape) - 1)
        lab = Y[tuple(ijk.T)]; ctx = lab > 0
        bz = z(be[:, ctx]); best = []
        for k in range(1, len(ynames) + 1):
            m = (lab[ctx] == k).astype(float); m = (m - m.mean()) / m.std()
            best.append(float((bz @ m / ctx.sum()).max()))
        n = expected_counts(C, th, be); b = C["nodes"].index("background")
        own = n[b] / n[b].sum(); top = int(np.argmax(own)); owned = n[b, top] / n[:, top].sum()
        print(f"{prefix} (K {K}): originals recovered at r >= {RECOVERED_R}: {(rr >= RECOVERED_R).sum()} of 16, "
              f"mean assigned r {rr.mean():.3f}; Yeo-17 best r mean {np.mean(best):.3f} "
              f"(networks with r >= 0.4: {sum(x >= 0.4 for x in best)}); shared author: {own[top]:.2f} of its mass in "
              f"network {top}, which it owns {owned:.2f} of")
        print("   Yeo-17 best r: " + ", ".join(f"{nm} {x:.2f}" for nm, x in zip(ynames, best)))


if __name__ == "__main__":
    main()
