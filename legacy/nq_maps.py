"""NeuroQuery maps (Dockès et al. 2020, eLife 9:e53385) for every construct node and every task of a construct corpus,
sampled at the task-networks grid voxels, for the held-out-task baseline (baselines_heldout.py).

For each text (a construct's label; a task's label), NeuroQuery's `brain_map` (its predicted smoothed z map, MNI152)
is read at each grid voxel's MNI coordinate (nearest voxel). A text whose words are all outside NeuroQuery's
vocabulary gives an all-zero map and is flagged. Note that NeuroQuery's model was trained on ~13,000 neuroimaging
papers that overlap with the task-networks corpus, so this baseline has seen the held-out tasks' papers: an
advantage it keeps in the comparison.

Writes OUT.npz: names (node ids), texts, kind ("construct" | "task"), in_vocab, maps (n x voxels, float32).

usage: python nq_maps.py CORPUS_JSON XYZ_NPY OUT.npz   (venv-nq; NEUROQUERY_DATA set)
"""
import json
import os
import sys

import numpy as np
from neuroquery import NeuroQueryModel, fetch_neuroquery_model


def main():
    corpus, xyz_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    H = json.load(open(corpus)); lab = H["labels"]
    xyz = np.load(xyz_path)
    model = NeuroQueryModel.from_data_dir(fetch_neuroquery_model(os.environ["NEUROQUERY_DATA"]))
    nodes = [n for n in H["parents"] if n.startswith("c:")]
    tasks = [n for n in H["parents"] if not n.startswith(("c:", "m:")) and n != "background"
             and any(p.startswith("c:") for p in H["parents"][n])]
    items = [(n, lab.get(n, n[2:]), "construct") for n in nodes] + [(t, lab.get(t, t), "task") for t in tasks]
    maps, invoc = [], []
    ijk = None
    for n, text, kind in items:
        res = model(text)
        img = res["brain_map"]
        if ijk is None:
            inv = np.linalg.inv(img.affine)
            ijk = np.round((np.c_[xyz, np.ones(len(xyz))] @ inv.T)[:, :3]).astype(int)
            ijk = np.clip(ijk, 0, np.array(img.shape[:3]) - 1)
        v = np.asarray(img.get_fdata())[tuple(ijk.T)].astype(np.float32)
        maps.append(v); invoc.append(bool(np.any(v != 0)))
    np.savez_compressed(out, names=np.array([i[0] for i in items]), texts=np.array([i[1] for i in items]),
                        kind=np.array([i[2] for i in items]), in_vocab=np.array(invoc), maps=np.array(maps))
    k = np.array([i[2] for i in items]); iv = np.array(invoc)
    print(f"{out}: {len(items)} maps ({(k == 'construct').sum()} constructs, {(k == 'task').sum()} tasks); "
          f"outside NeuroQuery's vocabulary: {(~iv[k == 'construct']).sum()} constructs, {(~iv[k == 'task']).sum()} tasks")


if __name__ == "__main__":
    main()
