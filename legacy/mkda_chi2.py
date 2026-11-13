"""What the papers found, without the model: NiMARE's MKDAChi2 for every task and construct (James 2026-11-11: the
paper-peak maps should be MKDAChi2's uniformity and association maps).

MKDAChi2 (Wager et al. 2007; nimare.meta.cbma.mkda.MKDAChi2) compares the contrasts that carry a feature with all
other contrasts, from MKDA maps (binary 10 mm spheres around each contrast's peaks, 2 mm MNI152 grid):
- uniformity test: where the feature's contrasts activate more often than their own average voxel (one-way
  chi-square), i.e. consistent activation, as Neurosynth's former "uniformity" map;
- association test: where activation is more likely in the feature's contrasts than in all other contrasts (two-way
  chi-square), Neurosynth's "association" (reverse-inference) map.
Here the MKDA maps of all contrasts are made once with NiMARE's MKDAKernel (r = 10 mm) and kept as a contrast-by-
voxel sparse matrix; each feature's statistics are then computed with NiMARE's own functions (nimare.stats.one_way,
two_way_counts, chi2_to_nlogp) exactly as MKDAChi2._fit does, so the result equals MKDAChi2 run on the two slices
(checked by `check`; chi2_to_nlogp is in nimare.transforms). Contrasts are the corpus documents (one per contrast, as the model's); a construct's contrasts
are those of every task below it in the hierarchy. Maps are thresholded at FDR q < Q (Benjamini-Hochberg over voxels,
NiMARE's "indep" FDR) and only positive z is shown.

usage:
  python mkda_chi2.py check CORPUS CACHE.npz          # equality with nimare MKDAChi2 on two features
(module: build_ma, feature_maps, fdr_keep; used by export_taxonomy.py)
"""
import json
import sys

import numpy as np
from scipy import sparse

Q = 0.05     # FDR level for the shown maps
R_MM = 10    # MKDA sphere radius (NiMARE's MKDAKernel default)


def dataset_from_corpus(corpus):
    """a NiMARE Dataset with one contrast per corpus document; ids in load_corpus document order"""
    from nimare.dataset import Dataset
    H = json.load(open(corpus)); docs = H["docs"]; keys = sorted(docs)
    d = {}
    for k in keys:
        v = docs[k]; f = np.array([x[:3] for x in v["foci"]], float) if v["foci"] else np.zeros((0, 3))
        if len(f) == 0:
            continue
        d.setdefault(v["group"], {"contrasts": {}})["contrasts"][k] = {
            "coords": {"space": "MNI", "x": f[:, 0].tolist(), "y": f[:, 1].tolist(), "z": f[:, 2].tolist()}}
    dset = Dataset(d, target="mni152_2mm")
    id_of = {k: f"{docs[k]['group']}-{k}" for k in keys if docs[k]["foci"]}
    return dset, keys, id_of


def build_ma(corpus, cache):
    """contrast-by-voxel MKDA matrix (CSR, 0/1) in load_corpus document order (rows of documents without foci are
    empty), the masker, and the document keys; cached in CACHE.npz"""
    from pathlib import Path
    from nimare.meta.kernel import MKDAKernel
    dset, keys, id_of = dataset_from_corpus(corpus)
    if Path(cache).exists():
        X = sparse.load_npz(cache)
    else:
        ma = MKDAKernel(r=R_MM).transform(dset, return_type="sparse")
        ma = sparse.csr_matrix(ma) if not sparse.issparse(ma) else ma.tocsr()
        row_of = {i: r for r, i in enumerate(dset.ids)}
        order = [row_of.get(id_of.get(k), -1) for k in keys]
        rows = [ma[r] if r >= 0 else sparse.csr_matrix((1, ma.shape[1])) for r in order]
        X = sparse.vstack(rows).tocsr(); X.data[:] = 1; X = X.astype(np.float32)
        sparse.save_npz(cache, X)
    return X, dset.masker, keys, dset


def feature_maps(X, rows, has_foci):
    """MKDAChi2 statistics for the contrasts in rows against every other contrast with foci (as MKDAChi2._fit)"""
    from nimare.stats import one_way, two_way_counts
    from nimare.transforms import chi2_to_nlogp
    sel = np.zeros(X.shape[0], bool); sel[rows] = True; sel &= has_foci; uns = has_foci & ~sel
    n1, n2 = int(sel.sum()), int(uns.sum())
    a1 = np.asarray(X[sel].sum(0)).ravel(); a2 = np.asarray(X[uns].sum(0)).ravel()
    pAgF, pAgU = a1 / n1, a2 / n2
    c_u = one_way(a1, n1); z_u = np.sqrt(c_u) * np.sign(a1 - np.mean(a1)); p_u = np.exp(chi2_to_nlogp(c_u, 1))
    c_a = two_way_counts(a1, a2, n1, n2); z_a = np.sqrt(c_a) * np.sign(pAgF - pAgU); p_a = np.exp(chi2_to_nlogp(c_a, 1))
    return {"z_uniformity": z_u, "p_uniformity": p_u, "z_association": z_a, "p_association": p_a, "n": n1, "pAgF": pAgF}


def fdr_keep(p, q=Q):
    """Benjamini-Hochberg: voxels significant at FDR q"""
    o = np.argsort(p); m = len(p); thr = q * np.arange(1, m + 1) / m
    ok = p[o] <= thr; k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    keep = np.zeros(m, bool); keep[o[:k]] = True
    return keep


def check(corpus, cache):
    from nimare.meta.cbma.mkda import MKDAChi2
    X, masker, keys, dset = build_ma(corpus, cache)
    H = json.load(open(corpus)); has = np.array([bool(H["docs"][k]["foci"]) for k in keys])
    for task in ("tct_simple_choice_rt", "hedtsk_stop_signal"):
        rows = [i for i, k in enumerate(keys) if task in H["docs"][k]["ids"] and H["docs"][k]["foci"]]
        mine = feature_maps(X, rows, has)
        ids1 = [f"{H['docs'][keys[i]]['group']}-{keys[i]}" for i in rows]
        ids2 = [f"{H['docs'][k]['group']}-{k}" for i, k in enumerate(keys) if has[i] and i not in set(rows)]
        res = MKDAChi2(kernel__r=R_MM, prior=None).fit(dset.slice(ids1), dset.slice(ids2))
        for name in ("z_uniformity", "z_association"):
            ref = np.asarray(res.get_map(name.replace("z_", "z_desc-"), return_type="array")).ravel()
            print(f"{task} {name}: {len(rows)} vs {len(ids2)} contrasts; max abs diff from nimare MKDAChi2 "
                  f"{np.max(np.abs(ref - mine[name])):.2e} (max |z| {np.max(np.abs(ref)):.1f})")


if __name__ == "__main__":
    if sys.argv[1] == "check":
        check(sys.argv[2], sys.argv[3])
