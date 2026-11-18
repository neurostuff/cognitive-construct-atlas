"""3D volumes for the explorer's NiiVue viewer (James 2026-11-17: "show all the explorable images using niivue, so people
can look more carefully at all the brains").

Every map the page shows as projections, sampled at the model's 28,549 grid points (the 2 mm MNI152 brain-mask voxels
with even indices, in load_corpus order):
- constructs and tasks: the release's probability and enrichment volumes (MAPS_DIR/nifti, MAPS_DIR/tasks/nifti) and
  MKDAChi2's uniformity and association z (MAPS_DIR/mkda/<constructs|tasks>/<page id>_<test>_z.nii.gz, 2 mm);
- contrasts: CONTRAST_MODEL_DIR/<x id>_<p|enrichment>.nii.gz and CONTRAST_MKDA_DIR/<x id>_<test>_z.nii.gz;
- networks: the most typical fit's beta over its maximum, and log2 of beta over the corpus-wide map (export_networks.py).

Encoding per map (bytes, one per grid point): "p" uint8 = round(255 P / max P); "e" int8 = round(40 log2 enrichment),
clipped at +-3 (8x), and -128 (no value) where P < ENRICH_MASK of its maximum, the page's rule for drawing a ratio;
"u" and "a" int8 = round(z / scale), scale = max |z| / 127 per map (kept as "us" / "as"), so no z is clipped. MKDAChi2's FDR cut (Benjamini-Hochberg at q = 0.05 over the 2 mm mask,
positive z, mkda_chi2.fdr_keep on p = chi2 sf(z^2, 1), which is how mkda_chi2.feature_maps derives z from p) is kept as
the smallest kept z, the viewer's default threshold.

Writes OUT_DIR/vol_index.json (grid: shape, origin, ijk base64 in the condition atlas's convention, i = (x + 70) / 4,
j = (y + 106) / 4, k = (z + 72) / 4; per kind and id: shard, which maps, MKDA cuts and n) and OUT_DIR/vol_<kind>_<i>.json
({"maps": {id: base64 of gzip(concatenated maps in index order)}}), shards of about SHARD_BYTES each.

usage: (beast, pilot-results/atm; cbmr-gpu venv, PYTHONPATH = task-networks src)
  python ../code/export_volumes.py CORPUS MAPS_DIR FIT_DIR CONTRAST_MODEL_DIR CONTRAST_MKDA_DIR TAXONOMY OUT_DIR
"""
import base64, gzip, json, sys
from pathlib import Path
import numpy as np

SHARD_BYTES = 2_500_000
ORIGIN = np.array([-70.0, -106.0, -72.0])   # the grid's lowest corner in MNI mm (min over the 4 mm points)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import nibabel as nib
    from scipy.stats import chi2
    from task_networks.corpus import load_corpus
    import construct_map as cm
    from construct_maps_all import ENRICH_MASK, grid4
    from export_taxonomy import construct_page_ids
    from mkda_chi2 import fdr_keep
    corpus, maps, fits, cmod, cmk, tax, out = sys.argv[1:8]
    maps, cmod, cmk, out = Path(maps), Path(cmod), Path(cmk), Path(out); out.mkdir(parents=True, exist_ok=True)
    C = load_corpus(corpus); H = json.load(open(corpus)); xyz = C["xyz"]; V = len(xyz)
    ijk4, shape4, aff4, mask4, _, _ = grid4(xyz)
    gijk = np.rint((xyz - ORIGIN) / 4).astype(int); assert gijk.min() == 0 and V == 28549, (gijk.min(), V)
    shape = [int(x) + 1 for x in gijk.max(0)]
    T = json.load(open(tax)); cid = construct_page_ids(T, H)

    m2 = None

    def at4(path):        # a 4 mm release volume at the grid points
        img = nib.load(str(path)); assert np.allclose(img.affine, aff4), path
        return np.asarray(img.dataobj, dtype=np.float32)[tuple(ijk4.T)]

    def mkda(path):       # a 2 mm z volume: z at the grid points and the FDR cut
        nonlocal m2
        img = nib.load(str(path)); z2 = np.asarray(img.dataobj, dtype=np.float32)
        if m2 is None:
            from nimare.utils import get_template
            t = get_template("mni152_2mm", mask="brain"); assert np.allclose(t.affine, img.affine); m2 = np.asarray(t.dataobj) > 0
        zm = z2[m2]; keep = fdr_keep(chi2.sf(zm.astype(float) ** 2, 1)) & (zm > 0)
        cut = float(zm[keep].min()) if keep.any() else None
        return z2[tuple((ijk4 * 2).T)], cut

    def enc_p(p):
        return np.round(255 * np.clip(p, 0, None) / max(float(p.max()), 1e-30)).astype(np.uint8)

    def enc_e(e, p):
        x = np.round(40 * np.clip(e, -3, 3)).astype(np.int8); x[p < ENRICH_MASK * p.max()] = -128
        return x

    def enc_z(z):
        sc = max(float(np.abs(z).max()), 1e-6) / 127
        return np.round(z / sc).astype(np.int8), sc

    items = {}            # kind -> id -> (bytes, info)

    def add(kind, key, p=None, e=None, u=None, a=None, **info):
        parts, have = [], []
        if p is not None:
            parts.append(enc_p(p).tobytes()); have.append("p")
            if e is not None: parts.append(enc_e(e, p).tobytes()); have.append("e")
        for nm, z in (("u", u), ("a", a)):
            if z is not None:
                q, sc = enc_z(z); parts.append(q.tobytes()); have.append(nm); info[nm + "s"] = sc
        if parts: items.setdefault(kind, {})[key] = (gzip.compress(b"".join(parts), 9, mtime=0), {"m": "".join(have), **info})

    def mk_pair(d, key):
        r = {}
        for nm, test in (("u", "uniformity"), ("a", "association")):
            f = d / f"{key}_{test}_z.nii.gz"
            if f.exists(): r[nm], r[nm + "c"] = mkda(f)
        return r

    # constructs and tasks: the release manifests name each map's file id and node
    M = json.load(open(maps / "manifest.json")); TM = json.load(open(maps / "tasks" / "manifest.json"))
    folded = set(H.get("folded", {}))     # duplicate nodes pooled into their twin (fold_duplicates.py): no page
    for kind, man, nd, key_of in (("constructs", M["constructs"], maps / "nifti", lambda m: cid.get(m["node"])),
                                  ("tasks", TM["tasks"], maps / "tasks" / "nifti", lambda m: T["tasks"].get(m["node"], {}).get("id"))):
        seen = set()
        for m in man:
            key = key_of(m)
            if not key or (kind == "constructs" and m["node"] in folded) or key in seen: continue
            seen.add(key)
            p, e = at4(nd / f"{m['id']}_p.nii.gz"), at4(nd / f"{m['id']}_enrichment.nii.gz")
            z = mk_pair(maps / "mkda" / kind, key)
            add(kind, key, p, e, z.get("u"), z.get("a"), uc=z.get("uc"), ac=z.get("ac"))
        # MKDAChi2 maps for pages with no model map
        for f in sorted((maps / "mkda" / kind).glob("*_uniformity_z.nii.gz")):
            key = f.name[:-len("_uniformity_z.nii.gz")]
            if key in items.get(kind, {}): continue
            z = mk_pair(maps / "mkda" / kind, key); add(kind, key, u=z.get("u"), a=z.get("a"), uc=z.get("uc"), ac=z.get("ac"))
        print(kind, len(items.get(kind, {})), flush=True)

    # task contrasts ("x<n>")
    xs = sorted({f.name.split("_")[0] for f in cmk.glob("x*_uniformity_z.nii.gz")} | {f.name.split("_")[0] for f in cmod.glob("x*_p.nii.gz")}, key=lambda s: int(s[1:]))
    for x in xs:
        p = at4(cmod / f"{x}_p.nii.gz") if (cmod / f"{x}_p.nii.gz").exists() else None
        e = at4(cmod / f"{x}_enrichment.nii.gz") if p is not None else None
        z = mk_pair(cmk, x); add("contrasts", x, p, e, z.get("u"), z.get("a"), uc=z.get("uc"), ac=z.get("ac"))
    print("contrasts", len(items.get("contrasts", {})), flush=True)

    # networks of the most typical fit, as export_networks.py
    net = json.load(open(maps / "networks.json")); pre = f"{fits}/gibbs_s{net['most_typical_seed']}"
    be = np.load(maps / "most_typical_beta.npy").astype(float)
    th, n = np.load(pre + "_theta.npy").astype(float), np.load(pre + "_n.npy")
    tc = cm.task_content(C, th, be) if cm.constructs_author(C) else None
    _, g, _ = cm.pooled_mass(C, n, tc); pall = g @ be
    for e_ in net["networks"]:
        k = e_["index"]; v = be[k]; add("networks", f"n{k + 1}", v, np.log2(v / pall))
    print("networks", len(items.get("networks", {})), flush=True)

    index = {"grid": {"shape": shape, "mm": 4, "origin": ORIGIN.tolist(), "ijk": base64.b64encode(gijk.astype(np.uint8).tobytes()).decode()},
             "encoding": {"p": "uint8, 255 P / max P", "e": "int8, 40 log2 enrichment, -128 none", "u": "int8, z / us", "a": "int8, z / as"},
             "kinds": {}}
    total = 0
    for kind, its in items.items():
        shard, size, part, ent = 1, 0, {}, {}

        def flush():
            json.dump({"maps": part}, open(out / f"vol_{kind}_{shard}.json", "w"), separators=(",", ":"))
        for key, (blob, info) in its.items():
            b = base64.b64encode(blob).decode()
            if size and size + len(b) > SHARD_BYTES:
                flush(); shard += 1; size, part = 0, {}
            part[key] = b; size += len(b); total += len(b)
            ent[key] = {"s": shard, **{k_: (round(v_, 3) if k_ in ("uc", "ac") else round(v_, 6) if isinstance(v_, float) else v_) for k_, v_ in info.items() if v_ is not None}}
        flush(); index["kinds"][kind] = {"shards": shard, "items": ent}
    json.dump(index, open(out / "vol_index.json", "w"), separators=(",", ":"))
    print("total base64 MB", round(total / 1e6, 1), {k: v["shards"] for k, v in index["kinds"].items()})


if __name__ == "__main__":
    main()
