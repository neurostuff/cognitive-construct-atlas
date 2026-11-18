"""Brain maps for every construct node of a construct-hierarchy fit, plus a manifest and browser-ready projections.

The maps are construct_map.py's, computed for all nodes at once: one E-step with the fitted theta and beta gives
every node's expected counts, which are pooled bottom-up into thetabar (construct_map.pooled). Then, with the
shared-pattern network removed and renormalized:
- P(voxel | construct) = thetabar_c . beta;
- enrichment = log2 P(voxel | construct) / P(voxel | the average task) (construct_map.reference).

Writes to OUT_DIR:
- nifti/<label>_p.nii.gz and nifti/<label>_enrichment.nii.gz (4 mm MNI152 grid; enrichment unmasked);
- manifest.json, one record per construct with data:
  - label, vocabulary key, Cognitive Atlas id and name (or "new");
  - parent and child constructs, and the tasks directly below;
  - contrasts and papers in its subtree, and its credited mass (the task-author share of those contrasts, in
    contrast-equivalents: what the map actually rests on);
  - its five largest networks, named by their best match to the original K=16 maps;
  - its most enriched voxels;
- mips.json: maximum-intensity projections (sagittal, coronal, axial) of P scaled to its maximum, and of the
  enrichment where P >= 20% of its maximum (clipped to [0, 3], so 8x), quantized to uint8 and base64-encoded.
  There is one brain-mask projection for the outline. This is what the explorable page draws.

usage: python construct_maps_all.py FIT_PREFIX CORPUS_JSON OUT_DIR   (cbmr-gpu venv; from pilot-results/atm)
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402

from construct_map import constructs_author, pooled, pooled_mass, reference, shared_network  # noqa: E402

ENRICH_MASK = 0.2  # enrichment is shown where P >= this fraction of its maximum (construct_map.py's rule)
ENRICH_MAX = 3.0   # log2 enrichment clipped at 8x for the projections
VOCAB = Path(__file__).resolve().parent.parent / "vocab" / "vocab_v0.json"


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode()


def main():
    prefix, corpus, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    th = np.load(f"{prefix}_theta.npy").astype(float); be = np.load(f"{prefix}_beta.npy").astype(float)
    xyz = np.load(f"{prefix}_xyz.npy"); nodes = json.load(open(f"{prefix}.json"))["nodes"]
    H = json.load(open(corpus))
    C = load_corpus(corpus); assert C["nodes"] == nodes
    # networks named by their best match to the original K=16 maps (Hungarian assignment on Pearson r)
    from scipy.optimize import linear_sum_assignment
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    K = len(be)
    if K == len(b0):
        r = z(be) @ z(b0).T / be.shape[1]; i, j = linear_sum_assignment(-r)
        name = {a: names[str(b)].split(":")[0] for a, b in zip(i, j)}
    else:  # another K: name each network by its most correlated original network
        r = z(be) @ z(b0).T / be.shape[1]
        name = {a: f"{names[str(int(np.argmax(r[a])))].split(':')[0]} ({r[a].max():.2f})" for a in range(K)}
    bar, n, children = pooled(C, th, be)
    shared = [shared_network(C, n)]  # the network the shared-pattern author owns most of
    name[shared[0]] = "Shared pattern"
    glob = reference(C, n); glob[shared] = 0; glob /= glob.sum(); p_all = glob @ be
    mass = pooled_mass(C, n)[0].sum(1)  # contrast-equivalents of task mass behind each node's map
    items = {}
    for k, node in enumerate(nodes):
        if node.startswith("c:"):
            mix = bar[k].copy(); mix[shared] = 0; mix /= mix.sum()
            p = mix @ be
            items[k] = {"p": p, "enrichment": np.log2(p / p_all), "credited_mass": float(mass[k]),
                        "networks": [{"name": name[q], "weight": round(float(mix[q]), 3)} for q in np.argsort(-mix)[:5]]}
    write_maps(out, C, H, xyz, children, items,
               {"fit": prefix, "corpus": corpus, "K": K, "networks": [name[q] for q in range(K)],
                "shared_network": [name[q] for q in shared]})


def grid4(xyz):
    """the fit's 4 mm grid (the 2 mm MNI152 voxels with even indices): voxel indices of the grid points, volume shape
    and affine, brain mask, and a maximum-intensity-projection encoder (sagittal, coronal, axial; uint8 base64)"""
    from nimare.utils import get_template
    tmpl = get_template("mni152_2mm", mask="brain"); aff2 = tmpl.affine
    ijk = np.round((np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(aff2).T)[:, :3]).astype(int) // 2
    shape = tuple(int(x) for x in np.array(tmpl.shape) // 2 + 1); aff4 = aff2 @ np.diag([2, 2, 2, 1])
    mask = np.zeros(shape, bool); mask[tuple(ijk.T)] = True

    def mips(vol):
        return [b64(np.flipud(vol.max(axis=a).T)) for a in (0, 1, 2)]
    vol_shape = {"sagittal": [shape[2], shape[1]], "coronal": [shape[2], shape[0]], "axial": [shape[1], shape[0]]}
    return ijk, shape, aff4, mask, mips, vol_shape


def write_maps(out, C, H, xyz, children, items, header, kind="constructs"):
    """NIfTI volumes, manifest.json and mips.json for the constructs in items (node index -> p, enrichment,
    credited_mass, networks, and optionally "extra" manifest fields and "volumes", more voxel arrays written as
    <label>_<tag>.nii.gz), for every node in items with data. kind names the manifest's list ("constructs", or
    "tasks" for task maps written to their own directory)"""
    out = Path(out); (out / "nifti").mkdir(parents=True, exist_ok=True)
    nodes = C["nodes"]; lab = H["labels"]
    entries = {r["key"]: r for r in json.load(open(VOCAB))}
    # subtree contrasts and papers; with construct authors a construct counts only the contrasts it or a construct below
    # it authors (construct_map.pooled_mass), so a task's contrasts do not climb to the task's constructs
    docs_under = [set() for _ in nodes]
    authored = constructs_author(C)
    for d, auth in enumerate(C["auth"]):
        stack, seen = list(auth), set()
        while stack:
            x = int(stack.pop())
            if x in seen:
                continue
            seen.add(x); docs_under[x].add(d)
            stack.extend(q for q in C["parents"][x] if not (authored and nodes[q].startswith("c:") and not nodes[x].startswith("c:")))
    paper_of = {d: v["group"] for d, v in H["docs"].items()}
    import nibabel as nib
    ijk, shape, aff4, mask, mips, vol_shape = grid4(xyz)
    manifest, proj = [], {"shape": vol_shape, "mask": mips(mask.astype(np.uint8) * 255), "maps": {}}
    used = set()
    for k, node in enumerate(nodes):
        if not docs_under[k] or k not in items:
            continue
        key = node[2:] if node.startswith("c:") else node; e = (entries.get(key) or {}) if node.startswith("c:") else {}
        label = lab.get(node, key)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label.lower())[:80]
        while safe in used:
            safe += "_"
        used.add(safe)
        it = items[k]; p, enr = it["p"], it["enrichment"]
        vols = {"p": p, "enrichment": enr, **it.get("volumes", {})}   # extra volumes, e.g. bands (ensemble build)
        for tag, vals in vols.items():
            v = np.zeros(shape, np.float32); v[tuple(ijk.T)] = vals
            nib.save(nib.Nifti1Image(v, aff4), out / "nifti" / f"{safe}_{tag}.nii.gz")
        pv = np.zeros(shape, np.float32); pv[tuple(ijk.T)] = p / p.max()
        ev = np.zeros(shape, np.float32)
        ev[tuple(ijk.T)] = np.where(p >= ENRICH_MASK * p.max(), np.clip(enr, 0, ENRICH_MAX), 0) / ENRICH_MAX
        proj["maps"][safe] = {"p": mips(np.round(pv * 255)), "enrichment": mips(np.round(ev * 255))}
        top = [t for t in np.argsort(-enr) if p[t] >= ENRICH_MASK * p.max()][:8]
        kids = [nodes[c] for c in children[k]]
        rec = {
            "id": safe, "node": node, "label": label, "key": key,
            "cogat_id": e.get("cogat_id"), "cogat_name": e.get("cogat_name"),
            "status": e.get("status", "category" if not e else "new") if node.startswith("c:") else "task",
            "parents": [lab.get(nodes[q], nodes[q]) for q in C["parents"][k]],
            "child_constructs": sorted(lab.get(c, c) for c in kids if c.startswith("c:")),
            "tasks": sorted(lab.get(c, c) for c in kids if not c.startswith(("c:", "m:", "x:"))),
            "contrasts": len(docs_under[k]), "credited_mass": round(float(it["credited_mass"]), 1),
            "papers": len({paper_of.get(C["ids"][d]) for d in docs_under[k]}),
            "claim_papers": e.get("papers"),
            "networks": it["networks"],
            "peaks": [{"mni": [int(x) for x in xyz[t]], "log2_enrichment": round(float(enr[t]), 2)} for t in top],
            "files": {tag: f"nifti/{safe}_{tag}.nii.gz" for tag in vols},
        }
        rec.update(it.get("extra", {}))
        manifest.append(rec)
    manifest.sort(key=lambda m: -m["contrasts"])
    json.dump({**header, kind: manifest},
              open(out / "manifest.json", "w"), indent=1, default=lambda o: o.item())  # numpy scalars
    json.dump(proj, open(out / "mips.json", "w"), default=lambda o: o.item())
    print(f"{out}: {len(manifest)} {kind} with data "
          f"({sum(m['cogat_id'] is not None for m in manifest)} linked to Cognitive Atlas); K {header['K']}")


if __name__ == "__main__":
    main()
