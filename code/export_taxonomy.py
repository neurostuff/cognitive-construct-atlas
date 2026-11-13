"""Export the construct taxonomy and the release maps' thumbnails for the explorer page (James 2026-11-11: an
artifact explaining the methods, with a utility to search and explore the taxonomy of cognitive constructs).

Writes OUT_DIR/taxonomy.json (constructs with their parents, children, tasks, counts, reliability and networks; tasks
with families, counts, reliability and networks; aliases from the vocabulary's own wordings; groups of constructs that
share a task set, hence a map), OUT_DIR/thumbs_constructs_<i>.json and OUT_DIR/thumbs_tasks_<i>.json (the probability and
enrichment maximum-intensity projections from the release's mips.json files, and the brain outline).

Cognitive Atlas links: an entry's own link, or else a concept of exactly the same name (cogat_concepts.json; marked
"cogat_by_name"). Part-of relations among constructs (the corpus's "partof", display only: they do not pool tasks)
are exported as part_of / has_parts.

What the papers found, without the model (James 2026-11-11: MKDAChi2's uniformity and association maps): for every
task and construct, NiMARE's MKDAChi2 of its contrasts against all other contrasts (mkda_chi2.py; a construct's
contrasts are those of every task below it). Unthresholded z maps are written as NIfTI (2 mm) to
RELEASE_DIR/mkda/<constructs|tasks>/<id>_<uniformity|association>_z.nii.gz; the thumbnails
(thumbs_mkda_constructs_<i>.json / thumbs_mkda_tasks_<i>.json, "p" = uniformity, "e" = association; every thumbnail
set is split into files of SHARD_MAPS maps, each saying how many there are) show positive z where
FDR q < 0.05, clipped at MKDA_ZMAX, on the model's 4 mm grid.

Construct authors (corpus B): a construct's contrasts are those it or a construct below it (kind-of) authors, for
MKDAChi2, its task list ("all_tasks": the tasks of those contrasts, most first) and its counts; constructs do not
share maps by task set there, so "same_map" is empty.

usage: python export_taxonomy.py CORPUS RELEASE_DIR VOCAB_JSON OUT_DIR   (cbmr-gpu venv, PYTHONPATH = task-networks src;
       from pilot-results/atm)
"""
import json
import sys
from pathlib import Path

N_ALIASES = 8


def hid(s):
    """a hash-safe id: ASCII letters, digits and hyphens"""
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-")


MKDA_ZMAX = 10.0  # z shown at full colour in the MKDAChi2 thumbnails (a display default)
SHARD_MAPS = 600  # maps per thumbnail file: the artifact host takes files up to 16 MB (1,013 construct maps were 19.5 MB)


def write_shards(out, name, base, maps):
    """thumbs_<name>_<i>.json, i = 1..n, each with the shape and mask, its share of the maps and n ("shards")"""
    keys = list(maps); n = max(1, -(-len(keys) // SHARD_MAPS))
    for i in range(n):
        part = {k: maps[k] for k in keys[i * SHARD_MAPS:(i + 1) * SHARD_MAPS]}
        json.dump({**base, "shards": n, "maps": part}, open(Path(out) / f"thumbs_{name}_{i + 1}.json", "w"), separators=(",", ":"))
    return n


def construct_authored(H):
    return any(i.startswith("c:") for d in H["docs"].values() for i in d["ids"])


def is_nested(H):
    """contrast nodes (nested_corpus.py) author the documents; tasks and constructs are their parents"""
    return any(n.startswith("x:") for n in H["parents"])


def cons_below(node, children, seen=None):
    """the node and the construct nodes below it along construct-to-construct edges"""
    seen = {node} if seen is None else seen
    for c in children.get(node, []):
        if c.startswith("c:") and c not in seen:
            seen.add(c); cons_below(c, children, seen)
    return seen


def construct_tasks(H, tasks, children):
    """with contrast-level constructs: a function giving, for a construct node, the tasks of the contrasts it or a
    narrower construct authors, as [(task, contrasts)], most first; a contrast counts once per task"""
    nested = is_nested(H); P = H["parents"]; docs_of, task_of = {}, {}
    for k, d in H["docs"].items():
        ids_ = list(d["ids"]) + (list(dict.fromkeys(P.get("x:" + k, []))) if nested else [])
        task_of[k] = [i for i in ids_ if i in tasks]
        for i in ids_:
            if i.startswith("c:"):
                docs_of.setdefault(i, set()).add(k)

    def of(node):
        cnt = {}
        for k in set().union(*(docs_of.get(c, set()) for c in cons_below(node, children))):
            for t in task_of[k]:
                cnt[t] = cnt.get(t, 0) + 1
        return sorted(cnt.items(), key=lambda x: -x[1])
    return of


def mkda_maps(corpus, rel, out, ids, T, tasks):
    import nibabel as nib
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from task_networks.corpus import load_corpus
    from construct_maps_all import grid4
    import mkda_chi2 as mk
    C = load_corpus(corpus); H = json.load(open(corpus))
    X, masker, keys, _ = mk.build_ma(corpus, str(Path(rel) / "mkda_ma.npz"))
    has = np.array([bool(H["docs"][k]["foci"]) for k in keys])
    rows_of = {}
    for i, k in enumerate(keys):
        for a in H["docs"][k]["ids"]:
            rows_of.setdefault(a, []).append(i)
    children = {}
    for c, ps in H["parents"].items():
        for p in ps:
            children.setdefault(p, []).append(c)
    nested = is_nested(H)
    if nested:   # a contrast node's task and construct parents take its rows
        for a in [a for a in rows_of if a.startswith("x:")]:
            for p in set(H["parents"][a]):
                rows_of.setdefault(p, []).extend(rows_of[a])

    def below(n, seen):
        s_ = set()
        for c in children.get(n, []):
            if c in seen:
                continue
            seen.add(c)
            s_ |= {c} if c in tasks else (below(c, seen) if c.startswith("c:") else set())
        return s_
    ijk, shape, aff4, mask, mips, vol_shape = grid4(C["xyz"])
    mimg = masker.mask_img if hasattr(masker, "mask_img") else masker.mask_img_
    m2 = np.asarray(mimg.get_fdata()) > 0; pos = -np.ones(m2.shape, int); pos[m2] = np.arange(m2.sum())
    gpos = pos[tuple((ijk * 2).T)]                                 # masked index of each 4 mm grid point (-1 outside)
    base = {"shape": vol_shape, "mask": mips(mask.astype(np.uint8) * 255)}
    stats = {}

    def one(kind, key, rows):
        r = mk.feature_maps(X, rows, has)
        d = Path(rel) / "mkda" / kind; d.mkdir(parents=True, exist_ok=True)
        th = {}; stats.setdefault(kind, {})[key] = {"n": r["n"]}
        for name in ("uniformity", "association"):
            z = r[f"z_{name}"].astype(np.float32)
            nib.save(masker.inverse_transform(z), d / f"{key}_{name}_z.nii.gz")
            keep = mk.fdr_keep(r[f"p_{name}"]) & (z > 0)
            v = np.where(gpos >= 0, np.where(keep, z, 0)[np.maximum(gpos, 0)], 0)
            vol = np.zeros(shape, np.float32); vol[tuple(ijk.T)] = np.clip(v, 0, MKDA_ZMAX) / MKDA_ZMAX
            th[name] = mips(np.round(vol * 255)); stats[kind][key][f"{name}_voxels"] = int(keep.sum())
        return {"p": th["uniformity"], "e": th["association"], "n": r["n"]}
    tm = {T[t]["id"]: one("tasks", T[t]["id"], rows_of[t]) for t in T if rows_of.get(t)}
    cm = {}; authored = construct_authored(H) or nested
    for node, cid in ids.items():
        src = cons_below(node, children) if authored else below(node, set())
        rows = sorted({i for t in src for i in rows_of.get(t, [])})
        if rows:
            cm[cid] = one("constructs", cid, rows)
    for name, maps in (("constructs", cm), ("tasks", tm)):
        write_shards(out, f"mkda_{name}", base, maps)
    json.dump(stats, open(Path(rel) / "mkda" / "summary.json", "w"))
    print(f"MKDAChi2 maps: {len(cm)} constructs, {len(tm)} tasks")


def main():
    corpus, rel, vocab, out = sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    H = json.load(open(corpus)); P = H["parents"]; lab = H["labels"]; nc = H.get("n_contrasts", {})
    M = json.load(open(rel / "manifest.json")); recs = {m["node"]: m for m in M["constructs"]}
    from task_networks.corpus import load_corpus
    Cc = load_corpus(corpus)   # credited mass is in tokens (the fits' E-step counts); shown as contrast-equivalents
    tpc = sum(len(t) for t in Cc["toks"]) / len(Cc["toks"])
    V = {e["key"]: e for e in json.load(open(vocab))}
    cog = {}
    cpath = Path(vocab).resolve().parent.parent / "code" / "cogat_concepts.json"
    if cpath.exists():
        for cc in json.load(open(cpath)):
            cog.setdefault(cc["name"].strip().lower(), (cc["id"], cc["name"]))
    partof = H.get("partof", {})
    used, ids = set(), {}
    for node in sorted(recs, key=lambda n: -recs[n]["contrasts"]):
        h = hid(recs[node]["label"]) or "c"
        while h in used:
            h += "-x"
        used.add(h); ids[node] = h
    fams = sorted({p for t, ps in P.items() for p in ps if not p.startswith(("c:", "m:"))})
    tasks = {t: ps for t, ps in P.items() if not t.startswith(("c:", "m:", "x:")) and t != "background" and t not in fams}
    children = {}
    for c, ps in P.items():
        for p in ps:
            children.setdefault(p, []).append(c)

    def below(node, seen=None):
        seen = set() if seen is None else seen
        s = set()
        for c in children.get(node, []):
            if c in seen:
                continue
            seen.add(c)
            if c in tasks:
                s.add(c)
            elif c.startswith("c:"):
                s |= below(c, seen)
        return s
    nested = is_nested(H); authored = construct_authored(H) or nested
    via_tasks = construct_tasks(H, tasks, children) if authored else None
    groups, cons = {}, []
    for node, m in recs.items():
        key = node[2:]; e = V.get(key, {})
        aliases = [s for s, _ in e.get("strings", []) if s.lower() != m["label"].lower()]
        seen, al = set(), []
        for s in aliases:
            if s.lower() not in seen:
                seen.add(s.lower()); al.append(s)
        ts = via_tasks(node) if authored else frozenset(below(node))   # authored: [(task, contrasts)], most first
        if not authored:
            groups.setdefault(ts, []).append(ids[node])
        cid, cnm, byname = m.get("cogat_id"), m.get("cogat_name"), False
        if not cid and m["label"].strip().lower() in cog:
            (cid, cnm), byname = cog[m["label"].strip().lower()], True
        cons.append({
            "id": ids[node], "label": m["label"], "cogat_id": cid, "cogat_name": cnm, "cogat_by_name": byname,
            "part_of": [ids[p] for p in partof.get(node, []) if p in ids],
            "has_parts": [ids[c] for c, ps in partof.items() if node in ps and c in ids],
            "status": m.get("status"),
            "parents": [ids[p] for p in P.get(node, []) if p in ids],
            "children": [ids[c] for c in children.get(node, []) if c in ids],
            "tasks": sorted(t for t in children.get(node, []) if t in tasks),
            "all_tasks": [t for t, _ in ts] if authored else sorted(ts),
            "task_counts": [n for _, n in ts] if authored else None,
            "contrasts": m["contrasts"], "papers": m["papers"], "claim_papers": m.get("claim_papers"),
            "mass": round(m["credited_mass"] / tpc, 1), "rel_p": m.get("reliability_p"), "rel_e": m.get("reliability_enrichment"),
            "band_seed": m.get("seed_band_median_width"), "enriched_seed": m.get("reliably_enriched_seed"),
            "band_boot": m.get("boot_band_median_width"), "enriched_boot": m.get("reliably_enriched_boot"),
            "nets": [[n["name"], n["weight"], n.get("recurrence")] for n in m["networks"][:4]],
            "peaks": [p["mni"] + [p["log2_enrichment"]] for p in m["peaks"][:3]],
            "aliases": al[:N_ALIASES], "vocab_papers": e.get("papers"),
        })
    same = {}
    for ts, members in groups.items():
        for i in members:
            same[i] = [j for j in members if j != i]
    for c in cons:
        c["same_map"] = same.get(c["id"], [])
    TM = json.load(open(rel / "tasks" / "manifest.json")); trec = {m["node"]: m for m in TM["tasks"]}
    try:                                           # the task inventory's names ("Simple/Choice Reaction Time Task")
        sys.path.insert(0, str(Path(__file__).resolve().parent / "contrast_hub"))
        import hub
        inv_name = {x["id"]: x["name"] for x in hub.INVENTORY["tasks"] if x.get("name")}
    except ImportError:
        inv_name = {}
    T = {}
    for t, ps in tasks.items():
        m = trec.get(t, {})
        T[t] = {"id": "t-" + hid(lab.get(t, t)), "label": inv_name.get(t) or ("Deactivation (baseline > task)" if t == "deactivation" else lab.get(t, t)), "family": next((p for p in ps if p in fams), None),
                "contrasts": m.get("contrasts", nc.get(t)), "papers": m.get("papers"),
                "constructs": [ids[p] for p in ps if p in ids],
                "mass": round(m["credited_mass"] / tpc, 1) if m.get("credited_mass") is not None else None,
                "rel_p": m.get("reliability_p"), "rel_e": m.get("reliability_enrichment"),
                "band_seed": m.get("seed_band_median_width"), "enriched_seed": m.get("reliably_enriched_seed"),
                "band_boot": m.get("boot_band_median_width"), "enriched_boot": m.get("reliably_enriched_boot"),
                "nets": [[n["name"], n["weight"], n.get("recurrence")] for n in m.get("networks", [])[:4]],
                "peaks": [p["mni"] + [p["log2_enrichment"]] for p in m.get("peaks", [])[:3]], "has_map": bool(m)}
    N = json.load(open(rel / "networks.json"))
    meta = {"K": M["K"], "seeds": len(M.get("seeds", [])), "most_typical_seed": M.get("most_typical_seed"),
            "tokens_per_contrast": round(tpc, 1),
            "networks": [[n["name"], n["recurrence"]] for n in N["networks"]],
            "bootstrap": any(c["band_boot"] is not None for c in cons)}
    json.dump({"meta": meta, "constructs": cons, "tasks": T, "families": {f: lab.get(f, f) for f in fams}},
              open(out / "taxonomy.json", "w"), separators=(",", ":"))
    n_thumbs = {}
    for name, man, src, key_of in (("constructs", M["constructs"], rel / "mips.json", lambda m: ids[m["node"]]),
                                   ("tasks", TM["tasks"], rel / "tasks" / "mips.json", lambda m: T[m["node"]]["id"])):
        X = json.load(open(src))
        safe_to = {m["id"]: key_of(m) for m in man if name == "constructs" or m["node"] in T}  # task maps only (no family maps)
        thumbs = {"shape": X["shape"], "mask": X["mask"],
                  "maps": {safe_to[s]: {"p": v["p"], "e": v["enrichment"]} for s, v in X["maps"].items() if s in safe_to}}
        write_shards(out, name, {"shape": thumbs["shape"], "mask": thumbs["mask"]}, thumbs["maps"]); n_thumbs[name] = len(thumbs["maps"])
    mkda_maps(corpus, rel, out, ids, T, tasks)
    print(f"{len(cons)} constructs, {len(T)} tasks ({sum(t['has_map'] for t in T.values())} with maps), {len(fams)} families, "
          f"{len(groups) if not authored else 'no shared'} task-set map groups; thumbnails {n_thumbs}")


if __name__ == "__main__":
    main()
