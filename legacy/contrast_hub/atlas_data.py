"""Data for the atlas explorer artifact: tasks -> their contrasts (directed hub entries) -> the concepts those
contrasts' analyses carry, and the concept hierarchy (James 2026-11-12).

From the within-task corpus B (corpus_w_B.json: 27,905 sample analyses, construct authors after the Cognitive Atlas
normalization, attachment and add-only connections), the hub (signatures), corpus_assign (each analysis's directed
entry) and contrast_constructs_full.json (where each analysis's constructs came from). Removal flags from
cogat_connect.json are marked on the edges they name.

usage: (code/contrast_hub on beast) python atlas_data.py ATM_DIR HUB_DIR CONSTRUCTS_FULL.json COGAT.json OUT.json
"""
import collections
import json
import sys

import hub as H

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from contrast_construct_authors import kind as entry_kind  # noqa: E402

SOURCE = {"claims": "papers' claims", "codex": "attribution", "atlas": "Cognitive Atlas", "task_contrast": "task",
          "paper": "paper reading"}


def side(v):
    return {"@rest": "rest", "@control": "control", "": "∅"}.get(v, (v or "").replace("_", " "))


def label(k, sign, entries):
    e = entries.get(k)
    if not e:
        return k.split("|", 1)[-1]
    comps = json.loads(e["sig_json"])[1]
    out = []
    for ax, a, b in comps:
        if b is None:
            out.append(f"{ax.replace('_', ' ')}: {side(a)}" + (" (deactivation)" if sign < 0 else ""))
        else:
            a, b = (a, b) if sign > 0 else (b, a)
            out.append(f"{ax.replace('_', ' ')}: {side(a)} > {side(b)}")
    return "; ".join(out)


def main():
    atm, hub_dir, consp, cogp, outp = sys.argv[1:6]
    B = json.load(open(f"{atm}/corpus_w_B.json")); P = B["parents"]
    hubj = json.load(open(f"{hub_dir}/hub.json")); entries = {**hubj.get("parents", {}), **hubj["contrasts"]}
    assign = json.load(open(f"{hub_dir}/corpus_assign.json")); R = json.load(open(consp))
    C = json.load(open(f"{atm}/../contrast_hub/contrasts_all.json"))
    cog = {c["name"].strip().lower().replace("-", " ") for c in json.load(open(cogp)) if c.get("name")}
    conn = json.load(open(f"{atm}/cogat_connect.json"))
    flags = {(n, "c:" + r["parent"].strip().lower()): r["reason"] for n, v in conn.items() for r in v["remove"]}
    added = {(n, "c:" + a) for n, v in conn.items() for a in v["add"]}
    inv = {t["id"]: t for t in H.INVENTORY["tasks"]}
    families = {t["family"] for t in H.INVENTORY["tasks"] if t.get("family")}
    tasks = collections.defaultdict(lambda: {"analyses": 0, "papers": set(), "contrasts": collections.defaultdict(lambda: {
        "analyses": 0, "papers": set(), "concepts": collections.Counter(), "sources": collections.Counter()})})
    cuse = collections.Counter(); ctasks = collections.defaultdict(collections.Counter)   # concept -> task -> analyses
    for aid, d in B["docs"].items():
        ts = [i for i in d["ids"] if not i.startswith(("c:", "m:")) and i not in ("background", "deactivation")]
        cs = [i for i in d["ids"] if i.startswith("c:")] + (["deactivation"] if "deactivation" in d["ids"] else [])
        a = assign.get(aid)
        if a:
            key = ("rev:" if a.get("sign", 1) < 0 else "") + a["hub"]
            lab = label(a["hub"], a.get("sign", 1), entries)
            ek = entry_kind(a["hub"], entries)
            kd = ("deactivation" if ek != "condition" else "reversed") if a.get("sign", 1) < 0 else \
                 {"task_vs_baseline": "task vs baseline", "simple": "simple effect", "condition": "condition"}[ek]
        else:
            key, lab, kd = "unnormalized", "not normalized (read from the paper)", "not normalized"
        src = SOURCE.get((R.get(aid) or {}).get("tier") or "", "task" if (R.get(aid) or {}).get("tier") else "none")
        for t in ts[:1] or ["(no task)"]:
            T = tasks[t]; T["analyses"] += 1; T["papers"].add(d.get("group"))
            X = T["contrasts"][key]; X["label"] = lab; X["kind"] = kd; X["analyses"] += 1; X["papers"].add(d.get("group"))
            X["concepts"].update(cs); X["sources"][src] += 1
            for c in cs:
                ctasks[c][t] += 1
        cuse.update(cs)
    out_tasks = []
    for t, T in tasks.items():
        x = inv.get(t) or {}
        bare = t.split(":", 1)[1] if t.startswith("family:") else t
        fam = x.get("family") or (bare if bare in families else "")
        name = x.get("name") or (bare.replace("_", " ").capitalize() + " (family label, no task)" if bare in families else t)
        tc = [p[2:] for p in P.get(t, []) if p.startswith("c:")]
        cons = sorted(({"key": k, "label": X["label"], "kind": X["kind"],
                        "analyses": X["analyses"], "papers": len(X["papers"]),
                        "concepts": [[c[2:] if c.startswith("c:") else c, n] for c, n in X["concepts"].most_common(8)],
                        "sources": dict(X["sources"])} for k, X in T["contrasts"].items()), key=lambda x: -x["analyses"])
        out_tasks.append({"id": t, "name": name, "family": fam.replace("_", " "), "analyses": T["analyses"],
                          "papers": len(T["papers"]), "concepts": tc, "contrasts": cons})
    out_tasks.sort(key=lambda x: -x["analyses"])
    concepts = {}
    for n, ps in P.items():
        if not n.startswith("c:"):
            continue
        concepts[n[2:]] = {"cogat": n[2:] in cog, "analyses": cuse.get(n, 0),
                           "tasks": [[t, k] for t, k in ctasks.get(n, collections.Counter()).most_common()], "task_level": [],
                           "parents": [[p[2:], "flagged" if (n, p) in flags else "added" if (n, p) in added else "",
                                        flags.get((n, p), "")] for p in ps if p.startswith("c:")]}
    for t in out_tasks:                               # tasks whose own concepts include it
        for c in t["concepts"]:
            concepts.setdefault(c, {"cogat": c in cog, "analyses": 0, "tasks": [], "task_level": [], "parents": []})
            concepts[c]["task_level"].append(t["id"])
    D = {"tasks": out_tasks, "concepts": concepts,
         "stats": {"analyses": len(B["docs"]), "tasks": len(out_tasks), "concepts": len(concepts),
                   "cogat_concepts": sum(v["cogat"] for v in concepts.values()),
                   "edges": sum(len(v["parents"]) for v in concepts.values()),
                   "edges_added": sum(1 for v in concepts.values() for p in v["parents"] if p[1] == "added"),
                   "edges_flagged": sum(1 for v in concepts.values() for p in v["parents"] if p[1] == "flagged"),
                   "contrasts": sum(len(t["contrasts"]) for t in out_tasks),
                   "deactivation": cuse.get("deactivation", 0)}}
    json.dump(D, open(outp, "w"), separators=(",", ":"))
    print(json.dumps(D["stats"]))


if __name__ == "__main__":
    main()
