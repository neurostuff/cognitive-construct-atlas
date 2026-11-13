"""Release tables of the Cognitive Construct Atlas: tasks, contrasts, concepts and their relations, aligned with the
Cognitive Atlas, with every analysis linked to Neurostore (James 2026-11-13: "formalize the data ... aligning the
updated cognitive atlas entries (including new tasks, contrasts, and concepts) ... the specific analyses on the
neurostore api to be linked and the neurostore study linked").

From the audited contrast-level corpus (corpus_w_B.json), the hub (hub.json, corpus_assign.json), the constructs file
(source of each analysis's constructs), the Cognitive Atlas concept dump and the task inventory. No text from the
papers is exported: names of analyses, quotes and sentences stay out (the Neurostore links lead to them).

Writes OUT_DIR:
  concepts.tsv           id, name, cognitive_atlas_id (empty: proposed new concept), status, analyses, papers
  concept_relations.tsv  child, parent, relation (kind_of | part_of), source
  removed_relations.tsv  kind-of edges the audit removed, with the reason
  tasks.tsv              id, name, family, status (inventory | provisional | family), cognitive_atlas_task_ids, analyses, papers
  contrasts.tsv          id (task|signature, with direction), task, label, kind, direction, analyses, papers
  contrast_concepts.tsv  contrast, concept, analyses (how many of the contrast's analyses carry the concept), share
  task_concepts.tsv      task, concept (what papers claim the task as a whole measures)
  analyses.tsv           one row per analysis: Neurostore analysis id and URL, base study id and URL, task, contrast,
                         direction, concepts, concept_source, confidence
  cognitive_atlas/*.json the same as Cognitive Atlas-shaped records (concepts, tasks, contrasts, assertions), existing
                         entries by their Cognitive Atlas id, new ones with id null and "proposed": true
  summary.json           counts

usage: (beast, pilot-results/atm; venv-cc) python ../code/contrast_hub/export_release.py OUT_DIR
"""
import collections
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import hub as HUB  # noqa: E402
from atlas_data import label as contrast_label  # noqa: E402
from contrast_construct_authors import kind as entry_kind  # noqa: E402

CORPUS, HUBDIR = "corpus_w_B.json", "../contrast_hub/v7"
CONS = "../contrast_hub/v7/constructs_dir/contrast_constructs_phase.json"
NS_ANALYSIS, NS_STUDY = "https://neurostore.org/api/analyses/{}", "https://neurostore.org/api/base-studies/{}"
CA_CONCEPT, CA_TASK = "https://www.cognitiveatlas.org/concept/id/{}", "https://www.cognitiveatlas.org/task/id/{}"
SOURCE = {"claims": "paper claims", "codex": "model attribution", "atlas": "cognitive atlas", "task_contrast": "task claims",
          "paper": "paper reading", "paper_phase": "paper reading (trial phase)"}
SPECIAL = ("background", "deactivation")


def tsv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n"); w.writerow(header); w.writerows(rows)


def main():
    out = Path(sys.argv[1]); (out / "cognitive_atlas").mkdir(parents=True, exist_ok=True)
    H = json.load(open(CORPUS)); P = H["parents"]; lab = H["labels"]
    hubj = json.load(open(f"{HUBDIR}/hub.json")); entries = {**hubj.get("parents", {}), **hubj["contrasts"]}
    A = json.load(open(f"{HUBDIR}/corpus_assign.json")); F = json.load(open(CONS))
    cog = {c["name"].strip().lower().replace("-", " "): c for c in json.load(open(HERE.parent / "cogat_concepts.json")) if c.get("name")}
    inv = {t["id"]: t for t in HUB.INVENTORY["tasks"]}
    conn = json.load(open("cogat_connect.json")); added = {(n, "c:" + a) for n, v in conn.items() for a in v["add"]}
    audited = set()
    for f in ("audit/proposals.json", "audit/proposals2.json"):
        if Path(f).exists():
            audited |= {x["analysis"] for x in json.load(open(f))["wrong_construct"]}
    is_task = lambda i: not i.startswith(("c:", "m:")) and i not in SPECIAL  # noqa: E731
    fams = {p for t, ps in P.items() if is_task(t) for p in ps if is_task(p)}
    # analyses
    rows, c_docs, c_pap, t_docs, t_pap = [], collections.Counter(), collections.defaultdict(set), collections.Counter(), collections.defaultdict(set)
    contrasts = {}; cc = collections.defaultdict(collections.Counter)
    for aid, d in sorted(H["docs"].items()):
        tasks = [i for i in d["ids"] if is_task(i)]; cs = [i[2:] for i in d["ids"] if i.startswith("c:")]; t = tasks[0] if tasks else ""
        a = A.get(aid); rev = bool(a) and a.get("sign", 1) < 0; deact = "deactivation" in d["ids"]
        cid = ""
        if a:
            sig = a["hub"].split("|", 1)[-1]; cid = f"{t}|{sig}" + ("|reversed" if rev else "")
            if cid not in contrasts:
                ek = entry_kind(a["hub"], entries)
                contrasts[cid] = {"task": t, "label": contrast_label(a["hub"], a.get("sign", 1), entries),
                                  "kind": {"task_vs_baseline": "task vs baseline", "simple": "simple effect", "condition": "condition comparison"}[ek],
                                  "direction": "reversed" if rev else "forward", "n": 0, "papers": set()}
            contrasts[cid]["n"] += 1; contrasts[cid]["papers"].add(d["group"]); cc[cid].update(cs)
        f = F.get(aid) or {}
        src = "audit" if aid in audited else SOURCE.get(f.get("tier") or "", "task claims" if cs else "")
        rows.append([aid, NS_ANALYSIS.format(aid), d["group"], NS_STUDY.format(d["group"]), t, ";".join(tasks[1:]), cid,
                     "deactivation" if deact else "reversed" if rev else "forward", ";".join(cs), src, f.get("confidence") or ""])
        for c in cs:
            c_docs[c] += 1; c_pap[c].add(d["group"])
        for x in tasks:
            t_docs[x] += 1; t_pap[x].add(d["group"])
    tsv(out / "analyses.tsv", ["neurostore_analysis_id", "neurostore_analysis_url", "neurostore_base_study_id", "neurostore_base_study_url",
                               "task", "other_tasks", "contrast", "direction", "concepts", "concept_source", "confidence"], rows)
    # concepts and relations
    cnodes = sorted(n[2:] for n in P if n.startswith("c:"))
    cid_of = lambda n: (cog.get(n) or {}).get("id") or ""  # noqa: E731
    tsv(out / "concepts.tsv", ["id", "name", "cognitive_atlas_id", "cognitive_atlas_url", "status", "analyses", "papers"],
        [[n.replace(" ", "-"), n, cid_of(n), CA_CONCEPT.format(cid_of(n)) if cid_of(n) else "",
          "cognitive atlas" if cid_of(n) else "proposed", c_docs[n], len(c_pap[n])] for n in cnodes])
    rel = [[n[2:], p[2:], "kind_of", "added 2026-11 (model-proposed, add-only pass)" if (n, p) in added else "cognitive atlas or literature"]
           for n in sorted(P) if n.startswith("c:") for p in P[n] if p.startswith("c:")]
    rel += [[n[2:], p[2:], "part_of", "cognitive atlas"] for n, ps in sorted(H.get("partof", {}).items()) for p in ps if n.startswith("c:") and p.startswith("c:")]
    tsv(out / "concept_relations.tsv", ["child", "parent", "relation", "source"], rel)
    removed = []
    if Path("audit/proposals.json").exists():
        removed = [[e["child"], e["parent"], "kind_of", e["reason"]] for e in json.load(open("audit/proposals.json")).get("edges", []) if e["decision"] == "remove"]
    tsv(out / "removed_relations.tsv", ["child", "parent", "relation", "reason"], removed)
    # tasks
    trows, tcon = [], []
    for t in sorted(x for x in P if is_task(x)):
        if not t_docs[t] and t not in fams:
            continue
        x = inv.get(t) or {}; ca = (x.get("crosswalk") or {}).get("cognitive_atlas") or []
        ca_ids = [c if isinstance(c, str) else (c.get("id") or "") for c in ca]
        status = "family" if t in fams else "inventory" if t in inv else "provisional"
        trows.append([t, x.get("name") or lab.get(t, t), "" if t in fams else next((p for p in P[t] if p in fams), ""), status,
                      ";".join(i for i in ca_ids if i), (x.get("definition") or "").replace("\n", " ").replace("\t", " "), t_docs[t], len(t_pap[t])])
        tcon += [[t, p[2:]] for p in P[t] if p.startswith("c:")]
    tsv(out / "tasks.tsv", ["id", "name", "family", "status", "cognitive_atlas_task_ids", "definition", "analyses", "papers"], trows)
    tsv(out / "task_concepts.tsv", ["task", "concept"], tcon)
    # contrasts
    tsv(out / "contrasts.tsv", ["id", "task", "label", "kind", "direction", "analyses", "papers"],
        [[k, v["task"], v["label"], v["kind"], v["direction"], v["n"], len(v["papers"])] for k, v in sorted(contrasts.items())])
    tsv(out / "contrast_concepts.tsv", ["contrast", "concept", "analyses", "share"],
        [[k, c, n, round(n / contrasts[k]["n"], 3)] for k in sorted(cc) for c, n in cc[k].most_common()])
    # Cognitive Atlas-shaped records
    ca = out / "cognitive_atlas"
    json.dump([{"id": cid_of(n) or None, "name": n, "proposed": not cid_of(n),
                "is_a": [{"id": cid_of(p[2:]) or None, "name": p[2:]} for p in P["c:" + n] if p.startswith("c:")],
                "part_of": [{"id": cid_of(p[2:]) or None, "name": p[2:]} for p in H.get("partof", {}).get("c:" + n, []) if p.startswith("c:")]}
               for n in cnodes], open(ca / "concepts.json", "w"), indent=1)
    by_task = collections.defaultdict(list)
    for k, v in sorted(contrasts.items()):
        by_task[v["task"]].append({"id": None, "name": v["label"], "key": k, "kind": v["kind"], "direction": v["direction"], "proposed": True})
    json.dump([{"id": (r[4].split(";")[0] or None), "cognitive_atlas_ids": [i for i in r[4].split(";") if i], "name": r[1], "key": r[0],
                "status": r[3], "proposed": not r[4], "definition_text": r[5], "contrasts": by_task.get(r[0], [])} for r in trows],
              open(ca / "tasks.json", "w"), indent=1)
    json.dump([{"relation": "measured_by", "concept": {"id": cid_of(c) or None, "name": c}, "contrast_key": k, "task_key": contrasts[k]["task"],
                "analyses": n, "share_of_contrast": round(n / contrasts[k]["n"], 3)} for k in sorted(cc) for c, n in cc[k].most_common()],
              open(ca / "assertions.json", "w"), indent=1)
    summ = {"analyses": len(rows), "papers": len({r[2] for r in rows}), "analyses_with_a_contrast": sum(bool(r[6]) for r in rows),
            "contrasts": len(contrasts), "tasks": sum(r[3] != "family" for r in trows), "provisional_tasks": sum(r[3] == "provisional" for r in trows),
            "tasks_with_cognitive_atlas_id": sum(bool(r[4]) for r in trows), "families": sum(r[3] == "family" for r in trows),
            "concepts": len(cnodes), "concepts_in_cognitive_atlas": sum(bool(cid_of(n)) for n in cnodes),
            "concepts_proposed": sum(not cid_of(n) for n in cnodes), "kind_of_relations": sum(r[2] == "kind_of" for r in rel),
            "part_of_relations": sum(r[2] == "part_of" for r in rel), "removed_relations": len(removed),
            "contrast_concept_assertions": sum(len(v) for v in cc.values()), "corpus": CORPUS}
    json.dump(summ, open(out / "summary.json", "w"), indent=1); print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
