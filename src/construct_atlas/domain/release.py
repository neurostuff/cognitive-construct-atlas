"""The release tables: tasks, contrasts, concepts, their relations, and every analysis linked to Neurostore.

Pure: everything it needs arrives as arguments (the corpus, each analysis's contrast assignment and concept source,
and three lookups given as ports). No text from the papers is put in the tables.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ..ports.catalogs import ConceptRegistry, ContrastCatalog, TaskInventory
from .corpus import CONCEPT, Corpus, is_concept, is_task

NS_ANALYSIS = "https://neurostore.org/api/analyses/{}"
NS_STUDY = "https://neurostore.org/api/base-studies/{}"
CA_CONCEPT = "https://www.cognitiveatlas.org/concept/id/{}"
SOURCE = {"claims": "paper claims", "codex": "model attribution", "atlas": "cognitive atlas",
          "task_contrast": "task claims", "paper": "paper reading", "paper_phase": "paper reading (trial phase)"}
KIND = {"task_vs_baseline": "task vs baseline", "simple": "simple effect", "condition": "condition comparison"}
ADDED = "added 2026-11 (model-proposed, add-only pass)"


@dataclass
class Release:
    tables: dict[str, tuple[list[str], list[list]]] = field(default_factory=dict)   # name -> (header, rows)
    records: dict[str, Any] = field(default_factory=dict)                            # cognitive_atlas/<name> -> JSON
    summary: dict[str, Any] = field(default_factory=dict)


def build_release(corpus: Corpus, assignments: Mapping[str, Mapping], sources: Mapping[str, Mapping],
                  contrasts: ContrastCatalog, concepts: ConceptRegistry, inventory: TaskInventory,
                  added_edges: Iterable[tuple[str, str]] = (), audited: Iterable[str] = (),
                  removed_edges: Iterable[Mapping[str, str]] = (), corpus_name: str = "") -> Release:
    """assignments: analysis -> {"hub": contrast key, "sign": 1 | -1}; sources: analysis -> {"tier", "confidence"};
    added_edges: (child node, parent node) pairs added by the add-only pass; audited: analyses whose concepts the audit
    set; removed_edges: {"child", "parent", "reason"} the audit removed."""
    P, lab = corpus.parents, corpus.labels
    added, audited = set(added_edges), set(audited)
    fams = corpus.families()
    rows = []
    c_docs: collections.Counter[str] = collections.Counter()
    t_docs: collections.Counter[str] = collections.Counter()
    c_pap: dict[str, set] = collections.defaultdict(set)
    t_pap: dict[str, set] = collections.defaultdict(set)
    con: dict[str, dict] = {}
    cc: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for aid, d in sorted(corpus.docs.items()):
        tasks = corpus.tasks_of(d)
        cs = [i[2:] for i in corpus.concepts_of(d)]
        t = tasks[0] if tasks else ""
        a = assignments.get(aid)
        rev = bool(a) and a.get("sign", 1) < 0
        cid = ""
        if a:
            cid = f"{t}|{a['hub'].split('|', 1)[-1]}" + ("|reversed" if rev else "")
            if cid not in con:
                con[cid] = {"task": t, "label": contrasts.label(a["hub"], a.get("sign", 1)), "kind": KIND[contrasts.kind(a["hub"])],
                            "direction": "reversed" if rev else "forward", "n": 0, "papers": set()}
            con[cid]["n"] += 1
            con[cid]["papers"].add(d["group"])
            cc[cid].update(cs)
        f = sources.get(aid) or {}
        src = "audit" if aid in audited else SOURCE.get(f.get("tier") or "", "task claims" if cs else "")
        direction = "deactivation" if "deactivation" in d["ids"] else "reversed" if rev else "forward"
        rows.append([aid, NS_ANALYSIS.format(aid), d["group"], NS_STUDY.format(d["group"]), t, ";".join(tasks[1:]), cid,
                     direction, ";".join(cs), src, f.get("confidence") or ""])
        for c in cs:
            c_docs[c] += 1
            c_pap[c].add(d["group"])
        for x in tasks:
            t_docs[x] += 1
            t_pap[x].add(d["group"])
    R = Release()
    R.tables["analyses"] = (["neurostore_analysis_id", "neurostore_analysis_url", "neurostore_base_study_id",
                             "neurostore_base_study_url", "task", "other_tasks", "contrast", "direction", "concepts",
                             "concept_source", "confidence"], rows)
    cnodes = sorted(n[2:] for n in P if is_concept(n))
    ca = concepts.cognitive_atlas_id
    R.tables["concepts"] = (["id", "name", "cognitive_atlas_id", "cognitive_atlas_url", "status", "analyses", "papers"],
                            [[n.replace(" ", "-"), n, ca(n), CA_CONCEPT.format(ca(n)) if ca(n) else "",
                              "cognitive atlas" if ca(n) else "proposed", c_docs[n], len(c_pap[n])] for n in cnodes])
    partof = corpus.extra.get("partof", {})
    rel = [[n[2:], p[2:], "kind_of", ADDED if (n, p) in added else "cognitive atlas or literature"]
           for n in sorted(P) if is_concept(n) for p in P[n] if is_concept(p)]
    rel += [[n[2:], p[2:], "part_of", "cognitive atlas"] for n, ps in sorted(partof.items()) for p in ps
            if is_concept(n) and is_concept(p)]
    R.tables["concept_relations"] = (["child", "parent", "relation", "source"], rel)
    removed = [[e["child"], e["parent"], "kind_of", e["reason"]] for e in removed_edges]
    R.tables["removed_relations"] = (["child", "parent", "relation", "reason"], removed)
    trows, tcon = [], []
    for t in sorted(x for x in P if is_task(x)):
        if not t_docs[t] and t not in fams:
            continue
        x = inventory.get(t) or {}
        xw = (x.get("crosswalk") or {}).get("cognitive_atlas") or []
        ca_ids = [c if isinstance(c, str) else (c.get("id") or "") for c in xw]
        status = "family" if t in fams else "inventory" if inventory.get(t) else "provisional"
        trows.append([t, x.get("name") or lab.get(t, t), "" if t in fams else next((p for p in P[t] if p in fams), ""), status,
                      ";".join(i for i in ca_ids if i), (x.get("definition") or "").replace("\n", " ").replace("\t", " "),
                      t_docs[t], len(t_pap[t])])
        tcon += [[t, p[2:]] for p in P[t] if is_concept(p)]
    R.tables["tasks"] = (["id", "name", "family", "status", "cognitive_atlas_task_ids", "definition", "analyses", "papers"], trows)
    R.tables["task_concepts"] = (["task", "concept"], tcon)
    R.tables["contrasts"] = (["id", "task", "label", "kind", "direction", "analyses", "papers"],
                             [[k, v["task"], v["label"], v["kind"], v["direction"], v["n"], len(v["papers"])]
                              for k, v in sorted(con.items())])
    R.tables["contrast_concepts"] = (["contrast", "concept", "analyses", "share"],
                                     [[k, c, n, round(n / con[k]["n"], 3)] for k in sorted(cc) for c, n in cc[k].most_common()])
    R.records["concepts"] = [
        {"id": ca(n) or None, "name": n, "proposed": not ca(n),
         "is_a": [{"id": ca(p[2:]) or None, "name": p[2:]} for p in P[CONCEPT + n] if is_concept(p)],
         "part_of": [{"id": ca(p[2:]) or None, "name": p[2:]} for p in partof.get(CONCEPT + n, []) if is_concept(p)]}
        for n in cnodes]
    by_task: dict[str, list] = collections.defaultdict(list)
    for k, v in sorted(con.items()):
        by_task[v["task"]].append({"id": None, "name": v["label"], "key": k, "kind": v["kind"], "direction": v["direction"],
                                   "proposed": True})
    R.records["tasks"] = [
        {"id": (r[4].split(";")[0] or None), "cognitive_atlas_ids": [i for i in r[4].split(";") if i], "name": r[1], "key": r[0],
         "status": r[3], "proposed": not r[4], "definition_text": r[5], "contrasts": by_task.get(r[0], [])} for r in trows]
    R.records["assertions"] = [
        {"relation": "measured_by", "concept": {"id": ca(c) or None, "name": c}, "contrast_key": k, "task_key": con[k]["task"],
         "analyses": n, "share_of_contrast": round(n / con[k]["n"], 3)} for k in sorted(cc) for c, n in cc[k].most_common()]
    R.summary = {
        "analyses": len(rows), "papers": len({r[2] for r in rows}), "analyses_with_a_contrast": sum(bool(r[6]) for r in rows),
        "contrasts": len(con), "tasks": sum(r[3] != "family" for r in trows),
        "provisional_tasks": sum(r[3] == "provisional" for r in trows),
        "tasks_with_cognitive_atlas_id": sum(bool(r[4]) for r in trows), "families": sum(r[3] == "family" for r in trows),
        "concepts": len(cnodes), "concepts_in_cognitive_atlas": sum(bool(ca(n)) for n in cnodes),
        "concepts_proposed": sum(not ca(n) for n in cnodes), "kind_of_relations": sum(r[2] == "kind_of" for r in rel),
        "part_of_relations": sum(r[2] == "part_of" for r in rel), "removed_relations": len(removed),
        "contrast_concept_assertions": sum(len(v) for v in cc.values()), "corpus": corpus_name}
    return R
