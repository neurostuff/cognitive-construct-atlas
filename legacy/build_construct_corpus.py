"""The K=16 task-networks corpus with constructs in its hierarchy: task -> its constructs -> their broader constructs.

Same documents, tokens and shared-pattern author as the original (task-networks-repro/corpus.json: one document per
contrast, authors = its task labels + "background"). Only `parents` changes:
- a task's parents are its passing constructs (vocab/task_constructs_pass.json: distinctive at log-odds z >= 2 in
  >= 5 papers, or the most-claimed fallback), kept when the construct passes for >= MIN_SHARE of the corpus's tasks
  (a construct of one task only is the task itself); the model splits a child's counts equally among its parents
  and shrinks it toward their mean, so the constructs are weighted equally;
- a construct's parents: the parent the text states (vocab/vocab_v0.json "parent", relation "stated"), and, for a
  construct linked to Cognitive Atlas, its KINDOF parents there (cogat/relations.jsonl, direction "parent" from the
  child's record or "child" from the parent's); an edge that would close a cycle is skipped. Only kind-of edges pool:
  a task measuring a kind of X measures X, but a task measuring a part of X does not measure X (2026-11-11: with
  PARTOF edges, "set shifting" pooled every decision-making task, from its part "decision making", and "task
  switching" inherited their reward activity). PARTOF edges between constructs in the hierarchy are kept, for
  display only, in H["partof"];
- a task named after a construct gets it as a parent (NAMESAKE): in a task's own papers the task's name is read as
  the task, not as a construct claim, so "task switching" never reached the task-switching task's profile. Applied
  when the name is a Cognitive Atlas concept or a construct some other task already has;
- category tasks that were never searched by name (motor execution, pain stimulation, ...) get the constructs hand-
  mapped in vocab/category_constructs.json;
- a task with no shared construct keeps its curated family as its parent (MODE "constructs"), or every task keeps
  its family besides its constructs (MODE "both").
Construct nodes are "c:<entry key>"; they have no documents of their own, so their mixture is the pooled subtree.

usage: python build_construct_corpus.py ORIGINAL_CORPUS OUT_CORPUS [MODE] [MIN_SHARE]   (from pilot-results/)
"""
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
V = ROOT / "vocab"
NLTK = None


def ckey(name):
    """the vocabulary key of a Cognitive Atlas name (vocab_normalize.key)"""
    sys.path.insert(0, str(ROOT / "code"))
    import vocab_normalize
    return vocab_normalize.key(name)


def main():
    src, out = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "constructs"
    min_share = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    H = json.load(open(src))
    old = H["parents"]
    fams = {p for ps in old.values() for p in ps}
    tasks = [n for n in old if n not in fams and n != "background"]
    passing = json.load(open(V / "task_constructs_pass.json"))
    entries = {r["key"]: r for r in json.load(open(V / "vocab_v0.json"))}
    # entries linked to one Cognitive Atlas concept are one construct ("executive function" is an alias of cognitive
    # control there): the concept's own key stands for all of them
    concepts0 = json.load(open(ROOT / "code" / "cogat_concepts.json"))
    cname0 = {c["id"]: c["name"] for c in concepts0}
    canon = {k: (ckey(cname0[e["cogat_id"]]) if e.get("cogat_id") else k) for k, e in entries.items()}
    tc = {t: list(dict.fromkeys(canon.get(c["entry"], c["entry"]) for c in passing.get(t, []))) for t in tasks}
    manual = Path(V / "category_constructs.json")
    if manual.exists():  # hand-mapped category tasks (no literature search): constructs added as given
        for t, ks in json.load(open(manual)).items():
            if t in tc:
                tc[t] = list(dict.fromkeys(tc[t] + [canon.get(k, ckey(k)) for k in ks]))
    share = collections.Counter(k for t in tasks for k in tc[t])
    task_cons = {t: [k for k in tc[t] if share[k] >= min_share] for t in tasks}
    # construct hierarchy: stated parents and Cognitive Atlas parents
    concepts = json.load(open(ROOT / "code" / "cogat_concepts.json"))
    cname = {c["id"]: c["name"] for c in concepts}
    cparent = collections.defaultdict(set); cpartof = collections.defaultdict(set)
    for l in open(ROOT / "code" / "cogat_relations.jsonl"):
        r = json.loads(l)
        if not r.get("other_id") or r.get("relationship") not in ("KINDOF", "PARTOF"):
            continue
        target = cparent if r["relationship"] == "KINDOF" else cpartof
        if r["direction"] == "parent":
            target[r["id"]].add(r["other_id"])
        elif r["direction"] == "child":
            target[r["other_id"]].add(r["id"])
    id_of_key = {}
    for e in entries.values():
        if e.get("cogat_id"):
            id_of_key[e["key"]] = e["cogat_id"]
    key_of_id = {i: ckey(cname[i]) for i in cname}

    by_canon = collections.defaultdict(list)
    for k, c in canon.items():
        by_canon[c].append(k)
    for i, n in cname.items():
        id_of_key.setdefault(ckey(n), i)

    # NAMESAKE: a task named after a Cognitive Atlas concept, or after a construct another task has, measures it
    have = {k for ks in task_cons.values() for k in ks}
    namesake = {}
    for t in tasks:
        nm = re.sub(r"\s*\(.*?\)", "", H["labels"].get(t, t)).strip()
        k = canon.get(ckey(nm), ckey(nm))
        if k and k not in task_cons[t] and (k in id_of_key or k in have):
            task_cons[t] = [k] + task_cons[t]; namesake[t] = k

    def parents_of(k):
        ps = set()
        for member in by_canon.get(k, [k]):
            p = (entries.get(member) or {}).get("parent")
            if p and p.get("relation") == "stated":
                ps.add(canon.get(p["key"], p["key"]))
        cid = id_of_key.get(k)
        if cid:
            ps |= {key_of_id[x] for x in cparent.get(cid, ()) if x in key_of_id}
        ps.discard(k)
        return ps
    cons = {k for ks in task_cons.values() for k in ks}
    edges = collections.defaultdict(set)
    todo = sorted(cons, reverse=True); seen = set(todo)  # sorted: which edge a cycle drops does not depend on hashing

    def reaches(a, b):  # is b reachable upward from a
        stack, vis = [a], set()
        while stack:
            x = stack.pop()
            if x == b:
                return True
            if x in vis:
                continue
            vis.add(x); stack.extend(edges.get(x, ()))
        return False
    skipped = 0
    while todo:
        k = todo.pop()
        for p in sorted(parents_of(k)):
            if reaches(p, k):
                skipped += 1; continue
            edges[k].add(p)
            if p not in seen:
                seen.add(p); todo.append(p)
        todo.sort(reverse=True)
    node = lambda k: f"c:{k}"
    P = {}
    for t in tasks:
        ps = [node(k) for k in task_cons[t]]
        if mode == "both" or not ps:
            ps = ps + old[t]
        P[t] = ps
    for k in seen:
        P[node(k)] = sorted(node(p) for p in edges.get(k, ()))
    for f in fams:
        P.setdefault(f, [])
    P["background"] = []
    H["parents"] = P
    lab = H["labels"]
    for k in seen:
        e = entries.get(k)
        lab[node(k)] = cname[id_of_key[k]] if k in id_of_key else ((e.get("cogat_name") or e["label"]) if e else k)
    H["labels"] = lab
    papers = collections.defaultdict(set)
    for d in H["docs"].values():
        stack = list(d["ids"]); vis = set()
        while stack:
            x = stack.pop()
            if x in vis:
                continue
            vis.add(x); papers[x].add(d["group"]); stack.extend(P.get(x, ()))
    H["n_papers"] = {k: len(papers[k]) for k in P}
    H["partof"] = {node(k): sorted(node(key_of_id[x]) for x in cpartof.get(id_of_key.get(k), ()) if x in key_of_id
                                   and key_of_id[x] in seen) for k in seen}
    H["partof"] = {k: v for k, v in H["partof"].items() if v}
    H["namesake"] = namesake
    H["construct_build"] = {"mode": mode, "min_share": min_share, "tasks": len(tasks),
                            "tasks_with_constructs": sum(bool(v) for v in task_cons.values()),
                            "construct_nodes": len(seen), "task_construct_edges": sum(len(v) for v in task_cons.values()),
                            "construct_parent_edges": sum(len(v) for v in edges.values()), "cycle_edges_skipped": skipped,
                            "namesake_edges": len(namesake), "partof_display_edges": sum(len(v) for v in H["partof"].values())}
    json.dump(H, open(out, "w"))
    print(json.dumps(H["construct_build"]))
    for t in sorted(tasks, key=lambda t: -H["n_contrasts"].get(t, 0))[:8]:
        print(f"  {lab.get(t, t)}: " + ", ".join(lab[p] for p in P[t]))


if __name__ == "__main__":
    main()
