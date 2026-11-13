"""Apply an audit round's agreed changes to an author-topic corpus (DESIGN.md, Audit workflow, step 5).

- tasks (TASK_MOVES.json: paper -> {to, name, family}): every document of the paper filed under the task the readers
  found wrong (the proposals' task_id) moves to `to`; a provisional task is added under its family with its name.
- constructs (only with CONSTRUCTS=1, the contrast-level corpus): for each agreed wrong_construct, the constructs both
  readers called wrong are removed and replacements added: those both readers named, else the first reader's; each
  mapped through the Cognitive Atlas normalization and kept only if it is already a node.
- edges (only with EDGES=1): the kind-of edges both edge readers said to remove are removed (James 2026-11-13: "make
  the fixes for the existing audit"; until then the hierarchy was add-only).

usage: [CONSTRUCTS=1] python audit_apply.py CORPUS_IN PROPOSALS.json TASK_MOVES.json COGAT_NORMALIZE.json COGAT_OVERRIDES.json CORPUS_OUT
"""
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cogat_apply import mapping, node  # noqa: E402


def main():
    cin, propp, movep, normp, overp, cout = sys.argv[1:7]
    H = json.load(open(cin)); P = H["parents"]; R = json.load(open(propp)); MV = {k: v for k, v in json.load(open(movep)).items() if not k.startswith("_")}
    M = mapping(json.load(open(normp)), json.load(open(overp))["overrides"]); n = collections.Counter()
    wrong = {x["paper"]: x["task_id"] for x in R["wrong_task"]}
    moves = {}                                       # (paper, filed task) -> move; keys "paper" (round 1) or "paper|task"
    for k, m in MV.items():
        moves[tuple(k.split("|", 1)) if "|" in k else (k, wrong[k])] = m
    for m in moves.values():
        if m["to"] not in P:
            assert m.get("family") in P, m; P[m["to"]] = [m["family"]]; H["labels"][m["to"]] = m["name"]; n["tasks added"] += 1
    for d in H["docs"].values():
        for i in list(d["ids"]):
            m = moves.get((d.get("group"), i))
            if m:
                d["ids"] = [m["to"] if j == i else j for j in d["ids"]]; n["analyses moved"] += 1
    if os.environ.get("CONSTRUCTS"):
        for x in R["wrong_construct"]:
            d = H["docs"].get(x["analysis"])
            if d is None:
                continue
            rem = {M.get(node(c), (node(c), None))[0] for c in x["remove"]}
            both = [c for c in x["add_first"] if c in x["add_second"]] or x["add_first"]
            add = []
            for c in both:
                t = M.get(node(c), (node(c), None))[0]
                if t in P and t not in add:
                    add.append(t)
                elif t not in P:
                    n["replacement not a node, skipped"] += 1
            keep = [i for i in d["ids"] if not (i.startswith("c:") and i in rem)]
            if not any(i.startswith("c:") for i in keep) and not add:
                n["would leave no construct, skipped"] += 1; continue
            d["ids"] = keep + [c for c in add if c not in keep]; n["analyses with constructs changed"] += 1
    if os.environ.get("EDGES"):
        for e in R.get("edges", []):
            c, q = "c:" + e["child"], "c:" + e["parent"]
            if e["decision"] == "remove" and q in P.get(c, []):
                P[c].remove(q); n["edges removed"] += 1
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"]); pap = collections.defaultdict(set)
    for d in H["docs"].values():
        for i in d["ids"]:
            pap[i].add(d.get("group"))
    H["n_contrasts"] = {k: cnt.get(k, 0) for k in P}; H["n_papers"] = {k: len(pap.get(k, ())) for k in P}
    H.setdefault("audit", []).append({"proposals": propp, "moves": movep, "counts": dict(n)})
    json.dump(H, open(cout, "w")); print(cout, json.dumps(dict(n)))


if __name__ == "__main__":
    main()
