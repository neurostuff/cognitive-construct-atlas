"""Check the relations mined from the text against Cognitive Atlas, then propose what Cognitive Atlas lacks.

Inputs: vocab/relations.jsonl (mine_relations.py), vocab/abbrev_defs.tsv (mine_abbrevs.py), cogat/concepts.json and
cogat/relations.jsonl (the public API: for concept X, a related concept Y with direction "child" is narrower than X).
A term is a Cognitive Atlas concept when it equals a concept name or alias (lowercased).

Mined hierarchy: "broader" a->b and "specifies" a->b say a contains b; "narrower" a->b says b contains a. A pair's
direction is the one with more papers; it is kept when that side has >= MIN papers and at least 3x the other side.

1. Agreement, for kept pairs whose two terms are both Cognitive Atlas concepts: the same direct edge in Cognitive
   Atlas; an ancestor path (transitive, through KINDOF and PARTOF); the reverse; or no path either way (absent).
2. Proposals (vocab/cogat_proposals.jsonl):
   - edge: two Cognitive Atlas concepts, mined parent -> child, absent from Cognitive Atlas;
   - new_child: a term that is not a Cognitive Atlas concept under a parent that is one;
   - synonym: an abbreviation the papers define for a Cognitive Atlas concept (>= 2 papers) and not already its alias.
   Each with its paper count and an example sentence.

usage: python3 cogat_fill.py [MIN]   (from cc/; MIN papers, default 3)
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MIN = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def main():
    concepts = json.load(open(ROOT / "cogat" / "concepts.json"))
    term2id = {}
    for c in concepts:
        term2id.setdefault(c["name"].lower().strip(), c["id"])
        for a in (c.get("alias") or "").split(","):
            if a.strip():
                term2id.setdefault(a.strip().lower(), c["id"])
    name = {c["id"]: c["name"] for c in concepts}
    alias = {c["id"]: {a.strip().lower() for a in (c.get("alias") or "").split(",") if a.strip()} for c in concepts}
    children = collections.defaultdict(set); fetched = set()
    for l in open(ROOT / "cogat" / "relations.jsonl"):
        r = json.loads(l); fetched.add(r["id"])
        if not r.get("other_id"):
            continue
        if r["direction"] == "child":
            children[r["id"]].add(r["other_id"])
        elif r["direction"] == "parent":
            children[r["other_id"]].add(r["id"])
    def descendants(x):
        seen, todo = set(), [x]
        while todo:
            for y in children.get(todo.pop(), ()):
                if y not in seen:
                    seen.add(y); todo.append(y)
        return seen
    down = collections.Counter(); ex = {}
    for l in open(ROOT / "vocab" / "relations.jsonl"):
        r = json.loads(l)
        if r["relation"] in ("broader", "specifies"):
            key = (r["a"], r["b"])
        elif r["relation"] == "narrower":
            key = (r["b"], r["a"])
        else:
            continue
        down[key] += r["papers"]; ex.setdefault(key, r["example"])
    kept = []
    for (p, c), n in down.items():
        if n >= MIN and n >= 3 * down.get((c, p), 0):
            kept.append((p, c, n))
    agree = collections.Counter(); props = []
    for p, c, n in kept:
        ip, ic = term2id.get(p), term2id.get(c)
        if ip and ic and ip != ic:
            if ip not in fetched or ic not in fetched:
                continue
            if ic in children.get(ip, ()):
                agree["direct edge"] += 1
            elif ic in descendants(ip):
                agree["ancestor path"] += 1
            elif ip in descendants(ic):
                agree["reversed"] += 1
            else:
                agree["absent"] += 1
                props.append({"kind": "edge", "parent": name[ip], "parent_id": ip, "child": name[ic], "child_id": ic,
                              "papers": n, "example": ex[(p, c)]})
        elif ip and not ic:
            props.append({"kind": "new_child", "parent": name[ip], "parent_id": ip, "child": c, "papers": n,
                          "example": ex[(p, c)]})
    for l in list(open(ROOT / "vocab" / "abbrev_defs.tsv"))[1:]:
        lf, sf, n_def, n_pap = l.rstrip("\n").split("\t")
        i = term2id.get(lf)
        initials = "".join(w[0] for w in lf.replace("-", " ").split())
        letters = [ch for ch in sf.lower() if ch.isalpha()]
        it = iter(lf)
        ok = (not any(ch.isdigit() for ch in sf) and letters and letters[0] == initials[0]
              and all(any(ch == x for x in it) for ch in letters))
        if i and ok and int(n_pap) >= 2 and sf.lower() not in alias[i] and sf.lower() != name[i].lower():
            props.append({"kind": "synonym", "concept": name[i], "concept_id": i, "term": lf, "synonym": sf,
                          "papers": int(n_pap)})
    # one row per (parent concept, child): several surface parents ("executive functions", "cognitive control") are
    # one Cognitive Atlas concept
    merged = {}
    for r in props:
        if r["kind"] == "synonym":
            merged[("s", r["concept_id"], r["synonym"].lower())] = r; continue
        key = (r["kind"], r["parent_id"], r.get("child_id") or r["child"])
        if key in merged:
            merged[key]["papers"] += r["papers"]
        else:
            merged[key] = dict(r)
    props = sorted(merged.values(), key=lambda r: (r["kind"], -r["papers"]))
    with open(ROOT / "vocab" / "cogat_proposals.jsonl", "w") as f:
        for r in props:
            f.write(json.dumps(r) + "\n")
    both = sum(agree.values())
    print(f"Cognitive Atlas concepts fetched {len(fetched)} of {len(concepts)}; kept mined parent->child pairs {len(kept)} "
          f"(>= {MIN} papers, 3x the reverse)")
    print(f"pairs with both terms in Cognitive Atlas: {both}: " + ", ".join(f"{k} {v} ({v / max(both, 1):.0%})" for k, v in agree.most_common()))
    kinds = collections.Counter(r["kind"] for r in props)
    print("proposals:", dict(kinds))
    for kind in ("edge", "new_child", "synonym"):
        print(f"== {kind}")
        for r in [x for x in props if x["kind"] == kind][:15]:
            if kind == "synonym":
                print(f"  {r['papers']:4d}  {r['concept']} ({r['term']}) = {r['synonym']}")
            else:
                print(f"  {r['papers']:4d}  {r['parent']} > {r['child']}")


if __name__ == "__main__":
    main()
