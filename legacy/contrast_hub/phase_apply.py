"""Put the phase-aware readings' constructs on corpus B's contrasts (James 2026-11-12: "change the associated
constructs for those particular contrasts"; DESIGN.md, Trial phase).

For every analysis paper_attribute.py read in phase mode (phase_scope.py's mixed-phase signatures), its construct
authors are replaced by the reading's constructs, each mapped to a corpus node by the Cognitive Atlas normalization the
corpus was built with (cogat_apply.mapping: merges followed, dropped names removed). A reading with no usable construct
leaves the contrast as it was. Deactivations (the deactivation author: reversed task-vs-baseline and simple effects)
keep B's rule, no construct authors. A construct that is not yet a node is added with no parent and listed: run
cogat_attach.py on them before fitting.

usage: python phase_apply.py CORPUS_B_IN PHASE_ATTRIBUTION.json COGAT_NORMALIZE.json COGAT_OVERRIDES.json CORPUS_OUT
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # cogat_apply: here (laptop) or code/ (beast)
from cogat_apply import mapping, node  # noqa: E402


def main():
    cin, attrp, normp, overp, cout = sys.argv[1:6]
    H = json.load(open(cin)); A = json.load(open(attrp)); P = H["parents"]
    M = mapping(json.load(open(normp)), json.load(open(overp))["overrides"])
    n = collections.Counter(); new_nodes = collections.Counter(); changed = collections.Counter()
    for aid, v in A.items():
        d = H["docs"].get(aid)
        if d is None:
            n["not in corpus"] += 1; continue
        if "deactivation" in d["ids"]:
            n["deactivation kept"] += 1; continue
        cs = []
        for c in v["constructs"]:
            t = M.get(node(c), (node(c), None))[0]
            if t and t not in cs:
                cs.append(t)
        if not cs:
            n["no usable construct, kept"] += 1; continue
        old = [i for i in d["ids"] if i.startswith("c:")]
        for c in cs:
            if c not in P:
                P[c] = []; new_nodes[c] += 1
        d["ids"] = [i for i in d["ids"] if not i.startswith("c:")] + cs
        n["unchanged" if set(old) == set(cs) else "changed"] += 1
        if set(old) != set(cs):
            changed[(tuple(sorted(old))[:2], tuple(cs[:2]))] += 1
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"])
    pap = collections.defaultdict(set)
    for d in H["docs"].values():
        for i in d["ids"]:
            pap[i].add(d.get("group"))
    H["n_contrasts"] = {k: cnt.get(k, 0) for k in P}
    H["n_papers"] = {k: len(pap.get(k, ())) for k in P}
    H["phase_constructs"] = {"readings": attrp, "counts": dict(n), "new_nodes": sorted(new_nodes)}
    json.dump(H, open(cout, "w"))
    print(json.dumps({"counts": dict(n), "new nodes (no parent yet)": len(new_nodes)}, indent=1))
    print("new nodes:", sorted(new_nodes)[:40])
    print("most common changes:", [f"{list(a)} -> {list(b)} x{k}" for (a, b), k in changed.most_common(12)])


if __name__ == "__main__":
    main()
