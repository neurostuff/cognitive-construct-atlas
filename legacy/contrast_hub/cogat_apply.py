"""Apply the Cognitive Atlas normalization (cogat_normalize.py, with the hand overrides) to an author-topic corpus.

For each construct node: `same` merges it into the named node (a Cognitive Atlas concept, or a kept construct its
near-synonyms are merged into); `narrower` keeps it and adds the named Cognitive Atlas concept as its parent, so its
contrasts also pool into the concept; `new` keeps it, with the named broader concept as its parent if any; `drop`
removes it (not a construct: "cognitive function", "brain function"). Applied to the documents' authors and to every
parent list (a task's construct parents too). A node referenced but absent is added with no parent; a parent edge that
would close a cycle is skipped. Nodes the normalization did not judge (Cognitive Atlas concepts) are unchanged.

usage: python cogat_apply.py CORPUS_IN NORMALIZE.json OVERRIDES.json CORPUS_OUT
"""
import collections
import json
import sys


def node(label):
    return "c:" + label.strip().lower().replace("-", " ")


def mapping(norm, over):
    """construct node -> (target node or None, parent node or None)"""
    out = {}
    for n, v in norm.items():
        rel, con = v["relation"], v["concept"]
        o = over.get(n[2:])
        if o:
            rel, con = o
        if rel == "drop":
            out[n] = (None, None)
        elif rel == "same" and con:
            out[n] = (node(con), None)
        else:
            out[n] = (n, node(con) if con else None)
    for n in over:                                    # an override may name a construct the run did not see
        if node(n) not in out:
            rel, con = over[n]
            out[node(n)] = (None, None) if rel == "drop" else (node(con), None) if rel == "same" else (node(n), node(con) if con else None)
    for n, (t, p) in list(out.items()):               # follow merges to their end: a -> b -> c
        seen = {n}
        while t in out and out[t][0] not in (t, None) and t not in seen:
            seen.add(t); t = out[t][0]
        out[n] = (t, p)
    return out


def main():
    cin, normp, overp, cout = sys.argv[1:5]
    H = json.load(open(cin)); M = mapping(json.load(open(normp)), json.load(open(overp))["overrides"])
    to = lambda x: M[x][0] if x in M else x  # noqa: E731
    n = collections.Counter()
    for d in H["docs"].values():
        ids = []
        for x in d["ids"]:
            y = to(x) if x.startswith("c:") else x
            if y is None:
                n["author slots dropped"] += 1; continue
            if y not in ids:
                ids.append(y)
        n["author slots merged"] += sum(1 for x in d["ids"] if x.startswith("c:")) - sum(1 for x in ids if x.startswith("c:"))
        d["ids"] = ids
    P = {}
    for k, ps in H["parents"].items():
        k2 = to(k) if k.startswith("c:") else k
        if k2 is None:
            continue
        mapped = [to(p) if p.startswith("c:") else p for p in ps]
        P.setdefault(k2, [])
        P[k2] += [p for p in mapped if p and p != k2 and p not in P[k2]]
    for k, (t, p) in M.items():                       # parents from the normalization
        if t == k and p and t in P and p != t:
            if p not in P:
                P[p] = []
            if p not in P[t]:
                P[t].append(p); n["parent edges added"] += 1
    for d in H["docs"].values():
        for x in d["ids"]:
            if x not in P:
                P[x] = []; n["nodes added"] += 1
    for k in list(P):
        for p in P[k]:
            if p not in P:
                P[p] = []; n["nodes added"] += 1

    def reaches(a, b, seen=None):                     # is b an ancestor of a
        seen = seen or set()
        for p in P.get(a, []):
            if p == b or (p not in seen and (seen.add(p) or reaches(p, b, seen))):
                return True
        return False
    for k in P:
        keep = []
        for p in P[k]:
            if reaches(p, k):
                n["cycle edges skipped"] += 1
            else:
                keep.append(p)
        P[k] = keep
    H["parents"] = P
    H["labels"] = {k: H["labels"].get(k, k[2:] if k.startswith("c:") else k) for k in P}
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"])
    pap = collections.defaultdict(set)
    for d in H["docs"].values():
        for i in d["ids"]:
            pap[i].add(d.get("group"))
    H["n_contrasts"] = {k: cnt.get(k, 0) for k in P}
    H["n_papers"] = {k: len(pap.get(k, ())) for k in P}
    H["cogat_normalization"] = {"normalize": normp, "overrides": overp, "counts": dict(n)}
    json.dump(H, open(cout, "w"))
    used = [k for k, v in cnt.items() if k.startswith("c:")]
    print(json.dumps(dict(n)), f"| construct authors used {len(used)}; construct nodes {sum(k.startswith('c:') for k in P)}")


if __name__ == "__main__":
    main()
