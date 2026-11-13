"""Contrasts nested within tasks (James 2026-11-12: "contrasts are nested within task"; "the tasks are just mixtures of
the contrasts"). From a corpus with construct authors on contrasts (contrast_construct_authors.py, corpus B), every
contrast gets its own author node, x:<contrast id>, whose parents are its task(s) and its constructs:

- a contrast's authors are its own node, its modality and material authors and the shared-pattern author: 2-9
  authors (by modality and material), never constructs, so the shared and modality shares no longer shrink as
  constructs are added (in B they ranged from 1/2 to 1/35 of a contrast);
- its mixture is shrunk toward its task's and its constructs' pooled mixtures (the model's prior, s), the constructs
  weighted more (James 2026-11-12: "I want the contrasts to borrow from their constructs more than from their tasks,
  since the same task can measure very different constructs"): together they get SHARE of the prior, split equally
  among them, and the task(s) the rest. The model's prior is the plain mean over the parent list, so the weights are
  written as repeats: with SHARE = a/b, t tasks and c constructs, each task is listed (b - a) * c times and each
  construct a * t times. A contrast with no constructs borrows from its task alone. (The model's pooling splits a
  child among its parent entries, so a construct's pooled mixture also takes the larger share of each contrast's
  counts; a pooled mixture is normalized per node, so only the relative weight of its children changes.)
- direction (James 2026-11-12: "incongruent > congruent relative to vice versa"): a reversed condition contrast has
  its own constructs (contrast_construct_authors.py). A reversed task-vs-baseline or simple effect, a deactivation,
  has none: there B's deactivation author becomes a parent in the constructs' place, so a deactivation borrows from all
  deactivations (SHARE) and from its task (the rest), not from its task's activations alone.
- task and construct nodes author nothing: a task's pooled mixture is its contrasts', a construct's the contrasts it
  is tagged on plus its kind-of child constructs'. Tasks keep their family parents and lose their construct parents:
  a task's constructs already sit on its task-vs-baseline contrasts (B's rule), and a task under a construct would pool
  every contrast of the task into it (simple choice RT's feedback contrasts into processing speed).

usage: python nested_corpus.py CORPUS_B SHARE CORPUS_OUT     (SHARE as a/b, e.g. 3/4)
"""
import collections
import json
import sys
from fractions import Fraction

SPECIAL = ("background", "deactivation")
CONCEPT = ("c:", "deactivation")  # parents in the constructs' place


def is_task(i):
    return not i.startswith(("c:", "m:", "x:")) and i not in SPECIAL


def main():
    cin, share, cout = sys.argv[1], Fraction(sys.argv[2]), sys.argv[3]
    a, b = share.numerator, share.denominator
    assert 0 < share < 1
    H = json.load(open(cin)); P = H["parents"]
    fams = {p for t, ps in P.items() if is_task(t) for p in ps if is_task(p)}
    for t in list(P):
        if is_task(t) and t not in fams:
            P[t] = [p for p in P[t] if not p.startswith("c:")]
    under = collections.Counter(); pap = collections.defaultdict(set)
    for k, d in H["docs"].items():
        x = "x:" + k
        content = [i for i in d["ids"] if is_task(i) or i.startswith(CONCEPT)]
        ts = [i for i in content if is_task(i)]; cs = [i for i in content if i.startswith(CONCEPT)]
        P[x] = ([t for t in ts for _ in range((b - a) * len(cs))] + [c for c in cs for _ in range(a * len(ts))]
                if ts and cs else content)
        H["labels"][x] = k
        d["ids"] = [x] + [i for i in d["ids"] if i not in content]
        for i in content:
            under[i] += 1; pap[i].add(d.get("group"))
        for i in d["ids"]:
            under[i] += 1; pap[i].add(d.get("group"))
    H["n_contrasts"] = {k: under.get(k, 0) for k in P}
    H["n_papers"] = {k: len(pap.get(k, ())) for k in P}
    H["nested"] = {"from": cin, "construct_share": str(share), "contrast_nodes": len(H["docs"]),
                   "authors_per_contrast": dict(collections.Counter(len(d["ids"]) for d in H["docs"].values()))}
    json.dump(H, open(cout, "w"))
    print(json.dumps(H["nested"]))


if __name__ == "__main__":
    main()
