"""Constructs as authors: each contrast also gets, as authors, the constructs its tasks have as parents.

In the hierarchy corpora a construct is only a parent: it sets its child tasks' prior, and its map is the pooled
map of the tasks below it. As an author, it competes with the task, modality and shared-pattern authors for the
contrast's voxels (each voxel picks one of the contrast's authors uniformly), so its map is what its tasks share.
Parents, labels and every other author are unchanged, so held-out-task groups match the input corpus's.

usage: python add_construct_authors.py CORPUS_IN CORPUS_OUT   (from pilot-results/atm)
"""
import collections
import json
import sys


def main():
    H = json.load(open(sys.argv[1])); P = H["parents"]
    n = collections.Counter(); per = []
    for d in H["docs"].values():
        tasks = [x for x in d["ids"] if not x.startswith(("c:", "m:")) and x != "background"]
        cons = sorted({p for t in tasks for p in P.get(t, []) if p.startswith("c:")})
        d["ids"] = d["ids"] + [c for c in cons if c not in d["ids"]]
        n.update(cons); per.append(len(cons))
    for c, k in n.items():
        H["n_contrasts"][c] = k
    json.dump(H, open(sys.argv[2], "w"))
    print(f"{sys.argv[2]}: {sum(x > 0 for x in per)} of {len(per)} contrasts get construct authors; "
          f"per contrast mean {sum(per) / len(per):.2f}, max {max(per)}; construct authors used {len(n)}")


if __name__ == "__main__":
    main()
