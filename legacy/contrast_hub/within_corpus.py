"""Keep only the within-task analyses (the sample) in an author-topic corpus (James 2026-11-12: the corpus holds only
comparisons of conditions within a task; the Neurostore export predates that exclusion and still has between-subject,
group and session comparisons). Parents and labels are unchanged; counts are recomputed.

usage: python within_corpus.py CORPUS_IN SAMPLE.json CORPUS_OUT
"""
import collections
import json
import sys


def main():
    cin, samplep, cout = sys.argv[1:4]
    H = json.load(open(cin)); keep = set(json.load(open(samplep)))
    n0 = len(H["docs"])
    H["docs"] = {k: d for k, d in H["docs"].items() if k in keep}
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"])
    pap = collections.defaultdict(set)
    for d in H["docs"].values():
        for i in d["ids"]:
            pap[i].add(d.get("group"))
    H["n_contrasts"] = {k: cnt.get(k, 0) for k in H["parents"]}
    H["n_papers"] = {k: len(pap.get(k, ())) for k in H["parents"]}
    H["within_task"] = {"sample": samplep, "documents_before": n0, "documents": len(H["docs"])}
    json.dump(H, open(cout, "w"))
    print(f"{cout}: {len(H['docs'])} of {n0} documents")


if __name__ == "__main__":
    main()
