"""Move a paper's analyses from one task author to another (James 2026-11-12: processing speed must not map to the
ventral striatum). A paper's task label can name the response format rather than the paradigm its analyses measure:
a choice RT task with performance feedback whose every analysis is a feedback-valence contrast. task_overrides.json
lists such papers by Neurostore base-study id (the corpus's document group), with the task to move from, the task to
move to, and the reason. Only documents of that paper that carry the "from" task are changed. Parents and labels are
unchanged; counts are recomputed.

usage: python task_overrides.py CORPUS_IN task_overrides.json CORPUS_OUT
"""
import collections
import json
import sys


def main():
    cin, overp, cout = sys.argv[1:4]
    H = json.load(open(cin)); O = json.load(open(overp)); moved = collections.Counter()
    for k, o in O.items():
        assert o["to"] in H["parents"], f"{o['to']} is not a node of {cin}"
    for d in H["docs"].values():
        o = O.get(d.get("group"))
        if o and o["from"] in d["ids"]:
            d["ids"] = [o["to"] if i == o["from"] else i for i in d["ids"]]; moved[d["group"]] += 1
    missing = set(O) - set(moved)
    assert not missing, f"no documents moved for {sorted(missing)}"
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"])
    pap = collections.defaultdict(set)
    for d in H["docs"].values():
        for i in d["ids"]:
            pap[i].add(d.get("group"))
    H["n_contrasts"] = {k: cnt.get(k, 0) for k in H["parents"]}
    H["n_papers"] = {k: len(pap.get(k, ())) for k in H["parents"]}
    H["task_overrides"] = {"file": overp, "moved": dict(moved)}
    json.dump(H, open(cout, "w"))
    print(f"{cout}: moved {dict(moved)}")


if __name__ == "__main__":
    main()
