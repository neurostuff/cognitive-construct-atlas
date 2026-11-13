"""Score an LLM sentence judge (labels/judge/<id>.<judge>.jsonl) against the adjudicated gold (ADJ2), on the gold
papers it judged. Only the sentences shown to it (stage-1 candidates, silver/papers/<id>.json) are scored, so
recall here is recall within stage 1. Also scores its construct against labeller A's phrases (score_asserted.match).

usage: python score_judge.py JUDGE   (run from cc/)
"""
import glob
import json
import sys
from collections import Counter
from pathlib import Path

import importlib.util

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("sa", ROOT / "score_asserted.py")
sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
POS = {"own_task", "own_contrast"}


def main():
    judge = sys.argv[1]
    c = Counter(); confusion = Counter(); papers = 0
    for f in sorted(glob.glob(str(ROOT / "labels" / "judge" / f"*.{judge}.jsonl"))):
        a = Path(f).name.split(".")[0]
        g = ROOT / "labels" / f"{a}.ADJ2.jsonl"
        if not g.exists():
            continue
        papers += 1
        gold = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(g)}
        A = {json.loads(l)["i"]: json.loads(l) for l in open(ROOT / "labels" / f"{a}.A.jsonl")}
        cand = set(json.load(open(ROOT / "silver" / "papers" / f"{a}.json"))["candidates"])
        pred = {}
        for l in open(f):
            if l.strip():
                r = json.loads(l); i = int(r["i"])
                if i not in pred or r["category"] in POS:
                    pred[i] = r
        for i in cand:
            gy = gold.get(i) in POS; py = pred.get(i, {}).get("category") in POS
            c["gold"] += gy; c["pred"] += py; c["tp"] += gy and py
            confusion[(gold.get(i, "none"), pred.get(i, {}).get("category", "none"))] += 1
            if gy and py and A.get(i, {}).get("constructs"):
                c["cons"] += 1
                c["cons_ok"] += any(sa.match(x, y) for x in A[i]["constructs"] for y in pred[i].get("constructs") or [])
        c["outside"] += sum(1 for i in pred if i not in cand)
    P = c["tp"] / max(c["pred"], 1); R = c["tp"] / max(c["gold"], 1)
    print(f"{judge} on {papers} gold papers (stage-1 sentences): claims predicted {c['pred']}, gold {c['gold']}, "
          f"precision {P:.2f}, recall {R:.2f}, F1 {2 * P * R / max(P + R, 1e-9):.2f}; construct right on "
          f"{c['cons_ok']}/{c['cons']} found claims; {c['outside']} lines outside the candidates")
    print("gold -> judge (counts, top 15):")
    for (g, p), n in confusion.most_common(15):
        print(f"  {g:13s} -> {p:13s} {n}")


if __name__ == "__main__":
    main()
