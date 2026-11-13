"""Second-look renders: stage-1 sentences the first judge pass did not list as own_task/own_contrast but a classifier
ranks as possible claims.

- gold papers (validation): out-of-fold probability p >= 0.01 from preds/clf_biomedbert.jsonl (8% of all units);
- new papers: the top FRACTION of stage-1 units by the all-gold model's margin (preds/silver_goldfull.jsonl).

Writes silver/render2/<id>.md, one line per sentence with the previous and next sentences as context.

usage: python render2.py JUDGE FRACTION   (run from cc/), e.g. python render2.py sonnet 0.15
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
POS = {"own_task", "own_contrast"}


def listed(judge, a):
    f = ROOT / "labels" / "judge" / f"{a}.{judge}.jsonl"
    return {int(json.loads(l)["i"]) for l in open(f) if l.strip() and json.loads(l)["category"] in POS}


def main():
    judge, frac = sys.argv[1], float(sys.argv[2])
    gold_ids = {a for g in json.load(open(ROOT / "groups.json")) for a in g}
    old = {(r["article"], r["i"]): r["p"] for r in map(json.loads, open(ROOT / "preds" / "clf_biomedbert.jsonl"))}
    new = [json.loads(l) for l in open(ROOT / "preds" / "silver_goldfull.jsonl")]
    zs = sorted((r["z"] for r in new), reverse=True)
    cut = zs[int(frac * len(zs))]
    newz = {(r["article"], r["i"]): r["z"] for r in new}
    out = ROOT / "silver" / "render2"; out.mkdir(exist_ok=True)
    tot = {"gold": 0, "new": 0}
    for f in sorted((ROOT / "silver" / "papers").glob("*.json")):
        s = json.load(open(f)); a = s["article"]
        if not (ROOT / "labels" / "judge" / f"{a}.{judge}.jsonl").exists():
            continue
        done = listed(judge, a); isgold = a in gold_ids
        pick = [i for i in s["candidates"] if i not in done and
                (old.get((a, i), 0) >= 0.01 if isgold else newz.get((a, i), -99) >= cut)]
        by = {u["i"]: k for k, u in enumerate(s["units"])}; U = s["units"]
        lines = [f"# Paper {a}", ""]
        for i in pick:
            k = by[i]
            prev = U[k - 1]["text"][-200:] if k else ""
            nxt = U[k + 1]["text"][:200] if k + 1 < len(U) else ""
            lines.append(f"[{i}] ({U[k]['section']}) task: {'; '.join(s['target_names'])} «{prev}» {U[k]['text']} «{nxt}»")
        (out / f"{a}.md").write_text("\n".join(lines) + "\n")
        tot["gold" if isgold else "new"] += len(pick)
    print(f"cutoff z {cut:.2f}; sentences for a second look: {tot}")


if __name__ == "__main__":
    main()
