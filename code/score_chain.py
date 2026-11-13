"""Claim sentence + construct: the no-LLM chain (classifier -> span tagger) against GLiNER and the LLM extractors.

A pair (sentence, construct) is correct when the sentence is a claim in the adjudicated gold (ADJ2) and the construct
matches one of labeller A's construct phrases for it (score_asserted.match: content-word Jaccard >= 0.5 or
containment). Claims the adjudication added have no A phrases and are reported apart, as "claim, construct unjudged".

usage: python score_chain.py   (run from cc/; reads preds/spans_biomedbert.jsonl, preds/gliner*_spans.jsonl, llm/)
"""
import glob
import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("sa", ROOT / "score_asserted.py")
sa = importlib.util.module_from_spec(spec); spec.loader.exec_module(sa)
POS = {"own_task", "own_contrast"}


def gold():
    claims, phrases = {}, {}
    for f in glob.glob(str(ROOT / "labels" / "*.ADJ2.jsonl")):
        a = Path(f).name.split(".")[0]
        claims[a] = {json.loads(l)["i"] for l in open(f) if l.strip() and json.loads(l)["category"] in POS}
        phrases[a] = {}
        for l in open(ROOT / "labels" / f"{a}.A.jsonl"):
            r = json.loads(l)
            if r["category"] in POS and r.get("constructs"):
                phrases[a][r["i"]] = r["constructs"]
    return claims, phrases


def judge(claims, phrases, a, i, construct):
    if i not in claims.get(a, set()):
        return "not a claim"
    if i not in phrases.get(a, {}):
        return "claim, construct unjudged"
    return "right" if construct and any(sa.match(g, construct) for g in phrases[a][i]) else "claim, construct wrong"


def tally(rows):
    n = len(rows); right = sum(r == "right" for r in rows); unj = sum(r == "claim, construct unjudged" for r in rows)
    wrong = sum(r == "claim, construct wrong" for r in rows); nc = sum(r == "not a claim" for r in rows)
    judged = n - unj
    return (f"{n:4d} kept: right {right/max(1,judged):.2f} of judged | not a claim {nc}, construct wrong {wrong}, "
            f"unjudged claims {unj}")


def main():
    claims, phrases = gold()
    S = [json.loads(l) for l in open(ROOT / "preds" / "spans_biomedbert.jsonl")]
    # 1) tagger alone on held-out gold claim sentences; GLiNER on the same sentences
    g = [r for r in S if r["kind"] == "gold"]
    top = lambda pred: max(pred, key=lambda x: x[1])[0] if pred else None
    ok_top = sum(bool(top(r["pred"])) and any(sa.match(x, top(r["pred"])) for x in phrases[r["article"]].get(r["i"], [])) for r in g)
    ok_any = sum(any(sa.match(x, p) for x in phrases[r["article"]].get(r["i"], []) for p, _ in r["pred"]) for r in g)
    print(f"BiomedBERT span tagger on {len(g)} held-out gold claim sentences: top span right {ok_top/len(g):.2f}, "
          f"some span right {ok_any/len(g):.2f}, no span {sum(not r['pred'] for r in g)}")
    keys = {(r["article"], r["i"]) for r in g}
    for f in sorted(glob.glob(str(ROOT / "preds" / "gliner*_spans.jsonl"))):
        rows = [json.loads(l) for l in open(f)]
        rows = [r for r in rows if (r["article"], r["i"]) in keys]
        t = sum(bool(r["pred"]) and any(sa.match(x, max(r["pred"], key=lambda z: z[1])[0]) for x in r["gold"]) for r in rows)
        print(f"  {Path(f).name:28s} same sentences ({len(rows)}): top span right {t/max(1,len(rows)):.2f}")
    # 2) chain: classifier rank -> tagger's top span
    c = sorted([r for r in S if r["kind"] == "cand"], key=lambda r: -r["p"])
    print("\nchain: classifier's top-ranked sentences, each with the tagger's top span")
    for n in (100, 151, 300, 500):
        print(f"  top {n}: " + tally([judge(claims, phrases, r["article"], r["i"], top(r["pred"])) for r in c[:n]]))
    # 3) LLM extractors: their claims with their first construct, on the papers each covers
    print("\nLLM extractors: every claim they list, with its first construct")
    for tag in ("codex-astra-linked", "codex-terra-linked", "sonnetv2", "sonnetguide"):
        rows, papers = [], 0
        for f in glob.glob(str(ROOT / "llm" / f"*.{tag}.jsonl")):
            a = Path(f).name.split(".")[0]; papers += 1; seen = set()
            for l in open(f):
                r = json.loads(l)
                if "i" not in r or r.get("category") not in POS or r["i"] in seen:
                    continue
                seen.add(r["i"])
                cons = (r.get("constructs") or [None])[0]
                rows.append(judge(claims, phrases, a, int(r["i"]), cons))
        print(f"  {tag:20s} {papers} papers: " + tally(rows))


if __name__ == "__main__":
    main()
