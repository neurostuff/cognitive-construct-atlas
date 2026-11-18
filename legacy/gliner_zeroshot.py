"""Zero-shot GLiNER tests on the pilot gold (CPU).

decide: fastino/GLiNER2.5-Decide labels every non-back unit (>= 25 chars) construct_claim vs other; the score is the
        construct_claim probability. Two inputs: the sentence alone, and "Task: <the paper's task names>. Sentence: ...".
        Scored against the ADJ2 gold (own_task/own_contrast = positive).
spans:  a GLiNER model extracts "psychological construct" spans (with a separate task/condition label to absorb task
        names) from each gold claim sentence of labeller A; a gold phrase is found when a predicted span overlaps it
        (token Jaccard >= 0.5), a predicted span is correct when it overlaps a gold phrase of its sentence.

usage: python -I gliner_zeroshot.py decide OUT.jsonl | spans MODEL_ID OUT.jsonl   (run from pilot-results/)
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
POS = {"own_task", "own_contrast"}
CLAIM = {"construct_claim": "the sentence says what psychological process or ability a task, condition, contrast or "
                            "behavioural measure assesses, isolates, engages, requires or reflects",
         "other": "procedure, stimuli, results, brain regions, or anything else"}
ENTITIES = {"psychological construct": "a cognitive, affective, motivational or social process or ability that a task "
                                       "measures, e.g. working memory, inhibitory control, reward anticipation, "
                                       "theory of mind, conflict monitoring",
            "experimental task or condition": "the name of a task, paradigm, condition, trial type or contrast, "
                                              "e.g. Stroop task, 2-back, NoGo trials, incongruent > congruent"}


def papers():
    for f in sorted((ROOT / "pilot" / "papers").glob("*.json")):
        yield json.loads(f.read_text())


def tokens(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def overlap(a, b):
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / max(1, len(ta | tb)) >= 0.5


def decide(out):
    from gliner2 import AutoExtractor
    m = AutoExtractor.from_pretrained("fastino/GLiNER2.5-Decide")
    t0, n = time.time(), 0
    with open(out, "w") as f:
        for p in papers():
            gold = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(ROOT / "labels" / f"{p['article']}.ADJ2.jsonl")}
            names = "; ".join(t["name"] or "" for t in p["tasks"])
            for u in p["units"]:
                if u["section"] == "back" or len(u["text"]) < 25:
                    continue
                row = {"article": p["article"], "i": u["i"], "y": int(gold.get(u["i"]) in POS)}
                for k, text in (("plain", u["text"]), ("task", f"Task: {names}. Sentence: {u['text']}")):
                    r = m.classify_text(text[:2000], {"claim": {"labels": CLAIM}}, include_confidence=True)["claim"]
                    row[k] = round(r["confidence"] if r["label"] == "construct_claim" else 1 - r["confidence"], 4)
                f.write(json.dumps(row) + "\n"); n += 1
            print(f"{p['article']} done, {n} units, {time.time() - t0:.0f}s", flush=True)


def spans(model_id, out):
    from gliner2 import AutoExtractor
    m = AutoExtractor.from_pretrained(model_id)
    with open(out, "w") as f:
        for p in papers():
            for line in open(ROOT / "labels" / f"{p['article']}.A.jsonl"):
                r = json.loads(line)
                if r["category"] not in POS or not r.get("constructs"):
                    continue
                s = p["units"][r["i"]]["text"]
                ents = m.extract_entities(s[:2000], ENTITIES, threshold=0.3, include_confidence=True)["entities"]
                pred = [(e["text"], round(e["confidence"], 3)) for e in ents.get("psychological construct", [])]
                f.write(json.dumps({"article": p["article"], "i": r["i"], "gold": r["constructs"], "pred": pred}) + "\n")
    rows = [json.loads(l) for l in open(out)]
    for t in (0.3, 0.5, 0.7, 0.9):
        found = sum(any(overlap(g, ps) for ps, c in r["pred"] if c >= t) for r in rows for g in r["gold"])
        ngold = sum(len(r["gold"]) for r in rows)
        preds = [(ps, r) for r in rows for ps, c in r["pred"] if c >= t]
        right = sum(any(overlap(g, ps) for g in r["gold"]) for ps, r in preds)
        print(json.dumps({"model": model_id, "threshold": t, "gold_found": f"{found}/{ngold} = {found / ngold:.2f}",
                          "pred_correct": f"{right}/{len(preds)} = {right / max(1, len(preds)):.2f}"}))


if __name__ == "__main__":
    if sys.argv[1] == "decide":
        decide(sys.argv[2])
    else:
        spans(sys.argv[2], sys.argv[3])
