"""Score extractors on what James needs (2026-11-10): asserted claim sentences, their level, their contrast, their
construct.

Gold: labeller A's own_task/own_contrast units that the adjudicated gold (ADJ2) keeps as claims, with polarity
"asserted". They carry A's level, contrast_text and constructs.
Per method, over a set of papers it covers:
- recall: share of gold asserted claims the method lists as a claim (any polarity), and as an asserted claim;
- precision: share of the method's asserted claims that are gold claims (ADJ2, any polarity), and gold asserted claims;
- level: own_task vs own_contrast agrees, on found gold asserted claims;
- contrast: on found gold own_contrast claims with a contrast_text, the method's contrast_text shares >= 50% of its
  content words (token Jaccard), or one contains the other;
- construct: on found claims, some predicted construct overlaps some gold construct the same way.

usage: python score_asserted.py [papers=common|all]   (run from cc/)
"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
POS = {"own_task", "own_contrast"}
STOP = set("the a an of and or in on for to with by vs versus task tasks condition conditions trial trials contrast "
           "effect effects activation activity during between".split())


def toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP}


def match(a, b):
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= 0.5 or ta <= tb or tb <= ta


def gold():
    out = {}
    for f in glob.glob(str(ROOT / "labels" / "*.A.jsonl")):
        a = Path(f).name.split(".")[0]
        adj = {json.loads(l)["i"]: json.loads(l) for l in open(ROOT / "labels" / f"{a}.ADJ2.jsonl") if l.strip()}
        A = {json.loads(l)["i"]: json.loads(l) for l in open(f) if l.strip()}
        out[a] = {"asserted": {i: r for i, r in A.items() if r["category"] in POS and r.get("polarity") == "asserted"
                               and adj.get(i, {}).get("category") in POS},
                  "claims": {i for i, r in adj.items() if r["category"] in POS}}
    return out


def method(tag, arts):
    out = {}
    for a in arts:
        f = ROOT / "llm" / f"{a}.{tag}.jsonl"
        if not f.exists() or not f.read_text().strip():
            return None
        d = {}
        for l in f.read_text().splitlines():
            r = json.loads(l)
            if "i" not in r or r.get("category") not in POS:
                continue
            cur = d.setdefault(int(r["i"]), {"category": r["category"], "polarity": r.get("polarity"),
                                             "contrast": [], "constructs": []})
            cur["contrast"].append(r.get("contrast_text") or "")
            cur["constructs"] += list(r.get("constructs") or []) + list(r.get("construct_terms") or [])
        out[a] = d
    return out


def score(G, M, arts):
    c = dict(gold=0, found=0, found_asserted=0, pred_asserted=0, pa_claim=0, pa_asserted=0, lvl=0, lvl_ok=0,
             con=0, con_ok=0, cons=0, cons_ok=0)
    for a in arts:
        g, m = G[a], M[a]
        for i, r in g["asserted"].items():
            c["gold"] += 1
            if i not in m:
                continue
            p = m[i]; c["found"] += 1; c["found_asserted"] += p["polarity"] == "asserted"
            c["lvl"] += 1; c["lvl_ok"] += p["category"] == r["category"]
            if r["category"] == "own_contrast" and r.get("contrast_text"):
                c["con"] += 1; c["con_ok"] += any(match(r["contrast_text"], x) for x in p["contrast"])
            if r.get("constructs"):
                c["cons"] += 1; c["cons_ok"] += any(match(gc, pc) for gc in r["constructs"] for pc in p["constructs"])
        for i, p in m.items():
            if p["polarity"] == "asserted":
                c["pred_asserted"] += 1; c["pa_claim"] += i in g["claims"]; c["pa_asserted"] += i in g["asserted"]
    f = lambda n, d: c[n] / c[d] if c[d] else float("nan")
    return {"gold asserted": c["gold"], "recall": f("found", "gold"), "recall as asserted": f("found_asserted", "gold"),
            "asserted preds": c["pred_asserted"], "P (is a claim)": f("pa_claim", "pred_asserted"),
            "P (is asserted claim)": f("pa_asserted", "pred_asserted"), "level": f("lvl_ok", "lvl"),
            "contrast": f("con_ok", "con"), "contrast n": c["con"], "construct": f("cons_ok", "cons")}


def main():
    G = gold()
    arts = sorted(G)
    tags = ["codex-astra-linked", "codex-terra-linked", "codex-luna-linked", "sonnetv2", "sonnetguide", "haikuguide",
            "qwen4b"]
    have = {t: {Path(f).name.split(".")[0] for f in glob.glob(str(ROOT / "llm" / f"*.{t}.jsonl"))} for t in tags}
    common = [a for a in arts if all(a in have[t] for t in ("codex-astra-linked", "sonnetv2"))]
    sets = {"common": common, "all": arts}
    which = sys.argv[1] if len(sys.argv) > 1 else "common"
    sub = sets[which]
    print(f"papers: {len(sub)} ({which}); gold asserted claims {sum(len(G[a]['asserted']) for a in sub)}")
    hdr = ["recall", "recall as asserted", "asserted preds", "P (is a claim)", "P (is asserted claim)", "level",
           "contrast", "construct"]
    print(f"{'method':20s} " + " ".join(f"{h[:16]:>16s}" for h in hdr))
    for t in tags:
        papers = [a for a in sub if a in have[t]]
        if len(papers) < len(sub):
            continue
        M = method(t, papers)
        if M is None:
            continue
        s = score(G, M, papers)
        print(f"{t:20s} " + " ".join(f"{s[h]:16.2f}" if isinstance(s[h], float) else f"{s[h]:16d}" for h in hdr))
    print("contrast is scored on", score(G, method("sonnetv2", sub), sub)["contrast n"], "gold contrast claims with wording")


if __name__ == "__main__":
    main()
