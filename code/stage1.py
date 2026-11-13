"""Stage 1: the sentences of a paper that may mention its searched task, from what a search gives (no normalized
task records). A unit is kept when it, or the unit before it, names a target task (search terms or the abbreviations
the paper defines for them, offtarget.Checker), or when it uses a generic task word (task, paradigm, condition,
trials, contrast, ...), a claim verb (methods.CUE: measures, reflects, engages, requires, ...) or a word about
performance or stimuli (accuracy, reaction time, errors, targets, scores, ...). Back matter and units under 25
characters are dropped. On the gold: keeps 55% of units and 89% of claims (the misses are interpretive sentences
in condition vocabulary, such as "cheater detection is only activated by ...").

usage: python stage1.py   (run from cc/; recall of the adjudicated gold claims when targets are the search tasks)
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

from methods import CUE
from offtarget import POS, Checker, mentions

ROOT = Path(__file__).parent
GENERIC = re.compile(r"\b(task|paradigm|test|game|procedure|condition|trials?|contrast|block|stimul\w*|cue|probe|"
                     r"versus|vs\.?|minus)s?\b|>", re.I)
PERF = re.compile(r"\b(trials?|conditions?|blocks?|stimul\w*|cues?|targets?|responses?|performance|accuracy|"
                  r"reaction times?|RTs?|errors?|effect|scores?|measures?|participants|subjects)\b", re.I)


def keep(units, targets, rxs, ments=None):
    """indices i of units kept by stage 1; units = [{"i", "text", "section"}] (ments: their mentions(), if known)"""
    named = [bool((ments[k] if ments is not None else mentions(u["text"], rxs)) & targets) for k, u in enumerate(units)]
    out = []
    for k, u in enumerate(units):
        if u["section"] == "back" or len(u["text"]) < 25:
            continue
        t = u["text"]
        if named[k] or (k and named[k - 1]) or GENERIC.search(t) or CUE.search(t) or PERF.search(t):
            out.append(u["i"])
    return out


def main():
    ck = Checker()
    search = json.load(open(ROOT / "pilot" / "search_tasks.json"))
    c = Counter()
    for f in sorted(glob.glob(str(ROOT / "pilot" / "papers" / "*.json"))):
        p = json.load(open(f)); a = p["article"]
        gold = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(ROOT / "labels" / f"{a}.ADJ2.jsonl")}
        targets = set(search.get(a) or ({t["canonical"] for t in p["tasks"] if t["canonical"]} |
                                        {p["pilot_task"]["canonical"]}))
        U = p["units"]
        rxs = ck.paper(" ".join(u["text"] for u in U))
        kept = set(keep(U, targets, rxs))
        for u in U:
            if u["section"] == "back" or len(u["text"]) < 25:
                continue
            y = gold.get(u["i"]) in POS
            c["units"] += 1; c["claims"] += y; c["kept"] += u["i"] in kept; c["kept claims"] += y and u["i"] in kept
    print(f"stage 1 keeps {c['kept']}/{c['units']} = {c['kept'] / c['units']:.0%} of units and "
          f"{c['kept claims']}/{c['claims']} = {c['kept claims'] / c['claims']:.0%} of gold claims")


if __name__ == "__main__":
    main()
