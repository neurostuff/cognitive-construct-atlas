"""Off-target task check: does a sentence talk about a task other than the one(s) the paper was searched for?

Task mentions come from the search terms in queries.jsonl (the names the task searches used; the inventory's regex
patterns are left out, since they match sibling tasks' construct words such as "dual-task" or "extinction"), plus abbreviations the paper itself defines for them ("stop-signal task (SST)"). Category
tasks that were not searched by name (make_queries.NOT_BY_NAME: "response inhibition task", "interference task")
are left out: their names are construct words. Questionnaires and rating scales count as other "tasks", since a claim
that the BIS/BAS measures reward sensitivity is not a claim about the searched task.

A sentence is off-target when it names another task (or a questionnaire) and does not name a target task. A sentence
that names neither inherits the verdict of the previous sentence when it opens with a pointer back to it ("This
task", "These paradigms", "In such tasks", "It"), and is otherwise on target.

usage: python offtarget.py   (run from cc/; scores the check against the adjudicated gold, ADJ2)
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
POS = {"own_task", "own_contrast"}
QUESTIONNAIRE = re.compile(r"\b(questionnaires?|inventory|inventories|self-reports?|rating scales?|subscales?|"
                           r"(?<!visual analog )(?<!analogue )scales?\b(?! (of|for) (the )?(task|stimul))|survey)\b", re.I)
BACK = re.compile(r"^\s*(this|these|those|such|it|they|both|either|each|the latter|the former|in (this|these|such|both)|"
                  r"(in|with|for|during) (this|these|such|both|each|either|the latter|the former))\b", re.I)
ABBR = r"\s*\(\s*([A-Z][A-Za-z0-9-]{1,9})\s*[;,)]"


def task_regexes(inventory, queries):
    """task id -> one compiled regex over its search terms (searched tasks only)"""
    terms = {q["canonical"]: q["terms"] for q in queries if q.get("terms")}
    out = {}
    for t in inventory["tasks"]:
        if t["id"] not in terms:
            continue
        alts = [re.escape(a).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+") for a in terms[t["id"]]]
        if alts:
            try:
                out[t["id"]] = re.compile(r"\b(" + "|".join(alts) + r")", re.I)
            except re.error:
                out[t["id"]] = re.compile(r"\b(" + "|".join(a for a in alts if _ok(a)) + r")", re.I)
    return out


def _ok(p):
    try:
        re.compile(p)
        return True
    except re.error:
        return False


def paper_regexes(text, regexes, spans=None):
    """add the abbreviations this paper defines right after a task mention, e.g. 'stop-signal task (SST)'
    (spans: a dict to fill with each task's (start, end) matches in text)"""
    out = {}
    for tid, rx in regexes.items():
        abbrs = set()
        for m in rx.finditer(text):
            if spans is not None:
                spans.setdefault(tid, []).append((m.start(), m.end()))
            tail = text[m.end():m.end() + 40]
            a = re.match(r"[\w-]*(?:\s+(?:task|paradigm|test|game))?" + ABBR, tail, re.I)
            if a and a.group(1).isupper() or (a and sum(c.isupper() for c in a.group(1)) >= 2):
                abbrs.add(a.group(1))
        out[tid] = (rx, re.compile(r"\b(" + "|".join(map(re.escape, abbrs)) + r")\b") if abbrs else None)
    # one combined pattern per kind, so a sentence that names no task is ruled out with two searches
    abbr = [ab.pattern for _, ab in out.values() if ab is not None]
    out[ANY] = (re.compile("|".join(f"(?:{rx.pattern})" for rx in regexes.values()), re.I),
                re.compile("|".join(f"(?:{p})" for p in abbr)) if abbr else None)
    return out


ANY = "__any__"


def mentions(s, rxs):
    if ANY in rxs:
        rx, ab = rxs[ANY]
        if not rx.search(s) and not (ab and ab.search(s)):
            return set()
    return {tid for tid, (rx, ab) in rxs.items() if tid != ANY and (rx.search(s) or (ab and ab.search(s)))}


class Checker:
    def __init__(self, inventory_path=ROOT / "task_inventory_v3.json", queries_path=ROOT / "queries.jsonl"):
        inventory = json.load(open(inventory_path))
        self.regexes = task_regexes(inventory, [json.loads(l) for l in open(queries_path)])
        self.names = {t["id"]: t["name"] for t in inventory["tasks"] if t["id"] in self.regexes}

    def paper(self, full_text):
        return paper_regexes(full_text, self.regexes)

    def unit_mentions(self, texts):
        """(rxs, [mentions(t, rxs) for t in texts]) with each pattern run once over the joined text: a match counts
        for the unit it lies inside; a match across two units counts for neither, as mentions() would not see it"""
        import bisect
        full = " ".join(texts)
        found = {}
        rxs = paper_regexes(full, self.regexes, found)
        starts, pos = [], 0
        for t in texts:
            starts.append(pos); pos += len(t) + 1
        ments = [set() for _ in texts]
        for tid, (rx, ab) in rxs.items():
            if tid == ANY:
                continue
            hits = found.get(tid, []) + ([(m.start(), m.end()) for m in ab.finditer(full)] if ab is not None else [])
            for s, e in hits:
                k = bisect.bisect_right(starts, s) - 1
                if e <= starts[k] + len(texts[k]):
                    ments[k].add(tid)
        return rxs, ments

    def verdicts(self, sentences, targets, rxs, ments=None):
        """one of 'target', 'other', 'neither' per sentence, with neither resolved through BACK pointers
        (ments: the sentences' mentions(), when already computed)"""
        out, prev = [], "neither"
        for k, s in enumerate(sentences):
            m = ments[k] if ments is not None else mentions(s, rxs)
            tgt, oth = bool(m & targets), bool(m - targets) or bool(QUESTIONNAIRE.search(s))
            v = "target" if tgt else "other" if oth else (prev if BACK.match(s) and prev != "neither" else "neither")
            out.append(v); prev = v
        return out


def main():
    ck = Checker()
    clf = {(r["article"], r["i"]): r["p"] for r in map(json.loads, open(ROOT / "preds" / "clf_biomedbert.jsonl"))}
    search = json.load(open(ROOT / "pilot" / "search_tasks.json"))  # gold paper -> tasks whose search found it
    c = Counter()
    for f in sorted(glob.glob(str(ROOT / "pilot" / "papers" / "*.json"))):
        p = json.load(open(f)); a = p["article"]
        gold = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(ROOT / "labels" / f"{a}.ADJ2.jsonl")}
        targets = set(search.get(a) or ({t["canonical"] for t in p["tasks"] if t["canonical"]} |
                                        {p["pilot_task"]["canonical"]}))
        units = p["units"]
        rxs = ck.paper(" ".join(u["text"] for u in units))
        v = ck.verdicts([u["text"] for u in units], targets, rxs)
        for u, verdict in zip(units, v):
            key = (a, u["i"])
            if key not in clf:
                continue
            g = gold.get(u["i"], "none"); g = "claim" if g in POS else g
            c[("all", g, verdict)] += 1
            if clf[key] >= 0.99:
                c[("p>=0.99", g, verdict)] += 1
    for scope in ("all", "p>=0.99"):
        print(f"\n{scope}: gold category -> verdict (off-target = 'other')")
        for g in ("claim", "other_task", "region", "label_usage", "none"):
            row = {v: c[(scope, g, v)] for v in ("target", "neither", "other")}
            n = sum(row.values())
            if n:
                print(f"  {g:12s} n {n:5d}  target {row['target']:5d}  neither {row['neither']:5d}  "
                      f"other {row['other']:4d} ({row['other'] / n:.0%})")


if __name__ == "__main__":
    main()
