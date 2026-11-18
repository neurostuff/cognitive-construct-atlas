"""Construct evidence for a task's contrasts from every sentence of that task's papers, not only comparison syntax.

scan_contrasts.py reads only sentences with explicit comparison syntax (vs, compared to, minus, >): in 300 Stroop
papers that is 5% of sentences, and 87% of the sentences that name a Stroop condition and a construct have none.
This search reads every sentence of a paper, but only for the conditions of the paper's own tasks (`targets`: the
task searches that found it, or its ns-pond tasks), so "congruent" in a Simon paper is never a Stroop condition.

1. Scope: a paper is searched for its target tasks' conditions only.
2. Conditions per task: the side names the task's analyses use (contrasts_all.json) and the lexicon's strings for its
   family, read to atoms by hub.side_from_text, kept when the atom occurs in one of the task's hub entries. A generic
   one-word name counts only before a condition noun (trials, stimuli, condition, ...) or beside another value on its
   axis; a name that is itself a construct term (conflict, interference) is not a condition.
3. Sentences: any sentence of the main sections that names a condition of the task and a taxonomy construct outside
   the condition spans; or a sentence that starts with "this effect / difference / contrast ..." and names a
   construct, read with the sentence before it that names the conditions.
4. Entries: two values on one axis -> the task's entries comparing them (either direction: the stance label decides
   which; the two the sample uses most); one value -> the task's most used entry with it as target against another
   condition (else as target, else as baseline).
5. Guards: a sentence that names another task (inventory name, alias or pattern) is skipped; a construct term that
   shares a word with the task's name or the entry's axes and values is dropped (valence on a valence axis, fear in
   fear conditioning).

Output: OUT_items.jsonl in stance.py's item shape (id, article, i, A, B, before, text, constructs; plus task, hub,
kind, construct_ids), and OUT.json with counts per task and the sample's reach.

usage: (code/contrast_hub on beast) python condition_search.py HUB_DIR LEXICON CONTRASTS.json SAMPLE.json OUT.json [WORKERS] [TASKS]
"""
import collections
import json
import os
import re
import sys
import multiprocessing as mp
from pathlib import Path

import hub as H
from constructs import directed, vocab_maps

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = {"results", "methods", "discussion", "introduction", "abstract"}
MAX_COND, MAX_TERM = 4, 6
COND_NOUNS = set("trial trials condition conditions stimuli stimulus items item block blocks words pictures images faces "
                 "cues cue targets distractors sentences scenes events responses outcomes feedback runs".split())
ANAPHOR = re.compile(r"^\s*(?:this|these|such|that|the(?: observed| present| same)?)\s+(?:\w+\s+)?"
                     r"(?:effect|difference|contrast|comparison)s?\b", re.I)
#: everyday words that are also condition values: they count only before a condition noun, even beside another value
AMBIG = set("high low fast slow consistent inconsistent same different positive negative neutral emotional social "
            "individual auditory visual word words conflict large small short long early late new old first second "
            "correct incorrect easy hard other self control target standard".split())
#: construct terms that are methods vocabulary in these papers
METHOD_TERMS = set("manipulation manipulations resources resource knowledge representation representations selection "
                   "performance processing cognition behavior behaviour feedback".split())
STOP = set("of and or the a an in on for to with by from as at is are was were be than that this these which".split())
MAX_PAIR = 2
#: a table row read as a sentence: tabs, or mostly numbers
TABLE = re.compile(r"\t|(?:\d[\d.,()±%-]*\s+){4,}")
G = {}


def comps_of(e):
    return [tuple(c) for c in json.loads(e["sig_json"])[1]]


def build_dicts(hubj, lex, C, assign, sample, terms, only=None):
    """per task: condition dictionary (folded string -> atoms), entries by atom, a word block-list for constructs"""
    inv = {t["id"]: t for t in H.INVENTORY["tasks"]}
    by_task = collections.defaultdict(dict)
    for k, e in hubj["contrasts"].items():
        if e.get("scope") in inv and (only is None or e["scope"] in only):
            by_task[e["scope"]][k] = comps_of(e)
    use = collections.Counter()
    for aid in sample:
        a = assign.get(aid)
        if a:
            use[a["hub"]] += 1
    names = collections.defaultdict(set)
    for c in C.values():
        for t in (c.get("tasks") or "").split(","):
            if t in by_task:
                for s in (c.get("pos") or "").split(" | ") + (c.get("neg") or "").split(" | "):
                    s = H.fold(s)
                    if s and 3 <= len(s) and len(s.split()) <= MAX_COND:
                        names[t].add(s)
    T = {}
    for t, ents in by_task.items():
        fam = H.TASK_FAMILY.get(t)
        atoms = {(ax, v) for cs in ents.values() for ax, a, b in cs for v in (a, b) if v and v not in H.BASES and v != H.TASK}
        strings = names[t] | {s for s in (lex.m.get(fam) or {}) if len(s.split()) <= MAX_COND and len(s) >= 3}
        cond = {}
        for s in strings:
            if s in terms:                              # "conflict", "interference": a construct word, not a condition
                continue
            sd = H.side_from_text(s, t, fam, lex)
            hit = frozenset((ax, v) for ax, v in (sd.get("atoms") or {}).items() if (ax, v) in atoms)
            if hit:
                cond[s] = hit
        by_atom = collections.defaultdict(list)
        for k, cs in ents.items():
            for ax, a, b in cs:
                if a and a not in H.BASES:
                    by_atom[(ax, a)].append((k, "A", b is not None and b not in H.BASES and b != H.ABSENT))
                if b and b not in H.BASES:
                    by_atom[(ax, b)].append((k, "B", a is not None and a not in H.BASES and a != H.ABSENT))
        name = inv[t]
        block = set(H.fold(" ".join([name["name"], *(name.get("aliases") or [])])).split())
        block |= {w for (ax, v) in atoms for w in H.fold(f"{ax} {v}").split()}
        T[t] = {"cond": cond, "first": {s.split()[0] for s in cond}, "ents": ents, "by_atom": dict(by_atom),
                "block": block, "use": {k: use[k] for k in ents}}
    pats = {}
    for t, x in inv.items():
        terms = [re.escape(x["name"])] + [re.escape(a) for a in x.get("aliases") or []] + list(x.get("patterns") or [])
        terms = [p for p in terms if p and len(p) >= 4]
        if terms:
            try:
                pats[t] = re.compile(r"\b(?:" + "|".join(terms) + r")", re.I)
            except re.error:
                pats[t] = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in terms) + r")", re.I)
    return T, pats


def build_terms():
    to_ids, _, _ = vocab_maps()
    s2e = json.load(open(ROOT / "vocab" / "string_to_entries.json"))
    terms = {}
    for s in s2e:
        k = H.fold(s)
        w = k.split()
        if k and len(k) >= 4 and len(w) <= MAX_TERM and w[0] not in STOP and w[-1] not in STOP:
            ids = to_ids([s])
            if ids:
                terms.setdefault(k, set()).update(ids)
    return terms, {k.split()[0] for k in terms}


def find(words, dic, first, maxn):
    """longest matches, left to right: [(start, end, key)]"""
    out = []; i = 0
    while i < len(words):
        hit = None
        if words[i] in first:
            for n in range(min(maxn, len(words) - i), 0, -1):
                k = " ".join(words[i:i + n])
                if k in dic:
                    hit = (i, i + n, k); break
        if hit:
            out.append(hit); i = hit[1]
        else:
            i += 1
    return out


def conditions(words, td):
    """condition mentions; a one-word name counts only before a condition noun ("incongruent trials") or beside
    another value on its axis ("congruent and incongruent")"""
    ms = find(words, td["cond"], td["first"], MAX_COND)
    vals = collections.defaultdict(set)
    for s, e, k in ms:
        for ax, v in td["cond"][k]:
            vals[ax].add(v)
    keep = []
    for s, e, k in ms:
        noun = e < len(words) and words[e] in COND_NOUNS
        if e - s == 1 and not noun and (k in AMBIG or not any(len(vals[ax]) > 1 for ax, _ in td["cond"][k])):
            continue
        keep.append((s, e, k))
    return keep


def constructs(words, used, td):
    out = {}
    for s, e, k in find(words, G["terms"], G["tfirst"], MAX_TERM):
        if any(s < ue and us < e for us, ue in used):
            continue                                 # inside a condition name
        if set(k.split()) & td["block"] or k in METHOD_TERMS:
            continue                                 # names the task or the entry's own axis; methods vocabulary
        out[k] = sorted(G["terms"][k])
    return out


def side_text(comps, side, spans):
    out = []
    for ax, a, b in comps:
        v = a if side == "A" else b
        if v is None or v == H.ABSENT:
            continue
        out.append({H.REST: "rest / low-level baseline", H.CONTROL: "control condition"}.get(v) or spans.get((ax, v)) or v.replace("_", " "))
    return " and ".join(out)


def entries_for(ms, td):
    """(kind, [entry keys]) for the matched condition atoms"""
    atoms = {}
    for s, e, k in ms:
        for at in td["cond"][k]:
            atoms.setdefault(at, k)
    by_ax = collections.defaultdict(set)
    for ax, v in atoms:
        by_ax[ax].add(v)
    pair = set()                                     # every component's two values named
    for k, cs in td["ents"].items():
        if all(a in by_ax.get(ax, ()) and b in by_ax.get(ax, ()) and a != b for ax, a, b in cs):
            pair.add(k)
    if pair:
        return "pair", sorted(pair, key=lambda k: -td["use"].get(k, 0))[:MAX_PAIR], atoms
    # one condition: the task's most used entry with it as target against another condition, else against
    # anything, else with it as baseline
    rank = lambda x: (-td["use"].get(x[0], 0), len(td["ents"][x[0]]))  # noqa: E731
    opts = [x for at in atoms for x in td["by_atom"].get(at, ())]
    for want in (lambda x: x[1] == "A" and x[2], lambda x: x[1] == "A", lambda x: True):
        hit = sorted((x for x in opts if want(x)), key=rank)
        if hit:
            return "single", [hit[0][0]], atoms
    return "single", [], atoms


def scan_paper(path):
    try:
        d = json.load(open(path))
    except Exception:  # noqa: BLE001
        return [], collections.Counter()
    tasks = [t for t in d.get("targets") or [] if t in G["T"]]
    n = collections.Counter()
    if not tasks:
        return [], n
    n["papers"] += 1; src = "imaging" if d["article"].startswith("ns_") else "lit"
    units = [u for u in d["units"] if u.get("section") in SECTIONS and not TABLE.search(u["text"])]
    out = []; seen = set()
    prev = None
    for j, u in enumerate(units):
        n["sentences"] += 1
        words = H.fold(u["text"]).split()
        for t in tasks:
            td = G["T"][t]
            ms = conditions(words, td)
            kind_override = None; text = u["text"]; before = units[j - 1]["text"] if j else ""
            if not ms and prev is not None and ANAPHOR.match(u["text"]):
                pw = H.fold(prev["text"]).split(); pms = conditions(pw, td)
                if pms:
                    cs = constructs(words, [], td)
                    if cs:
                        ms, kind_override = pms, "anaphor"; text = prev["text"] + " " + u["text"]
                        before = units[j - 2]["text"] if j > 1 else ""
                        words_for_spans = pw
            if not ms:
                continue
            n[f"{t}|cond"] += 1
            if kind_override:
                cs = constructs(words, [], td)
            else:
                cs = constructs(words, [(s, e) for s, e, _ in ms], td); words_for_spans = words
            if not cs:
                continue
            others = [o for o, rx in G["pats"].items() if o not in tasks and rx.search(text)]
            if others:
                n["other_task"] += 1; continue
            kind, ks, atoms = entries_for(ms, td)
            if not ks:
                n["no_entry"] += 1; continue
            spans = {at: k for at, k in atoms.items()}
            for k in ks:
                key = (u.get("i", j), k)
                if key in seen:
                    continue
                seen.add(key)
                comps = td["ents"][k]
                out.append({"id": f"{d['article']}:{u.get('i', j)}|{k}", "article": d["article"], "i": u.get("i", j),
                            "section": u.get("section"), "src": src, "task": t, "hub": k, "kind": kind_override or kind,
                            "A": side_text(comps, "A", spans), "B": side_text(comps, "B", spans), "before": before[-300:],
                            "text": text, "constructs": list(cs)[:4], "construct_ids": cs})
                n[f"{t}|items"] += 1; n[f"kind_{kind_override or kind}"] += 1
        prev = u
    return out, n


def main():
    hub_dir, lexp, contrastsp, samplep, outp = sys.argv[1:6]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 16
    only = set(sys.argv[7].split(",")) if len(sys.argv) > 7 else None
    hubj = json.load(open(f"{hub_dir}/hub.json")); lex = H.Lexicon(lexp)
    C = json.load(open(contrastsp)); assign = json.load(open(f"{hub_dir}/corpus_assign.json")); sample = json.load(open(samplep))
    G["terms"], G["tfirst"] = build_terms()
    G["T"], G["pats"] = build_dicts(hubj, lex, C, assign, sample, G["terms"], only)
    print(f"tasks {len(G['T'])}, condition strings {sum(len(x['cond']) for x in G['T'].values())}, construct terms {len(G['terms'])}", flush=True)
    papers = [ROOT / d / f for d in ("corpus_papers", "imaging_papers") for f in os.listdir(ROOT / d)]
    n = collections.Counter(); per_entry = collections.defaultdict(set); per_c = collections.defaultdict(lambda: collections.defaultdict(set))
    with mp.get_context("fork").Pool(workers) as pool, open(outp.replace(".json", "_items.jsonl"), "w") as fo:
        for k, (out, c) in enumerate(pool.imap_unordered(scan_paper, papers, chunksize=64)):
            n.update(c)
            for r in out:
                fo.write(json.dumps(r) + "\n")
                per_entry[r["hub"]].add(r["article"])
                for ids in r["construct_ids"].values():
                    for cid in ids:
                        per_c[r["hub"]][cid].add(r["article"])
            if k % 10000 == 0:
                print(k, len(papers), n["papers"], sum(v for x, v in n.items() if x.endswith("|items")), flush=True)
    sk = [a for a in sample if a in assign]
    und = {a: assign[a]["hub"] for a in sk}       # the stance label decides the direction; reach is per entry here
    pooled = {k for k, d in per_c.items() if any(len(a) >= 2 for a in d.values())}
    tasks = collections.defaultdict(dict)
    for x, v in n.items():
        if "|" in x:
            t, what = x.split("|"); tasks[t][what] = v
    R = {"papers": n["papers"], "sentences": n["sentences"], "items": sum(v for x, v in n.items() if x.endswith("|items")),
         "kinds": {x: v for x, v in n.items() if x.startswith("kind_")}, "other_task_skipped": n["other_task"],
         "no_entry": n["no_entry"], "entries_with_items": len(per_entry),
         "entries_with_2plus_articles": sum(len(v) >= 2 for v in per_entry.values()), "entries_with_pooled_construct": len(pooled),
         "sample_mapped": len(sk), "sample_entry_with_items": sum(und[a] in per_entry for a in sk),
         "sample_entry_with_pooled_construct": sum(und[a] in pooled for a in sk)}
    print(json.dumps(R, indent=1))
    json.dump({"summary": R, "tasks": tasks, "articles": {k: len(v) for k, v in per_entry.items()}}, open(outp, "w"), indent=1)


if __name__ == "__main__":
    main()
