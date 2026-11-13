"""Judge: is a contrast no other paper repeats the same comparison as an existing entry of its task, or a new contrast?

Items: corpus condition contrasts whose exact signature is not established (not L0 / L0a / L0h / L0m) and whose task
has established entries. Candidates per item (at most MAX_CAND): established entries of the same task with the same
axes, a shared two-valued component, or a subset of its components, most papers first; each shown with up to two
example side-name pairs from its own contrasts. Codex (no tools, batches by task) sees the item's side names, its
paper's condition descriptions and abbreviations (that paper's only), and the candidates, and answers per item:
`same` (candidate id and whether the item's target > baseline runs the candidate's way or reversed) or `new`, with a
confidence. hub.py build reads the answers: a high-confidence `same` puts the contrast in that entry (level L0j,
reported on its own line); `new` keeps it as its own entry, marked judged.
WIDE=1 judges the contrasts left without candidates: any established entry of the task (at most MAX_CAND_WIDE), most
shared atoms first, then shared axes, then papers, since a misread item has the wrong axes.

usage: [WIDE=1] [REJUDGE=same] [SAMPLE=CONTRASTS.json] python judge.py HUB_DIR CONTRASTS.json GLOSSARY.json ABBREV.json OUT.json [MODEL EFFORT]
"""
import collections
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import codex_extract as cx  # noqa: E402

import hub as H  # noqa: E402

MAX_CAND, MAX_CAND_WIDE, BATCH = 6, 12, 30
WIDE = os.environ.get("WIDE") == "1"
REJUDGE = os.environ.get("REJUDGE") == "same"    # re-judge the earlier "same" rulings and reversed items (after a prompt change)
#: SAMPLE=CONTRASTS.json judges only the contrasts in that file (the brain-map sample) when the hub is built from more
SAMPLE = set(json.load(open(os.environ["SAMPLE"]))) if os.environ.get("SAMPLE") else None
TASK_NAME = {t["id"]: t["name"] for t in H.INVENTORY["tasks"]}

RULES = """You decide whether a contrast from a neuroimaging paper is the same comparison as an existing contrast of the
same task, or a new one. Contrasts are written `task|axis:target>baseline` (`@rest` = rest or fixation, `@control` = a
matched control, `∅` = no value on that side, no `>` = a simple effect). Each candidate comes with example condition
names from other papers.

Same means the two sides compare the same conditions of this task, as researchers would pool them in a meta-analysis
(different wording or abbreviations count as same). Different levels of the manipulation (3-back vs 1-back against
2-back vs 0-back), a different baseline, or a different condition are new. Use the paper's condition descriptions and
abbreviations to read the item.
- Match on the candidate's signature, not its examples: if the item's manipulated factor differs from the signature's
  axis or values, answer new even when an example looks like the item.
- A qualifier is incidental only when the paper does not manipulate it. Experimental phase (acquisition, extinction,
  reinstatement; encoding, retrieval), study or test modality, trial role (target, standard, novel) and memory type
  (source vs item) are never incidental.
- `novelty:new>old` is only study or repetition history with both sides in the same trial role: novel vs target is not
  novelty, and the familiarity of a meaning is not novelty. A reward or loss cue is incentive, never valence.
- Classify the item's baseline from its description: `@rest` is fixation, blank, no stimulus, an off period, or a
  low-level display with a random or no response; `@control` is a task with matched stimuli or responses (shape
  matching, tones with tapping, reversed speech, a parity decision). If the candidate's baseline class differs, answer
  new.

Answer per item: decision "same" or "new"; for same, the candidate id and direction "same" (the item's target side
matches the candidate's target) or "reversed"; confidence "high" only when the reading is clear. Answer with the JSON
object the schema defines, one entry per item, with its id."""
ITEM = {"type": "object", "additionalProperties": False, "required": ["id", "decision", "candidate", "direction", "confidence"],
        "properties": {"id": {"type": "integer"}, "decision": {"type": "string", "enum": ["same", "new"]},
                       "candidate": {"type": ["integer", "null"]}, "direction": {"type": ["string", "null"], "enum": ["same", "reversed", None]},
                       "confidence": {"type": "string", "enum": ["high", "low"]}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": ITEM}}}


def main():
    hub_dir, contrasts, glossp, abbrp, outp = sys.argv[1:6]
    model, effort = (sys.argv[6], sys.argv[7]) if len(sys.argv) > 7 else ("gpt-6.1-sol", "low")
    h = json.load(open(f"{hub_dir}/hub.json")); E = h["contrasts"]; A = json.load(open(f"{hub_dir}/corpus_assign.json"))
    C = json.load(open(contrasts)); gloss = json.load(open(glossp)); abbr = json.load(open(abbrp))
    est = collections.defaultdict(list); examples = collections.defaultdict(list)
    for aid, a in A.items():             # examples in the entry's own direction: a reversed contrast's sides swapped
        if len(examples[a["hub"]]) < 2:
            t, b = C[aid]['pos'].split(' | ')[0][:40], C[aid]['neg'].split(' | ')[0][:40] or '-'
            examples[a["hub"]].append(f"{t} vs {b}" if a.get("sign", 1) > 0 else f"{b} vs {t}")
    for k, e in E.items():
        if e["established"] and not k.startswith("task:"):
            sc, comp = H._sig_from_json(e["sig_json"]); est[sc].append((comp, k, e["papers"]))
    two = lambda c: c[1] not in (H.ABSENT, None) and c[2] not in (H.ABSENT, None)  # noqa: E731
    done = json.load(open(outp)) if Path(outp).exists() else {}
    items = collections.defaultdict(list)
    for aid, a in A.items():
        if SAMPLE is not None and aid not in SAMPLE:
            continue
        redo = REJUDGE and (done.get(aid, {}).get("decision") == "same" or a.get("sign", 1) < 0)   # reversed items saw a wrong "read as" before
        if a["kind"] != "condition" or a["level"] in ("L0", "L0a", "L0h", "L0m") or (aid in done and not redo):
            continue
        sc, comp = H._sig_from_json(E[a["hub"]]["sig_json"]); axes = {c[0] for c in comp}
        if WIDE:        # any established entry of the task: the item's own axes may be misread
            atoms = lambda cc: {(c[0], v) for c in cc for v in c[1:] if v not in (H.ABSENT, None)}  # noqa: E731
            cands = [((len(atoms(cc) & atoms(comp)), len({c[0] for c in cc} & axes), n), k) for cc, k, n in est.get(sc, [])]
        else:
            cands = [(n, k) for cc, k, n in est.get(sc, []) if {c[0] for c in cc} == axes or set(cc) < set(comp)
                     or any(two(c) and c in cc for c in comp)]
        if cands:
            items[sc].append((aid, [k for _, k in sorted(cands, reverse=True)[:MAX_CAND_WIDE if WIDE else MAX_CAND]]))
    batches = [(sc, its[i:i + BATCH]) for sc, its in items.items() for i in range(0, len(its), BATCH)]
    print(f"{sum(len(v) for v in items.values())} contrasts to judge in {len(batches)} batches", flush=True)
    disabled = [t for t in cx.TOOLS if t in cx.known_features()]

    def read_as(aid):
        """the item's reading in the direction of its named sides (the hub stores an entry one way; sign -1 = reversed)"""
        k = A[aid]["hub"]
        if A[aid].get("sign", 1) > 0:
            return k.split("|", 1)[1]
        f = H.flip(H._sig_from_json(E[k]["sig_json"]))
        return H.sig_str(f).split("|", 1)[1] if f is not None else k.split("|", 1)[1] + " (reversed: baseline side > target side)"

    def text(aid, cands):
        a = C[aid]; g = gloss.get(a["paper"], {})
        conds = [f"{c['name']}: {c['description'][:110]}" for t in g.get("tasks", []) for c in t["conditions"]][:8]
        toks = set(re.findall(r"[A-Za-z][A-Za-z0-9+\-]{0,9}", a["pos"] + " " + a["neg"]))
        ab = [f"{k} = {v}" for k, v in abbr.get(a["paper"], {}).items() if k in toks][:6]
        cl = "\n".join(f"  C{j}: {k.split('|', 1)[1]} (e.g. {'; '.join(examples.get(k, []))})" for j, k in enumerate(cands))
        return (f"target side: {a['pos'][:160]}\nbaseline side: {a['neg'][:160] or '(none)'}\nread as: {read_as(aid)}\n"
                f"paper conditions: {' || '.join(conds) or '(none)'}\nabbreviations: {'; '.join(ab) or '(none)'}\ncandidates:\n{cl}")

    def run(b):
        sc, its = b
        prompt = f"task: {TASK_NAME.get(sc, sc)}\n\n" + "\n\n".join(f"id {i}\n{text(aid, c)}" for i, (aid, c) in enumerate(its))
        try:
            ans, usage, secs = cx.call(model, effort, RULES, prompt, disabled, schema=SCHEMA)
            return b, ans["items"], f"{secs}s in {usage.get('input_tokens')} out {usage.get('output_tokens')}"
        except Exception as exc:  # noqa: BLE001
            return b, None, str(exc)[:200]
    st = collections.Counter()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        for (sc, its), res, note in pool.map(run, batches):
            if res is None:
                print(f"FAILED {sc}: {note}", flush=True); continue
            for it in res:
                if not 0 <= it["id"] < len(its):
                    continue
                aid, cands = its[it["id"]]
                if it["decision"] == "same" and it["candidate"] is not None and 0 <= it["candidate"] < len(cands):
                    done[aid] = {"decision": "same", "entry": cands[it["candidate"]],
                                 "sign": -1 if it["direction"] == "reversed" else 1, "confidence": it["confidence"]}
                else:
                    done[aid] = {"decision": "new", "confidence": it["confidence"]}
                st[f"{done[aid]['decision']}:{it['confidence']}"] += 1
            json.dump(done, open(outp, "w"))
            print(f"{sc}: {len(its)} {note}", flush=True)
    print(json.dumps(dict(st)), flush=True)


if __name__ == "__main__":
    main()
