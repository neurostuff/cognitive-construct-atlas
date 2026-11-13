"""Task assignment for contrasts the corpus labels only with a task family (ns-pond found no inventory task for them).

Contrasts are scoped to their task (James 2026-11-11), so a contrast labelled only `language` or `perceptual_judgment`
has nothing to be matched within. Codex reads each such contrast with its paper's record tasks (name, description,
condition names) and the family's inventory tasks (name and definition), and answers per contrast: the inventory task
it is an instance of, or none and a short generic name for the task it is (`word monitoring`, `infant cry listening`).

Phase 2 consolidates the new names: per family, Codex groups the names that are the same task under one canonical
name. A new task gets the id `new_<family>_<slug of its canonical name>`; it is a provisional task for the hub, a
candidate for ns-pond's task inventory, never written back to it.

Output TASKS.json: aid -> {"task": inventory id | null, "new_task": canonical name | null, "task_id": the id the hub
scopes the contrast to, "confidence": "high" | "low"}. Resumes from TASKS.json.

usage: python task_assign.py CONTRASTS.json GLOSSARY.json TASKS.json [MODEL EFFORT]
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

PAPERS_PER_BATCH, MAX_PER_UNIT = 15, 14
TASK_ID = {t["id"] for t in H.INVENTORY["tasks"]}
BY_FAMILY = collections.defaultdict(list)
for t in H.INVENTORY["tasks"]:
    BY_FAMILY[t["family"]].append(t)

RULES = """You assign neuroimaging contrasts to the experimental task they come from. Each paper comes with the tasks its
record describes and its contrasts (target side vs baseline side, as condition names). Candidate tasks are the known
tasks of one task family, each with a definition.

Per contrast, answer `task`: the candidate index when the contrast's task is an instance or close variant of that
candidate (same procedure and manipulation, different stimuli or wording allowed), else null. When null, `new_task`:
a short generic name for the task as a researcher would name it across papers (lower case, no paper-specific details,
no stimulus counts, e.g. "word monitoring", "infant cry listening", "visual motion localizer"); else null. A localizer
is named by what it localizes. confidence "high" only when the record makes the task clear. Answer with the JSON object
the schema defines, one entry per contrast, with its id."""
ITEM = {"type": "object", "additionalProperties": False, "required": ["id", "task", "new_task", "confidence"],
        "properties": {"id": {"type": "integer"}, "task": {"type": ["integer", "null"]},
                       "new_task": {"type": ["string", "null"]}, "confidence": {"type": "string", "enum": ["high", "low"]}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": ITEM}}}

MERGE_RULES = """These are names of experimental tasks within one task family, each with the number of contrasts that
carry it. Group the names that are the same task (synonyms, word order, singular/plural, "task"/"paradigm" suffixes,
the same procedure named by its stimulus class at a finer grain only when the procedure is the same). Different
procedures stay apart. Answer with the JSON object the schema defines: one group per task, its canonical name (short,
generic, lower case) and the indices of its member names; every index appears in exactly one group."""
GROUP = {"type": "object", "additionalProperties": False, "required": ["name", "members"],
         "properties": {"name": {"type": "string"}, "members": {"type": "array", "items": {"type": "integer"}}}}
MERGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["groups"],
                "properties": {"groups": {"type": "array", "items": GROUP}}}


def slug(name):
    s = re.sub(r"\b(task|paradigm|experiment)s?\b", " ", name.lower())
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def family_labelled(C):
    return {aid: a for aid, a in C.items() if (a["tasks"].split(",") or [""])[0] not in TASK_ID}


def apply(C, tasks):
    """C with each assigned contrast's task replaced by its assigned task id (high confidence only); registers the new
    task ids' families in H.TASK_FAMILY. Returns the number of contrasts reassigned."""
    n = 0
    for aid, t in tasks.items():
        if aid in C and t.get("confidence") == "high" and t.get("task_id"):
            fam = (C[aid]["families"].split(",") or [""])[0] or C[aid]["tasks"].split(",")[0]
            H.TASK_FAMILY.setdefault(t["task_id"], fam)
            C[aid] = {**C[aid], "tasks": t["task_id"], "task_assigned": "inventory" if t["task"] else "new"}
            n += 1
    return n


def main():
    contrasts, glossp, outp = sys.argv[1:4]
    model, effort = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ("gpt-6.1-sol", "low")
    C = json.load(open(contrasts)); gloss = json.load(open(glossp))
    done = json.load(open(outp)) if Path(outp).exists() else {}
    disabled = [t for t in cx.TOOLS if t in cx.known_features()]
    units = collections.defaultdict(list)                 # (family, paper) -> contrast ids
    for aid, a in family_labelled(C).items():
        fam = (a["families"].split(",") or [""])[0] or a["tasks"].split(",")[0]
        if fam in BY_FAMILY and aid not in done:
            units[(fam, a["paper"])].append(aid)
    by_fam = collections.defaultdict(list)
    for (fam, p), ids in units.items():
        for i in range(0, len(ids), MAX_PER_UNIT):
            by_fam[fam].append((p, ids[i:i + MAX_PER_UNIT]))
    batches = [(fam, us[i:i + PAPERS_PER_BATCH]) for fam, us in by_fam.items() for i in range(0, len(us), PAPERS_PER_BATCH)]
    print(f"{sum(len(i) for us in by_fam.values() for _, i in us)} contrasts in {len(batches)} batches", flush=True)

    def run(b):
        fam, us = b
        cands = BY_FAMILY[fam]
        head = "candidate tasks:\n" + "\n".join(f"  T{j}: {t['name']}: {(t.get('definition') or '')[:160]}" for j, t in enumerate(cands))
        body, ids = [], []
        for p, aids in us:
            g = gloss.get(p, {})
            rt = "; ".join(f"{t['name']}: {t['description'][:160]} (conditions: {', '.join(c['name'] for c in t['conditions'])[:120]})"
                           for t in g.get("tasks", [])[:4]) or "(none)"
            lines = []
            for aid in aids:
                lines.append(f"  id {len(ids)}: {C[aid]['pos'][:100]} vs {C[aid]['neg'][:100] or '(none)'}"); ids.append(aid)
            body.append(f"paper record tasks: {rt}\ncontrasts:\n" + "\n".join(lines))
        prompt = f"task family: {fam}\n{head}\n\n" + "\n\n".join(body)
        try:
            ans, usage, secs = cx.call(model, effort, RULES, prompt, disabled, schema=SCHEMA)
            return b, ids, ans["items"], f"{secs}s in {usage.get('input_tokens')} out {usage.get('output_tokens')}"
        except Exception as exc:  # noqa: BLE001
            return b, ids, None, str(exc)[:200]
    st = collections.Counter()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        for (fam, _), ids, res, note in pool.map(run, batches):
            if res is None:
                print(f"FAILED {fam}: {note}", flush=True); continue
            cands = BY_FAMILY[fam]
            for it in res:
                if not 0 <= it["id"] < len(ids):
                    continue
                t = cands[it["task"]]["id"] if it["task"] is not None and 0 <= it["task"] < len(cands) else None
                nt = None if t else (it["new_task"] or "").strip().lower() or None
                done[ids[it["id"]]] = {"task": t, "new_task": nt, "family": fam, "confidence": it["confidence"],
                                       "task_id": t or (f"new_{fam}_{slug(nt)}" if nt else None)}
                st[("inventory" if t else "new" if nt else "none") + ":" + it["confidence"]] += 1
            json.dump(done, open(outp, "w"))
            print(f"{fam}: {len(ids)} {note}", flush=True)
    print(json.dumps(dict(st)), flush=True)
    consolidate(done, outp, model, effort, disabled)


def consolidate(done, outp, model, effort, disabled):
    """phase 2: per family, merge the new task names that are one task under a canonical name"""
    names = collections.defaultdict(collections.Counter)
    for v in done.values():
        if v["new_task"] and not v.get("merged"):
            names[v["family"]][v["new_task"]] += 1
    for fam, cnt in names.items():
        if len(cnt) < 2:
            continue
        ns = [n for n, _ in cnt.most_common(400)]
        prompt = f"task family: {fam}\n" + "\n".join(f"{i}: {n} ({cnt[n]})" for i, n in enumerate(ns))
        try:
            ans, _, secs = cx.call(model, effort, MERGE_RULES, prompt, disabled, schema=MERGE_SCHEMA)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED merge {fam}: {str(exc)[:200]}", flush=True); continue
        canon = {}
        for g in ans["groups"]:
            for m in g["members"]:
                if 0 <= m < len(ns):
                    canon.setdefault(ns[m], g["name"].strip().lower())
        for v in done.values():
            if v["family"] == fam and v["new_task"] in canon:
                v["new_task_read"] = v["new_task"]; v["new_task"] = canon[v["new_task"]]
                v["task_id"] = f"new_{fam}_{slug(v['new_task'])}"; v["merged"] = True
        json.dump(done, open(outp, "w"))
        print(f"merge {fam}: {len(ns)} names -> {len(set(canon.values()))} tasks ({secs}s)", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "consolidate":
        o = sys.argv[2]; d = json.load(open(o))
        consolidate(d, o, "gpt-6.1-sol", "low", [t for t in cx.TOOLS if t in cx.known_features()])
    else:
        main()
