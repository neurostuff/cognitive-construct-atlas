"""Contrast-level normalization by Codex for corpus contrasts the rules leave unnormalized (hub growth, imaging).

Each item is one corpus contrast: its task family and task, the two sides' names (a cell's level, then its
conditions), the paper's conditions with their record descriptions (glossary.py), and the paper's abbreviations that
occur in the names (abbrev.py). Codex answers with each side's atoms in the hub vocabulary (the family's atoms in use,
most frequent first, and every axis in use), the contrast type, and a confidence. The answers are kept per contrast
(overrides, read by imaging.read) and, for high-confidence answers, each side's level name becomes a paper lexicon
entry (paper -> folded name -> atoms): names decoded from a paper's own descriptions and abbreviations hold only in
that paper. Atoms (the vocabulary) are shared; name readings are not.

REREAD=ft re-reads the contrasts whose reading came from the fine-tuned model (source "ft") and replaces it.

usage: [REREAD=ft] python codex_normalize.py HUB_DIR CONTRASTS.json GLOSSARY.json ABBREV.json OVERRIDES.json PAPER_LEXICON.json [MODEL EFFORT LIMIT]
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

BATCH = int(__import__("os").environ.get("BATCH", "60"))

RULES = """You normalize contrasts from neuroimaging papers into a shared vocabulary of condition atoms. A contrast compares
a target side with a baseline side. Each side is described by the atoms `axis:value` on which it differs from the
other side (incongruent > congruent is target congruency:incongruent, baseline congruency:congruent; 2-back > 0-back is
load:2 vs load:0; fearful faces > neutral faces is emotion:fear vs valence:neutral).

You get, per contrast: the task family and task, the side names (a level name, then the conditions it pools), the
paper's conditions with their descriptions, and the paper's abbreviations. Decode codes and abbreviations from them.

Answer per contrast:
- target, baseline: lists of atoms for each side. Only atoms that distinguish the sides; leave out what both share
  (the task's material). Reuse the family's atoms and the axes listed whenever one fits, even if worded differently;
  a new axis or value only when nothing fits, lower_snake_case and general (masking:masked, fiber_type:c_fiber,
  perspective:egocentric, color:chromatic, saccade:anti, block_position:first). Never a paper-specific code.
- baseline_special: "@base" when the baseline is rest, fixation, passive viewing or a low-level control (then
  baseline is []); null otherwise.
- type: "difference" (two conditions), "baseline" (condition vs rest/fixation/control), "main_effect" (one factor's
  levels pooled over another factor's), "parametric" (a quantity varying over trials), "interaction", "task_vs_base"
  (the task as a whole vs baseline, no finer condition), or "not_conditions" (groups, sessions of a treatment,
  regions, or a comparison of tasks across papers that is not a condition contrast).
- confidence: "high" when the reading follows from the names, descriptions and abbreviations; "low" otherwise.

Answer with the JSON object the schema defines, one item per contrast, in the order given, with its id."""

ITEM = {"type": "object", "additionalProperties": False,
        "required": ["id", "target", "baseline", "baseline_special", "type", "confidence"],
        "properties": {"id": {"type": "integer"}, "target": {"type": "array", "items": {"type": "string"}},
                       "baseline": {"type": "array", "items": {"type": "string"}},
                       "baseline_special": {"type": ["string", "null"], "enum": ["@base", None]},
                       "type": {"type": "string", "enum": ["difference", "baseline", "main_effect", "parametric",
                                                           "interaction", "task_vs_base", "not_conditions"]},
                       "confidence": {"type": "string", "enum": ["high", "low"]}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": ITEM}}}

TASK_NAME = {t["id"]: t["name"] for t in H.INVENTORY["tasks"]}


def vocab(hubj):
    """family -> atoms in use, most papers first; and all axes in use"""
    cnt = collections.defaultdict(collections.Counter)
    for e in hubj["contrasts"].values():
        for ax, a, b in json.loads(e["sig_json"])[1]:
            for x in (a, b):
                if x and x not in ("@base", "@rest", "@control", "*", "@task"):
                    cnt[e["family"]][f"{ax}:{x}"] += e["papers"]
    axes = collections.Counter(a.split(":")[0] for c in cnt.values() for a in c.elements())
    return {f: [a for a, _ in c.most_common(160)] for f, c in cnt.items()}, [a for a, _ in axes.most_common(140)]


def item_text(i, a, gloss, abbr):
    task = (a["tasks"].split(",") or [""])[0]
    g = gloss.get(a["paper"], {})
    conds = [f"{c['name']}: {c['description'][:140]}" for t in g.get("tasks", []) for c in t["conditions"]][:12]
    toks = set(re.findall(r"[A-Za-z][A-Za-z0-9+\-]{0,9}", a["pos"] + " " + a["neg"]))
    ab = abbr.get(a["paper"], {})
    abl = [f"{k} = {v}" for k, v in ab.items() if k in toks or k.lower() in {t.lower() for t in toks}][:10]
    return (f"id {i}\nfamily: {a['families'].split(',')[0]}; task: {TASK_NAME.get(task, task)}\n"
            f"target side: {a['pos'][:200]}\nbaseline side: {a['neg'][:200] or '(none)'}\n"
            f"paper conditions: {' || '.join(conds) or '(none)'}\nabbreviations: {'; '.join(abl) or '(none)'}")


def main():
    hub_dir, contrasts, glossp, abbrp, ovp, lexp = sys.argv[1:7]
    model, effort = (sys.argv[7], sys.argv[8]) if len(sys.argv) > 8 else ("gpt-6.1-sol", "low")
    limit = int(sys.argv[9]) if len(sys.argv) > 9 else None
    hubj = json.load(open(f"{hub_dir}/hub.json")); fam_atoms, axes = vocab(hubj)
    C = json.load(open(contrasts)); gloss = json.load(open(glossp)); abbr = json.load(open(abbrp))
    todo = json.load(open(f"{hub_dir}/corpus_unnormalized.json"))
    ov = json.load(open(ovp)) if Path(ovp).exists() else {}
    if os.environ.get("REREAD") == "ft":        # replace the fine-tuned model's readings with Codex's
        todo = [a for a, o in ov.items() if o.get("source") == "ft"][:limit]
    else:
        todo = [a for a in todo if a not in ov][:limit]
    by_fam = collections.defaultdict(list)
    for aid in todo:
        by_fam[C[aid]["families"].split(",")[0] or "?"].append(aid)
    batches = [(f, ids[i:i + BATCH]) for f, ids in by_fam.items() for i in range(0, len(ids), BATCH)]
    print(f"{len(todo)} contrasts in {len(batches)} batches", flush=True)
    disabled = [t for t in cx.TOOLS if t in cx.known_features()]

    def run(b):
        fam, ids = b
        head = (f"family atoms in use (most papers first): {', '.join(fam_atoms.get(fam, [])[:160])}\n"
                f"axes in use anywhere: {', '.join(axes)}\n\n")
        prompt = head + "\n\n".join(item_text(i, C[aid], gloss, abbr) for i, aid in enumerate(ids))
        try:
            ans, usage, secs = cx.call(model, effort, RULES, prompt, disabled, schema=SCHEMA)
            return fam, ids, ans["items"], f"{secs}s in {usage.get('input_tokens')} out {usage.get('output_tokens')}"
        except Exception as exc:  # noqa: BLE001
            return fam, ids, None, str(exc)[:300]

    lex = json.load(open(lexp)) if Path(lexp).exists() else {}
    st = collections.Counter(); done = 0
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        for fam, ids, items, note in pool.map(run, batches):
            if items is None:
                print(f"FAILED {fam}: {note}", flush=True); st["failed_batches"] += 1; continue
            for it in items:
                if not 0 <= it["id"] < len(ids):
                    continue
                aid = ids[it["id"]]; a = C[aid]
                clean = lambda xs: sorted({x.strip().lower().replace(" ", "_") for x in xs if ":" in x})
                rec = {"target": clean(it["target"]), "baseline": clean(it["baseline"]),
                       "baseline_special": it["baseline_special"], "type": it["type"], "confidence": it["confidence"]}
                ov[aid] = rec; st[f"{it['type']}:{it['confidence']}"] += 1
                if it["confidence"] == "high" and it["type"] not in ("not_conditions", "interaction"):
                    for field, atoms in ((a["pos"], rec["target"]), (a["neg"], rec["baseline"])):
                        name = field.split(" | ")[0].strip()
                        if name and atoms:
                            lex.setdefault(a["paper"], {}).setdefault(H.fold(name), atoms)
                    if rec["baseline_special"] == "@base" and a["neg"]:
                        lex.setdefault(a["paper"], {}).setdefault(H.fold(a["neg"].split(" | ")[0]), ["@base"])
            done += len(ids)
            print(f"{fam}: {len(ids)} ({done}/{len(todo)}) {note}", flush=True)
            if done % 500 < BATCH:
                json.dump(ov, open(ovp, "w")); json.dump(lex, open(lexp, "w"), indent=1, sort_keys=True)
    json.dump(ov, open(ovp, "w")); json.dump(lex, open(lexp, "w"), indent=1, sort_keys=True)
    print(json.dumps(dict(st)), flush=True)


if __name__ == "__main__":
    main()
