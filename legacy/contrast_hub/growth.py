"""Hub growth loop: condition strings the atoms miss -> lexicon entries (contrast hub, step 3).

One round:
1. collect the strings that left a contrast unnormalized, per family: imaging (the residue tokens of corpus keys,
   hub build's corpus_unmatched.json, weighted by contrasts) and claims (match_claims' unmatched_sides);
2. keep those seen at least MIN_IMAGING times on the imaging side, or MIN_CLAIM times on the claim side;
3. send them in batches to Codex with the family's tasks and its current condition vocabulary (axis:value atoms) and
   ask for each: the atoms it denotes (existing ones first; a new axis or value only when none fits), `@base` for a
   rest / fixation / low-level control condition, or `@none` (a task name alone, a group, a measure, an unreadable
   abbreviation), with a confidence;
4. accept the `high` confidence answers into lexicon.json (family -> folded string -> atoms); everything else is
   left for the next round or a person.
The lexicon is read by hub.side_from_text / side_from_tokens, so rebuilding the hub and re-matching claims measures
the round.

usage: python growth.py HUB_DIR CLAIMS_MATCH.json LEXICON.json [MODEL EFFORT]   (Codex through ../codex_extract.call)
"""
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import codex_extract as cx  # noqa: E402

import hub as H  # noqa: E402

MIN_IMAGING, MIN_CLAIM = 3, 1
BATCH = 60

RULES = """You normalize experimental condition names from neuroimaging and cognitive psychology papers into a shared
condition vocabulary. A condition is one state of a task (incongruent trials, 2-back blocks, fearful faces, a rest
block). Its normalized form is a set of atoms `axis:value`. Two conditions of a task are compared on the axes where
their values differ (incongruent > congruent is congruency:incongruent vs congruency:congruent).

For each string (given with its task family, the tasks it came from, and the family's current atoms) answer:
- atoms: the atoms the string denotes. Reuse an existing axis and value whenever one fits (same meaning, even if
  worded differently). Propose a new value on an existing axis, or a new axis, only when nothing fits; new names are
  lower_snake_case and general (cue_type:double, saccade:anti, cpt_trial:ay, tmt_part:b, task_load:dual), never a
  paper-specific code. Use as few atoms as the string supports.
- special: "@base" when the string is a rest, fixation, passive viewing or low-level control/baseline condition;
  "@none" when it is not a task condition (a task name on its own, a participant group, a measure such as RT, a
  statistic, or an abbreviation whose meaning cannot be known from the string and family); null otherwise.
- confidence: "high" only when a researcher in the field would read the string this way without seeing the paper;
  "low" otherwise (ambiguous abbreviations, strings that could be several things).

Answer with the JSON object the schema defines, one item per string, in the order given, with its id."""

ITEM = {"type": "object", "additionalProperties": False, "required": ["id", "atoms", "special", "confidence"],
        "properties": {"id": {"type": "integer"}, "atoms": {"type": "array", "items": {"type": "string"}},
                       "special": {"type": ["string", "null"], "enum": ["@base", "@none", None]},
                       "confidence": {"type": "string", "enum": ["high", "low"]}}}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": ITEM}}}


def candidates(hub_dir, claims_match):
    img = collections.Counter(); clm = collections.Counter()
    for fam, s, n in json.load(open(f"{hub_dir}/corpus_unmatched.json")):
        for part in [s]:
            img[(fam, H.fold(part))] += n
    for fam, s, n in json.load(open(claims_match))["unmatched_sides"]:
        clm[(fam, H.fold(s))] += n
    keys = {k for k, n in img.items() if n >= MIN_IMAGING and k[1]} | {k for k, n in clm.items() if n >= MIN_CLAIM and k[1]}
    return sorted(keys, key=lambda k: -(img[k] + clm[k])), img, clm


def main():
    hub_dir, claims_match, lex_path = sys.argv[1:4]
    model, effort = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ("gpt-6.1-sol", "low")
    hubj = json.load(open(f"{hub_dir}/hub.json"))
    fam_tasks = collections.defaultdict(list)
    for t in H.INVENTORY["tasks"]:
        fam_tasks[t["family"]].append(t["name"])
    keys, img, clm = candidates(hub_dir, claims_match)
    lex = json.load(open(lex_path)) if Path(lex_path).exists() else {}
    keys = [k for k in keys if k[1] not in lex.get(k[0], {})]
    print(f"{len(keys)} strings to map ({sum(img[k] for k in keys)} imaging contrasts, {sum(clm[k] for k in keys)} claim sides)", flush=True)
    by_fam = collections.defaultdict(list)
    for k in keys:
        by_fam[k[0]].append(k[1])
    batches = []
    for fam, ss in by_fam.items():
        vocab = sorted(hubj["conditions"].get(fam, {}))
        for i in range(0, len(ss), BATCH):
            batches.append((fam, vocab, ss[i:i + BATCH]))
    disabled = [t for t in cx.TOOLS if t in cx.known_features()]
    log = open(Path(lex_path).with_suffix(".answers.jsonl"), "a")

    def run(b):
        fam, vocab, ss = b
        head = (f"family: {fam}\ntasks: {', '.join(fam_tasks[fam][:25])}\ncurrent atoms: {', '.join(vocab[:250])}\n"
                f"all axes in use: {', '.join(sorted({a.split(':')[0] for v in hubj['conditions'].values() for a in v}))}\n\n")
        prompt = head + "\n".join(f"id {i}: {s}" for i, s in enumerate(ss))
        try:
            ans, usage, secs = cx.call(model, effort, RULES, prompt, disabled, schema=SCHEMA)
        except Exception as exc:  # noqa: BLE001
            return fam, ss, None, str(exc)[:300]
        return fam, ss, ans["items"], f"{secs}s in {usage.get('input_tokens')} out {usage.get('output_tokens')}"

    acc = collections.Counter()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool:
        for fam, ss, items, note in pool.map(run, batches):
            if items is None:
                print(f"FAILED {fam}: {note}", flush=True); continue
            for it in items:
                if not 0 <= it["id"] < len(ss):
                    continue
                s = ss[it["id"]]
                log.write(json.dumps({"family": fam, "string": s, **it, "imaging": img[(fam, s)], "claims": clm[(fam, s)]}) + "\n")
                if it["confidence"] != "high":
                    acc["low"] += 1; continue
                val = [it["special"]] if it["special"] else sorted({a.strip().lower() for a in it["atoms"] if ":" in a})
                if not val:
                    acc["empty"] += 1; continue
                lex.setdefault(fam, {})[s] = val; acc["accepted"] += 1
            print(f"{fam}: {len(ss)} strings, {note}", flush=True)
    json.dump(lex, open(lex_path, "w"), indent=1, sort_keys=True)
    print(json.dumps(dict(acc)), flush=True)


if __name__ == "__main__":
    main()
