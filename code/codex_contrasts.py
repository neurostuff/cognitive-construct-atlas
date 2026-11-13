"""Silver labels for contrast mentions in construct-claim sentences, from Codex (James 2026-11-11: use Codex `sol` to
extract contrasts from sentences where they exist, and derive a training set for a non-LLM contrast tagger).

Input: contrast_labels/contrast_label_sample.json (claim sentences with their constructs; split by article). Each
call sends a batch of BATCH sentences with the rules below and a strict schema, through codex_extract.call (no tools,
account-billed, empty working directory). Spans must be verbatim substrings of the sentence: the answer is checked,
and a span that is not found is dropped and counted. Output: contrast_labels/<tag>.jsonl, one record per sentence.

usage: [SAMPLE=contrast_label_sample2.json] python codex_contrasts.py MODEL EFFORT TAG [LIMIT]   (run from cc/)
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import codex_extract as cx

ROOT = Path(__file__).parent
BATCH = 40

RULES = """You label sentences from neuroimaging papers. Each sentence was flagged because it claims that a task, a
measure or a brain activation reflects a cognitive construct (the constructs found are listed with it).

For each sentence decide whether it names a CONTRAST: an explicit comparison of experimental conditions within a task
(e.g. "incongruent > congruent", "stop versus go trials", "2-back minus 0-back", "the conflict effect (incongruent -
congruent)", "successful stop trials compared with failed stop trials"), or a condition compared with a baseline
("faces versus fixation", "task > rest"), or a parametric effect ("activation increasing with memory load").

Not contrasts: a group comparison alone (patients vs controls, old vs young), a task name alone ("the Stroop task"), a
correlation with a trait, a comparison between studies or tasks in a review.

Fields (copy every span verbatim from the sentence, character for character; null when absent):
- has_contrast: true or false.
- contrast: the shortest span that states the comparison ("incongruent > congruent trials").
- condition_a, condition_b: the spans naming the two sides, when stated (condition_b is null for a parametric effect
  or a baseline that is not named).
- kind: "conditions", "baseline", "parametric" or null.
- measure: what the contrast is measured on: "brain" (activation, BOLD, ERP, connectivity), "behavior" (RT,
  accuracy, errors) or "unclear"; null when has_contrast is false.
- construct_link: true if the sentence attributes one of its listed constructs to this contrast (the contrast
  reflects, indexes or isolates it), false if the contrast is mentioned but the construct is attributed to something
  else (the task as a whole, a region, a group), null when has_contrast is false.

Answer with the JSON object the schema defines, one item per sentence, in the order given, with its id."""

ITEM = {
    "type": "object", "additionalProperties": False,
    "required": ["id", "has_contrast", "contrast", "condition_a", "condition_b", "kind", "measure", "construct_link"],
    "properties": {
        "id": {"type": "integer"}, "has_contrast": {"type": "boolean"},
        "contrast": {"type": ["string", "null"]}, "condition_a": {"type": ["string", "null"]},
        "condition_b": {"type": ["string", "null"]},
        "kind": {"type": ["string", "null"], "enum": ["conditions", "baseline", "parametric", None]},
        "measure": {"type": ["string", "null"], "enum": ["brain", "behavior", "unclear", None]},
        "construct_link": {"type": ["boolean", "null"]},
    },
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": ITEM}}}


def prompt_for(batch):
    return "\n\n".join(f"id {r['id']}\nconstructs: {', '.join(r['constructs'])}\nsentence: {r['text']}" for r in batch)


def run_batch(batch, model, effort, disabled):
    ans, usage, secs = cx.call(model, effort, RULES, prompt_for(batch), disabled, schema=SCHEMA)
    by_id = {r["id"]: r for r in batch}; out, dropped = [], 0
    for it in ans["items"]:
        r = by_id.get(it["id"])
        if r is None:
            continue
        for f in ("contrast", "condition_a", "condition_b"):
            if it[f] is not None and it[f] not in r["text"]:
                it[f] = None; dropped += 1
        if it["has_contrast"] and it["contrast"] is None:
            it["has_contrast_unspanned"] = True
        out.append({**r, "label": it})
    return out, usage, secs, dropped, len(batch) - len(out)


def main():
    model, effort, tag = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
    sample = os.environ.get("SAMPLE", "contrast_label_sample.json")   # contrast_label_sample2.json: 4,000 more
    rows = json.load(open(ROOT / "contrast_labels" / sample))[:limit]
    outp = ROOT / "contrast_labels" / f"{tag}.jsonl"
    done = {json.loads(l)["id"] for l in open(outp)} if outp.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    disabled = [t for t in cx.TOOLS if t in cx.known_features()]
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool:
        for res in pool.map(lambda b: _safe(b, model, effort, disabled), batches):
            if isinstance(res, str):
                print(res, flush=True); continue
            out, usage, secs, dropped, missing = res
            with open(outp, "a") as f:
                for r in out:
                    f.write(json.dumps(r) + "\n")
            print(f"{tag}: {len(out)} labelled, {missing} missing, {dropped} non-verbatim spans dropped, {secs}s, "
                  f"in {usage.get('input_tokens')} out {usage.get('output_tokens')}", flush=True)


def _safe(b, model, effort, disabled):
    try:
        return run_batch(b, model, effort, disabled)
    except Exception as exc:  # noqa: BLE001
        return f"FAILED batch starting id {b[0]['id']}: {str(exc)[:300]}"


if __name__ == "__main__":
    main()
