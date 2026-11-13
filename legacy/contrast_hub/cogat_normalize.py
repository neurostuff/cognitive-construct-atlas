"""Normalize constructs that are not Cognitive Atlas concepts onto the Cognitive Atlas (James 2026-11-12: new concepts
keep being generated, so each must map onto an existing concept or be motivated as different).

For every construct node a corpus uses (taxonomy constructs without a cogat_id; Cognitive Atlas concepts are left as
they are), the Claude CLI (claude_extract.call) judges it against the whole Cognitive Atlas concept list:
  same      it names an existing concept (a synonym or a wording variant): merge into that concept
  narrower  a kind or a part of an existing concept (motor response inhibition < response inhibition): keep, with
            that concept as its parent, so its contrasts also pool into the concept's map
  new       no Cognitive Atlas concept covers it: keep; `parent` the broader concept, if any
A `same` or `narrower` answer must name a concept on the list; answers that do not are dropped and reported. Whether
a kept construct is different enough is then tested on the fitted model (a child whose mixture cannot be told from
its parent's is merged), not here.

usage: python cogat_normalize.py CORPUS.json TAXONOMY.json COGAT_CONCEPTS.json OUT.json [MODEL EFFORT]
       (laptop: the claude CLI; resumable, OUT.json keeps finished constructs)
"""
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import claude_extract as cx  # noqa: E402

RULES = """You map cognitive constructs onto the Cognitive Atlas, an ontology of cognitive concepts. Each construct was
extracted from neuroimaging papers, where it names what a task or a contrast between conditions measures; its most
common tasks are given. For each construct, decide against the Cognitive Atlas concept list:
- same: it names a concept on the list (a synonym, an inflection, a spelling or wording variant, or the same thing
  under another name). Give that concept.
- narrower: it is a specific kind or part of a concept on the list, and the difference is real (a subtype, a
  modality-, effector- or stimulus-specific version, a sub-process). Give the closest broader concept.
- new: no concept on the list names it or contains it. If a concept on the list is clearly broader, give it as the
  parent; otherwise leave it empty.
Prefer same over narrower when the difference is only wording; prefer narrower over new when a broader concept
fits. Use only concept names exactly as listed. Answer with the JSON object the schema defines, one item per
construct, with its id."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "relation", "concept", "reason"],
    "properties": {"id": {"type": "integer"}, "relation": {"type": "string", "enum": ["same", "narrower", "new"]},
                   "concept": {"type": "string"}, "reason": {"type": "string"}}}}}}
BATCH = 30


def main():
    corpusp, taxp, cogp, outp = sys.argv[1:5]
    model, effort = (sys.argv[5], sys.argv[6]) if len(sys.argv) > 6 else ("claude-opus-5-5", "medium")
    H = json.load(open(corpusp)); tax = {c["id"]: c for c in json.load(open(taxp))["constructs"]}
    cog = sorted({c["name"].strip() for c in json.load(open(cogp)) if c.get("name")}, key=str.lower)
    cogl = {c.lower(): c for c in cog}
    cnt = collections.Counter(x for d in H["docs"].values() for x in d["ids"] if x.startswith("c:"))
    nodes = sorted({x for x in H["parents"] if x.startswith("c:")} | set(cnt), key=lambda x: -cnt[x])
    todo = []
    for n in nodes:
        tid = n[2:].replace(" ", "-")
        t = tax.get(tid)
        if (t and t.get("cogat_id")) or n[2:].lower() in cogl:
            continue
        todo.append({"node": n, "label": n[2:], "contrasts": cnt[n],
                     "tasks": [x for x in (t or {}).get("tasks", [])][:3], "aliases": (t or {}).get("aliases", [])[:4]})
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [x for x in todo if x["node"] not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(todo)} constructs in {len(batches)} batches", flush=True)
    listing = "Cognitive Atlas concepts:\n" + "; ".join(cog) + "\n\n"

    def run(b):
        prompt = listing + "\n".join(
            f"id {i}: {x['label']}" + (f" (aliases: {', '.join(x['aliases'])})" if x["aliases"] else "") +
            (f"; tasks: {', '.join(x['tasks'])}" if x["tasks"] else "") for i, x in enumerate(b))
        try:
            ans, _, secs = cx.call(model, effort, RULES, prompt, (), schema=SCHEMA)
            return b, ans["items"]
        except Exception as exc:  # noqa: BLE001
            print("FAILED", str(exc)[:200], flush=True); return b, None
    bad = collections.Counter()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool:
        for b, res in pool.map(run, batches):
            if res is None:
                continue
            for it in res:
                if not 0 <= it["id"] < len(b):
                    continue
                x = b[it["id"]]; c = cogl.get(it["concept"].strip().lower(), "") if it["concept"] else ""
                if it["concept"] and not c:
                    bad[it["relation"]] += 1
                rel = it["relation"] if (c or it["relation"] == "new") else "new"
                done[x["node"]] = {"relation": rel, "concept": c, "reason": it["reason"], "contrasts": x["contrasts"]}
            json.dump(done, open(outp, "w"), indent=1)
    n = collections.Counter(v["relation"] for v in done.values())
    slots = collections.Counter()
    for v in done.values():
        slots[v["relation"]] += v["contrasts"]
    print(json.dumps({"constructs": dict(n), "contrast_slots": dict(slots), "concept_not_on_list": dict(bad)}, indent=1))


if __name__ == "__main__":
    main()
