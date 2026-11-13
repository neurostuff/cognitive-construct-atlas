"""Sentence-level verification by a local model: the classifier's top candidates per paper, one request each.

Candidates: for each paper, the K units with the highest out-of-fold BiomedBERT score (preds/clf_biomedbert.jsonl).
Request: the rules (guide_v2.md's "What counts" and "Categories" sections) as the system message; the paper's tasks,
the previous unit and the unit as the user message; a small JSON schema (category, task, contrast_text, constructs,
polarity) under vLLM structured outputs. Writes preds/verify_<tag>.jsonl.

usage: python verify_sentences.py BASE_URL MODEL TAG K   (run from pilot-results/)
"""
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path.cwd()
CATS = ["own_task", "own_contrast", "label_usage", "other_task", "region", "none"]
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["category", "task", "contrast_text", "constructs", "polarity"],
          "properties": {"category": {"type": "string", "enum": CATS},
                         "task": {"type": ["string", "null"]},
                         "contrast_text": {"type": ["string", "null"]},
                         "constructs": {"type": "array", "items": {"type": "string"}},
                         "polarity": {"type": "string", "enum": ["asserted", "hedged", "negated"]}}}


def rules():
    g = (ROOT / "guide_v2.md").read_text(encoding="utf-8")
    body = g[g.index("## What counts as a construct claim"):g.index("## Output")]
    return (body + "\nJudge one sentence (UNIT) at a time. Answer with the JSON object the schema defines: its "
            "category (`none` when it is not a construct claim of any kind), the paper's task id it is about "
            "(or null), the contrast as worded (or null), the constructs copied word for word from the UNIT, and "
            "its polarity.")


def call(base, model, system, user):
    body = {"model": model, "temperature": 0.0, "max_tokens": 512,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "verdict", "schema": SCHEMA, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(json.load(r)["choices"][0]["message"]["content"])


def main():
    base, model, tag, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    system = rules()
    by = defaultdict(list)
    for l in open(ROOT / "preds" / "clf_biomedbert.jsonl"):
        r = json.loads(l); by[r["article"]].append(r)
    papers = {a: json.loads((ROOT / "pilot" / "papers" / f"{a}.json").read_text()) for a in by}
    jobs = []
    for a, rows in by.items():
        p = papers[a]
        head = "\n".join(f"- {t['task']}: {t['name']!r}" for t in p["tasks"])
        for r in sorted(rows, key=lambda r: -r["p"])[:k]:
            U = p["units"]; i = r["i"]
            user = (f"Paper's tasks:\n{head}\n\nPrevious unit: {U[i - 1]['text'] if i else ''}\n\n"
                    f"UNIT ({U[i]['section']}): {U[i]['text']}")
            jobs.append((r, user))

    def run(job):
        r, user = job
        try:
            v = call(base, model, system, user)
        except Exception as exc:  # noqa: BLE001
            v = {"error": str(exc)[:200]}
        return {**r, "verdict": v}

    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "8"))) as pool, \
            open(ROOT / "preds" / f"verify_{tag}.jsonl", "w") as f:
        for n, out in enumerate(pool.map(run, jobs), 1):
            f.write(json.dumps(out) + "\n")
            if n % 100 == 0:
                print(n, "of", len(jobs), flush=True)
    print("done", len(jobs))


if __name__ == "__main__":
    main()
