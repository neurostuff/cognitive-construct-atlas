"""Construct-claim extraction with a local model behind a vLLM OpenAI-compatible server.

Same request as codex_extract.py: rules v2 (+ the answer-shape note) as the system message, the rendered paper as
the user message, and the same JSON schema, here enforced by vLLM's structured outputs (`response_format`
json_schema). Writes llm/<article>.<tag>.jsonl in the format score.py reads.

usage: python vllm_extract.py BASE_URL MODEL TAG [--think] ARTICLE ...   (run from pilot-results/)
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
from codex_extract import SCHEMA, SHAPE  # noqa: E402


def call(base, model, system, paper, think):
    body = {"model": model, "temperature": 0.0, "max_tokens": 8192,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": paper}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "claims", "schema": SCHEMA, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": think}}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return json.loads(d["choices"][0]["message"]["content"]), d.get("usage", {}), round(time.time() - t0, 1)


def one(base, model, tag, system, think, art):
    out = ROOT / "llm" / f"{art}.{tag}.jsonl"
    if out.exists() and out.stat().st_size:
        return f"{art}: cached"
    paper = (ROOT / "rendered" / f"{art}.txt").read_text(encoding="utf-8")
    try:
        ans, usage, secs = call(base, model, system, paper, think)
    except Exception as exc:  # noqa: BLE001
        return f"{art}: FAILED {exc}"
    with open(out, "w") as f:
        f.write(json.dumps({"paper": {"participants": ans["participants"]}}) + "\n")
        for c in ans["claims"]:
            f.write(json.dumps(c) + "\n")
    with open(ROOT / "llm" / f"usage.{tag}.jsonl", "a") as f:
        f.write(json.dumps({"article": art, "seconds": secs, **usage}) + "\n")
    return f"{art}: {len(ans['claims'])} lines, {secs}s, {usage}"


def main():
    base, model, tag, *rest = sys.argv[1:]
    think = "--think" in rest
    arts = [a for a in rest if not a.startswith("--")]
    system = (ROOT / "guide_v2.md").read_text(encoding="utf-8") + SHAPE
    (ROOT / "llm").mkdir(exist_ok=True)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool:
        for msg in pool.map(lambda a: one(base, model, tag, system, think, a), arts):
            print(f"{tag} {msg}", flush=True)


if __name__ == "__main__":
    main()
