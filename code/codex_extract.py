"""Construct-claim extraction through `codex exec` on the `codex login` account (pondie's CodexCaller recipe).

One call per paper: the rules (guide_v2.md, plus a short note on the JSON shape) as the system prompt, the rendered
paper as the prompt, a strict output schema. Codex's tools are switched off, no user config or rules are read, and
each call runs in an empty directory with API-key variables removed, so the account (not an API key) is billed.
The answer is written as llm/<article>.<tag>.jsonl, the format score.py reads.

usage: python codex_extract.py MODEL EFFORT TAG ARTICLE [ARTICLE ...]   (run from cc/)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
# pondie/extraction/llm.py _CODEX_TOOLS and _CODEX_EXEC_FLAGS (codex-cli 0.161)
TOOLS = ("apps", "browser_use", "browser_use_external", "computer_use", "goals", "hooks", "image_generation",
         "in_app_browser", "multi_agent", "plugins", "shell_tool", "skill_search", "sleep_tool", "tool_suggest",
         "unified_exec", "view_image")
FLAGS = ("--json", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only", "--ignore-user-config",
         "--ignore-rules")
API_KEY_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL")

SHAPE = """

## Answer shape for this call

Answer with the JSON object the output schema defines: `participants` (the paper line above) and `claims`, one entry
per listed unit with the fields above. Use null for a field that does not apply (`task` for other_task/region,
`contrast_text` outside own_contrast) and [] for an empty list."""

CATEGORIES = ["own_task", "own_contrast", "label_usage", "other_task", "region"]
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["participants", "claims"],
    "properties": {
        "participants": {"type": "string", "enum": ["human", "non-human", "both", "none"]},
        "claims": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["i", "category", "task", "analyses", "contrast_text", "constructs", "construct_terms",
                         "polarity"],
            "properties": {
                "i": {"type": "integer"},
                "category": {"type": "string", "enum": CATEGORIES},
                "task": {"type": ["string", "null"]},
                "analyses": {"type": "array", "items": {"type": "string"}},
                "contrast_text": {"type": ["string", "null"]},
                "constructs": {"type": "array", "items": {"type": "string"}},
                "construct_terms": {"type": "array", "items": {"type": "string"}},
                "polarity": {"type": "string", "enum": ["asserted", "hedged", "negated"]},
            }}},
    },
}


def known_features():
    listed = subprocess.run(["codex", "features", "list"], capture_output=True, text=True, check=True).stdout
    return {line.split()[0] for line in listed.splitlines() if line.strip()}


def call(model, effort, system, prompt, disabled, schema=None):
    env = {k: v for k, v in os.environ.items() if k not in API_KEY_ENV}
    with tempfile.TemporaryDirectory(prefix="cc-codex-") as work:
        (Path(work) / "instructions.md").write_text(system, encoding="utf-8")
        (Path(work) / "schema.json").write_text(json.dumps(schema or SCHEMA), encoding="utf-8")
        config = {"model_reasoning_effort": json.dumps(effort),
                  "model_instructions_file": json.dumps(str(Path(work) / "instructions.md")),
                  "include_environment_context": "false", "include_permissions_instructions": "false",
                  "include_apps_instructions": "false", "web_search": '"disabled"'}
        args = ["codex", "exec", *FLAGS, "-m", model,
                *(a for k, v in config.items() for a in ("-c", f"{k}={v}")),
                *(a for name in disabled for a in ("--disable", name)),
                "--output-schema", str(Path(work) / "schema.json"), "-"]
        t0 = time.time()
        done = subprocess.run(args, input=prompt, capture_output=True, text=True, cwd=work, env=env, timeout=3600)
    body, usage, failure = "", None, ""
    for line in done.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
            body = ev["item"].get("text", "")
        elif ev.get("type") == "turn.completed":
            usage = ev.get("usage") or {}
        elif ev.get("type") in ("turn.failed", "error"):
            failure = failure or json.dumps(ev)[:400]
    if usage is None:
        raise RuntimeError(failure or done.stderr.strip()[-400:] or f"exit {done.returncode}")
    return json.loads(body), usage, round(time.time() - t0, 1)


def one(article, model, effort, tag, system, disabled):
    out = ROOT / "llm" / f"{article}.{tag}.jsonl"
    if out.exists() and out.stat().st_size:
        return article, "cached", None
    prompt = (ROOT / "rendered" / f"{article}.txt").read_text(encoding="utf-8")
    try:
        answer, usage, secs = call(model, effort, system, prompt, disabled)
    except Exception as exc:  # noqa: BLE001
        return article, f"FAILED {exc}", None
    with open(out, "w") as f:
        f.write(json.dumps({"paper": {"participants": answer["participants"]}}) + "\n")
        for c in answer["claims"]:
            f.write(json.dumps(c) + "\n")
    with open(ROOT / "llm" / f"usage.{tag}.jsonl", "a") as f:
        f.write(json.dumps({"article": article, "seconds": secs, **usage}) + "\n")
    return article, f"{len(answer['claims'])} lines, {secs}s, in {usage.get('input_tokens')} out {usage.get('output_tokens')}", usage


def main():
    model, effort, tag, *articles = sys.argv[1:]
    system = (ROOT / "guide_v2.md").read_text(encoding="utf-8") + SHAPE
    disabled = [t for t in TOOLS if t in known_features()]
    workers = int(os.environ.get("WORKERS", "4"))
    with ThreadPoolExecutor(workers) as pool:
        for art, msg, _ in pool.map(lambda a: one(a, model, effort, tag, system, disabled), articles):
            print(f"{tag} {art}: {msg}", flush=True)


if __name__ == "__main__":
    main()
