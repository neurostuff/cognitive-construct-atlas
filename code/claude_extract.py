"""Claude through the Claude Code CLI as a drop-in for codex_extract.call (James 2026-11-12: "call claude via cli like
you do codex"), so codex_attribute and the other Codex steps can run on Claude when Codex's quota is out.

`call(model, effort, system, prompt, disabled, schema)` runs `claude -p` with the system prompt replaced, no tools,
no MCP servers, no settings files and no saved session, the prompt on stdin and the answer constrained by
`--json-schema`; it returns (answer, usage, seconds) as codex_extract.call does. `disabled` is accepted and ignored
(Codex feature flags). TOOLS and known_features() mirror codex_extract's, empty.
"""
import json
import subprocess
import tempfile
import time
from pathlib import Path

TOOLS = []


def known_features():
    return set()


def call(model, effort, system, prompt, disabled=(), schema=None):
    with tempfile.TemporaryDirectory(prefix="cc-claude-") as work:
        (Path(work) / "system.md").write_text(system, encoding="utf-8")
        args = ["claude", "-p", "--model", model, "--effort", effort, "--output-format", "json",
                "--system-prompt-file", str(Path(work) / "system.md"), "--tools", "", "--strict-mcp-config",
                "--setting-sources", "", "--no-session-persistence", "--json-schema", json.dumps(schema)]
        t0 = time.time()
        done = subprocess.run(args, input=prompt, capture_output=True, text=True, cwd=work, timeout=3600)
    try:
        out = json.loads(done.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(done.stderr.strip()[-400:] or done.stdout.strip()[-400:] or f"exit {done.returncode}")
    if out.get("is_error") or out.get("subtype") not in (None, "success"):
        raise RuntimeError(json.dumps({k: out.get(k) for k in ("subtype", "result", "is_error")})[:400])
    ans = out.get("structured_output")
    if ans is None:
        ans = json.loads(out.get("result") or "")
    return ans, out.get("usage") or {}, round(time.time() - t0, 1)
