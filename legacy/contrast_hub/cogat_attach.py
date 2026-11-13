"""Attach construct nodes that have no parent to the corpus hierarchy (James 2026-11-12).

The Cognitive Atlas concepts that entered a corpus through attribution or the paper readings arrive as new nodes with
no parent, so their contrasts pool with nothing and a thin node shrinks to the global mixture. For each such node the
Claude CLI (claude_extract.call) picks, from every construct node in the corpus, the one it is a kind or a part of
(right hand response execution -> response execution), or none when it is a top-level concept. An answer that is not
a node is dropped; an edge that would close a cycle is skipped.

  judge  CORPUS.json OUT.json [MODEL EFFORT]   (laptop: the claude CLI; resumable)
  apply  CORPUS_IN ATTACH.json OVERRIDES.json CORPUS_OUT   (OVERRIDES: {"overrides": {node label: parent label or ""}})
"""
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RULES = """You place cognitive concepts in a hierarchy. Each concept below has no parent yet. For each, pick from the
list of existing concepts the one it is a specific kind or a part of: the closest broader concept, not merely a related
one (speech perception -> auditory perception; right hand response execution -> response execution; visual scene
perception -> visual perception). Answer "" when it is a top-level concept with nothing broader on the list (emotion,
memory). Never answer the concept itself. Use names exactly as listed. Answer with the JSON object the schema defines,
one item per concept, with its id.

Existing concepts:
"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "parent", "reason"],
    "properties": {"id": {"type": "integer"}, "parent": {"type": "string"}, "reason": {"type": "string"}}}}}}
BATCH = 40


def orphans(H):
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"] if i.startswith("c:"))
    kids = {p for ps in H["parents"].values() for p in ps}
    return [n for n in sorted(cnt, key=lambda n: -cnt[n]) if not H["parents"].get(n) and n not in kids], cnt


def judge():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import claude_extract as cx
    corpusp, outp = sys.argv[2:4]
    model, effort = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ("claude-opus-5-5", "medium")
    H = json.load(open(corpusp)); todo, cnt = orphans(H)
    names = sorted({n[2:] for n in H["parents"] if n.startswith("c:")})
    ok = set(names)
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [n for n in todo if n not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(todo)} nodes in {len(batches)} batches", flush=True)
    system = RULES + "; ".join(names)
    bad = collections.Counter()

    def run(b):
        prompt = "\n".join(f"id {i}: {n[2:]}" for i, n in enumerate(b))
        try:
            ans, _, _ = cx.call(model, effort, system, prompt, (), schema=SCHEMA)
            return b, ans["items"]
        except Exception as exc:  # noqa: BLE001
            print("FAILED", str(exc)[:200], flush=True); return b, None
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "4"))) as pool:
        for b, res in pool.map(run, batches):
            if res is None:
                continue
            for it in res:
                if not 0 <= it["id"] < len(b):
                    continue
                n = b[it["id"]]; p = it["parent"].strip().lower()
                if p and (p not in ok or p == n[2:]):
                    bad["not a node or itself"] += 1; p = ""
                done[n] = {"parent": p, "reason": it["reason"], "contrasts": cnt[n]}
            json.dump(done, open(outp, "w"), indent=1)
    print(json.dumps({"nodes": len(done), "attached": sum(bool(v["parent"]) for v in done.values()),
                      "top level": sum(not v["parent"] for v in done.values()), "dropped answers": dict(bad)}, indent=1))


def apply():
    cin, attp, overp, cout = sys.argv[2:6]
    H = json.load(open(cin)); A = json.load(open(attp))
    over = json.load(open(overp))["overrides"] if os.path.exists(overp) else {}
    P = H["parents"]; n = collections.Counter()

    def reaches(a, b, seen):                          # is b an ancestor of a (or a itself)
        if a == b:
            return True
        for p in P.get(a, []):
            if p not in seen:
                seen.add(p)
                if reaches(p, b, seen):
                    return True
        return False
    for node, v in A.items():
        p = over.get(node[2:], v["parent"])
        if not p:
            n["top level"] += 1; continue
        pn = "c:" + p
        if pn not in P:
            n["parent not a node"] += 1; continue
        if reaches(pn, node, set()):
            n["cycle skipped"] += 1; continue
        if pn not in P[node]:
            P[node].append(pn); n["attached"] += 1
    H["attach"] = {"judgements": attp, "overrides": overp, "counts": dict(n)}
    json.dump(H, open(cout, "w")); print(json.dumps(dict(n)))


if __name__ == "__main__":
    {"judge": judge, "apply": apply}[sys.argv[1]]()
