"""Fill in the hierarchy between construct nodes, adding only (James 2026-11-12: "only add new connections, do not
remove any; flag relationships it would like to remove").

For every construct node of a corpus, the Claude CLI (claude_extract.call) sees its current parents and the list of
all construct nodes, and answers: further parents it is a kind of (is-a), from the list; and current parents it would
remove, with a reason. Only kind-of edges are asked for: a task's or construct's map pools its descendants', and a
part-of edge pools wrongly (with part-of edges, "set shifting" once pooled every decision-making task). Additions are
checked (a node on the list, not itself, not already a parent) and applied with a cycle check; removals are only
reported.

  judge  CORPUS.json OUT.json [MODEL EFFORT]           (laptop: the claude CLI; resumable)
  apply  CORPUS_IN CONNECT.json CORPUS_OUT              adds the proposed parents; prints the removal flags' count
"""
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RULES = """You complete a hierarchy of cognitive concepts (Cognitive Atlas concepts and constructs from the literature).
A parent must be a broader concept that the child is a kind of (is-a): working memory updating is a kind of updating;
speech perception is a kind of auditory perception. Do not use part-of, cause, measure or association: encoding is a
part of memory, not a kind of it, so memory is not its parent.
For each concept below you are given its current parents. Answer:
- add: other concepts from the list that it is directly a kind of and that are missing from its parents (at most 3,
  the closest broader ones; none is a fine answer). Never the concept itself, never one of its own kinds.
- remove: current parents that are wrong (not a kind-of relation: a part, a cause, a related concept, or wrong), with
  the reason. Leave correct parents out of this list.
Use names exactly as listed. Answer with the JSON object the schema defines, one item per concept, with its id.

Concepts:
"""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "add", "remove"],
    "properties": {"id": {"type": "integer"},
                   "add": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                   "remove": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["parent", "reason"],
                                                         "properties": {"parent": {"type": "string"}, "reason": {"type": "string"}}}}}}}}}
BATCH = 40


def judge():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import claude_extract as cx
    corpusp, outp = sys.argv[2:4]
    model, effort = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ("claude-opus-5-5", "medium")
    H = json.load(open(corpusp)); P = H["parents"]
    nodes = sorted(n for n in P if n.startswith("c:"))
    names = {n[2:] for n in nodes}
    cnt = collections.Counter(i for d in H["docs"].values() for i in d["ids"] if i.startswith("c:"))
    done = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [n for n in sorted(nodes, key=lambda n: -cnt[n]) if n not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(todo)} nodes in {len(batches)} batches", flush=True)
    system = RULES + "; ".join(sorted(names))
    bad = collections.Counter()

    def run(b):
        prompt = "\n".join(f"id {i}: {n[2:]} (current parents: {', '.join(p[2:] for p in P[n] if p.startswith('c:')) or 'none'})"
                           for i, n in enumerate(b))
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
                n = b[it["id"]]; cur = {p[2:] for p in P[n]}
                add = []
                for a in it["add"]:
                    a = a.strip().lower()
                    if a in names and a != n[2:] and a not in cur and a not in add:
                        add.append(a)
                    else:
                        bad["addition not a new node on the list"] += 1
                rem = [r for r in it["remove"] if r["parent"].strip().lower() in cur]
                done[n] = {"add": add, "remove": rem, "contrasts": cnt[n]}
            json.dump(done, open(outp, "w"), indent=1)
    print(json.dumps({"nodes": len(done), "nodes with additions": sum(bool(v["add"]) for v in done.values()),
                      "edges proposed": sum(len(v["add"]) for v in done.values()),
                      "removal flags": sum(len(v["remove"]) for v in done.values()), "dropped": dict(bad)}, indent=1))


def apply():
    cin, conp, cout = sys.argv[2:5]
    H = json.load(open(cin)); C = json.load(open(conp)); P = H["parents"]; n = collections.Counter()

    def reaches(a, b, seen):                          # is b a (or an ancestor of a)
        if a == b:
            return True
        for p in P.get(a, []):
            if p not in seen:
                seen.add(p)
                if reaches(p, b, seen):
                    return True
        return False
    for node, v in C.items():
        if node not in P:
            continue
        for a in v["add"]:
            pa = "c:" + a
            if pa not in P or pa in P[node]:
                n["skipped (absent or present)"] += 1; continue
            if reaches(pa, node, set()):
                n["cycle skipped"] += 1; continue
            P[node].append(pa); n["edges added"] += 1
        n["removal flags (not applied)"] += len(v["remove"])
    H["connect"] = {"judgements": conp, "counts": dict(n)}
    json.dump(H, open(cout, "w")); print(json.dumps(dict(n)))


if __name__ == "__main__":
    {"judge": judge, "apply": apply}[sys.argv[1]]()
