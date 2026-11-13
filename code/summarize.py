"""Per-canonical-task and per-contrast-set counts from inventory.py output.

usage: python -I summarize.py INVDIR TASK_INVENTORY_JSON
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

inv = Path(sys.argv[1])
ti = json.load(open(sys.argv[2]))
meta = {t["id"]: t for t in ti["tasks"]}

tasks = [json.loads(l) for l in open(inv / "tasks.jsonl")]
by_task = {(t["article"], t["task"]): t for t in tasks}
status = Counter(t["canonical_status"] for t in tasks)
method = Counter(t["method"] for t in tasks)
canon = defaultdict(lambda: {"tasks": 0, "papers": set(), "analyses": 0, "sets": Counter(), "set_papers": defaultdict(set)})
fam_only = Counter()
for t in tasks:
    if t["canonical"]:
        c = canon[t["canonical"]]; c["tasks"] += 1; c["papers"].add(t["article"])
    elif t["family"]:
        fam_only[t["family"]] += 1

n_an = 0; an_set = 0; an_task = 0; an_canon = 0; set_labels = {}; set_total = Counter()
for line in open(inv / "analyses.jsonl"):
    a = json.loads(line); n_an += 1
    if a["set"]:
        an_set += 1; set_labels[a["set"]] = a["set_label"]; set_total[a["set"]] += 1
    if a["tasks"]:
        an_task += 1
    for tid in a["tasks"][:1]:
        t = by_task.get((a["article"], tid))
        if t and t["canonical"]:
            an_canon += 1
            c = canon[t["canonical"]]; c["analyses"] += 1
            if a["set"]:
                c["sets"][a["set"]] += 1; c["set_papers"][a["set"]].add(a["article"])

rows = []
for cid, c in canon.items():
    m = meta.get(cid, {})
    cw = (m.get("crosswalk") or {}).get("cognitive_atlas") or []
    top = sorted(c["sets"].items(), key=lambda kv: -len(c["set_papers"][kv[0]]))
    rows.append({
        "canonical": cid, "name": m.get("name"), "family": m.get("family"), "source": m.get("source"),
        "tasks": c["tasks"], "papers": len(c["papers"]), "analyses": c["analyses"],
        "sets": len(c["sets"]), "sets_ge3_papers": sum(1 for s in c["sets"] if len(c["set_papers"][s]) >= 3),
        "cogat": [x["id"] for x in cw], "cogat_match": [x.get("match") for x in cw],
        "top_sets": [(s, set_labels[s], n, len(c["set_papers"][s])) for s, n in top[:8]],
    })
rows.sort(key=lambda r: -r["papers"])
summary = {
    "task_rows": len(tasks), "articles_with_tasks": len({t["article"] for t in tasks}),
    "canonical_status": status, "method": method, "canonical_ids_used": len(canon),
    "inventory_ids": len(meta), "family_only_tasks": sum(fam_only.values()),
    "analyses": n_an, "analyses_with_set": an_set, "analyses_with_task": an_task,
    "analyses_with_canonical_task": an_canon, "sets_used": len(set_total),
    "sets_ge3_papers_any": None,
}
json.dump(summary, open(inv / "summary.json", "w"), indent=1, default=list)
with open(inv / "canonical_tasks.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(json.dumps(summary, indent=1, default=list))
for r in rows[:60]:
    print(f'{r["papers"]:6d} {r["tasks"]:6d} {r["analyses"]:6d} {r["sets"]:4d} {r["sets_ge3_papers"]:4d} {r["family"]:32s} {r["canonical"]:45s} cogat={len(r["cogat"])}')
print("...")
for r in rows[-25:]:
    print(f'{r["papers"]:6d} {r["tasks"]:6d} {r["analyses"]:6d} {r["sets"]:4d} {r["sets_ge3_papers"]:4d} {r["family"]:32s} {r["canonical"]:45s} cogat={len(r["cogat"])}')
