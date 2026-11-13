"""Every Cognitive Atlas concept's relationships (KINDOF, PARTOF, with direction) from the public API, one request per
concept, 0.5 s apart. Writes cogat/relations.jsonl: {id, name, other_id, other_name, relationship, direction}.

usage: python3 cogat_relations.py   (from cc/)
"""
import json
import time
import urllib.request

concepts = json.load(open("cogat/concepts.json"))
done = set()
try:
    done = {json.loads(l)["id"] for l in open("cogat/relations.jsonl")}
except FileNotFoundError:
    pass
with open("cogat/relations.jsonl", "a") as f:
    for c in concepts:
        if c["id"] in done:
            continue
        url = f"https://www.cognitiveatlas.org/api/v-alpha/concept?id={c['id']}"
        for attempt in range(3):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=30)); break
            except Exception:
                time.sleep(5)
        else:
            continue
        d = d[0] if isinstance(d, list) else d
        rows = d.get("relationships") or []
        f.write(json.dumps({"id": c["id"], "name": c["name"], "n": len(rows)}) + "\n") if not rows else None
        for r in rows:
            f.write(json.dumps({"id": c["id"], "name": c["name"], "other_id": r.get("id"), "other_name": r.get("name"),
                                "relationship": r.get("relationship"), "direction": r.get("direction")}) + "\n")
        f.flush(); time.sleep(0.5)
print("done")
