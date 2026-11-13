"""Europe PMC hit counts per canonical task (counts only: pageSize=1, nothing downloaded).

Query per task: name + distinctive aliases (>= 5 chars, or containing task/test/paradigm), quoted, OR-ed, searched in
title/abstract/keywords... via the default field set. Three counts: any, + fMRI terms, + fMRI and open-access full text.

usage: python -I epmc_counts.py task_inventory_v3.json canonical_tasks.jsonl out.jsonl
"""
import json
import sys
import time
import urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FMRI = '(fMRI OR "functional magnetic resonance" OR "functional MRI")'


def count(q):
    url = API + "?" + urllib.parse.urlencode({"query": q, "format": "json", "pageSize": 1, "resultType": "idlist"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)["hitCount"]
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    return None


def terms(t):
    out = [t["name"]]
    for a in t.get("aliases", []):
        if len(a) >= 5 or any(w in a.lower() for w in ("task", "test", "paradigm")):
            out.append(a)
    seen, res = set(), []
    for x in out:
        x = x.replace('"', "").strip()
        if x and x.lower() not in seen:
            seen.add(x.lower()); res.append(x)
    return res


def main():
    inv = json.load(open(sys.argv[1]))
    used = {json.loads(l)["canonical"]: json.loads(l) for l in open(sys.argv[2])}
    with open(sys.argv[3], "w") as f:
        for t in inv["tasks"]:
            if t["id"] not in used or t["family"] == "pseudo_tasks":
                continue
            ts = terms(t)
            base = "(" + " OR ".join(f'"{x}"' for x in ts) + ")"
            row = {"canonical": t["id"], "source": t["source"], "terms": ts, "ns_pond_papers": used[t["id"]]["papers"],
                   "any": count(base), "fmri": count(f"{base} AND {FMRI}"),
                   "fmri_oa_ft": count(f"{base} AND {FMRI} AND OPEN_ACCESS:y AND HAS_FT:y")}
            f.write(json.dumps(row) + "\n"); f.flush()
            time.sleep(0.3)


if __name__ == "__main__":
    main()
