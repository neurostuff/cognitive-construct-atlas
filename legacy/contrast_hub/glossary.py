"""Per-paper condition glossary from ns-pond's record artifacts (read-only), for the corpus papers.

For each corpus paper (base_study id), finds its ns-pond article (aliases kind 'neurostore'), reads the newest `record`
artifact blob (gzipped JSON, study_schema record) and keeps, per task: the task name and description, and each
condition's name, description, kind and stimulus content; and every analysis's name and definition. The catalog is
opened read-only (sqlite URI mode=ro); nothing is written in the ns-pond tree.

usage: python glossary.py CATALOG_DIR CORPUS.json OUT.json
"""
import gzip
import json
import sqlite3
import sys
from pathlib import Path


def v(x):
    return x.get("value") if isinstance(x, dict) else x


def text(x):
    x = v(x)
    return "; ".join(map(str, x)) if isinstance(x, list) else (str(x) if x is not None else "")


def main():
    cat, corpus, out = sys.argv[1:4]
    groups = sorted({d["group"] for d in json.load(open(corpus))["docs"].values()})
    c = sqlite3.connect(f"file:{cat}/catalog.sqlite?mode=ro", uri=True)
    res, miss = {}, 0
    for g in groups:
        row = c.execute("select article_id from aliases where kind='neurostore' and value=?", (g,)).fetchone()
        if not row:
            miss += 1; continue
        b = c.execute("select blob from artifacts where article_id=? and stage='record' and status='ok' order by updated_at desc",
                      (row[0],)).fetchone()
        if not b or not b[0]:
            miss += 1; continue
        p = Path(cat) / "blobs" / b[0][:2] / f"{b[0]}.json.gz"
        try:
            r = json.load(gzip.open(p))
        except Exception:  # noqa: BLE001
            miss += 1; continue
        tasks = []
        for t in r.get("tasks") or []:
            if not isinstance(t, dict):
                continue
            tasks.append({"name": text(t.get("name")), "description": text(t.get("description"))[:400],
                          "conditions": [{"name": text(x.get("name")), "description": text(x.get("description"))[:300],
                                          "kind": text(x.get("condition_kind")),
                                          "stimulus": text(x.get("stimulus_content"))[:120]}
                                         for x in t.get("conditions") or [] if isinstance(x, dict)]})
        res[g] = {"article": row[0], "tasks": tasks,
                  "analyses": [{"name": text(a.get("name")), "definition": text(a.get("definition"))[:300]}
                               for a in r.get("analyses") or [] if isinstance(a, dict)]}
    json.dump(res, open(out, "w"))
    print(f"{len(res)} papers with a record, {miss} without")


if __name__ == "__main__":
    main()
