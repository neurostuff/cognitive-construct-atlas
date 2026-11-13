"""Inventory of normalized tasks and contrasts in the production ns-pond catalog (read-only).

Reads every `normalize` artifact (ns-27b) from the catalog, opens its blob (the record with
`normalized` sidecars) and writes one row per task and one per analysis.

usage: python -I inventory.py OUTDIR [WORKERS]
"""
import gzip
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

CAT = "file:/data/alejandro/projects/ns-pond/catalog/catalog.sqlite?mode=ro"
BLOBS = Path("/data/alejandro/projects/ns-pond/catalog/blobs")


def blob(digest):
    return json.loads(gzip.decompress((BLOBS / digest[:2] / f"{digest}.json.gz").read_bytes()))


def norm(slot, vocab_prefix):
    for e in (slot or {}).get("normalized") or []:
        if e.get("vocabulary", "").startswith(vocab_prefix):
            yield e


def val(slot):
    return (slot or {}).get("value")


def one(row):
    art, digest, fit = row
    try:
        r = blob(digest)
    except Exception as exc:  # noqa: BLE001
        return art, None, None, repr(exc)
    tasks, analyses = [], []
    for t in r.get("tasks") or []:
        canon = next(norm(t.get("name"), "astra canonical task"), {})
        fam = next(norm(t.get("name"), "astra task family"), {})
        tasks.append({
            "article": art, "task": t.get("local_id"), "name": val(t.get("name")),
            "canonical": canon.get("value"), "canonical_status": canon.get("status"),
            "family": fam.get("value"), "method": canon.get("method"), "confidence": canon.get("confidence"),
            "conditions": [val(c.get("name")) for c in t.get("conditions") or []],
        })
    for a in r.get("analyses") or []:
        key = next(norm(a.get("name"), "astra contrast key"), {})
        cset = next(norm(a.get("name"), "ns-pond contrast set"), {})
        analyses.append({
            "article": art, "analysis": a.get("local_id"), "name": val(a.get("name")),
            "tasks": a.get("tasks") or [], "contrast_key": key.get("value"),
            "set": cset.get("value") if cset.get("status") == "mapped" else None,
            "set_label": cset.get("label"), "set_method": cset.get("method"),
        })
    return art, tasks, analyses, fit


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    con = sqlite3.connect(CAT, uri=True)
    rows = con.execute(
        "select a.article_id, a.blob, json_extract(a.summary,'$.fit') from artifacts a join articles r on r.id=a.article_id "
        "where a.stage='normalize' and a.status='ok' and r.merged_into is null and a.blob is not null").fetchall()
    con.close()
    fits, errors = Counter(), []
    with Pool(workers) as pool, open(out / "tasks.jsonl", "w") as ft, open(out / "analyses.jsonl", "w") as fa:
        for art, tasks, analyses, fit in pool.imap_unordered(one, rows, chunksize=64):
            if tasks is None:
                errors.append((art, fit)); continue
            fits[fit] += 1
            for t in tasks: ft.write(json.dumps(t) + "\n")
            for a in analyses: fa.write(json.dumps(a) + "\n")
    json.dump({"articles": len(rows), "fits": fits, "errors": errors[:50], "n_errors": len(errors)},
              open(out / "inventory_meta.json", "w"), indent=1)
    print(len(rows), fits, len(errors))


if __name__ == "__main__":
    main()
