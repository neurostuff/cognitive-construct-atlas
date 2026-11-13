"""ScienceDirect hit counts per task, before searching: full text (ns-pond's default for ScienceDirect) and title,
for the paradigm-naming query (query_strict) of every task, through ns-pond's query parser and ScienceDirect client.
One request per count (1 s apart, the client's pace). Writes out/sd_counts.tsv: canonical, fulltext, title.

usage: python sd_counts.py   (from the work dir, run/env.sh sourced; ELSEVIER_API_KEY in the environment)
"""
import json
import os
import time

from ingestion_workflow.clients.sciencedirect import ScienceDirectClient
from ingestion_workflow.services.query_dialect import Years, parse, to_sciencedirect

client = ScienceDirectClient(os.environ["ELSEVIER_API_KEY"])
years = Years.parse("1960:2026")
with open("out/sd_counts.tsv", "w") as f:
    f.write("canonical\tfulltext\ttitle\n")
    for r in map(json.loads, open("out/queries.jsonl")):
        q = r.get("query_strict")
        if not q:
            continue
        n = []
        for text in (q, f"title:({q})"):
            try:
                n.append(client.count(to_sciencedirect(parse(text, years))))
            except Exception as e:  # a query ScienceDirect cannot say is reported, not fatal
                n.append(f"error: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(1.0)
        f.write(f"{r['canonical']}\t{n[0]}\t{n[1]}\n"); f.flush()
print("done")
