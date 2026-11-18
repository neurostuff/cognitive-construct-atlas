"""ns-pond's sectionizer (services.paper_parse.sections / section_at) against the pilot's heading regex.

For each pilot paper: its production text (the extract whose sha256 is the record's text_sha256, read-only), ev1 units
with character offsets, the section ns-pond assigns each unit start, and the pilot's own tag. Prints agreement, the
ns-pond kinds seen, and where gold claims fall under each.

usage: python sections_check.py PILOT_DIR LABEL_DIR   (PYTHONPATH = the ns-pond clone)
"""
import collections
import gzip
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from ingestion_workflow.services.paper_parse import section_at, sections

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot import units  # noqa: E402

CAT = "file:/data/alejandro/projects/ns-pond/catalog/catalog.sqlite?mode=ro"
BLOBS = Path("/data/alejandro/projects/ns-pond/catalog/blobs")
POS = {"own_task", "own_contrast"}


def blob(d):
    return json.loads(gzip.decompress((BLOBS / d[:2] / f"{d}.json.gz").read_bytes()))


def text_of(con, art, sha):
    for b, m in con.execute("select blob, summary from artifacts where article_id=? and stage='extract' and status='ok'", (art,)):
        if json.loads(m).get("text_sha256") == sha:
            t = Path(blob(b)["full_text_path"]).read_text(encoding="utf-8")
            if hashlib.sha256(t.encode()).hexdigest() == sha:
                return t
    return None


def main():
    pilot, labels = Path(sys.argv[1]), Path(sys.argv[2])
    con = sqlite3.connect(CAT, uri=True)
    pair, kinds, claim_kind, no_sections = collections.Counter(), collections.Counter(), collections.Counter(), 0
    for f in sorted((pilot / "papers").glob("*.json")):
        p = json.loads(f.read_text()); a = p["article"]
        t = text_of(con, a, p["text_sha256"])
        if t is None:
            continue
        U = units(t)
        assert [t[x:y] for x, y in U] == [u["text"] for u in p["units"]], a
        spans = sections(t)
        no_sections += spans is None
        gold = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(labels / f"{a}.ADJ2.jsonl")}
        for u, (x, _) in zip(p["units"], U):
            s = section_at(spans, x)
            k = s.kind.value if s is not None and hasattr(s.kind, "value") else (str(s.kind) if s is not None else "none")
            kinds[k] += 1
            pair[(u["section"], k)] += 1
            if gold.get(u["i"]) in POS:
                claim_kind[(u["section"], k)] += 1
    print("papers without any section:", no_sections)
    print("ns-pond section kinds over all units:", dict(kinds.most_common()))
    print("pilot tag -> ns-pond kind (units):")
    for (mine, theirs), n in sorted(pair.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {mine:13s} -> {theirs:20s} {n}")
    print("gold claims, pilot tag -> ns-pond kind:")
    for (mine, theirs), n in sorted(claim_kind.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {mine:13s} -> {theirs:20s} {n}")


if __name__ == "__main__":
    main()
