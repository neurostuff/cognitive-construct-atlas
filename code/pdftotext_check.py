"""How much of the production PDF text (MinerU/Docling) and of the gold claim sentences pdftotext recovers.

For each pilot paper whose production text came from a PDF: run the clone's pdftotext engine function on the
same PDF (read-only from the production catalog), cut both texts into ev1 units, and count a unit as recovered when
its normalized form (lower-case alphanumerics) occurs in the pdftotext text's normalized form.

usage: python pdftotext_check.py PILOT_DIR ADJ_LABEL_DIR OUT_DIR   (PYTHONPATH = the ns-pond clone)
"""
import gzip
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

from ingestion_workflow.extractors.pdf_extractor import reflow_pdftotext

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot import units  # noqa: E402  (ev1's splitter)

CAT = "file:/data/alejandro/projects/ns-pond/catalog/catalog.sqlite?mode=ro"
BLOBS = Path("/data/alejandro/projects/ns-pond/catalog/blobs")


def blob(d):
    return json.loads(gzip.decompress((BLOBS / d[:2] / f"{d}.json.gz").read_bytes()))


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def main():
    pilot, labels, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CAT, uri=True)
    rows = []
    for f in sorted((pilot / "papers").glob("*.json")):
        p = json.loads(f.read_text()); art = p["article"]
        arts = {(s, src): (b, json.loads(m)) for s, src, b, m in con.execute(
            "select stage, source, blob, summary from artifacts where article_id=? and status='ok'", (art,))}
        ext = arts.get(("extract", "pdf"))
        if not ext or ext[1].get("text_sha256") != p["text_sha256"]:
            continue  # production text did not come from the PDF
        dl = arts.get(("download", "pdf"))
        if not dl:
            continue
        pdfs = [x["file_path"] for x in blob(dl[0]).get("files", []) if str(x.get("file_path", "")).endswith(".pdf")]
        if not pdfs or not Path(pdfs[0]).exists():
            continue
        raw = subprocess.run(["pdftotext", "-enc", "UTF-8", pdfs[0], "-"], capture_output=True, timeout=300).stdout
        text = reflow_pdftotext(raw.decode("utf-8", "replace"))
        (out / f"{art}.txt").write_text(text)
        hay = norm(text)
        prod = [u["text"] for u in p["units"] if u["section"] != "back" and len(u["text"]) >= 40]
        got = sum(norm(t) in hay for t in prod)
        gold = []
        lf = labels / f"{art}.ADJ.jsonl"
        for line in lf.read_text().splitlines() if lf.exists() else []:
            r = json.loads(line)
            if r["category"] in ("own_task", "own_contrast"):
                gold.append(p["units"][r["i"]]["text"])
        ggot = sum(norm(t) in hay for t in gold)
        tu = [text[a:b] for a, b in units(text)]
        rows.append({"article": art, "engine": ext[1].get("text_source"), "units_prod": len(prod),
                     "recovered": got, "gold": len(gold), "gold_recovered": ggot,
                     "units_pdftotext": len(tu), "median_unit_chars": sorted(map(len, tu))[len(tu) // 2] if tu else 0})
    for r in rows:
        print(json.dumps(r))
    tp = sum(r["recovered"] for r in rows); tn = sum(r["units_prod"] for r in rows)
    gp = sum(r["gold_recovered"] for r in rows); gn = sum(r["gold"] for r in rows)
    print(json.dumps({"papers": len(rows), "units_recovered": f"{tp}/{tn}", "gold_recovered": f"{gp}/{gn}"}))


if __name__ == "__main__":
    main()
