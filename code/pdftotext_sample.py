"""Sentence recovery of pdftotext against MinerU text on random production PDFs (read-only on the ns-pond catalog).

For N random articles whose production extract text came from MinerU: cut the MinerU text into ev1 units (>= 40
chars, before the References heading), run pdftotext + the clone's reflow on the same PDF, and report the share of
units whose normalized form occurs in pdftotext's normalized text. Engines to compare: plain pdftotext, and
pdftotext -layout? No: -layout keeps columns side by side, so only plain mode is measured.

usage: python pdftotext_sample.py N SEED OUT_JSONL   (PYTHONPATH = the ns-pond clone)
"""
import gzip
import json
import random
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from ingestion_workflow.extractors.pdf_extractor import reflow_pdftotext

sys.path.insert(0, str(Path(__file__).parent))
from build_pilot import units  # noqa: E402

CAT = "file:/data/alejandro/projects/ns-pond/catalog/catalog.sqlite?mode=ro"
BLOBS = Path("/data/alejandro/projects/ns-pond/catalog/blobs")
REFS = re.compile(r"^(?:#+\s*)?(?:references|bibliography)\s*$", re.I | re.M)


def blob(d):
    return json.loads(gzip.decompress((BLOBS / d[:2] / f"{d}.json.gz").read_bytes()))


def norm(s):
    """lower-case alphanumerics after NFKC, so ligatures (U+FB01 "fi") compare equal to their letters"""
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", s).lower())


def main():
    n, seed, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    con = sqlite3.connect(CAT, uri=True)
    cand = [a for (a,) in con.execute(
        "select article_id from artifacts where stage='extract' and source='pdf' and status='ok' "
        "and json_extract(summary,'$.has_text') = 1")]
    random.Random(seed).shuffle(cand)
    rows = []
    for art in cand:
        if len(rows) == n:
            break
        arts = {(s, src): b for s, src, b in con.execute(
            "select stage, source, blob from artifacts where article_id=? and status='ok'", (art,))}
        try:
            ext = blob(arts[("extract", "pdf")]); dl = blob(arts[("download", "pdf")])
        except (KeyError, OSError):
            continue
        if (ext.get("text_source") or {}).get("method") != "mineru" or not ext.get("full_text_path"):
            continue
        pdfs = [x["file_path"] for x in dl.get("files", []) if str(x.get("file_path", "")).endswith(".pdf")]
        if not pdfs or not Path(pdfs[0]).exists() or not Path(ext["full_text_path"]).exists():
            continue
        ref = Path(ext["full_text_path"]).read_text(encoding="utf-8", errors="replace")
        m = REFS.search(ref); ref = ref[:m.start()] if m else ref
        t0 = time.time()
        raw = subprocess.run(["pdftotext", "-enc", "UTF-8", pdfs[0], "-"], capture_output=True, timeout=300).stdout
        text = reflow_pdftotext(raw.decode("utf-8", "replace"))
        dt = time.time() - t0
        prose = [ref[a:b] for a, b in units(ref) if b - a >= 40 and "\t" not in ref[a:b] and "|" not in ref[a:b]]
        hay = norm(text)
        got = sum(norm(u) in hay for u in prose)
        rows.append({"article": art, "units": len(prose), "recovered": got,
                     "share": round(got / len(prose), 3) if prose else None, "seconds": round(dt, 2),
                     "pdf_mb": round(Path(pdfs[0]).stat().st_size / 1e6, 2)})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    sh = sorted(r["share"] for r in rows if r["share"] is not None)
    print(json.dumps({"papers": len(rows), "micro": round(sum(r["recovered"] for r in rows) / sum(r["units"] for r in rows), 3),
                      "median": sh[len(sh) // 2], "p10": sh[len(sh) // 10], "below_0.6": sum(s < 0.6 for s in sh),
                      "below_0.8": sum(s < 0.8 for s in sh),
                      "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2)}))


if __name__ == "__main__":
    main()
