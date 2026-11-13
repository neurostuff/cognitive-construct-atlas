"""The imaging papers' own full texts as claim-chain papers, so the claim chain and the contrast scan can read what each
paper says its contrasts isolate ("the incongruent > congruent contrast indexes conflict monitoring").

Papers: every paper with a hub contrast (CONTRASTS.json: the brain-map sample and the rest of ns-pond's corpus). Each
paper's text is the full text its latest ok `passages` artifact read (ns-pond's chosen source), split into units with
claim_chain's own prep (`_prep_one`); its targets are the inventory tasks of its contrasts. Files are
`OUT_DIR/ns_<ns-pond article id>.json`, the prefix keeping them apart from the construct catalog's ids. The ns-pond
catalog is opened read-only.

These units feed construct evidence only: a paper's own text naming its own contrast is not an independent attestation,
so they never go into attest.py.

usage: (pilot-results/, venv-cc) python code/contrast_hub/imaging_prep.py CATALOG_DIR CONTRASTS.json OUT_DIR [WORKERS]
"""
import collections
import gzip
import json
import sqlite3
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import claim_chain as cc  # noqa: E402


def main():
    cat, contrasts, out = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    ck = cc.checker()
    tasks = collections.defaultdict(set)
    for c in json.load(open(contrasts)).values():
        tasks[c["paper"]].update(t for t in c["tasks"].split(",") if t in ck.names)
    con = sqlite3.connect(f"file:{cat}/catalog.sqlite?mode=ro", uri=True)
    art = {v: a for a, v in con.execute("select article_id, value from aliases where kind='neurostore'") if v in tasks}
    want = set(art.values())
    path = {}                                          # ordered by time: the latest passages artifact wins
    for a, b in con.execute("select article_id, blob from artifacts where stage='passages' and status='ok' order by updated_at"):
        if b and a in want:
            path[a] = b
    text = {}
    for a, b in path.items():
        if a in want:
            try:
                p = json.load(gzip.open(Path(cat) / "blobs" / b[:2] / f"{b}.json.gz")).get("full_text_path")
            except Exception:  # noqa: BLE001
                p = None
            if p and Path(p).exists():
                text[a] = p
    Path(out).mkdir(parents=True, exist_ok=True)
    done = {f.stem for f in Path(out).glob("*.json")}
    jobs = [(f"ns_{a}", tasks[p], text[a], out, ck.names) for p, a in art.items()
            if a in text and tasks[p] and f"ns_{a}" not in done]
    print(f"{len(tasks)} papers, {len(art)} in the catalog, {len(text)} with a text, {len(jobs)} to prepare", flush=True)
    errors = 0
    with Pool(workers) as pool:
        for k, err in enumerate(pool.imap_unordered(cc._prep_one, jobs, chunksize=20), 1):
            if err:
                errors += 1; print(err, flush=True)
            if k % 2000 == 0:
                print(f"prepared {k}/{len(jobs)} ({errors} errors)", flush=True)
    print(f"prep done: {len(jobs) - errors} written, {errors} errors", flush=True)


if __name__ == "__main__":
    main()
