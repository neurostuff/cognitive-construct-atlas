"""Human-only download manifest for the construct-claims catalog (James 2026-11-10: human literature only).

An article is left out when PubMed indexes it as animal-only: one of its PMIDs (its own, or the alternative and
preprint PMIDs ns-pond files in the catalog's aliases) matches `animals[mh] NOT humans[mh]` (PubMed's standard
species filter). Articles with no PMID of any kind, and PMIDs not yet MeSH-indexed, are kept.
Each PMID is checked once; verdicts are cached in out/species/pmid_animal_only.tsv.

Only articles a kept task search found are candidates (search_hits), so articles registered by a search that was
stopped before ns-pond kept it are never downloaded. A (search, provider) is dropped when Semantic Scholar or OpenAlex
returned more than max(2,000, 5 x PubMed's records for that task) -- a query that matched another field's
literature -- or when it is listed in run/bad_hits.tsv. Writes data/article_tasks.jsonl: each candidate article's
tasks and the providers that found it.

usage: PUBMED_API_KEY=... python -I species.py CATALOG_SQLITE OUT_MANIFEST
"""
import json
import os
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

KEY = os.environ.get("PUBMED_API_KEY")
CACHE = Path("out/species/pmid_animal_only.tsv")
CHUNK = 300
FILTER = "(animals[mh] NOT humans[mh])"


def animal_only(pmids):
    """the subset of `pmids` PubMed indexes as animal-only"""
    term = "(" + " OR ".join(f"{p}[uid]" for p in pmids) + f") AND {FILTER}"
    data = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": len(pmids), "retmode": "json",
                                   **({"api_key": KEY} if KEY else {})}).encode()
    for attempt in range(5):
        try:
            req = urllib.request.Request("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", data=data)
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.loads(r.read().decode("utf-8", "replace"), strict=False)["esearchresult"]
            return set(d["idlist"])
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
    raise RuntimeError("PubMed esearch failed 5 times")


GUARD_MIN, GUARD_RATIO = 2000, 5


def searches(con):
    """kept (search_id, provider) pairs, the task of each search, and the pairs the guard or run/bad_hits.tsv drops"""
    task_of = {}
    for r in map(json.loads, open("out/queries.jsonl")):
        for f in ("query", "query_strict"):
            if r.get(f):
                task_of.setdefault(r[f], r["canonical"])
    rows = [(sid, q, json.loads(p)) for sid, q, p in con.execute("select id, query, providers from searches")]
    pubmed = {}
    for sid, q, prov in rows:
        if "pubmed" in prov and q in task_of:
            pubmed[task_of[q]] = pubmed.get(task_of[q], 0) + prov["pubmed"]["records"]
    bad = set()
    if Path("run/bad_hits.tsv").exists():
        for line in Path("run/bad_hits.tsv").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                sid, provider = line.split("\t")[:2]; bad.add((int(sid), provider))
    flagged = []
    for sid, q, prov in rows:
        task = task_of.get(q)
        for provider in ("semantic_scholar", "openalex"):
            n = (prov.get(provider) or {}).get("records")
            if n is not None and n > max(GUARD_MIN, GUARD_RATIO * pubmed.get(task, 0)):
                bad.add((sid, provider)); flagged.append((sid, provider, task, n, pubmed.get(task, 0)))
    return {sid: task_of.get(q) for sid, q, _ in rows}, bad, flagged


def main():
    con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
    task_of_search, bad, flagged = searches(con)
    found = {}
    for sid, provider, art in con.execute("select search_id, provider, article_id from search_hits"):
        if (sid, provider) not in bad:
            found.setdefault(art, set()).add((task_of_search.get(sid), provider))
    rows = con.execute("select a.id, al.kind, al.value from articles a left join aliases al on al.article_id = a.id "
                       "where a.merged_into is null").fetchall()
    con.close()
    rows = [r for r in rows if r[0] in found]
    ids = {}  # every article, with each alias kind it holds (pmid, doi, pmcid, preprint_*, alternative_*)
    for art, kind, value in rows:
        d = ids.setdefault(art, {})
        if kind:
            d.setdefault(kind, []).append(value)
    ids = {a: {k: (v[0] if k in ("pmid", "doi", "pmcid") else sorted(v)) for k, v in d.items()} for a, d in ids.items()}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    seen = {}
    if CACHE.exists():
        for line in CACHE.read_text().splitlines():
            p, v = line.split("\t"); seen[p] = v == "1"
    def pmids(d):
        return ([d["pmid"]] if d.get("pmid") else []) + d.get("alternative_pmid", []) + d.get("preprint_pmid", [])
    todo = sorted({p for d in ids.values() for p in pmids(d) if p not in seen})
    with open(CACHE, "a") as f:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            hit = animal_only(chunk)
            for p in chunk:
                seen[p] = p in hit
                f.write(f"{p}\t{int(p in hit)}\n")
            f.flush()
            time.sleep(0.12 if KEY else 0.4)
    kept = dropped = no_pmid = 0
    with open(sys.argv[2], "w") as out:
        for art, d in sorted(ids.items()):
            if any(seen.get(p) for p in pmids(d)):
                dropped += 1
                continue
            no_pmid += not pmids(d)
            kept += 1
            row = {"neurostore": None, "pmid": d.get("pmid"), "doi": d.get("doi"), "pmcid": d.get("pmcid"),
                   "other_ids": {}}
            for kind in ("preprint_doi", "preprint_pmid", "preprint_pmcid", "alternative_doi", "alternative_pmid",
                         "alternative_pmcid"):
                if d.get(kind):
                    row[kind + "s"] = d[kind]
            unknown = set(d) - {"pmid", "doi", "pmcid", "preprint_doi", "preprint_pmid", "preprint_pmcid",
                                "alternative_doi", "alternative_pmid", "alternative_pmcid"}
            if unknown:
                raise SystemExit(f"alias kinds the manifest cannot carry: {sorted(unknown)} (article {art})")
            out.write(json.dumps(row) + "\n")
    with open(Path(sys.argv[2]).parents[1] / "article_tasks.jsonl", "w") as out:
        for art in sorted(found):
            if art in ids:
                out.write(json.dumps({"article": art, "found": sorted([list(x) for x in found[art]])}) + "\n")
    print(json.dumps({"candidates": len(ids), "checked_now": len(todo), "animal_only_dropped": dropped,
                      "kept": kept, "kept_without_pmid": no_pmid, "dropped_searches": sorted(bad),
                      "guard_flagged": flagged}))


if __name__ == "__main__":
    main()
