"""How many candidate evidence sentences does each directed hub entry have, before any stance check?

A candidate is a sentence where the contrast tagger found two sides (scan_contrasts.py rows, construct literature and
imaging papers) that match_one maps to a hub entry, and that names a taxonomy construct outside the two side spans
(vocab/string_to_entries.json -> taxonomy ids, matched on word n-grams). It is what a local stance checker would
read; the claim pool constructs.py uses today is the subset whose row also carries extracted `constructs`.

Reports, per directed entry and for the sample's analyses: candidate sentences and articles; constructs named in 2+
articles' candidates (the ceiling of a pooled evidence tier, before the stance check removes ruled-out, task-level
and unrelated mentions). Writes the candidates (OUT_candidates.jsonl) for stance labelling.

usage: (code/contrast_hub on beast) python retrieval_yield.py HUB_DIR LEXICON 'SCAN_GLOB[+..]' SAMPLE.json OUT.json [WORKERS]
"""
import collections
import glob
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import hub as H
import match_claims as M
from constructs import directed, vocab_maps

ROOT = Path(__file__).resolve().parents[2]
TOK = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
MAX_N = 6
G = {}


def norm(s):
    return " ".join(TOK.findall(s.lower()))


def init(hub_dir, lexp):
    G["hub"] = H.Hub.load(f"{hub_dir}/hub.json"); G["lex"] = H.Lexicon(lexp)
    to_ids, _, _ = vocab_maps()
    s2e = json.load(open(ROOT / "vocab" / "string_to_entries.json"))
    terms = {}
    for s in s2e:
        k = norm(s)
        if k and len(k) >= 4 and len(k.split()) <= MAX_N:
            ids = to_ids([s])
            if ids:
                terms.setdefault(k, set()).update(ids)
    G["terms"] = terms


def constructs_in(text, sides):
    t = " " + norm(text) + " "
    for s in sides:                                 # a side named "working memory task" is not a claim about it
        s = norm(s or "")
        if s:
            t = t.replace(" " + s + " ", " | ")
    w = t.split(); found = {}
    for n in range(MAX_N, 0, -1):
        for i in range(len(w) - n + 1):
            k = " ".join(w[i:i + n])
            if "|" not in k and k in G["terms"]:
                for c in G["terms"][k]:
                    found.setdefault(c, k)
    return found


def units(article, cache):
    if article not in cache:
        cache.clear()                               # rows come grouped by article
        d = ROOT / ("imaging_papers" if article.startswith("ns_") else "corpus_papers")
        try:
            u = json.load(open(d / f"{article}.json"))["units"]
            cache[article] = {x.get("i", j): x["text"] for j, x in enumerate(u)}
        except Exception:  # noqa: BLE001
            cache[article] = {}
    return cache[article]


def scan_file(f):
    out = []; n = collections.Counter(); cache = {}
    src = "imaging" if "imaging" in Path(f).name else "lit"
    for line in open(f):
        r = json.loads(line); n["rows"] += 1
        m = M.match_one({"tasks": r["tasks"], "a": r["a"], "b": r["b"]}, G["hub"], G["lex"])
        if "hub" not in m:
            continue
        n["matched"] += 1
        text = units(r["article"], cache).get(r["i"], "")
        if not text:
            n["no_text"] += 1; continue
        found = constructs_in(text, (r["a"], r["b"]))
        if not found:
            continue
        n["candidates"] += 1
        out.append({"key": directed(m["hub"], m["sign"]), "how": m["how"], "article": r["article"], "i": r["i"],
                    "section": r.get("section"), "src": src, "a": r["a"], "b": r["b"], "claim": "constructs" in r,
                    "constructs": found, "text": text})
    return f, out, n


def main():
    hub_dir, lexp, scan_glob, samplep, outp = sys.argv[1:6]
    workers = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    files = sorted(f for pat in scan_glob.split("+") for f in glob.glob(pat))
    assign = json.load(open(f"{hub_dir}/corpus_assign.json")); sample = json.load(open(samplep))
    per = collections.defaultdict(lambda: collections.defaultdict(set))      # key -> construct -> articles
    sent = collections.defaultdict(set); n = collections.Counter()
    with Pool(workers, initializer=init, initargs=(hub_dir, lexp)) as pool, \
            open(outp.replace(".json", "_candidates.jsonl"), "w") as fo:
        for f, out, c in pool.imap_unordered(scan_file, files):
            n.update(c); print(f, dict(c), flush=True)
            for r in out:
                fo.write(json.dumps(r) + "\n")
                sent[r["key"]].add(r["article"])
                n[f"cand_{r['src']}"] += 1; n["cand_with_claim_field"] += r["claim"]
                for cid in r["constructs"]:
                    per[r["key"]][cid].add(r["article"])
    pooled = {k: {c: len(a) for c, a in d.items() if len(a) >= 2} for k, d in per.items()}
    pooled = {k: v for k, v in pooled.items() if v}
    sk = [a for a in sample if a in assign]
    dk = {a: directed(assign[a]["hub"], assign[a].get("sign", 1)) for a in sk}
    R = {"scan": dict(n), "entries_with_candidates": len(sent), "entries_with_2plus_articles": sum(len(v) >= 2 for v in sent.values()),
         "entries_with_pooled_construct": len(pooled),
         "sample": len(sample), "sample_mapped": len(sk),
         "sample_with_candidate": sum(dk[a] in sent for a in sk),
         "sample_with_candidates_2plus_articles": sum(len(sent.get(dk[a], ())) >= 2 for a in sk),
         "sample_with_pooled_construct": sum(dk[a] in pooled for a in sk),
         "sample_distinct_entries": len(set(dk.values())),
         "sample_entries_with_pooled_construct": len({dk[a] for a in sk} & set(pooled))}
    print(json.dumps(R, indent=1))
    json.dump({"summary": R, "pooled": pooled, "articles": {k: len(v) for k, v in sent.items()}}, open(outp, "w"))


if __name__ == "__main__":
    main()
