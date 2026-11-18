"""How reliable are the construct claims imaging papers make about their own contrasts, and do sources agree?

Inputs: contrast-level construct claims (scan_contrasts.py rows that carry `constructs`) from the imaging papers' own
texts and from the construct literature, re-matched to a hub; Codex's attribution per entry. Per exact entry it pools
the constructs each source names (taxonomy ids, constructs.vocab_maps), counted in distinct articles, and reports:

- coverage: entries and sample contrasts reached by each source;
- cross-paper agreement: for entries named by 2+ imaging papers, how often two different papers name a shared construct
  (pairwise), and whether the entry has a construct named by 2+ papers (a consensus construct);
- cross-source agreement: for entries with both, whether imaging consensus / top constructs appear among the
  literature's or Codex's constructs;
- an audit sample of claim sentences (OUT_sample.json) for a reader to judge.

usage: (code/contrast_hub on beast) python imaging_reliability.py HUB_DIR LEXICON IMAGING.jsonl LIT.jsonl CODEX.json SAMPLE_CONTRASTS.json CORPUS_ASSIGN.json OUT.json
"""
import collections
import itertools
import json
import random
import sys

import hub as H
import match_claims as M
from constructs import directed, vocab_maps


def load(path, hub, lex, to_ids):
    """entry -> construct -> set of articles; plus the matched rows"""
    ev = collections.defaultdict(lambda: collections.defaultdict(set)); rows = []
    for line in open(path):
        r = json.loads(line)
        m = M.match_one({"tasks": r["tasks"], "a": r["a"], "b": r["b"]}, hub, lex)
        if "hub" not in m:
            continue
        ids = to_ids(r["constructs"])
        r["hub"] = directed(m["hub"], m["sign"]); r["construct_ids"] = sorted(ids); rows.append(r)   # per direction
        for c in ids:
            ev[r["hub"]][c].add(r["article"])
    return ev, rows


def main():
    hub_dir, lexp, imgp, litp, codexp, samplep, assignp, outp = sys.argv[1:9]
    hub = H.Hub.load(f"{hub_dir}/hub.json"); lex = H.Lexicon(lexp)
    to_ids, _, _ = vocab_maps()
    img, img_rows = load(imgp, hub, lex, to_ids)
    lit, lit_rows = load(litp, hub, lex, to_ids)
    codex = json.load(open(codexp))
    sample = set(json.load(open(samplep))); assign = json.load(open(assignp))
    R = {"imaging_rows_matched": len(img_rows), "lit_rows_matched": len(lit_rows),
         "imaging_entries": len(img), "lit_entries": len(lit)}

    papers = lambda d: set().union(*d.values()) if d else set()  # noqa: E731
    multi = {k: d for k, d in img.items() if len(papers(d)) >= 2}
    R["imaging_entries_2plus_papers"] = len(multi)
    # pairwise: two different papers on the same entry name a shared construct
    agree = tot = 0; consensus = 0
    for k, d in multi.items():
        per_paper = collections.defaultdict(set)
        for c, arts in d.items():
            for a in arts:
                per_paper[a].add(c)
        for p, q in itertools.combinations(sorted(per_paper), 2):
            tot += 1; agree += bool(per_paper[p] & per_paper[q])
        consensus += any(len(a) >= 2 for a in d.values())
    R["pairwise_paper_agreement"] = round(agree / tot, 3) if tot else None
    R["pairs"] = tot
    R["entries_with_consensus_construct"] = consensus

    def top(d):
        return max(d, key=lambda c: len(d[c])) if d else None
    both_lit = [k for k in img if k in lit]; both_cx = [k for k in img if k in codex]
    R["entries_imaging_and_lit"] = len(both_lit)
    R["imaging_top_in_lit"] = round(sum(top(img[k]) in lit[k] for k in both_lit) / len(both_lit), 3) if both_lit else None
    R["imaging_any_in_lit"] = round(sum(bool(set(img[k]) & set(lit[k])) for k in both_lit) / len(both_lit), 3) if both_lit else None
    R["entries_imaging_and_codex"] = len(both_cx)
    R["imaging_top_in_codex"] = round(sum(top(img[k]) in codex[k] for k in both_cx) / len(both_cx), 3) if both_cx else None
    R["imaging_any_in_codex"] = round(sum(bool(set(img[k]) & set(codex[k])) for k in both_cx) / len(both_cx), 3) if both_cx else None
    lc = [k for k in lit if k in codex]
    R["lit_any_in_codex"] = round(sum(bool(set(lit[k]) & set(codex[k])) for k in lc) / len(lc), 3) if lc else None
    R["entries_lit_and_codex"] = len(lc)
    # sample contrasts reached
    sk = [aid for aid in sample if aid in assign]
    R["sample_contrasts"] = len(sk)
    dk = {a: directed(assign[a]["hub"], assign[a].get("sign", 1)) for a in sk}
    R["sample_with_imaging_claim"] = sum(dk[a] in img for a in sk)
    R["sample_with_imaging_consensus"] = sum(dk[a] in img and any(len(x) >= 2 for x in img[dk[a]].values()) for a in sk)
    R["sample_with_lit_claim"] = sum(dk[a] in lit for a in sk)
    R["construct_ids_per_row"] = round(sum(len(r["construct_ids"]) for r in img_rows) / max(1, len(img_rows)), 2)
    R["rows_no_taxonomy_construct"] = sum(not r["construct_ids"] for r in img_rows)
    print(json.dumps(R, indent=1))
    random.seed(0)
    pick = random.sample([r for r in img_rows if r["construct_ids"]], min(40, len(img_rows)))
    json.dump([{k: r[k] for k in ("article", "i", "section", "a", "b", "tasks", "hub", "constructs", "construct_ids")}
               | {"lit": sorted(lit.get(r["hub"], {})), "codex": codex.get(r["hub"], [])} for r in pick],
              open(outp.replace(".json", "_sample.json"), "w"), indent=1)
    json.dump({"summary": R, "imaging": {k: {c: sorted(a) for c, a in d.items()} for k, d in img.items()}},
              open(outp, "w"))


if __name__ == "__main__":
    main()
