"""Contrast mentions across the claim catalog's full texts -> hub contrasts (contrast hub, claim route at scale).

Reads every article of the construct-claims catalog (corpus_papers/<article>.json: targets, units with text and
section), keeps units with explicit comparison syntax (PREFILTER, the design note's rule, which covers 6% of claim
sentences), tags condition A and B with the contrast tagger (contrast_tagger.py weights), normalizes the pair in the
family of each of the article's target tasks and looks it up in the hub (match_claims.match_one). Units that are
also construct claims (preds/corpus_claims*.jsonl, by article and unit) carry their constructs, so a matched one is a
contrast-level construct claim.

Output: one JSON line per tagged unit: article, i, section, a, b, tasks, match, constructs (claims only).

usage: python scan_contrasts.py PAPERS_DIR CLAIMS_GLOB TAGGER_WEIGHTS HUB_DIR OUT.jsonl [LEXICON.json] [SHARD NSHARDS]
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import hub as H  # noqa: E402
import match_claims as M  # noqa: E402
from contrast_tagger import MAXLEN, decode  # noqa: E402

PREFILTER = re.compile(r"\s>\s|[a-z]\s?>\s?[a-z]|\bvs\.?\s|\bversus\b|\bminus\b|\bsubtract|\brelative to\b|"
                       r"\bcompared (?:to|with)\b|(?<!in )\bcontrast(?:ed|ing|s)?\b", re.I)
SECTIONS = {"results", "methods", "discussion", "introduction", "abstract"}


def main():
    papers, claims_glob, weights, hub_dir, out = sys.argv[1:6]
    lex = H.Lexicon(sys.argv[6] if len(sys.argv) > 6 and sys.argv[6] != "-" else None)
    shard, nshards = (int(sys.argv[7]), int(sys.argv[8])) if len(sys.argv) > 8 else (0, 1)
    hub = H.Hub.load(f"{hub_dir}/hub.json")
    claims = {}
    for f in glob.glob(claims_glob):
        for line in open(f):
            r = json.loads(line); claims.setdefault((r["article"], r["i"]), []).extend(c for c, _ in r["constructs"])
    tok = AutoTokenizer.from_pretrained(weights)
    model = AutoModelForTokenClassification.from_pretrained(weights).cuda().eval().half()
    files = sorted(os.listdir(papers))[shard::nshards]
    with open(out, "w") as fo:
        buf = []

        def flush():
            if not buf:
                return
            enc = tok([u["text"] for u in buf], truncation=True, max_length=MAXLEN, padding=True,
                      return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping")
            with torch.no_grad():
                pred = model(**{k: v.cuda() for k, v in enc.items()}).logits.argmax(-1).cpu().tolist()
            for b, u in enumerate(buf):
                got = decode(pred[b], offs[b].tolist(), u["text"])
                if "A" not in got:
                    continue
                c = {"tasks": u["tasks"], "a": got["A"], "b": got.get("B")}
                m = M.match_one(c, hub, lex)
                rec = {"article": u["article"], "i": u["i"], "section": u["section"], "a": got["A"], "b": got.get("B"),
                       "tasks": u["tasks"], "match": m}
                k = (u["article"], u["i"])
                if k in claims:
                    rec["constructs"] = sorted(set(claims[k]))
                fo.write(json.dumps(rec) + "\n")
            buf.clear()

        for n, f in enumerate(files):
            d = json.load(open(os.path.join(papers, f)))
            for u in d["units"]:
                if u.get("section") in SECTIONS and PREFILTER.search(u["text"]):
                    buf.append({"article": d["article"], "i": u["i"], "section": u["section"], "text": u["text"],
                                "tasks": d.get("targets") or []})
                    if len(buf) >= 128:
                        flush()
            if n % 5000 == 0:
                print(f"shard {shard}: {n}/{len(files)} papers", flush=True)
        flush()
    print(f"shard {shard}: done", flush=True)


if __name__ == "__main__":
    main()
