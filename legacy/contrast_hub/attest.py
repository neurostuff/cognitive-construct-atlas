"""Literature attestation: how many articles of the claim catalog name each contrast signature in their text.

Every unit of the full-text scan (scan_contrasts.py: sentences with comparison syntax, condition spans from the
contrast tagger) is read like a claim contrast, within the article's task (as hub entries are): each span with that article's own abbreviations spelled out
(abbrev.py catalog; never another article's), then hub.side_from_text in each of the article's target task families,
oriented, as a signature, and put in its canonical direction with the hub's frozen directions (hub.canonical). The
article sets are unioned per canonical signature, so a signature named as A > B by one article and B > A by another
counts both. hub.py build reads the counts: an imaging signature that one corpus paper reports and 2+ articles name
is shared vocabulary too (level L0a).

usage: python attest.py 'SCAN_GLOB' LEXICON.json ABBREV_CATALOG.json DIRECTIONS.json OUT.json
"""
import collections
import glob
import json
import sys

import hub as H
import imaging


def main():
    pattern, lexp, abbrp, dirp, out = sys.argv[1:6]
    lex = H.Lexicon(lexp)
    abbr = json.load(open(abbrp)); frozen = set(json.load(open(dirp)))
    arts = collections.defaultdict(set); n = 0
    for f in sorted(glob.glob(pattern)):
        for line in open(f):
            r = json.loads(line); n += 1
            ab = abbr.get(r["article"], {})
            a = imaging.expand(r["a"], ab); b = imaging.expand(r["b"], ab) if r["b"] else None
            for tid in r["tasks"] or []:
                fam = H.TASK_FAMILY.get(tid)
                if not fam:
                    continue
                A = H.side_from_text(a, tid, fam, lex)
                B = H.side_from_text(b, tid, fam, lex) if b else None
                if B is not None and B["kind"] == "not_a_condition":
                    B = None
                T, B2, _ = H.orient(A, B)
                sig = H.scoped(H.signature(fam, T, B2), tid)    # counted within the article's task
                if sig:
                    c, _ = H.canonical(sig, {}, frozen)
                    arts[H.sig_str(c)].add(r["article"])
    res = {k: len(v) for k, v in arts.items()}
    json.dump(res, open(out, "w"))
    print(f"{n} units, {len(res)} canonical signatures, {sum(v >= 2 for v in res.values())} named by 2+ articles")


if __name__ == "__main__":
    main()
