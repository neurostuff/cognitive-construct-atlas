"""Per-paper abbreviation dictionaries from full text (Schwartz & Hearst 2003, "A simple algorithm for identifying
abbreviation definitions in biomedical text", PSB 8:451-462).

Candidates are "long form (SF)" with SF 1-10 characters, at least one letter, starting with a letter or digit, and at
most 2 words; the long form is searched for in the min(|SF| + 5, 2|SF|) words before the parenthesis by the paper's
backward character match: each SF character, right to left, must be found in the long form, the first SF character at
the start of a word. The shortest long form that matches is kept. Also "SF (long form)" when the parenthesis holds
more words than the SF has characters (the paper's reversed case). A paper's dictionary keeps, per SF, its most
frequent long form.

A long form never crosses a sentence boundary and starts with the short form's first character.

usage: python abbrev.py DATA_DIR CORPUS.json OUT.json       (imaging papers: DATA_DIR/<base_study>/processed/*/text.txt)
       python abbrev.py catalog PAPERS_DIR OUT.json [WORKERS] (claim-catalog articles: PAPERS_DIR/<article>.json units)
"""
import collections
import glob
import json
import re
import sys

PAREN = re.compile(r"\(([^()]{1,60})\)")


def best_long_form(sf, lf):
    """Schwartz-Hearst backward match; returns the long form or None"""
    s, l = len(sf) - 1, len(lf) - 1
    while s >= 0:
        c = sf[s].lower()
        if not c.isalnum():
            s -= 1; continue
        while l >= 0 and (lf[l].lower() != c or (s == 0 and l > 0 and lf[l - 1].isalnum())):
            l -= 1
        if l < 0:
            return None
        l -= 1; s -= 1
    start = lf.rfind(" ", 0, l + 1) + 1
    return lf[start:].strip()


def valid_sf(sf):
    return (1 <= len(sf) <= 10 and len(sf.split()) <= 2 and any(ch.isalpha() for ch in sf) and sf[0].isalnum())


SENT_END = re.compile(r"[.;:!?]\s")


def pairs(text):
    for m in PAREN.finditer(text):
        inner = m.group(1).strip().split(";")[0].split(",")[0].strip()
        before = text[max(0, m.start() - 300):m.start()].rstrip()
        cut = [x.end() for x in SENT_END.finditer(before)]
        if cut:                                        # a long form never crosses a sentence boundary
            before = before[cut[-1]:]
        if valid_sf(inner):
            words = before.split()
            n = min(len(inner) + 5, 2 * len(inner))
            lf = " ".join(words[-n:])
            got = best_long_form(inner, lf)
            if got and len(got) > len(inner) and len(got.split()) <= n and got[0].lower() == inner[0].lower():
                yield inner, got
        else:                                          # "SF (long form)"
            w = before.split()
            if w and valid_sf(w[-1]) and len(inner.split()) > len(w[-1]):
                got = best_long_form(w[-1], inner)
                if got and got[0].lower() == w[-1][0].lower() and len(got.split()) <= min(len(w[-1]) + 5, 2 * len(w[-1])):
                    yield w[-1], got


def main():
    data, corpus, out = sys.argv[1:4]
    groups = sorted({d["group"] for d in json.load(open(corpus))["docs"].values()})
    res = {}
    for g in groups:
        fs = sorted(glob.glob(f"{data}/{g}/processed/*/text.txt"))
        if not fs:
            continue
        cnt = collections.defaultdict(collections.Counter)
        for f in fs[:1]:
            for sf, lf in pairs(open(f, errors="ignore").read()):
                cnt[sf][lf.lower()] += 1
        res[g] = {sf: c.most_common(1)[0][0] for sf, c in cnt.items()}
    json.dump(res, open(out, "w"))
    print(f"{len(res)} papers, {sum(len(v) for v in res.values())} abbreviations")


def _article(path):
    d = json.load(open(path))
    cnt = collections.defaultdict(collections.Counter)
    for sf, lf in pairs("\n".join(u["text"] for u in d["units"])):
        cnt[sf][lf.lower()] += 1
    return d["article"], {sf: c.most_common(1)[0][0] for sf, c in cnt.items()}


def catalog(papers, out, workers=16):
    import os
    from multiprocessing import Pool
    files = [os.path.join(papers, f) for f in sorted(os.listdir(papers))]
    res = {}
    with Pool(workers) as pool:
        for i, (a, d) in enumerate(pool.imap_unordered(_article, files, chunksize=64)):
            if d:
                res[a] = d
            if i % 10000 == 0:
                print(i, flush=True)
    json.dump(res, open(out, "w"))
    print(f"{len(res)} articles, {sum(len(v) for v in res.values())} abbreviations")


if __name__ == "__main__":
    if sys.argv[1] == "catalog":
        catalog(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 16)
    elif sys.argv[1] == "test":
        t = ("visually guided saccades (VGS) and predictive saccades (pred) were compared; the stop-signal task (SST) "
             "and an ignore-signal task (UST); 0-back (0B); functional magnetic resonance imaging (fMRI). "
             "Moreover, in the two congruity tasks, variance in size (VMconSz)")
        print(list(pairs(t)))
    else:
        main()
