"""Synonym pairs the literature defines itself: "long form (ABBR)" in the corpus texts.

Short form: 2-10 characters, starts with a letter, at least two capitals or one capital plus a digit (WM, SSRT, ToM,
IC). Long form: the shortest run of the words before the parenthesis whose initials cover the short form's letters
in order, Schwartz & Hearst (2003) style (first letter must start the long form; at most len(SF) + 5 words).
Kept when the long form, normalized, is a construct string the tagger produced somewhere in the corpus.

Pairs: (long form, short form) per definition; and two different long forms defined with the same short form in
papers of the same task are recorded as long-form pairs, with the number of papers each was seen in.
Writes vocab/abbrev_defs.tsv and vocab/abbrev_pairs.jsonl.

usage: python mine_abbrevs.py WORKERS   (from pilot-results/)
"""
import collections
import glob
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path.cwd()
PAREN = re.compile(r"\(\s*([A-Za-z][A-Za-z0-9-]{1,9})\s*[),;]")
WORD = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
ART = re.compile(r"^(the|a|an|their|its|his|her|our) ")


def norm(s):
    return ART.sub("", re.sub(r"\s+", " ", s.lower()).strip(" .,;:()[]\"'"))


def is_short(sf):
    caps = sum(c.isupper() for c in sf)
    return caps >= 2 or (caps >= 1 and any(c.isdigit() for c in sf))


def long_form(before, sf):
    """the shortest suffix of `before` (a list of words) whose initials match sf's letters in order"""
    letters = [c.lower() for c in sf if c.isalpha()]
    for n in range(1, min(len(before), len(letters) + 5) + 1):
        cand = before[-n:]
        if cand[0][0].lower() != letters[0]:
            continue
        initials = "".join(w[0].lower() for w in cand)
        it = iter(initials + "".join(w[1:].lower() for w in cand))
        if all(any(ch == x for x in it) for ch in letters) and sum(w[0].lower() in letters for w in cand) >= min(2, len(letters)):
            return " ".join(cand)
    return None


def one(path):
    p = json.loads(Path(path).read_text())
    out = []
    for u in p["units"]:
        t = u["text"]
        for m in PAREN.finditer(t):
            sf = m.group(1)
            if not is_short(sf):
                continue
            before = WORD.findall(t[:m.start()])[-15:]
            lf = long_form(before, sf) if before else None
            if lf:
                out.append((p["article"], tuple(p["targets"]), norm(lf), sf))
    return out


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    constructs = set()
    for f in glob.glob(str(ROOT / "preds" / "corpus_claims.*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            constructs.update(norm(c) for c, _ in r["constructs"])
    files = sorted(glob.glob(str(ROOT / "corpus_papers" / "*.json")))
    defs = collections.Counter(); papers = collections.defaultdict(set); by_sf_task = collections.defaultdict(collections.Counter)
    with Pool(workers) as pool:
        for rows in pool.imap_unordered(one, files, chunksize=50):
            for a, targets, lf, sf in rows:
                if lf not in constructs:
                    continue
                defs[(lf, sf)] += 1; papers[(lf, sf)].add(a)
                for t in targets:
                    by_sf_task[(sf.lower(), t)][lf] += 1
    with open(ROOT / "vocab" / "abbrev_defs.tsv", "w") as f:
        f.write("long_form\tshort_form\tdefinitions\tpapers\n")
        for (lf, sf), n in defs.most_common():
            f.write(f"{lf}\t{sf}\t{n}\t{len(papers[(lf, sf)])}\n")
    pairs = collections.Counter()
    for (sf, t), lfs in by_sf_task.items():
        common = [lf for lf, n in lfs.items() if n >= 2]
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                pairs[tuple(sorted((common[i], common[j])))] += 1
    with open(ROOT / "vocab" / "abbrev_pairs.jsonl", "w") as f:
        for (lf, sf), n in defs.items():
            if len(papers[(lf, sf)]) >= 2:
                f.write(json.dumps({"a": lf, "b": sf, "kind": "defines", "papers": len(papers[(lf, sf)])}) + "\n")
        for (a, b), n in pairs.items():
            f.write(json.dumps({"a": a, "b": b, "kind": "same_abbreviation", "tasks": n}) + "\n")
    print(f"{len(files)} papers; {len(defs)} construct definitions ({sum(1 for k in defs if len(papers[k]) >= 2)} in 2+ papers); "
          f"{len(pairs)} long-form pairs sharing an abbreviation within a task")


if __name__ == "__main__":
    main()
