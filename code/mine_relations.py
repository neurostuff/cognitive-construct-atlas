"""Relations between construct terms that authors state in the text itself (no model, no LLM).

Lexicon: Cognitive Atlas concept names and aliases, plus the multi-word construct strings in >= 5 corpus papers
(vocab/corpus_min5.items.jsonl; corpus single words such as "positive" or "mechanisms" are mostly tagger noise),
lowercased; minus fragments (ending in a function word, or starting with one), comparatives ("more efficient"),
task names and search terms, and raw measures (reaction time, accuracy, errors, ...). Each unit of every corpus paper is lowercased and
its lexicon mentions found greedily (longest first, up to 7 words). For consecutive mentions m1, m2, the text
between them decides the relation:
- same:      "also/otherwise known/referred to/termed/called (as)"; abbreviation definitions come from
             mine_abbrevs.py
- specifies: "i.e.", "(i.e.": m1 made specific as m2 (often a narrower term, sometimes a renaming; the items of a
             list after it are specified too)
- broader:   m1 broader than m2 (and than the list that follows m2): "such as", "including", "e.g.", "especially",
             "particularly", "notably", "for example"
- narrower:  m1 narrower than m2: "and other", "or other", "is a component/form/type/kind/aspect/facet/subtype of",
             ", a component/form/... of", "as part of"
- contrast:  "versus", "vs", "rather than", "as opposed to", "compared with"
- coordinated: "and", "as well as", "and/or", "," (lists); items of a broader-list are coordinated too
- or:          "or", ", or": alternatives or a renaming ("mentalizing, or theory of mind"), so kept apart from
               "coordinated", which says two terms are distinct
Counts are papers, not sentences. Writes vocab/relations.jsonl: {a, b, relation, papers, sentences, example}.

usage: python mine_relations.py WORKERS   (from pilot-results/)
"""
import collections
import glob
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path.cwd()
FUNC = {"and", "or", "of", "to", "in", "the", "with", "for", "on", "by", "a", "an", "as", "from", "at", "that", "which"}
MEASURES = {"reaction time", "reaction times", "response time", "response times", "rt", "rts", "accuracy", "errors",
            "error rate", "error rates", "performance", "task performance", "iq", "scores", "latency", "latencies"}
COMPAR = {"more", "less", "greater", "better", "worse", "increased", "decreased", "reduced", "higher", "lower", "poor",
          "poorer", "good", "enhanced", "impaired", "improved", "larger", "smaller", "faster", "slower", "efficient"}
TOK = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\sa-z0-9]")
SAME = re.compile(r"^\s*,?\s*(?:\(\s*)?(?:(?:also|otherwise|sometimes|often)\s+)?(?:known|referred\s+to|termed|called|labell?ed|denoted)(?:\s+as)?\s*$")
IE = re.compile(r"^\s*,?\s*\(?\s*i\s*\.\s*e\s*\.\s*,?\s*$")
BROADER = re.compile(r"^\s*,?\s*\(?\s*(?:such\s+as|including|e\s*\.\s*g\s*\.\s*,?|especially|particularly|notably|for\s+example\s*,?)\s*$")
NARROWER = re.compile(r"^\s*,?\s*(?:and|or)\s+other\s*$|^\s*,?\s*(?:is\s+|as\s+)?(?:a|an|one)\s+(?:core\s+|key\s+|central\s+|major\s+)?"
                      r"(?:component|form|type|kind|aspect|facet|subtype|subcomponent|dimension|part|subprocess|subset)\s+of\s*$|^\s*as\s+part\s+of\s*$")
CONTRAST = re.compile(r"^\s*,?\s*(?:versus|vs\s*\.?|rather\s+than|as\s+opposed\s+to|compared\s+(?:with|to)|in\s+contrast\s+to)\s*$")
COORD = re.compile(r"^\s*(?:,\s*)?(?:and|or|and\s*/\s*or|as\s+well\s+as)?\s*$|^\s*,\s*$")
LEX = {}
MAXN = 7


def load_lexicon():
    inv = json.load(open(ROOT / "code" / "task_inventory_v3.json"))["tasks"]
    tasky = {t["name"].lower() for t in inv}
    for l in open(ROOT / "code" / "queries.jsonl"):
        tasky.update(x.lower() for x in json.loads(l).get("terms") or [])
    cogat = set()
    for c in json.load(open(ROOT / "code" / "cogat_concepts.json")):
        cogat.add(c["name"].lower().strip())
        cogat.update(a.strip().lower() for a in (c.get("alias") or "").split(",") if a.strip())
    lex = set(cogat)
    for l in open(ROOT / "vocab" / "corpus_min5.items.jsonl"):
        s = json.loads(l)["string"]
        if len(s.split()) >= 2:  # single words only from Cognitive Atlas: corpus single words are mostly noise
            lex.add(s)
    out = set()
    for s in lex:
        w = s.split()
        if not w or w[0] in FUNC or w[-1] in FUNC or s in MEASURES or s in tasky or len(s) < 3 or w[0] in COMPAR:
            continue
        if re.search(r"\b(task|test|paradigm|scale|questionnaire|inventory)\b", s):
            continue
        out.add(" ".join(TOK.findall(s)))
    return out


def init(lex):
    global LEX
    LEX = lex


def mentions(toks):
    out, i = [], 0
    while i < len(toks):
        hit = None
        for n in range(min(MAXN, len(toks) - i), 0, -1):
            s = " ".join(toks[i:i + n])
            if s in LEX:
                hit = (i, i + n, s); break
        if hit:
            out.append(hit); i = hit[1]
        else:
            i += 1
    return out


def one(path):
    p = json.loads(Path(path).read_text())
    found = []
    for u in p["units"]:
        if u["section"] == "back":
            continue
        toks = TOK.findall(u["text"].lower())
        ms = mentions(toks)
        if len(ms) < 2:
            continue
        head = None  # the broader term while inside its list ("X such as Y, Z and W")
        for k, ((s1, e1, a), (s2, e2, b)) in enumerate(zip(ms, ms[1:])):
            gap = " ".join(toks[e1:s2])
            gap = re.sub(r"\s+([,.;:)])", r"\1", gap).replace("( ", "(")
            rel = None
            if SAME.match(gap):
                rel = ("same", a, b)
            elif IE.match(gap):
                # "i.e." specifies rather than renames in this literature ("executive functions (i.e., working
                # memory)", "inhibitory control (i.e., response inhibition)"); before a list it names components
                rel = ("specifies", a, b); head = a
            elif BROADER.match(gap):
                rel = ("broader", a, b); head = a
            elif NARROWER.match(gap):
                rel = ("narrower", a, b)
            elif CONTRAST.match(gap):
                rel = ("contrast", a, b)
            elif COORD.match(gap) and e2 - s1 < 40:
                rel = ("or" if re.fullmatch(r"\s*,?\s*or\s*", gap) else "coordinated", a, b)
                if head:
                    found.append(("specifies" if IE.match(" ".join(toks[ms[max(0, k - 1)][1]:s1]) or "") else "broader",
                                  head, b, u["text"]))
            else:
                head = None
            if rel and rel[1] != rel[2]:
                found.append((*rel, u["text"]))
    return p["article"], found


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    lex = load_lexicon()
    files = sorted(glob.glob(str(ROOT / "corpus_papers" / "*.json")))
    papers = collections.defaultdict(set); sent = collections.Counter(); ex = {}
    with Pool(workers, initializer=init, initargs=(lex,)) as pool:
        for art, rows in pool.imap_unordered(one, files, chunksize=50):
            for rel, a, b, text in rows:
                if rel in ("coordinated", "or", "contrast", "same"):
                    a, b = sorted((a, b))
                key = (a, b, rel)
                papers[key].add(art); sent[key] += 1
                if key not in ex or len(text) < len(ex[key]):
                    ex[key] = text
    with open(ROOT / "vocab" / "relations.jsonl", "w") as f:
        for key, ps in sorted(papers.items(), key=lambda kv: -len(kv[1])):
            a, b, rel = key
            f.write(json.dumps({"a": a, "b": b, "relation": rel, "papers": len(ps), "sentences": sent[key],
                                "example": ex[key][:300]}) + "\n")
    by = collections.Counter(k[2] for k in papers)
    by2 = collections.Counter(k[2] for k, ps in papers.items() if len(ps) >= 2)
    print(f"lexicon {len(lex)} terms; {len(files)} papers; pairs by relation: {dict(by)}; in 2+ papers: {dict(by2)}")


if __name__ == "__main__":
    main()
