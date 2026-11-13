"""Construct vocabulary v0: the merges that need no judgment, Cognitive Atlas links, and new terms with parents.

Tier 1, surface normalization of every construct string the tagger produced (preds/corpus_claims*.jsonl):
  lowercase, articles and possessives dropped, hyphens and slashes as spaces, British -> American spelling (a short
  list), a trailing "and/or/of/in/..." and leading "in/of/on/the" trimmed (tagger splits), the last word replaced by
  its WordNet noun lemma (NLTK; biases -> bias, stimuli -> stimulus, abilities -> ability), and "functioning" ->
  "function" (a deliberate merge, not a lemma); a string that
  is an abbreviation the papers define with one dominant long form (>= 60% of its definitions, >= 5 papers;
  vocab/abbrev_defs.tsv) becomes that long form. Strings with the same key are one entry, labelled by its most
  frequent original wording.
  A closed aside is removed ("working memory (WM) capacity" -> working memory capacity).
Dropped as not constructs: raw measures (reaction time, accuracy, ...), task names and search terms, single words
  that are neither Cognitive Atlas terms nor seen as a construct in >= 20 papers, single adjectives (WordNet: an
  adjective and never a noun, "inhibitory"), and keys of under 3 characters. A Cognitive Atlas name or alias is never
  dropped as a single word, even when it is not linked ("reward").
Synonym merges between existing entries (vocab/synonym_merges.json), into the Cognitive Atlas-linked entry, else the
  one in more papers, never between two Cognitive Atlas concepts:
  - a verb phrase into its nominal form by WordNet derivation ("inhibit their response" -> response inhibition);
  - "X of Y" and "Y X" ("inhibition of responses" / response inhibition), into the more frequent form ("return of
    fear" stays), unless "X of Y" is a Cognitive Atlas term;
  - entries linked to one Cognitive Atlas concept, into the concept's own name ("motor inhibition" -> response
    inhibition);
  - an adjective modifier and its stem noun ("attentional bias" / "attention bias"), "-iveness" / "-ivity";
  - "X, also known as / also called / termed Y" in >= SAME_MIN papers, X and Y multi-word with one head word
    ("implicit bias" / "unconscious bias"), when more papers state it than coordinate, contrast or nest the two.
  A Cognitive Atlas alias links to its concept unless >= ALIAS_APART papers coordinate or contrast the two
  ("response selection and motor execution"): the papers use them as different constructs.
Tier 2, links:
  - cogat: the key equals a Cognitive Atlas concept name or alias key (same surface normalization);
  - new: otherwise. A new entry records a parent when the text states one (vocab/relations.jsonl broader /
    specifies / narrower, >= 2 papers, the parent being a Cognitive Atlas concept or another entry), and its
    nearest Cognitive Atlas concept by bge-base-en-v1.5 cosine, as a suggestion only.
Also: abbreviations inside a phrase written with 2+ capitals are expanded the same way ("WM load", "ToM
abilities"); a list ("alerting, orienting, and executive control") becomes its parts when every part is itself a
construct key; an adjective-like part takes the list's head ("visuospatial and executive functions"), a noun or
gerund does not.
Writes vocab/vocab_v0.json (entries) and vocab/string_to_entries.json (string -> its entry keys); prints coverage.

usage: python vocab_normalize.py   (GPU for the nearest-concept suggestions; from pilot-results/)
"""
import collections
import glob
import json
import os
import re
from pathlib import Path

ROOT = Path.cwd()
V = ROOT / "vocab"
FUNC_TAIL = {"and", "or", "of", "to", "in", "the", "with", "for", "on", "by", "a", "an", "as", "from", "at", "that",
             "which", "between", "during", "while", "both", "also"}
FUNC_HEAD = {"in", "of", "on", "the", "for", "to", "and", "or", "a", "an", "by", "with", "their", "its", "his", "her",
             "our", "both", "also"}
UK = {"behaviour": "behavior", "behavioural": "behavioral", "colour": "color", "organisation": "organization",
      "generalisation": "generalization", "categorisation": "categorization", "visualisation": "visualization",
      "recognising": "recognizing", "analysing": "analyzing", "centre": "center", "programme": "program"}
MEASURES = {"reaction time", "response time", "rt", "accuracy", "error", "error rate", "performance",
            "task performance", "iq", "score", "latency", "speed", "rt variability", "commission error",
            "omission error", "hit rate", "false alarm", "d prime"}
GENERIC_TAIL = ("capacity", "performance", "process", "processing", "ability", "skill", "function", "system",
                "mechanism", "load", "demand", "deficit", "impairment", "efficiency", "resource", "network",
                "component", "level", "task", "measure", "score", "index")
NOUN_END = re.compile(r"(tion|sion|ment|ness|ity|ance|ence|ory|ing|ism|ure|cy|ogy)$")
ALIAS_APART = 3  # papers coordinating or contrasting a Cognitive Atlas alias with its concept's name
SAME_MIN = 2     # papers stating "X, also known as Y"


_LEMMA = None


def singular(w):
    """the head noun's lemma, by WordNet (NLTK): dictionary forms stay as they are (bias, status, awareness, data);
    plurals and irregulars map to their lemma (biases -> bias, stimuli -> stimulus, analyses -> analysis)"""
    global _LEMMA
    if _LEMMA is None:
        from nltk.stem import WordNetLemmatizer
        _LEMMA = WordNetLemmatizer()
    return _LEMMA.lemmatize(w, "n")


def key(s, asides=False):
    """asides=True keeps a closed parenthesis: task names, where it qualifies ("Response Inhibition (unspecified)")"""
    s = s.lower().replace("’", "'")
    if s.count("(") > s.count(")"):  # an unclosed parenthesis is a cut span: "working memory (wm"
        s = s[:s.rfind("(")]
    if not asides:
        s = re.sub(r"\s+\([^()]*\)", " ", s)  # a closed aside, mostly the abbreviation: "working memory (wm) capacity"
    s = re.sub(r"[-/]", " ", s)
    s = re.sub(r"'s\b", "", s)
    w = re.findall(r"[a-z0-9]+", s)
    w = [UK.get(x, x) for x in w]
    while w and w[0] in FUNC_HEAD:
        w = w[1:]
    while w and w[-1] in FUNC_TAIL:
        w = w[:-1]
    if not w:
        return ""
    w[-1] = "function" if w[-1] == "functioning" else singular(w[-1])
    return " ".join(w)


_WN = None


def wordnet():
    global _WN
    if _WN is None:
        from nltk.corpus import wordnet as wn
        _WN = wn
    return _WN


def wn_adjective(w):
    """w itself is a WordNet adjective (exact lemma: the lookup's morphology would read "alerting" as "alert")"""
    wn = wordnet()
    return any(l.name() == w for x in wn.synsets(w, "a") + wn.synsets(w, "s") for l in x.lemmas())


def adjective_only(w):
    """WordNet knows w as an adjective and never as a noun ("inhibitory", "attentional"): not a construct alone; a
    gerund ("alerting", "orienting") is a noun here"""
    return not w.endswith("ing") and not wordnet().synsets(w, "n") and wn_adjective(w)


ADJ_END = re.compile(r"(al|ic|ive|ory|ous|ar|ary|ual|ile)$")


def adjective_like(w):
    """a list part that takes the list's head ("visuospatial and executive functions" -> visuospatial function): a word
    WordNet reads as an adjective, or an unknown word with an adjective ending; never a gerund ("alerting, orienting,
    and executive control" names three networks)"""
    if w.endswith("ing"):
        return False
    return wn_adjective(w) or (not wordnet().synsets(w) and bool(ADJ_END.search(w)))


OBJ_DROP = {"their", "the", "a", "an", "his", "her", "its", "one", "ones", "own", "oneself", "them", "our", "your"}


def nominal_forms(k):
    """a verb phrase's nominal forms, by WordNet derivation: "inhibit their response" -> response inhibition,
    "maintaining goals" -> goal maintenance. Only for a verb WordNet never lists as a noun, with an object of 1-3 words
    and no preposition; the caller keeps a form only when it already is an entry, since derivations can be wrong
    ("ignore" -> ignorance)"""
    w = k.split()
    if len(w) < 2:
        return []
    wn = wordnet()
    singular("x")  # loads the lemmatizer
    v = _LEMMA.lemmatize(w[0], "v")
    if not wn.synsets(v, "v") or wn.synsets(v, "n"):
        return []
    obj = [x for x in w[1:] if x not in OBJ_DROP]
    if not 1 <= len(obj) <= 3 or set(obj) & FUNC_TAIL:
        return []
    obj[-1] = singular(obj[-1])
    noms = {d.name().replace("_", " ") for l in wn.lemmas(v, "v") for d in l.derivationally_related_forms()
            if d.synset().pos() == "n" and not d.name().endswith(("er", "or"))}
    return [" ".join(obj) + " " + n for n in sorted(noms)]


def pertainym_nouns(a):
    """nouns an adjective pertains to that share its stem ("attentional" -> attention, "emotional" -> emotion);
    stems that differ ("verbal" -> word, "temporal" -> time) are left out"""
    wn = wordnet()
    out = set()
    for s in wn.synsets(a, "a") + wn.synsets(a, "s"):
        for l in s.lemmas():
            if l.name() == a:
                for p in l.pertainyms() + [d for d in l.derivationally_related_forms() if d.synset().pos() == "n"]:
                    n = p.name()
                    if n != a and len(os.path.commonprefix([a, n])) >= 5:
                        out.add(n)
    return sorted(out)


CAPS_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9]*\b")
LIST_SPLIT = re.compile(r"\s*,\s*(?:and\s+|or\s+)?|\s+(?:and/or|and|or)\s+")


def key_full(s, expand):
    """key() after expanding, inside the phrase, abbreviations written with 2+ capitals that the papers define with
    one dominant long form ("WM load" -> working memory load, "ToM abilities" -> theory of mind ability); lowercase
    tokens are left alone, so ordinary words ("us", "am") are not taken for abbreviations"""
    s = CAPS_TOKEN.sub(lambda m: expand.get(m.group(0).lower(), m.group(0))
                       if sum(c.isupper() for c in m.group(0)) >= 2 else m.group(0), s)
    k = key(s)
    return expand.get(k, k)


def main():
    concepts = json.load(open(ROOT / "code" / "cogat_concepts.json"))
    # relations stated in the text (mine_relations.py), by key: "X and Y" / "X versus Y" say two terms are distinct,
    # "X, also known as Y" that they are one
    rels = [json.loads(l) for l in open(V / "relations.jsonl")]
    apart, same, hier = collections.Counter(), collections.Counter(), collections.Counter()
    for r in rels:
        pair = tuple(sorted((key(r["a"]), key(r["b"]))))
        if r["relation"] in ("coordinated", "contrast"):
            apart[pair] += r["papers"]
        elif r["relation"] == "same":
            same[pair] += r["papers"]
        elif r["relation"] in ("broader", "narrower", "specifies"):
            hier[pair] += r["papers"]  # "or" (alternatives or a renaming) counts for neither
    # a Cognitive Atlas alias links to its concept unless the papers keep the two apart (>= ALIAS_APART papers say
    # "response selection and motor execution", "anticipation and prediction"); the alias is then a term of its own
    ckey, rejected, known = {}, [], set()  # known: every Cognitive Atlas name and alias, linked or kept apart
    for c in concepts:
        n = key(c["name"])
        ckey.setdefault(n, c["id"]); known.add(n)
        for t in [a for a in (c.get("alias") or "").split(",") if a.strip()]:
            a = key(t); known.add(a)
            abbrev = t.strip().isupper() or a == "".join(x[0] for x in n.split())  # "STM": "short-term memory (STM)" is no coordination
            if a != n and not abbrev and apart[tuple(sorted((a, n)))] >= ALIAS_APART and apart[tuple(sorted((a, n)))] > same[tuple(sorted((a, n)))]:
                rejected.append((t.strip(), c["name"], apart[tuple(sorted((a, n)))])); continue
            ckey.setdefault(a, c["id"])
    cname = {c["id"]: c["name"] for c in concepts}
    inv = json.load(open(ROOT / "code" / "task_inventory_v3.json"))["tasks"]
    tasky = {key(t["name"], asides=True) for t in inv}
    for l in open(ROOT / "code" / "queries.jsonl"):
        tasky.update(key(x, asides=True) for x in json.loads(l).get("terms") or [])
    # abbreviations with one dominant long form
    defs = collections.defaultdict(collections.Counter)
    for l in list(open(V / "abbrev_defs.tsv"))[1:]:
        lf, sf, n, papers = l.rstrip("\n").split("\t")
        defs[sf.lower()][key(lf)] += int(papers)
    expand = {}
    for sf, lfs in defs.items():
        (lf, n), tot = lfs.most_common(1)[0], sum(lfs.values())
        if n >= 5 and n >= 0.6 * tot:
            expand[sf] = lf
    # strings, claims and papers
    claims = collections.Counter(); papers = collections.defaultdict(set)
    for f in sorted(glob.glob(str(ROOT / "preds" / "corpus_claims*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if r["constructs"]:
                s = r["constructs"][0][0].strip()
                claims[s] += 1; papers[s].add(r["article"])
    entries = collections.defaultdict(lambda: {"strings": collections.Counter(), "papers": set()})
    dropped = collections.Counter()
    single_ok = collections.Counter()
    for s, n in claims.items():
        k = key(s)
        if k and len(k.split()) == 1:
            single_ok[k] += len(papers[s])
    def why_not(k):
        if len(k) < 3:
            return "short"
        if k in MEASURES:
            return "measure"
        if k in tasky or re.search(r"\b(task|test|paradigm|questionnaire|scale)\b", k):
            return "task name"
        if len(k.split()) == 1 and k not in known and adjective_only(k):
            return "adjective"  # "inhibitory", left of a cut "inhibitory and executive control"
        if len(k.split()) == 1 and k not in known and (single_ok[k] < 20 or not NOUN_END.search(k)):
            return "single word"
        return None
    # pass 1: the keys of strings that are not lists; pass 2: a list ("alerting, orienting, and executive control")
    # becomes its parts when every part is such a key, else it stays one string
    single = {key_full(s, expand) for s in claims if len(LIST_SPLIT.split(s)) == 1}
    single = {k for k in single if k and not why_not(k)}
    assign = {}
    n_split = 0
    for s, n in claims.items():
        parts = [p for p in LIST_SPLIT.split(s) if p.strip() and key(p)]  # "alerting, orienting, and": two parts
        ks = [key_full(p, expand) for p in parts] if len(parts) >= 2 else []
        if ks and len(ks[-1].split()) >= 2:  # shared head: "visuospatial and executive functions"
            head = ks[-1].split()[-1]
            ks = [k + " " + head if (len(k.split()) == 1 and k not in single and k not in ckey and adjective_like(k)) else k
                  for k in ks[:-1]] + ks[-1:]
        # every part a valid construct term (the single-word rule relaxed: "orienting" is rarely claimed alone) and
        # at least one part already a construct on its own
        if (len(ks) >= 2 and len(set(ks)) == len(ks) and any(k in single for k in ks)
                and all(k and (k in single or why_not(k) in (None, "single word")) and len(k) >= 3 for k in ks)):
            assign[s] = ks; n_split += n; continue
        k = key_full(s, expand)
        why = why_not(k)
        if why:
            dropped[why] += n; continue
        assign[s] = [k]
    key0 = {s: ks[0] for s, ks in assign.items() if len(ks) == 1}  # a string's own key, before synonym merges
    # synonym merges between entries (each target must already be an entry, so nothing is invented)
    kp = collections.defaultdict(set)
    for s, ks in assign.items():
        for k in ks:
            kp[k] |= papers[s]
    K = set(kp)
    merge, why_merged = {}, collections.Counter()
    def final(k):
        seen = set()
        while k in merge and k not in seen:
            seen.add(k); k = merge[k]
        return k

    def into(a, b, why):
        """merge a and b: into the Cognitive Atlas-linked one, else the one in more papers; two different
        Cognitive Atlas concepts stay apart"""
        a, b = final(a), final(b)
        if a == b:
            return
        ca, cb = ckey.get(a), ckey.get(b)
        if ca and cb and ca != cb:
            return
        src, dst = (a, b) if (cb and not ca) or (not ca and len(kp[b]) >= len(kp[a])) else (b, a)
        merge[src] = dst; why_merged[why] += 1
    for k in sorted(K, key=lambda k: -len(kp[k])):
        if k in ckey:
            continue
        nf = [n for n in nominal_forms(k) if n in K and n != k]  # "inhibit their response" -> response inhibition
        if nf:
            n = max(nf, key=lambda n: len(kp[n]))
            merge[k] = n; why_merged["verb phrase"] += 1; continue
        m = re.fullmatch(r"(\w+(?: \w+)?) of (?:the |their |a |an )?(\w+(?: \w+)?)", k)  # "inhibition of response"
        if m:
            w = m.group(2).split(); w[-1] = singular(w[-1])
            inv_k = " ".join(w) + " " + m.group(1)
            if inv_k in K:
                into(k, inv_k, "X of Y"); continue  # into the more frequent form: "return of fear" stays
    for k in sorted(K):
        w = k.split()
        for i, x in enumerate(w[:-1]):  # "attentional bias" / "attention bias"
            for n in pertainym_nouns(x):
                v = " ".join(w[:i] + [n] + w[i + 1:])
                if v in K:
                    into(k, v, "adjective / noun modifier")
        if w[-1].endswith("iveness") and " ".join(w[:-1] + [w[-1][:-7] + "ivity"]) in K:  # impulsiveness / impulsivity
            into(k, " ".join(w[:-1] + [w[-1][:-7] + "ivity"]), "-iveness / -ivity")
    for (a, b), n in sorted(same.items()):
        # "termed" also introduces a narrower or a generic term ("executive function, termed verbal fluency",
        # "hazard perception, termed this skill"); the statements that rename are two multi-word terms with one
        # head and another modifier ("implicit bias, also called unconscious bias"), stated more often than the
        # papers coordinate, contrast or nest the two
        a, b = expand.get(a, a), expand.get(b, b)
        wa, wb = a.split(), b.split()
        if (n >= SAME_MIN and len(wa) >= 2 and len(wb) >= 2 and wa[-1] == wb[-1] and a in K and b in K
                and n > max(apart[tuple(sorted((a, b)))], hier[tuple(sorted((a, b)))])):
            into(a, b, "also known as")

    for k in sorted(K):  # entries linked to one Cognitive Atlas concept ("motor inhibition" is an alias of response inhibition)
        cid = ckey.get(k)
        if cid:
            src, dst = final(k), final(key(cname[cid]))
            if dst in K and src != dst and ckey.get(dst) == cid:
                merge[src] = dst; why_merged["same Cognitive Atlas concept"] += 1
    assign = {s: list(dict.fromkeys(final(k) for k in ks)) for s, ks in assign.items()}
    for s, ks in assign.items():
        for k in ks:
            e = entries[k]; e["strings"][s] += claims[s]; e["papers"] |= papers[s]
    # tier 2: links, parents, suggestions
    parent_votes = collections.defaultdict(collections.Counter)
    for r in rels:
        if r["papers"] < 2:
            continue
        if r["relation"] in ("broader", "specifies"):
            p, c = key(r["a"]), key(r["b"])
        elif r["relation"] == "narrower":
            p, c = key(r["b"]), key(r["a"])
        else:
            continue
        p, c = expand.get(p, p), expand.get(c, c)
        if p != c:
            parent_votes[c][p] += r["papers"]
    keys = sorted(entries, key=lambda k: -sum(entries[k]["strings"].values()))
    from sentence_transformers import SentenceTransformer
    import numpy as np
    m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    new_keys = [k for k in keys if k not in ckey]
    E = m.encode(new_keys, normalize_embeddings=True, batch_size=512) if new_keys else np.zeros((0, 768))
    C = m.encode([c["name"] for c in concepts], normalize_embeddings=True, batch_size=256)
    near = {k: (concepts[j]["id"], float(E[i] @ C[j])) for i, k in enumerate(new_keys) for j in [int(np.argmax(E[i] @ C.T))]}
    out = []
    for k in keys:
        e = entries[k]; n = sum(e["strings"].values())
        own = [x for x, _ in e["strings"].most_common() if key0.get(x) == k]  # its own wording: not a list's or a merged synonym's
        row = {"key": k, "label": re.sub(r"\s+\([^()]*\)", "", own[0] if own else k).lower(), "claims": n, "papers": len(e["papers"]),
               "variants": len(e["strings"]), "strings": e["strings"].most_common(8)}
        if k in ckey:
            row.update({"status": "cogat", "cogat_id": ckey[k], "cogat_name": cname[ckey[k]]})
        else:
            row["status"] = "new"
            # the stated direction must win clearly: >= 3 papers and >= 3x the reverse ("X such as Y" vs "Y such as X")
            par = [(p, v) for p, v in parent_votes[k].most_common()
                   if (p in ckey or p in entries) and v >= 3 and v >= 3 * parent_votes[p].get(k, 0)]
            w = k.split()
            if len(w) >= 2 and w[-1] in GENERIC_TAIL and " ".join(w[:-1]) in entries and not why_not(" ".join(w[:-1])):
                base = " ".join(w[:-1])  # "working memory capacity" is an aspect of "working memory": a child, not merged
                row["parent"] = {"key": base, "papers": 0, "cogat_id": ckey.get(base), "relation": "variant",
                                 "name": cname[ckey[base]] if base in ckey else base}
            elif par:
                p = par[0][0]
                row["parent"] = {"key": p, "papers": par[0][1], "cogat_id": ckey.get(p), "relation": "stated",
                                 "name": cname[ckey[p]] if p in ckey else p}
            cid, sim = near[k]
            row["nearest_cogat"] = {"id": cid, "name": cname[cid], "cosine": round(sim, 3)}
        out.append(row)
    json.dump(out, open(V / "vocab_v0.json", "w"), indent=0)
    json.dump(assign, open(V / "string_to_entries.json", "w"))
    tot = sum(claims.values()); kept = sum(claims[x] for x in assign)
    cog = sum(r["claims"] for r in out if r["status"] == "cogat")
    newp = sum(r["claims"] for r in out if r["status"] == "new" and "parent" in r)
    print(f"{len(claims)} distinct strings, {tot} claims -> {len(out)} entries "
          f"({sum(r['status'] == 'cogat' for r in out)} linked to {len({r['cogat_id'] for r in out if r['status'] == 'cogat'})} "
          f"Cognitive Atlas concepts, {sum(r['status'] == 'new' for r in out)} new, of which "
          f"{sum('parent' in r for r in out)} with a parent from the text)")
    print(f"lists split into their parts: {n_split} claims")
    print(f"synonym merges: {dict(why_merged)}")
    print(f"Cognitive Atlas aliases kept apart (coordinated or contrasted with the name in >= {ALIAS_APART} papers): "
          + "; ".join(f"{a} / {n} ({k}p)" for a, n, k in sorted(rejected, key=lambda x: -x[2])))
    json.dump({k: final(k) for k in merge}, open(V / "synonym_merges.json", "w"), indent=0)
    print(f"claims: kept {kept / tot:.0%} (dropped: {', '.join(f'{k} {v / tot:.1%}' for k, v in dropped.most_common())}); "
          f"of kept, linked to Cognitive Atlas {cog / kept:.0%}, new with a parent {newp / kept:.0%}")
    big = [r for r in out if r["papers"] >= 10]
    print(f"entries in >= 10 papers: {len(big)} ({sum(r['status'] == 'cogat' for r in big)} linked, "
          f"{sum(r['status'] == 'new' for r in big)} new)")
    print("largest entries:", "; ".join(f"{r['label']} [{r['status']}] {r['papers']}p {r['variants']}v" for r in out[:15]))
    nv = sum(r.get("parent", {}).get("relation") == "variant" for r in out); ns = sum(r.get("parent", {}).get("relation") == "stated" for r in out)
    print(f"new entries with a parent: {ns} stated in the text, {nv} variants of a base construct (X capacity, X load, ...)")
    withp = [r for r in out if r["status"] == "new" and r.get("parent", {}).get("relation") == "stated"][:15]
    print("largest new entries with a parent:", "; ".join(f"{r['label']} < {r['parent']['name']} ({r['papers']}p)" for r in withp))
    nop = [r for r in out if r["status"] == "new" and "parent" not in r][:15]
    print("largest new entries without a parent:", "; ".join(
        f"{r['label']} (~{r['nearest_cogat']['name']} {r['nearest_cogat']['cosine']:.2f}; {r['papers']}p)" for r in nop))


if __name__ == "__main__":
    main()
