"""Inputs for the construct-vocabulary judge.

For each construct string to judge: its counts (claims, papers, tasks), its top tasks, one example sentence, its 8
nearest frequent strings and its 5 nearest Cognitive Atlas concepts (bge-base-en-v1.5 cosine; concept text = name,
aliases and definition). Embeddings only retrieve candidates: they do not separate synonyms from sibling constructs
(response inhibition ~ inhibitory control 0.83 vs proactive ~ reactive control 0.80), so the judge decides.

Writes vocab/<name>.items.jsonl and vocab/<name>.md (the rendering a judge reads, one block per string), plus
vocab/preferred_terms.txt (Cognitive Atlas names and the 300 most frequent strings) once.

usage: python vocab_items.py corpus MIN_PAPERS        the corpus claims (preds/corpus_claims.*.jsonl)
       python vocab_items.py gold                     labeller A's construct phrases in the 36 gold papers
(GPU; from pilot-results/)
"""
import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path.cwd()
OUT = ROOT / "vocab"
ART = re.compile(r"^(the|a|an|their|its|his|her|our) ")
POS = {"own_task", "own_contrast"}


def norm(s):
    return ART.sub("", re.sub(r"\s+", " ", s.lower()).strip(" .,;:()[]\"'"))


def corpus_strings():
    claims, papers, tasks, example = collections.Counter(), collections.defaultdict(set), collections.defaultdict(
        collections.Counter), {}
    for f in sorted(glob.glob(str(ROOT / "preds" / "corpus_claims.*.jsonl"))):
        for l in open(f):
            r = json.loads(l)
            if not r["constructs"]:
                continue
            c = norm(r["constructs"][0][0])
            claims[c] += 1; papers[c].add(r["article"]); tasks[c].update(r["task"])
            if c not in example or len(r["text"]) < len(example[c]):
                example[c] = r["text"]
    return claims, papers, tasks, example


def main():
    mode = sys.argv[1]
    OUT.mkdir(exist_ok=True)
    inv = {t["id"]: t["name"] for t in json.load(open(ROOT / "code" / "task_inventory_v3.json"))["tasks"]}
    claims, papers, tasks, example = corpus_strings()
    frequent = [c for c, _ in claims.most_common() if len(papers[c]) >= 5]
    concepts = json.load(open(ROOT / "code" / "cogat_concepts.json"))
    ctext = [f"{c['name']}" + (f" ({c['alias']})" if c.get("alias") else "") + f": {c.get('definition_text') or ''}"
             for c in concepts]
    if not (OUT / "preferred_terms.txt").exists():
        top = [c for c, _ in claims.most_common(300)]
        (OUT / "preferred_terms.txt").write_text(
            "# Cognitive Atlas concept names\n" + "\n".join(sorted({c["name"] for c in concepts})) +
            "\n\n# The 300 most frequent construct strings in the corpus\n" + "\n".join(top) + "\n")
    if mode == "gold":
        items = []
        for f in sorted(glob.glob(str(ROOT / "labels" / "*.A.jsonl"))):
            a = Path(f).name.split(".")[0]
            p = json.load(open(ROOT / "pilot" / "papers" / f"{a}.json"))
            for l in open(f):
                r = json.loads(l)
                if r["category"] in POS:
                    for c, t in zip(r.get("constructs") or [], r.get("construct_terms") or []):
                        items.append({"string": norm(c), "gold_term": t, "example": p["units"][r["i"]]["text"]})
        seen, todo = set(), []
        for it in items:
            if it["string"] not in seen:
                seen.add(it["string"]); todo.append(it)
        name = "gold"
    else:
        k = int(sys.argv[2])
        todo = [{"string": c, "example": example[c]} for c in frequent if len(papers[c]) >= k]
        name = f"corpus_min{k}"
    m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    E_c = m.encode(ctext, normalize_embeddings=True, batch_size=128)
    E_f = m.encode(frequent, normalize_embeddings=True, batch_size=256)
    E_t = m.encode([it["string"] for it in todo], normalize_embeddings=True, batch_size=256)
    near_c = np.argsort(-(E_t @ E_c.T), axis=1)[:, :5]
    near_f = np.argsort(-(E_t @ E_f.T), axis=1)[:, :9]
    with open(OUT / f"{name}.items.jsonl", "w") as fi, open(OUT / f"{name}.md", "w") as fm:
        for n, it in enumerate(todo):
            s = it["string"]
            it.update({"n": n, "claims": claims.get(s, 0), "papers": len(papers.get(s, ())),
                       "tasks": [inv.get(t, t) for t, _ in tasks[s].most_common(3)] if s in tasks else [],
                       "neighbors": [frequent[j] for j in near_f[n] if frequent[j] != s][:8],
                       "cogat": [{"id": concepts[j]["id"], "name": concepts[j]["name"],
                                  "definition": (concepts[j].get("definition_text") or "")[:160]} for j in near_c[n]]})
            fi.write(json.dumps(it) + "\n")
            fm.write(f"## [{n}] {s}\n")
            if it["claims"]:
                fm.write(f"claims {it['claims']}, papers {it['papers']}; tasks: {'; '.join(it['tasks'])}\n")
            fm.write(f"example: {it['example'][:260]}\n")
            fm.write(f"nearest strings: {' | '.join(it['neighbors'])}\n")
            fm.write("Cognitive Atlas candidates:\n" + "".join(
                f"  - {c['id']} {c['name']}: {c['definition']}\n" for c in it["cogat"]) + "\n")
    print(f"{name}: {len(todo)} strings to judge -> {OUT}/{name}.md")


if __name__ == "__main__":
    main()
