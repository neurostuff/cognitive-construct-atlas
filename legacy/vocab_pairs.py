"""Candidate construct-term pairs for relation labelling (training and testing a construct embedder).

Strings: construct strings in >= 5 corpus papers (vocab/corpus_min5.items.jsonl). Held out: a random 12% of them
(seed 0); every pair touching a held-out string is a test pair, so the test measures new terms too.
Per string: its 3 nearest other strings and its nearest Cognitive Atlas concept (bge-base-en-v1.5), plus one random
string from its 20-60th neighbours (related-or-not material). Pairs are deduplicated and shuffled.
Writes vocab/pairs.jsonl ({k, a, b, split, source, bge}) and vocab/pairs_<part>.md (one line per pair) in PARTS parts.

usage: python vocab_pairs.py PARTS   (GPU; from pilot-results/)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path.cwd()
V = ROOT / "vocab"


def main():
    parts = int(sys.argv[1])
    items = [json.loads(l) for l in open(V / "corpus_min5.items.jsonl")]
    strings = [it["string"] for it in items]
    concepts = json.load(open(ROOT / "code" / "cogat_concepts.json"))
    cnames = [c["name"] for c in concepts]
    rng = random.Random(0)
    held = set(rng.sample(strings, int(0.12 * len(strings))))
    m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    E = m.encode(strings, normalize_embeddings=True, batch_size=256)
    C = m.encode(cnames, normalize_embeddings=True, batch_size=256)
    S = E @ E.T; np.fill_diagonal(S, -1); SC = E @ C.T
    pairs = {}
    def add(a, b, src, sim):
        key = tuple(sorted((a, b)))
        if a != b and key not in pairs:
            pairs[key] = {"a": key[0], "b": key[1], "source": src, "bge": round(float(sim), 3),
                          "split": "test" if (key[0] in held or key[1] in held) else "train"}
    for i, s in enumerate(strings):
        order = np.argsort(-S[i])
        for j in order[:3]:
            add(s, strings[j], "neighbor", S[i, j])
        j = order[rng.randrange(20, 60)]
        add(s, strings[j], "mid", S[i, j])
        c = int(np.argmax(SC[i]))
        if cnames[c].lower() != s:
            add(s, cnames[c], "cogat", SC[i, c])
    out = list(pairs.values()); rng.shuffle(out)
    for k, p in enumerate(out):
        p["k"] = k
    with open(V / "pairs.jsonl", "w") as f:
        for p in out:
            f.write(json.dumps(p) + "\n")
    size = -(-len(out) // parts)
    for n in range(parts):
        chunk = out[n * size:(n + 1) * size]
        (V / f"pairs_{n}.md").write_text("".join(f"[{p['k']}] {p['a']} || {p['b']}\n" for p in chunk))
    print(f"{len(strings)} strings ({len(held)} held out); {len(out)} pairs "
          f"({sum(p['split'] == 'test' for p in out)} test) in {parts} parts of about {size}")


if __name__ == "__main__":
    main()
