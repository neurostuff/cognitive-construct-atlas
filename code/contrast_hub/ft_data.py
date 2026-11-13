"""Training data for a fine-tuned contrast normalizer (James 2026-11-11: fine-tune a model so the hub does not depend on
Codex for new papers).

One example per corpus contrast. Input (the same facts Codex saw, codex_normalize.item_text): task family and task, the
two sides' names, the paper's own conditions with descriptions, the paper's own abbreviations that occur in the names.
Output: JSON {"target": [atoms], "baseline": [atoms], "baseline_special": "@base"|null, "type": ...}.
- gold: Codex's high-confidence readings (overrides.json). Papers are split 90/10; the 10% is the test set and is never
  trained on, silver included.
- silver: the rule readings of contrasts in established exact entries (precision 98% lenient, 62% strict in the v4
  review), written in the same JSON from their signature. Sampled to SILVER_N, a few per entry so that frequent entries
  do not dominate.
Writes OUT_DIR/{train,test}.jsonl with {"id", "paper", "prompt", "answer", "source"}.

usage: python ft_data.py HUB_DIR CONTRASTS.json GLOSSARY.json ABBREV.json OVERRIDES.json OUT_DIR
"""
import collections
import os
import json
import random
import sys
from pathlib import Path

import hub as H
from codex_normalize import item_text

SILVER_N, PER_ENTRY, GOLD_REPEAT = int(os.environ.get("SILVER_N", 7000)), 6, int(os.environ.get("GOLD_REPEAT", 3))
SYSTEM = ("Normalize this neuroimaging contrast into condition atoms (axis:value). target/baseline: the atoms on which "
          "the sides differ; baseline_special @base for rest, fixation or a low-level baseline; type: difference, "
          "baseline, main_effect, parametric, interaction, task_vs_base or not_conditions. Answer with JSON.")


def answer_from_sig(sig, a):
    fam, comp = sig
    if fam.startswith("task:"):
        return {"target": [], "baseline": [], "baseline_special": "@base", "type": "task_vs_base"}
    tgt = sorted(f"{ax}:{x}" for ax, x, _ in comp if x not in (H.ABSENT, None))
    base = sorted(f"{ax}:{y}" for ax, _, y in comp if y not in (H.ABSENT, None) and y not in H.BASES)
    special = "@base" if any(y == H.REST for _, _, y in comp) else None
    if all(y is None for _, _, y in comp):
        typ = "parametric" if False else "difference"
    typ = "baseline" if special else ("main_effect" if " | " in a["pos"] else "difference")
    return {"target": tgt, "baseline": base, "baseline_special": special, "type": typ}


def main():
    hub_dir, contrasts, glossp, abbrp, ovp, out = sys.argv[1:7]
    C = json.load(open(contrasts)); A = json.load(open(f"{hub_dir}/corpus_assign.json"))
    E = json.load(open(f"{hub_dir}/hub.json"))["contrasts"]
    gloss = json.load(open(glossp)); abbr = json.load(open(abbrp)); ov = json.load(open(ovp))
    rng = random.Random(20261111)
    papers = sorted({C[a]["paper"] for a, o in ov.items() if o.get("source") != "ft"})
    test_papers = set(rng.sample(papers, max(1, len(papers) // 10)))
    rows = {"train": [], "test": []}

    def prompt(a):
        return item_text(0, a, gloss, abbr).split("\n", 1)[1]
    for aid, o in ov.items():
        if o["confidence"] != "high" or o.get("source") == "ft":     # gold is Codex's readings, never the model's own
            continue
        a = C[aid]
        ans = {k: o[k] for k in ("target", "baseline", "baseline_special", "type")}
        split = "test" if a["paper"] in test_papers else "train"
        for _ in range(1 if split == "test" else GOLD_REPEAT):
            rows[split].append({"id": aid, "paper": a["paper"], "prompt": prompt(a), "answer": json.dumps(ans), "source": "gold"})
    by_entry = collections.defaultdict(list)
    for aid, v in A.items():
        if v["level"] == "L0" and aid not in ov and C[aid]["paper"] not in test_papers:
            by_entry[v["hub"]].append(aid)
    pool = [aid for k, ids in by_entry.items() for aid in rng.sample(ids, min(PER_ENTRY, len(ids)))]
    for aid in rng.sample(pool, min(SILVER_N, len(pool))):
        a = C[aid]; sig = H._sig_from_json(E[A[aid]["hub"]]["sig_json"])
        if A[aid]["sign"] == -1:
            f = H.flip(sig)
            if f is None:
                continue
            sig = f
        rows["train"].append({"id": aid, "paper": a["paper"], "prompt": prompt(a), "answer": json.dumps(answer_from_sig(sig, a)),
                              "source": "silver"})
    rng.shuffle(rows["train"])
    Path(out).mkdir(parents=True, exist_ok=True)
    for k, r in rows.items():
        with open(f"{out}/{k}.jsonl", "w") as f:
            for x in r:
                f.write(json.dumps(x) + "\n")
    print({k: collections.Counter(x["source"] for x in v) for k, v in rows.items()}, f"{len(test_papers)} test papers")


if __name__ == "__main__":
    main()
