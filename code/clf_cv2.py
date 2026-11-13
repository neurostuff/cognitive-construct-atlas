"""Claim-sentence classifier, retrained for calibration, optionally with silver labels; cross-validated with whole gold
papers held out.

Changes from clf_cv.py:
- no class weighting (the 4.4x positive weight inflated the scores), and the raw logit margin z = l1 - l0 is saved;
- the input names what a search gives, not the normalized records: "section: S. tasks: <target task names>.
  names: <offtarget verdict>. previous: <prev unit>" paired with the unit. The verdict is "the searched task",
  "another task" or "no task" (offtarget.Checker);
- with SILVER set, every fold also trains on the stage-1 units of the silver papers, labelled by the judge
  (labels/judge/<id>.<judge>.jsonl, own_task/own_contrast = 1, "unsure" = left out). Units stage 1 drops are not
  used: unjudged.

Evaluation is on the 36 gold papers only (ADJ2), as before. Writes preds/clf_<tag>.jsonl with z per unit.
Calibration and cutoff transfer are scored by calib_report.py.

usage: python clf_cv2.py MODEL_ID TAG [SILVER_JUDGE] [EPOCHS]   (GPU; from pilot-results/)
       python clf_cv2.py MODEL_ID TAG score-silver   train once on all 36 gold papers, write the margin z of every
                                                     silver stage-1 unit to preds/silver_<tag>.jsonl
"""
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # pilot-results/
sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from offtarget import Checker  # noqa: E402

POS = {"own_task", "own_contrast"}
SAY = {"target": "the searched task", "other": "another task", "neither": "no task"}


def rows_for(article, units, targets, ck, label_of, only=None, rxs=None, ments=None):
    rxs = rxs if rxs is not None else ck.paper(" ".join(u["text"] for u in units))
    verdict = ck.verdicts([u["text"] for u in units], targets, rxs, ments)
    names = "; ".join(ck.names.get(t, t) for t in sorted(targets))
    out = []
    for k, u in enumerate(units):
        if u["section"] == "back" or len(u["text"]) < 25 or (only is not None and u["i"] not in only):
            continue
        y = label_of(u["i"])
        if y is None:  # judged unsure: not used for training
            continue
        prev = units[k - 1]["text"][-300:] if k else ""
        out.append({"article": article, "i": u["i"], "y": y, "names": verdict[k],
                    "a": f"section: {u['section']}. tasks: {names}. names: {SAY[verdict[k]]}. previous: {prev}",
                    "b": u["text"]})
    return out


def examples(judge):
    ck = Checker(ROOT / "code" / "task_inventory_v3.json", ROOT / "code" / "queries.jsonl")
    groups = json.load(open(ROOT / "groups.json"))
    fold_of = {a: k for k, g in enumerate(groups) for a in g}
    search = json.load(open(ROOT / "pilot" / "search_tasks.json"))
    gold, silver = [], []
    for f in sorted((ROOT / "pilot" / "papers").glob("*.json")):
        p = json.loads(f.read_text()); a = p["article"]
        lab = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(ROOT / "labels" / f"{a}.ADJ2.jsonl")}
        targets = set(search.get(a) or ({t["canonical"] for t in p["tasks"] if t["canonical"]} |
                                        {p["pilot_task"]["canonical"]}))
        for r in rows_for(a, p["units"], targets, ck, lambda i: int(lab.get(i) in POS)):
            r["fold"] = fold_of[a]; gold.append(r)
    if judge:
        for f in sorted((ROOT / "silver" / "papers").glob("*.json")):
            a = f.stem
            lf = ROOT / "labels" / "judge" / f"{a}.{judge}.jsonl"
            if a in fold_of or (judge != "score-silver" and not lf.exists()):
                continue
            s = json.loads(f.read_text())
            lab = {}
            for l in (lf.read_text().splitlines() if lf.exists() else []):
                if l.strip():
                    r = json.loads(l); lab[int(r["i"])] = lab.get(int(r["i"])) if lab.get(int(r["i"])) in POS else r["category"]
            silver += rows_for(a, s["units"], set(s["targets"]), ck,
                               lambda i: None if lab.get(i) == "unsure" else int(lab.get(i) in POS),
                               only=set(s["candidates"]))
    return gold, silver


def batches(tok, rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for k in range(0, len(idx), bs):
        sel = [rows[j] for j in idx[k:k + bs]]
        enc = tok([r["a"] for r in sel], [r["b"] for r in sel], truncation="longest_first", max_length=320,
                  padding=True, return_tensors="pt")
        yield enc, torch.tensor([r["y"] for r in sel]), sel


def train(model_id, tok, rows, epochs, dev):
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps = epochs * ((len(rows) + 15) // 16)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    lossf = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for enc, y, _ in batches(tok, rows, 16, True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
            loss = lossf(logits.float(), y.to(dev))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
    model.eval()
    return model


def predict(model, tok, rows, dev):
    with torch.no_grad():
        for enc, _, sel in batches(tok, rows, 64, False):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float()
            yield from zip(sel, (lg[:, 1] - lg[:, 0]).tolist())


def main():
    model_id, tag = sys.argv[1], sys.argv[2]
    judge = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    seed = int(os.environ.get("SEED", "0")); random.seed(seed); torch.manual_seed(seed)  # SEED: replicate runs
    gold, silver = examples(judge)
    print(f"gold {len(gold)} units ({sum(r['y'] for r in gold)} claims); silver {len(silver)} units "
          f"({sum(r['y'] for r in silver)} claims)", flush=True)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model_id)
    out = []
    if judge == "score-silver":
        model = train(model_id, tok, gold, epochs, dev)
        with open(ROOT / "preds" / f"silver_{tag}.jsonl", "w") as f:
            for r, z in predict(model, tok, silver, dev):
                f.write(json.dumps({"article": r["article"], "i": r["i"], "z": round(z, 4), "names": r["names"]}) + "\n")
        return
    for fold in range(6):
        train_rows = [r for r in gold if r["fold"] != fold] + silver; test = [r for r in gold if r["fold"] == fold]
        model = train(model_id, tok, train_rows, epochs, dev)
        for r, z in predict(model, tok, test, dev):
            out.append({"article": r["article"], "i": r["i"], "fold": fold, "y": r["y"], "z": round(z, 4),
                        "names": r["names"]})
        print(f"fold {fold}: train {len(train_rows)} ({sum(r['y'] for r in train_rows)} pos), test {len(test)}",
              flush=True)
        del model; torch.cuda.empty_cache()
    with open(ROOT / "preds" / f"clf_{tag}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
