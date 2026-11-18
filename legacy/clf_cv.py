"""Supervised construct-claim sentence classifier, cross-validated with whole papers held out (no LLM).

Each non-back unit of the 36 pilot papers (>= 25 chars) is one example; label 1 when the adjudicated gold (ADJ2)
calls it own_task or own_contrast. Input pair: ("section: S. tasks: <the paper's task names>. previous: <prev unit>",
"<unit>"). Six folds = the six labelling groups (groups.json), so a paper is never in train and test at once.
Out-of-fold probabilities are written to preds/clf_<tag>.jsonl and summarized as precision at fixed recall and
recall at fixed precision.

usage: python clf_cv.py MODEL_ID TAG [EPOCHS]   (GPU; HF_HOME set by the caller)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent  # pilot-results/
POS = {"own_task", "own_contrast"}


def examples():
    groups = json.load(open(ROOT / "groups.json"))
    fold_of = {a: k for k, g in enumerate(groups) for a in g}
    rows = []
    for f in sorted((ROOT / "pilot" / "papers").glob("*.json")):
        p = json.loads(f.read_text()); a = p["article"]
        gold = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(ROOT / "labels" / f"{a}.ADJ2.jsonl")}
        tasks = "; ".join(t["name"] or "" for t in p["tasks"])
        U = p["units"]
        for u in U:
            if u["section"] == "back" or len(u["text"]) < 25:
                continue
            prev = U[u["i"] - 1]["text"][-300:] if u["i"] else ""
            rows.append({"article": a, "i": u["i"], "fold": fold_of[a], "y": int(gold.get(u["i"]) in POS),
                         "a": f"section: {u['section']}. tasks: {tasks}. previous: {prev}", "b": u["text"]})
    return rows


def batches(tok, rows, bs, shuffle):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for k in range(0, len(idx), bs):
        sel = [rows[j] for j in idx[k:k + bs]]
        enc = tok([r["a"] for r in sel], [r["b"] for r in sel], truncation="longest_first", max_length=320,
                  padding=True, return_tensors="pt")
        yield enc, torch.tensor([r["y"] for r in sel]), sel


def main():
    model_id, tag = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    random.seed(0); torch.manual_seed(0)
    rows = examples()
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(model_id)
    out = []
    for fold in range(6):
        train = [r for r in rows if r["fold"] != fold]; test = [r for r in rows if r["fold"] == fold]
        model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2, ignore_mismatched_sizes=True).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
        steps = epochs * ((len(train) + 15) // 16)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        pos = sum(r["y"] for r in train); w = torch.tensor([1.0, (len(train) - pos) / max(pos, 1) / 3], device=dev)
        lossf = torch.nn.CrossEntropyLoss(weight=w)
        model.train()
        for _ in range(epochs):
            for enc, y, _ in batches(tok, train, 16, True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**{k: v.to(dev) for k, v in enc.items()}).logits
                loss = lossf(logits.float(), y.to(dev))
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        model.eval()
        with torch.no_grad():
            for enc, y, sel in batches(tok, test, 64, False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pr = model(**{k: v.to(dev) for k, v in enc.items()}).logits.float().softmax(-1)[:, 1]
                for r, s in zip(sel, pr.tolist()):
                    out.append({"article": r["article"], "i": r["i"], "fold": fold, "y": r["y"], "p": round(s, 5)})
        print(f"fold {fold}: train {len(train)} ({pos} pos), test {len(test)}", flush=True)
        del model; torch.cuda.empty_cache()
    Path(ROOT / "preds").mkdir(exist_ok=True)
    with open(ROOT / "preds" / f"clf_{tag}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    y = np.array([r["y"] for r in out]); p = np.array([r["p"] for r in out])
    order = np.argsort(-p); tp = np.cumsum(y[order]); k = np.arange(1, len(y) + 1)
    prec, rec = tp / k, tp / y.sum()
    summary = {"model": model_id, "examples": len(y), "positives": int(y.sum())}
    for r_ in (0.1, 0.2, 0.3, 0.5, 0.7):
        j = int(np.argmax(rec >= r_)); summary[f"P@R{r_}"] = round(float(prec[j]), 3)
    for p_ in (0.8, 0.9, 0.95):
        ok = np.where(prec >= p_)[0]
        summary[f"R@P{p_}"] = round(float(rec[ok.max()]), 3) if len(ok) else 0.0
        summary[f"n@P{p_}"] = int(ok.max() + 1) if len(ok) else 0
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9); j = int(np.argmax(f1))
    summary["best_F1"] = round(float(f1[j]), 3); summary["P_at_bestF1"] = round(float(prec[j]), 3)
    summary["R_at_bestF1"] = round(float(rec[j]), 3)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
