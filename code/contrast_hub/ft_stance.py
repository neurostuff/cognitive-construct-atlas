"""A fine-tuned stance classifier (stance.py labels) for claim sentences Codex has not labelled.

DeBERTa-v3-large (the MNLI/FEVER/ANLI checkpoint already on beast) with a 5-way head over stance.LABELS. Input:
"A: <side A> | B: <side B> | construct: <construct>" paired with "<previous sentence> <sentence>". Each training pair
is also shown with A and B swapped and the label A <-> B swapped, so the model has to read the sides, not their order.

  cv    ITEMS.jsonl CODEX.json OUT.json [FOLDS]   k-fold agreement with Codex (by article, so no article is in both)
  fit   ITEMS.jsonl CODEX.json MODEL_DIR          train on every Codex label
  apply ITEMS.jsonl MODEL_DIR OUT.json            label every (item, construct); stance.py's label file shape

(beast, iw-post venv: python ft_stance.py ...; one RTX 3070)
"""
import collections
import json
import os
import random
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from stance import LABELS

BASE = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
SWAP = {"A": "B", "B": "A"}
DEVICE = os.environ.get("DEVICE", "cuda")          # apply on CPU: DEVICE=cpu (fp32), threads from OMP_NUM_THREADS
EPOCHS, LR, BS, ACC, MAXLEN = int(os.environ.get("EPOCHS", 3)), 1e-5, 4, 4, 320   # EPOCHS=1: a quick look


def pairs(items, labels=None, swap=False):
    out = []
    for it in items:
        for c in it["constructs"][:4]:
            lab = labels.get(it["id"], {}).get(c) if labels is not None else None
            if labels is not None and lab is None:
                continue
            ctx = (it["before"] + " " + it["text"]).strip()
            out.append((f"A: {it['A'][:80]} | B: {it['B'][:80] or '(none)'} | construct: {c}", ctx, lab, it["id"], c))
            if swap and it["B"]:
                out.append((f"A: {it['B'][:80]} | B: {it['A'][:80]} | construct: {c}", ctx, SWAP.get(lab, lab), it["id"], c))
    return out


def model_new():
    tok = AutoTokenizer.from_pretrained(BASE)
    m = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(LABELS), ignore_mismatched_sizes=True,
                                                           id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
                                                           torch_dtype=torch.float32)
    for q in m.deberta.embeddings.parameters():   # frozen: fp32 Adam over the rest fits an 8 GB RTX 3070
        q.requires_grad = False
    m.gradient_checkpointing_enable()
    return tok, m.cuda()


def train(tok, m, data):
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(data) // (BS * ACC); sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / max(1, steps // 10)) * max(0.0, 1 - s / max(steps, 1)))
    scaler = torch.cuda.amp.GradScaler(); m.train(); k = 0
    for ep in range(EPOCHS):
        random.shuffle(data); tot = 0
        for i in range(0, len(data), BS):
            b = data[i:i + BS]
            enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt").to("cuda")
            y = torch.tensor([LABELS.index(x[2]) for x in b]).cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = m(**enc, labels=y).loss / ACC
            scaler.scale(loss).backward(); tot += loss.item(); k += 1
            if k % ACC == 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        print(f"epoch {ep} loss {tot * ACC / max(1, len(data) / BS):.3f}", flush=True)


@torch.no_grad()
def predict(tok, m, data):
    m.eval(); out = []
    for i in range(0, len(data), 32):
        b = data[i:i + 32]
        enc = tok([x[0] for x in b], [x[1] for x in b], truncation="only_second", max_length=MAXLEN, padding=True, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            p = torch.softmax(m(**enc).logits.float(), -1).cpu()
        out += [(LABELS[int(q.argmax())], float(q.max())) for q in p]
    return out


def main():
    cmd = sys.argv[1]; items = [json.loads(x) for x in open(sys.argv[2])]
    random.seed(0); torch.manual_seed(0)
    if cmd == "cv":
        labels = json.load(open(sys.argv[3])); outp = sys.argv[4]; K = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        arts = sorted({it["article"] for it in items if it["id"] in labels}); random.shuffle(arts)
        fold = {a: i % K for i, a in enumerate(arts)}
        conf = collections.Counter(); per = []
        for f in range(K):
            tr = [it for it in items if it["id"] in labels and fold[it["article"]] != f]
            te = [it for it in items if it["id"] in labels and fold[it["article"]] == f]
            tok, m = model_new(); train(tok, m, pairs(tr, labels, swap=True))
            test = pairs(te, labels); pred = predict(tok, m, test)
            for x, (p, q) in zip(test, pred):
                conf[(x[2], p)] += 1; per.append({"id": x[3], "construct": x[4], "codex": x[2], "pred": p, "p": round(q, 3)})
            del m; torch.cuda.empty_cache()
            print(f"fold {f}: {sum(a == b for (a, b), v in conf.items() for _ in range(v))}/{sum(conf.values())}", flush=True)
        keep = lambda l: l if l in ("A", "B") else "drop"  # noqa: E731
        dec = collections.Counter((keep(a), keep(b)) for (a, b), v in conf.items() for _ in range(v))
        R = {"pairs": sum(conf.values()), "agreement": round(sum(v for (a, b), v in conf.items() if a == b) / sum(conf.values()), 3),
             "decision_agreement": round(sum(v for (a, b), v in dec.items() if a == b) / sum(dec.values()), 3),
             "confusion": {f"{a}->{b}": v for (a, b), v in sorted(conf.items())},
             "decision_confusion": {f"{a}->{b}": v for (a, b), v in sorted(dec.items())}}
        for t in (0.7, 0.8, 0.9):
            hi = [r for r in per if r["p"] >= t]
            R[f"p>={t}"] = {"share": round(len(hi) / len(per), 3), "agreement": round(sum(r["codex"] == r["pred"] for r in hi) / max(1, len(hi)), 3)}
        print(json.dumps(R, indent=1)); json.dump({"summary": R, "predictions": per}, open(outp, "w"))
    elif cmd == "fit":
        labels = json.load(open(sys.argv[3])); tok, m = model_new()
        train(tok, m, pairs([it for it in items if it["id"] in labels], labels, swap=True))
        m.save_pretrained(sys.argv[4]); tok.save_pretrained(sys.argv[4])
    elif cmd == "apply":
        tok = AutoTokenizer.from_pretrained(sys.argv[3]); m = AutoModelForSequenceClassification.from_pretrained(sys.argv[3]).to(DEVICE)
        data = sorted(pairs(items), key=lambda x: len(x[1]))       # similar lengths per batch: little padding
        pred = predict(tok, m, data); out = collections.defaultdict(dict)
        for x, (p, q) in zip(data, pred):
            out[x[3]][x[4]] = {"label": p, "p": round(q, 3)}
        json.dump(out, open(sys.argv[4], "w"))


if __name__ == "__main__":
    main()
