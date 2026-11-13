"""Contrast tagger without an LLM at run time (James 2026-11-11: train a non-LLM extractor of conditions/contrasts
from Codex silver labels).

BIO token tagger on BiomedBERT, the construct tagger's recipe (MODEL.md of construct-claim-tagger v1: 6 epochs, lr
5e-5, batch 16; max length 128 here, claim sentences are short) with five labels: O, B-A, I-A, B-B, I-B, for the two
condition sides of a contrast named in a construct-claim sentence ("incongruent" > "congruent"). Training data:
Codex gpt-6.1-sol labels (codex_contrasts.py) of claim sentences, condition spans located verbatim (first
occurrence); sentences without a contrast are all O. Split by article (the sample's 80/20 split): trained on train,
scored on test.

Scores on the test papers:
- detection: a sentence has a contrast if the tagger finds any condition span; precision / recall against Codex;
- spans, on sentences where both have a contrast: exact match and content-token overlap F1 for condition A and B.

usage: python contrast_tagger.py LABELS_DIR OUT_DIR [EPOCHS]   (GPU; iw-post venv; MEM_FRACTION caps GPU memory)
"""
import glob
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
LABELS = ["O", "B-A", "I-A", "B-B", "I-B"]
MAXLEN = 128


def load(d):
    rows = {}
    for f in sorted(glob.glob(f"{d}/sol_low_*.jsonl")):
        for line in open(f):
            r = json.loads(line); rows[r["id"]] = r
    out = []
    for r in rows.values():
        L = r["label"]; spans = []
        if L["has_contrast"]:
            for side, key in (("A", "condition_a"), ("B", "condition_b")):
                s = L.get(key)
                if s:
                    i = r["text"].find(s)
                    if i >= 0:
                        spans.append((i, i + len(s), side))
        out.append({"id": r["id"], "split": r["split"], "text": r["text"], "spans": spans, "has": bool(L["has_contrast"])})
    return out


def encode(tok, rows, with_labels):
    enc = tok([r["text"] for r in rows], truncation=True, max_length=MAXLEN, padding=True, return_offsets_mapping=True,
              return_tensors="pt")
    if with_labels:
        lab = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
        for b, r in enumerate(rows):
            started = set()
            for t, (s, e) in enumerate(enc["offset_mapping"][b].tolist()):
                if s == e:
                    continue
                y = 0
                for j, (a, z, side) in enumerate(r["spans"]):
                    if s >= a and e <= z:   # the first token inside a span is B, the rest I
                        y = LABELS.index(("I-" if j in started else "B-") + side); started.add(j)
                        break
                lab[b, t] = y
        enc["labels"] = lab
    return enc


def decode(pred, offsets, text):
    """predicted label ids -> {side: span text} (longest run per side)"""
    runs, cur = {}, None
    for t, (s, e) in enumerate(offsets):
        if s == e:
            continue
        lab = LABELS[pred[t]]
        if lab == "O":
            cur = None; continue
        side = lab[-1]
        if lab.startswith("B-") or cur is None or cur[0] != side:
            cur = [side, s, e]; runs.setdefault(side, []).append(cur)
        else:
            cur[2] = e
    return {side: max((text[a:z] for _, a, z in v), key=len) for side, v in runs.items()}


def content(s):
    stop = {"the", "a", "an", "of", "in", "on", "for", "to", "and", "trials", "trial", "condition", "conditions", "blocks"}
    return {w for w in s.lower().replace("-", " ").split() if w not in stop}


def f1(a, b):
    A, B = content(a), content(b)
    if not A or not B:
        return float(A == B)
    i = len(A & B)
    return 0.0 if i == 0 else 2 * i / (len(A) + len(B))


def main():
    d, out = sys.argv[1], Path(sys.argv[2]); epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    frac = float(os.environ.get("MEM_FRACTION", "0"))
    if frac:
        torch.cuda.set_per_process_memory_fraction(frac, 0)
    random.seed(0); torch.manual_seed(0)
    rows = load(d); tr = [r for r in rows if r["split"] == "train"]; te = [r for r in rows if r["split"] == "test"]
    print(f"{len(rows)} sentences ({sum(r['has'] for r in rows)} with a contrast); train {len(tr)}, test {len(te)}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    model = AutoModelForTokenClassification.from_pretrained(BASE, revision=REVISION, num_labels=len(LABELS)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    steps = epochs * ((len(tr) + 15) // 16); sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler()
    model.train()
    for ep in range(epochs):
        random.shuffle(tr); tot = 0.0
        for k in range(0, len(tr), 16):
            enc = encode(tok, tr[k:k + 16], True); enc.pop("offset_mapping")
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**{a: v.cuda() for a, v in enc.items()}).loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step(); tot += loss.item()
        print(f"epoch {ep + 1}: loss {tot / ((len(tr) + 15) // 16):.4f}", flush=True)
    model.eval(); tp = fp = fn = 0; ex = {"A": [], "B": []}; ov = {"A": [], "B": []}; examples = []
    with torch.no_grad():
        for k in range(0, len(te), 32):
            batch = te[k:k + 32]; enc = encode(tok, batch, False); offs = enc.pop("offset_mapping")
            pred = model(**{a: v.cuda() for a, v in enc.items()}).logits.argmax(-1).cpu().tolist()
            for b, r in enumerate(batch):
                got = decode(pred[b], offs[b].tolist(), r["text"]); has_p = bool(got)
                tp += has_p and r["has"]; fp += has_p and not r["has"]; fn += (not has_p) and r["has"]
                if has_p and r["spans"]:
                    gold = {side: r["text"][a:z] for a, z, side in r["spans"]}
                    for side in ("A", "B"):
                        if side in gold:
                            p = got.get(side, ""); ex[side].append(p == gold[side]); ov[side].append(f1(p, gold[side]))
                if len(examples) < 12 and has_p:
                    examples.append({"text": r["text"][:200], "pred": got, "codex": {s: r["text"][a:z] for a, z, s in r["spans"]}})
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    res = {"train": len(tr), "test": len(te), "detection_precision": round(prec, 3), "detection_recall": round(rec, 3),
           **{f"exact_{s}": round(sum(ex[s]) / max(1, len(ex[s])), 3) for s in "AB"},
           **{f"overlap_f1_{s}": round(sum(ov[s]) / max(1, len(ov[s])), 3) for s in "AB"}, "n_span_pairs": len(ex["A"])}
    print(json.dumps(res), flush=True)
    out.mkdir(parents=True, exist_ok=True); model.save_pretrained(out / "weights"); tok.save_pretrained(out / "weights")
    json.dump({**res, "examples": examples, "labels": LABELS, "base": BASE, "revision": REVISION, "epochs": epochs},
              open(out / "eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
