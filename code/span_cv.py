"""Supervised construct span tagger (no LLM), cross-validated with whole papers held out.

Training sentences: labeller A's claim units that the adjudicated gold (ADJ2) keeps, with A's construct phrases
located in the sentence (case-insensitive; phrases not found verbatim are skipped). BIO token tagging on
BiomedBERT, six folds = the labelling groups. Each fold's model then tags, in its held-out papers, the gold claim
sentences and every unit among the classifier's top 30 per paper (preds/clf_biomedbert.jsonl), so the chain
classifier -> tagger can be scored on sentences the tagger never saw. Writes preds/spans_<tag>.jsonl.

usage: python span_cv.py MODEL_ID TAG [EPOCHS]   (GPU; run from pilot-results/)
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path.cwd()
POS = {"own_task", "own_contrast"}


def load():
    groups = json.load(open(ROOT / "groups.json"))
    fold_of = {a: k for k, g in enumerate(groups) for a in g}
    papers = {f.stem: json.loads(f.read_text()) for f in (ROOT / "pilot" / "papers").glob("*.json")}
    train = []
    for a, p in papers.items():
        adj = {json.loads(l)["i"]: json.loads(l)["category"] for l in open(ROOT / "labels" / f"{a}.ADJ2.jsonl")}
        for l in open(ROOT / "labels" / f"{a}.A.jsonl"):
            r = json.loads(l)
            if r["category"] not in POS or adj.get(r["i"]) not in POS:
                continue
            s = p["units"][r["i"]]["text"]; spans = []
            for c in r.get("constructs") or []:
                m = re.search(re.escape(c.strip()), s, re.I)
                if m:
                    spans.append((m.start(), m.end()))
            if spans:
                train.append({"article": a, "i": r["i"], "fold": fold_of[a], "text": s, "spans": spans})
    cand = defaultdict(list)
    for l in open(ROOT / "preds" / "clf_biomedbert.jsonl"):
        r = json.loads(l); cand[r["article"]].append(r)
    tests = [{"article": a, "i": r["i"], "fold": fold_of[a], "text": papers[a]["units"][r["i"]]["text"], "p": r["p"]}
             for a, rows in cand.items() for r in sorted(rows, key=lambda r: -r["p"])[:30]]
    return train, tests


def encode(tok, rows, with_labels):
    enc = tok([r["text"] for r in rows], truncation=True, max_length=256, padding=True,
              return_offsets_mapping=True, return_tensors="pt")
    if with_labels:
        labels = torch.full(enc["input_ids"].shape, -100)
        for b, r in enumerate(rows):
            prev_in = False
            for t, (s, e) in enumerate(enc["offset_mapping"][b].tolist()):
                if s == e:
                    continue
                inside = [(cs, ce) for cs, ce in r["spans"] if s >= cs and e <= ce]
                if not inside:
                    labels[b, t] = 0; prev_in = False
                else:
                    labels[b, t] = 1 if (s == inside[0][0] or not prev_in) else 2; prev_in = True
        enc["labels"] = labels
    return enc


def spans_of(probs, offsets, text):
    """contiguous B/I runs -> (span text, mean probability of its labels)"""
    out, cur = [], None
    lab = probs.argmax(-1).tolist()
    for t, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if lab[t] == 1 or (lab[t] == 2 and cur is None):
            if cur:
                out.append(cur)
            cur = [s, e, [probs[t][lab[t]].item()]]
        elif lab[t] == 2:
            cur[1] = e; cur[2].append(probs[t][2].item())
        elif cur:
            out.append(cur); cur = None
    if cur:
        out.append(cur)
    return [(text[s:e], round(sum(ps) / len(ps), 4)) for s, e, ps in out]


def main():
    model_id, tag = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    random.seed(0); torch.manual_seed(0)
    train, tests = load()
    tok = AutoTokenizer.from_pretrained(model_id)
    out = []
    for fold in range(6):
        tr = [r for r in train if r["fold"] != fold]
        te_gold = [r for r in train if r["fold"] == fold]
        te_cand = [r for r in tests if r["fold"] == fold]
        model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=3).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        steps = epochs * ((len(tr) + 15) // 16)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        model.train()
        for _ in range(epochs):
            random.shuffle(tr)
            for k in range(0, len(tr), 16):
                enc = encode(tok, tr[k:k + 16], True); enc.pop("offset_mapping")
                loss = model(**{x: v.cuda() for x, v in enc.items()}).loss
                loss.backward(); opt.step(); sched.step(); opt.zero_grad()
        model.eval()
        for kind, rows in (("gold", te_gold), ("cand", te_cand)):
            for k in range(0, len(rows), 32):
                batch = rows[k:k + 32]
                enc = encode(tok, batch, False); offs = enc.pop("offset_mapping")
                with torch.no_grad():
                    pr = model(**{x: v.cuda() for x, v in enc.items()}).logits.softmax(-1).cpu()
                for b, r in enumerate(batch):
                    out.append({"kind": kind, "article": r["article"], "i": r["i"], "fold": fold,
                                "pred": spans_of(pr[b], offs[b].tolist(), r["text"]),
                                **({"p": r["p"]} if "p" in r else {})})
        print(f"fold {fold}: trained on {len(tr)}; tagged {len(te_gold)} gold, {len(te_cand)} candidates", flush=True)
        del model; torch.cuda.empty_cache()
    with open(ROOT / "preds" / f"spans_{tag}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print("training sentences", len(train))


if __name__ == "__main__":
    main()
