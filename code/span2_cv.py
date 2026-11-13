"""Joint construct + contrast span tagger (no LLM), cross-validated with whole papers held out.

Training sentences: labeller A's claim units kept by the adjudicated gold (ADJ2), task- and contrast-level.
- construct spans: A's construct phrases located verbatim (case-insensitive);
- contrast spans: A's contrast_text is a description, often not a quote, so it is aligned to the window of sentence
  tokens with the best token-F1 against it, kept when F1 >= 0.5. Task-level claims have no contrast span.
Labels O, B-CON, I-CON, B-CTR, I-CTR on BiomedBERT; six folds = the labelling groups. Each fold's model tags its
held-out gold claim sentences and the classifier's top 30 units per paper. Writes preds/spans2_<tag>.jsonl.

usage: python span2_cv.py MODEL_ID TAG [EPOCHS]   (GPU; run from pilot-results/)
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
LABELS = ["O", "B-CON", "I-CON", "B-CTR", "I-CTR"]
WORD = re.compile(r"[A-Za-z0-9]+")


def align(text, phrase):
    """(start, end) of the window of `text`'s words with the best token-F1 against `phrase`, or None below 0.5"""
    want = {w.lower() for w in WORD.findall(phrase)}
    words = list(WORD.finditer(text))
    if not want or not words:
        return None
    best, span = 0.0, None
    for i in range(len(words)):
        got = set()
        for j in range(i, min(len(words), i + 3 * len(want) + 3)):
            got.add(words[j].group().lower())
            inter = len(got & want)
            if not inter:
                continue
            f1 = 2 * inter / (len(got) + len(want))
            if f1 > best:
                best, span = f1, (words[i].start(), words[j].end())
    return span if best >= 0.5 else None


def load():
    groups = json.load(open(ROOT / "groups.json"))
    fold_of = {a: k for k, g in enumerate(groups) for a in g}
    papers = {f.stem: json.loads(f.read_text()) for f in (ROOT / "pilot" / "papers").glob("*.json")}
    train, stats = [], defaultdict(int)
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
                    spans.append((m.start(), m.end(), "CON"))
            if r["category"] == "own_contrast" and r.get("contrast_text"):
                stats["contrast claims"] += 1
                w = align(s, r["contrast_text"])
                if w and not any(cs < w[1] and w[0] < ce for cs, ce, _ in spans):
                    spans.append((w[0], w[1], "CTR")); stats["contrast aligned"] += 1
            if spans:
                train.append({"article": a, "i": r["i"], "fold": fold_of[a], "text": s, "spans": spans,
                              "category": r["category"], "contrast_text": r.get("contrast_text"),
                              "constructs": r.get("constructs") or []})
    cand = defaultdict(list)
    for l in open(ROOT / "preds" / "clf_biomedbert.jsonl"):
        r = json.loads(l); cand[r["article"]].append(r)
    tests = [{"article": a, "i": r["i"], "fold": fold_of[a], "text": papers[a]["units"][r["i"]]["text"], "p": r["p"]}
             for a, rows in cand.items() for r in sorted(rows, key=lambda r: -r["p"])[:30]]
    return train, tests, dict(stats)


def encode(tok, rows, with_labels):
    enc = tok([r["text"] for r in rows], truncation=True, max_length=256, padding=True,
              return_offsets_mapping=True, return_tensors="pt")
    if with_labels:
        labels = torch.full(enc["input_ids"].shape, -100)
        for b, r in enumerate(rows):
            prev = None
            for t, (s, e) in enumerate(enc["offset_mapping"][b].tolist()):
                if s == e:
                    continue
                hit = [(cs, ce, k) for cs, ce, k in r["spans"] if s >= cs and e <= ce]
                if not hit:
                    labels[b, t] = 0; prev = None
                else:
                    cs, ce, k = hit[0]
                    labels[b, t] = LABELS.index(("I-" if prev == (cs, k) else "B-") + k); prev = (cs, k)
        enc["labels"] = labels
    return enc


def spans_of(probs, offsets, text):
    """runs of B-X/I-X -> {X: [(span text, mean prob)]}"""
    lab = probs.argmax(-1).tolist()
    out = {"CON": [], "CTR": []}; cur = None
    def close():
        if cur:
            out[cur[0]].append((text[cur[1]:cur[2]], round(sum(cur[3]) / len(cur[3]), 4)))
    for t, (s, e) in enumerate(offsets):
        if s == e:
            continue
        name = LABELS[lab[t]]
        if name == "O":
            close(); cur = None
        elif name.startswith("B-") or cur is None or cur[0] != name[2:]:
            close(); cur = [name[2:], s, e, [probs[t][lab[t]].item()]]
        else:
            cur[2] = e; cur[3].append(probs[t][lab[t]].item())
    close()
    return out


def main():
    model_id, tag = sys.argv[1], sys.argv[2]
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    random.seed(0); torch.manual_seed(0)
    train, tests, stats = load()
    print("labels:", stats, "training sentences", len(train), flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    out = []
    for fold in range(6):
        tr = [r for r in train if r["fold"] != fold]
        te_gold = [r for r in train if r["fold"] == fold]
        te_cand = [r for r in tests if r["fold"] == fold]
        model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=len(LABELS)).cuda()
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
                                **{x: r[x] for x in ("p", "category", "contrast_text", "constructs") if x in r}})
        print(f"fold {fold}: trained on {len(tr)}", flush=True)
        del model; torch.cuda.empty_cache()
    with open(ROOT / "preds" / f"spans2_{tag}.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
