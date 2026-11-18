"""Train one final model on all 36 gold papers plus the silver papers, and save it.

- classifier member: clf_cv2's input and training (3 epochs, lr 3e-5, no class weighting), on every gold unit and
  the silver stage-1 units judged by JUDGE (sonnetc: contested left out; sonnet: pass 1), seed SEED;
- tagger: span_cv2's BIO construct tagger (6 epochs, lr 5e-5) on the gold claim sentences plus the silver sentences
  the pass-1 judge listed, with constructs located verbatim.

usage: python train_final.py clf JUDGE SEED OUT_DIR     (GPU; from pilot-results/)
       python train_final.py tagger JUDGE OUT_DIR [SEED]
"""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # pilot-results/
sys.path.insert(0, str(ROOT / "code"))

import torch  # noqa: E402
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

import clf_cv2  # noqa: E402
import span_cv2  # noqa: E402

BASE = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"


def clf(judge, seed, out):
    random.seed(seed); torch.manual_seed(seed)
    gold, silver = clf_cv2.examples(judge)
    rows = gold + silver
    print(f"classifier: {len(rows)} units ({sum(r['y'] for r in rows)} claims), judge {judge}, seed {seed}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = clf_cv2.train(BASE, tok, rows, 3, "cuda")
    model.save_pretrained(out); tok.save_pretrained(out)


def tagger(judge, out, seed=0):
    random.seed(seed); torch.manual_seed(seed)
    train, tests = span_cv2.load()
    rows = train + span_cv2.load_silver(judge, {r["article"] for r in train} | {r["article"] for r in tests})
    print(f"tagger: {len(rows)} sentences", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    epochs = 6; steps = epochs * ((len(rows) + 15) // 16)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    model.train()
    for _ in range(epochs):
        random.shuffle(rows)
        for k in range(0, len(rows), 16):
            enc = span_cv2.encode(tok, rows[k:k + 16], True); enc.pop("offset_mapping")
            loss = model(**{x: v.cuda() for x, v in enc.items()}).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
    model.save_pretrained(out); tok.save_pretrained(out)


if __name__ == "__main__":
    if sys.argv[1] == "clf":
        clf(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        tagger(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
