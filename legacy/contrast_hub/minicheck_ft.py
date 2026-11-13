"""A LoRA fine-tune of Bespoke-MiniCheck-7B as a stance classifier (stance.py labels), scored against ft_stance.py.

The model reads one (claim sentence, construct) per prompt and its next-token logits over the five label words are
the classifier (the label words' first tokens; the rest of the vocabulary is ignored). Training data and the
article folds are ft_stance.py's (folds.json: random.seed(0) shuffle of the sorted labelled articles, i % 4), with the
same A <-> B swap augmentation, so a fold's score compares with DeBERTa's on the same test papers.

  python minicheck_ft.py ITEMS.jsonl CODEX.json FOLDS.json FOLD OUT_DIR     train on the other folds, score FOLD
  SMOKE=1 ...                                                               24 items each way, one epoch
  (LS6 gpu-a100, one GPU per fold; venv with transformers 4.46: MiniCheck's remote InternLM2 code loads wrongly on 5.x)
"""
import collections
import json
import math
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

NAME = "bespokelabs/Bespoke-MiniCheck-7B"
LABELS = ["A", "B", "none", "task", "unrelated"]          # stance.LABELS
SWAP = {"A": "B", "B": "A"}
EPOCHS, LR, MB, ACC, MAXLEN = 2, 1e-4, 4, 4, 512
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["wqkv", "wo", "w1", "w2", "w3"], task_type="CAUSAL_LM")

#: stance.py RULES, for one construct
SYSTEM = """You read a sentence from a neuroimaging paper (with the sentence before it), two conditions A and B that the
sentence compares, and one construct the sentence names. Say how the sentence relates the construct to the comparison
of A and B, with one word:
- A: A engages it more than B, or the A > B difference reflects, isolates or indexes it (a hypothesis the paper tests
  counts).
- B: B engages it more than A, or the B > A difference reflects it.
- none: the sentence says the comparison does not reflect it, rules it out, controls for it or matches it across A
  and B, or it is shared by both conditions and subtracted.
- task: it describes the task, the study or both conditions in general, not their difference.
- unrelated: the sentence is about something else (a group or individual difference, a brain region's role, another
  comparison, the behavioural measure itself).
Use the meaning, not the order the conditions are written in."""


def prompt(tok, it, c, swap=False):
    a, b = (it["B"], it["A"]) if swap else (it["A"], it["B"])
    user = (f"Sentence before: {it['before'][-300:]}\nSentence: {it['text'][:1200]}\nA: {a[:120]}\n"
            f"B: {b[:120] or '(none: a simple effect of A)'}\nConstruct: {c}\nAnswer:")
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)


def pairs(tok, items, labels, swap):
    out = []
    for it in items:
        for c in it["constructs"][:4]:                      # ft_stance.pairs: the first four constructs
            lab = labels.get(it["id"], {}).get(c)
            if lab is None:
                continue
            out.append((prompt(tok, it, c), lab, it["id"], c))
            if swap and it["B"]:
                out.append((prompt(tok, it, c, swap=True), SWAP.get(lab, lab), it["id"], c))
    return out


def scores(base, tok, ids, texts):
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN, add_special_tokens=False).to("cuda")
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)          # left padding
    h = base.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], position_ids=pos, use_cache=False)[0][:, -1, :]
    return base.output(h)[:, ids].float()                              # the head on the last position, label words only


def main():
    itemsp, codexp, foldsp, f, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    random.seed(0); torch.manual_seed(0)
    items = [json.loads(x) for x in open(itemsp)]; labels = json.load(open(codexp)); fold = json.load(open(foldsp))["fold"]
    tr = [it for it in items if it["id"] in labels and fold[it["article"]] != f]
    te = [it for it in items if it["id"] in labels and fold[it["article"]] == f]
    global EPOCHS
    if os.environ.get("SMOKE"):                                        # a few items, one epoch: loading and shapes
        tr, te, EPOCHS = tr[:24], te[:24], 1
    cfg = get_class_from_dynamic_module("configuration_internlm2.InternLM2Config", NAME)
    cfg._rope_scaling_validation = lambda self: None
    tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True); tok.padding_side = tok.truncation_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ids = [tok.encode(w, add_special_tokens=False)[0] for w in LABELS]
    assert len(set(ids)) == len(ids), ids
    model = AutoModelForCausalLM.from_pretrained(NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**LORA)); model.print_trainable_parameters()
    base = model.get_base_model()                                      # LoRA layers are injected into its modules
    train, test = pairs(tok, tr, labels, True), pairs(tok, te, labels, False)
    print(f"fold {f}: train items {len(tr)} pairs {len(train)}; test items {len(te)} pairs {len(test)}", flush=True)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=LR, weight_decay=0.0)
    steps = math.ceil(EPOCHS * len(train) / (MB * ACC)); warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * max(0.0, 1 - s / steps))
    model.train(); k = 0; t0 = time.time()
    for ep in range(EPOCHS):
        random.shuffle(train); tot = 0.0
        for i in range(0, len(train), MB):
            b = train[i:i + MB]
            y = torch.tensor([LABELS.index(x[1]) for x in b], device="cuda")
            loss = torch.nn.functional.cross_entropy(scores(base, tok, ids, [x[0] for x in b]), y) / ACC
            loss.backward(); tot += loss.item() * ACC; k += 1
            if k % ACC == 0:
                torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if k % 200 == 0:
                print(f"epoch {ep} batch {k} loss {tot / (i // MB + 1):.3f} {time.time() - t0:.0f}s", flush=True)
        print(f"epoch {ep} done loss {tot / max(1, len(train) / MB):.3f} {time.time() - t0:.0f}s", flush=True)
    model.save_pretrained(f"{out}/adapter")
    model.eval(); per = []
    with torch.no_grad():
        order = sorted(range(len(test)), key=lambda j: len(test[j][0]))
        for i in range(0, len(order), 16):
            ch = [test[j] for j in order[i:i + 16]]
            p = torch.softmax(scores(base, tok, ids, [x[0] for x in ch]), -1).cpu()
            for x, q in zip(ch, p):
                per.append({"id": x[2], "construct": x[3], "codex": x[1], "pred": LABELS[int(q.argmax())], "p": round(float(q.max()), 3)})
    k3 = lambda l: l if l in ("A", "B") else "drop"           # noqa: E731
    k4 = lambda l: l if l in ("A", "B", "task") else "drop"   # noqa: E731
    R = {"fold": f, "pairs": len(per), "agreement": round(sum(r["codex"] == r["pred"] for r in per) / len(per), 3),
         "decision_agreement": round(sum(k3(r["codex"]) == k3(r["pred"]) for r in per) / len(per), 3),
         "decision_task_agreement": round(sum(k4(r["codex"]) == k4(r["pred"]) for r in per) / len(per), 3),
         "confusion": {f"{a}->{b}": v for (a, b), v in sorted(collections.Counter((r["codex"], r["pred"]) for r in per).items())},
         "train_seconds": round(time.time() - t0)}
    for t in (0.7, 0.8, 0.9):
        hi = [r for r in per if r["p"] >= t]
        R[f"p>={t}"] = {"share": round(len(hi) / len(per), 3), "agreement": round(sum(r["codex"] == r["pred"] for r in hi) / max(1, len(hi)), 3)}
    print(json.dumps(R, indent=1), flush=True)
    json.dump({"summary": R, "predictions": per}, open(f"{out}/cv_fold{f}.json", "w"))


if __name__ == "__main__":
    main()
