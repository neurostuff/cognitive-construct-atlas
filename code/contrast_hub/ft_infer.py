"""Run the fine-tuned contrast normalizer (ft_train.py adapter) on prompts, greedy, batched.

Each answer is parsed as JSON (an answer that does not parse is dropped) and gets a confidence: the mean log-probability
of its generated tokens. `test` mode scores the held-out gold (ft_data.py test.jsonl) against Codex: the share of
contrasts whose signature (imaging.from_override on both readings) is the same, overall and above each confidence
threshold. `run` mode reads the contrasts in a hub's corpus_unnormalized.json that have no Codex reading, writes
readings in overrides.json's format (confidence "high" when the mean log-probability clears TAU), one shard of
NSHARDS.

usage: python ft_infer.py test ADAPTER DATA_DIR OUT.json
       python ft_infer.py run ADAPTER HUB_DIR CONTRASTS.json GLOSSARY.json ABBREV.json OVERRIDES.json OUT.json TAU SHARD NSHARDS
"""
import json
import sys

import torch
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig

import hub as H
import imaging
from ft_train import encode, load_base

BATCH = 8   # per-step scores over a 151k vocabulary are kept for the confidence: 8 fits an 8 GB card


def load(adapter):
    tok = AutoTokenizer.from_pretrained(adapter); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    return tok, PeftModel.from_pretrained(load_base(q, 0), adapter).eval()


def generate(tok, model, prompts):
    out = []
    for i in range(0, len(prompts), BATCH):
        ids = [encode(tok, p) for p in prompts[i:i + BATCH]]
        L = max(map(len, ids))
        x = torch.tensor([[tok.pad_token_id] * (L - len(s)) + s for s in ids]).cuda()
        att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in ids]).cuda()
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=att, max_new_tokens=96, do_sample=False,
                               output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        seq = g.sequences[:, L:]
        lp = torch.stack([torch.log_softmax(s.float(), -1) for s in g.scores], 1)        # batch, steps, vocab
        tlp = lp.gather(-1, seq.unsqueeze(-1)).squeeze(-1)
        for b in range(seq.shape[0]):
            keep = seq[b] != tok.pad_token_id
            text = tok.decode(seq[b][keep], skip_special_tokens=True)
            conf = float(tlp[b][keep].mean()) if keep.any() else -99.0
            try:
                ans = json.loads(text)
            except Exception:  # noqa: BLE001
                ans = None
            out.append((ans, conf))
        print(f"{min(i + BATCH, len(prompts))}/{len(prompts)}", flush=True)
    return out


def sig_of(ans, fam, task):
    if not isinstance(ans, dict) or not all(k in ans for k in ("target", "baseline", "baseline_special", "type")):
        return None
    o = {**ans, "confidence": "high"}
    try:
        return imaging.from_override(o, fam, task, frozenset())[0]
    except Exception:  # noqa: BLE001
        return None


def same(a, b):
    return a is not None and b is not None and (a == b or H.flip(a) == b)


def test(adapter, data, outp):
    rows = [json.loads(l) for l in open(f"{data}/test.jsonl")]
    tok, model = load(adapter)
    res = generate(tok, model, [r["prompt"] for r in rows])
    items = []
    for r, (ans, conf) in zip(rows, res):
        fam = r["prompt"].split("family: ", 1)[1].split(";", 1)[0]
        gold = json.loads(r["answer"])
        g = sig_of(gold, fam, None); p = sig_of(ans, fam, None)
        both_none = gold["type"] in ("not_conditions", "interaction") and isinstance(ans, dict) and ans.get("type") == gold["type"]
        items.append({"id": r["id"], "conf": conf, "agree": both_none or same(p, g), "type_agree": isinstance(ans, dict) and ans.get("type") == gold["type"],
                      "pred": ans, "gold": gold})
    table = []
    for tau in (-0.02, -0.05, -0.1, -0.2, -0.5, -99):
        a = [x["agree"] for x in items if x["conf"] >= tau]
        table.append({"tau": tau, "answered": round(len(a) / len(items), 3), "agree": round(sum(a) / max(1, len(a)), 3)})
    json.dump({"n": len(items), "agree": round(sum(x["agree"] for x in items) / len(items), 3),
               "type_agree": round(sum(x["type_agree"] for x in items) / len(items), 3), "by_tau": table, "items": items},
              open(outp, "w"), indent=1)
    print(json.dumps(table)); print("overall agree", round(sum(x["agree"] for x in items) / len(items), 3))


def run(adapter, hub_dir, contrasts, glossp, abbrp, ovp, outp, tau, shard, nshards):
    from codex_normalize import item_text
    C = json.load(open(contrasts)); gloss = json.load(open(glossp)); abbr = json.load(open(abbrp)); ov = json.load(open(ovp))
    todo = [a for a in json.load(open(f"{hub_dir}/corpus_unnormalized.json")) if a not in ov][shard::nshards]
    tok, model = load(adapter)
    res = generate(tok, model, [item_text(0, C[a], gloss, abbr).split("\n", 1)[1] for a in todo])
    out = {}
    for aid, (ans, conf) in zip(todo, res):
        if isinstance(ans, dict) and all(k in ans for k in ("target", "baseline", "baseline_special", "type")):
            out[aid] = {"target": ans["target"], "baseline": ans["baseline"], "baseline_special": ans["baseline_special"],
                        "type": ans["type"], "confidence": "high" if conf >= tau else "low", "source": "ft", "logprob": round(conf, 4)}
    json.dump(out, open(outp, "w"))
    print(f"{len(out)} readings of {len(todo)}")


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test(*sys.argv[2:5])
    else:
        a = sys.argv[2:]
        run(*a[:7], float(a[7]), int(a[8]), int(a[9]))
