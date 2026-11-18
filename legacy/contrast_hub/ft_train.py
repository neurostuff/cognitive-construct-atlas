"""QLoRA fine-tune of a small instruct model as the hub's contrast normalizer (ft_data.py examples).

Base: Qwen/Qwen3-4B-Instruct-2507, 4-bit NF4 with bf16 compute (bitsandbytes), LoRA r=16, alpha 32, dropout 0.05 on
all attention and MLP projections (the QLoRA recipe, Dettmers et al. 2023). Chat template with ft_data.SYSTEM as the
system message; the loss is on the answer tokens only. AdamW lr 2e-4, cosine, warmup 3%, effective batch 16, max
length 640, EPOCHS epochs (micro-batch 2: the 151k-token logits dominate memory on an 8 GB card). Saves the adapter and tokenizer to OUT/adapter.

usage: torchrun --nproc_per_node N ft_train.py DATA_DIR OUT_DIR [EPOCHS]   (pondie venv; HF_HOME with the base
downloaded; data parallel over N GPUs: each rank trains on its share and the LoRA gradients are averaged over
gloo on the CPU before every optimizer step, because NCCL fails on beast's 3070s with this torch build)
"""
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig,
                          get_cosine_schedule_with_warmup)

#: FT_MODEL=qwen (Qwen/Qwen3-4B-Instruct-2507, chat template) or nu (numind/NuExtract3, its template wire format as
#: pondie's astra-distill train_record.render writes it; the vision tower is dropped)
FT_MODEL = os.environ.get("FT_MODEL", "qwen")
BASE = {"qwen": "Qwen/Qwen3-4B-Instruct-2507", "nu": "numind/NuExtract3"}[FT_MODEL]
NU_TEMPLATE = json.dumps({"target": ["string"], "baseline": ["string"], "baseline_special": ["@base"],
                          "type": ["difference", "baseline", "main_effect", "parametric", "interaction", "task_vs_base", "not_conditions"]}, indent=2)
SYSTEM = ("Normalize this neuroimaging contrast into condition atoms (axis:value). target/baseline: the atoms on which "
          "the sides differ; baseline_special @base for rest, fixation or a low-level baseline; type: difference, "
          "baseline, main_effect, parametric, interaction, task_vs_base or not_conditions. Answer with JSON.")
MAXLEN, MICRO, ACCUM, LR = 640, (1 if FT_MODEL == "nu" else 2), (16 if FT_MODEL == "nu" else 8), 2e-4   # NuExtract3's 248k vocabulary: micro-batch 1


def encode(tok, prompt, answer=None):
    if FT_MODEL == "nu":
        head = (f"<|im_start|>user\n【task】structured\n【template_start】{NU_TEMPLATE}【template_end】\n【document_start】\n"
                f"{SYSTEM}\n\n{prompt}\n【document_end】<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    else:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(head, add_special_tokens=False)["input_ids"]
    if answer is None:
        return ids
    ans = tok(answer + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids, labels = (ids + ans)[:MAXLEN], ([-100] * len(ids) + ans)[:MAXLEN]
    return ids, labels


def load_base(q, device):
    if FT_MODEL == "nu":
        m = AutoModelForImageTextToText.from_pretrained(BASE, quantization_config=q, dtype=torch.bfloat16, device_map={"": device})
        for holder in (m, getattr(m, "model", None)):
            for attr in ("visual", "vision_tower", "vision_model"):
                if holder is not None and getattr(holder, attr, None) is not None:
                    setattr(holder, attr, None)
        return m
    return AutoModelForCausalLM.from_pretrained(BASE, quantization_config=q, dtype=torch.bfloat16, device_map={"": device})


def main():
    data, out = Path(sys.argv[1]), Path(sys.argv[2]); epochs = float(sys.argv[3]) if len(sys.argv) > 3 else 2
    random.seed(0); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(data / "train.jsonl")]
    ex = [encode(tok, r["prompt"], r["answer"]) for r in rows]
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                           bnb_4bit_use_double_quant=True)
    ddp = "RANK" in os.environ
    rank, world = (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])) if ddp else (0, 1)
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("gloo"); torch.cuda.set_device(local)
    model = load_base(q, local)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if rank == 0:
        model.print_trainable_parameters()
    core = model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    accum = max(1, ACCUM // world)                     # the effective batch stays MICRO * ACCUM
    steps = math.ceil(len(ex) * epochs / (MICRO * accum * world))
    sch = get_cosine_schedule_with_warmup(opt, int(0.03 * steps), steps)
    model.train(); step = 0; seen = 0; tot = 0.0
    order = []
    per = steps * MICRO * accum
    while len(order) < per * world:
        o = list(range(len(ex))); random.shuffle(o); order += o
    order = order[rank:per * world:world]               # this rank's share, same seed on every rank
    for i in range(0, per, MICRO):
        b = [ex[j] for j in order[i:i + MICRO]]
        L = max(len(x[0]) for x in b)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in b]).cuda()
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in b]).cuda()
        att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in b]).cuda()
        # A batch with nothing to learn (an answer cut off by MAXLEN) or a non-finite loss adds no gradient, but still
        # counts toward the accumulation, so every rank reaches each all-reduce together.
        if (lab != -100).any():
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / accum
            if torch.isfinite(loss):
                loss.backward(); tot += loss.item()
            else:
                print("skipping a non-finite loss", flush=True)
        seen += 1
        if seen % accum == 0:
            if ddp:                                    # average the LoRA gradients across ranks (gloo, CPU)
                flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().float().flatten().cpu()
                                  for p in params])
                dist.all_reduce(flat); flat /= world; o = 0
                for p in params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    n = p.numel(); p.grad.copy_(flat[o:o + n].view_as(p.grad).to(p.grad.dtype)); o += n
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(); step += 1
            if (step % 25 == 0 or step == steps) and rank == 0:
                print(f"step {step}/{steps} loss {tot / 25:.4f}", flush=True)
            if step % 25 == 0:
                tot = 0.0
    if rank != 0:
        dist.destroy_process_group(); return
    out.mkdir(parents=True, exist_ok=True)
    core.save_pretrained(out / "adapter"); tok.save_pretrained(out / "adapter")
    json.dump({"world": world, "model": FT_MODEL, "base": BASE, "epochs": epochs, "examples": len(ex), "lr": LR, "batch": MICRO * ACCUM, "maxlen": MAXLEN,
               "lora": {"r": 16, "alpha": 32, "dropout": 0.05}}, open(out / "train_config.json", "w"), indent=1)
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
