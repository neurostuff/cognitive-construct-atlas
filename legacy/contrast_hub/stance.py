"""Stance of a contrast-level construct claim: does the sentence give the construct to the contrast's target side,
to its baseline side, or rule it out?

A matched claim sentence names a contrast (sides A and B) and constructs. The audit (reports/audit_constructs) found
sentences that name a construct the comparison subtracts, controls for or rejects ("to control for item saliency we
compared positive vs negative", "the MID contrast was not designed to estimate reward prediction error"), and
sentences whose construct belongs to the other side. Labels per (sentence, construct):
  A         A engages it more than B, or the A > B difference reflects / isolates / indexes it (a tested hypothesis too)
  B         B engages it more than A
  none      the sentence says the comparison does not reflect it, rules it out, controls or matches it across A and B,
            or it is shared by both and subtracted
  task      it describes the task or both conditions in general, not their difference
  unrelated something else: a group difference, a brain region's role, another comparison, the dependent measure
constructs.py credits A to the contrast's own direction, B to its reverse (`rev:`), task to the task itself
(`task_claims`), and drops none and unrelated.

Subcommands:
  prep   (beast) HUB_DIR LEXICON 'SCAN_GLOB[+..]' NAMED.jsonl OUT.jsonl [CLAIMS_MATCH.json]  one item per matched claim sentence
  codex  (laptop) ITEMS.jsonl OUT.json [MODEL EFFORT] [ONLY_SRC]             Codex labels, resumable
  rules  ITEMS.jsonl OUT.json                                                regex cues for `none`
  nli    (beast GPU) ITEMS.jsonl OUT.json                                    DeBERTa-v3-large NLI scores per hypothesis
  minicheck (beast, 3 GPUs, venv-mc: transformers 4.46; 5.x loads its weights wrongly) ITEMS.jsonl OUT.json [ONLY_IDS.json]
  eval   ITEMS.jsonl CODEX.json RULES.json NLI.json [MINICHECK.json]         agreement of rules, NLI, MiniCheck with Codex
"""
import collections
import json
import os
import re
import sys
from pathlib import Path

LABELS = ["A", "B", "none", "task", "unrelated"]


def _side(v):
    return "" if v in (None, "∅") else str(v).replace("_", " ")


def prep():
    import hub as H
    import match_claims as M
    from constructs import vocab_maps
    hub_dir, lexp, scan_glob, namedp, outp = sys.argv[2:7]
    matchp = sys.argv[7] if len(sys.argv) > 7 else None      # claims_match.json: the Codex-read sample claims
    hub = H.Hub.load(f"{hub_dir}/hub.json"); lex = H.Lexicon(lexp); to_ids, _, _ = vocab_maps()
    root = Path(__file__).resolve().parents[2]
    units = {}

    def unit(article, i):
        if article not in units:
            d = root / ("imaging_papers" if article.startswith("ns_") else "corpus_papers")
            try:
                u = json.load(open(d / f"{article}.json"))["units"]
                units[article] = {x.get("i", j): x["text"] for j, x in enumerate(u)}
            except Exception:  # noqa: BLE001
                units[article] = {}
        return units[article].get(i, ""), units[article].get(i - 1, "")
    seen = set(); n = collections.Counter()
    with open(outp, "w") as fo:
        for pat in scan_glob.split("+"):
            for f in sorted(Path().glob(pat) if not pat.startswith("/") else Path("/").glob(pat[1:])):
                src = "imaging" if "imaging" in f.name else "lit"
                for line in open(f):
                    r = json.loads(line)
                    if "constructs" not in r:
                        continue
                    m = M.match_one({"tasks": r["tasks"], "a": r["a"], "b": r["b"]}, hub, lex)
                    cs = [c for c in r["constructs"] if to_ids([c])]
                    key = f"{r['article']}:{r['i']}"
                    if "hub" not in m or not cs or (key, m["hub"]) in seen:
                        continue
                    seen.add((key, m["hub"])); text, before = unit(r["article"], r["i"])
                    fo.write(json.dumps({"id": f"{key}|{m['hub']}", "src": src, "article": r["article"], "i": r["i"],
                                         "text": text, "before": before[:300], "A": r["a"], "B": r["b"] or "",
                                         "hub": m["hub"], "sign": m["sign"], "constructs": cs}) + "\n"); n[src] += 1
        for line in open(namedp):
            r = json.loads(line)
            cs = [c for c in r["constructs"] if to_ids([c])]
            key = f"{r['article']}:{r['i']}"
            if not cs or (key, r["hub"]) in seen:
                continue
            seen.add((key, r["hub"])); text, before = unit(r["article"], r["i"])
            comp = [x.split(":", 1)[1] for x in r["effect"].split(";")]
            a = " and ".join(_side(c.split(">")[0]) for c in comp if _side(c.split(">")[0]))
            b = " and ".join(_side(c.split(">")[1]) for c in comp if ">" in c and _side(c.split(">")[1]))
            # the effect is written in the entry's direction when sign is +1; A > B is then the entry
            fo.write(json.dumps({"id": f"{key}|{r['hub']}", "src": "named", "article": r["article"], "i": r["i"],
                                 "text": text, "before": before[:300], "A": a, "B": b, "hub": r["hub"],
                                 "sign": 1 if r.get("sign", 1) > 0 else -1, "effect": r["effect"], "constructs": cs}) + "\n")
            n["named"] += 1
        for r in (json.load(open(matchp))["claims"] if matchp else []):
            m = M.match_one({"tasks": r["tasks"], "a": r["a"], "b": r["b"]}, hub, lex)
            cs = [c for c in r["constructs"] if to_ids([c])]
            key = f"{r['article']}:{r.get('i', r['id'])}"
            if "hub" not in m or not r.get("construct_link") or not cs or (key, m["hub"]) in seen:
                continue
            seen.add((key, m["hub"]))
            fo.write(json.dumps({"id": f"{key}|{m['hub']}", "src": "sample", "article": r["article"], "i": r.get("i", r["id"]),
                                 "text": r["text"], "before": "", "A": r["a"], "B": r["b"] or "", "hub": m["hub"],
                                 "sign": m["sign"], "constructs": cs}) + "\n"); n["sample"] += 1
    print(json.dumps(dict(n)), flush=True)


RULES = """Each item is a sentence from a paper (with the sentence before it), two conditions A and B that the sentence
compares, and constructs the sentence names. For each construct, say how the sentence relates it to the comparison of
A and B:
- A: A engages it more than B, or the A > B difference reflects, isolates or indexes it (a hypothesis the paper tests
  counts).
- B: B engages it more than A, or the B > A difference reflects it.
- none: the sentence says the comparison does not reflect it, rules it out, controls for it or matches it across A
  and B, or it is shared by both conditions and subtracted.
- task: it describes the task, the study or both conditions in general, not their difference.
- unrelated: the sentence is about something else (a group or individual difference, a brain region's role, another
  comparison, the behavioural measure itself).
Use the meaning, not the order the conditions are written in: in "the difference between congruent and incongruent
trials reflects conflict", conflict is incongruent's; if A is incongruent the answer is A, if A is congruent it is B.
Answer with the JSON object the schema defines: per item its id and one label per construct, in the order given."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"], "properties": {"items": {"type": "array", "items": {
    "type": "object", "additionalProperties": False, "required": ["id", "labels"],
    "properties": {"id": {"type": "integer"}, "labels": {"type": "array", "items": {"type": "string", "enum": LABELS}}}}}}}


def codex():
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import codex_extract as cx
    itemsp, outp = sys.argv[2:4]
    model, effort = (sys.argv[4], sys.argv[5]) if len(sys.argv) > 5 else ("gpt-6.1-sol", "low")
    only = sys.argv[6].split(",") if len(sys.argv) > 6 else None
    items = [json.loads(x) for x in open(itemsp)]
    done = json.load(open(outp)) if Path(outp).exists() else {}
    todo = [it for it in items if it["id"] not in done and (only is None or it["src"] in only)]
    B = int(os.environ.get("BATCH", "40"))
    batches = [todo[i:i + B] for i in range(0, len(todo), B)]
    disabled = [t for t in cx.TOOLS if t in cx.known_features()]
    print(f"{len(todo)} items in {len(batches)} batches", flush=True)

    def run(b):
        prompt = "\n\n".join(
            f"id {j}\nbefore: {it['before'][:250]}\nsentence: {it['text'][:600]}\nA: {it['A'][:80]}\nB: {it['B'][:80] or '(none: a simple effect of A)'}\n"
            f"constructs: {' | '.join(it['constructs'][:4])}" for j, it in enumerate(b))
        try:
            ans, usage, secs = cx.call(model, effort, RULES, prompt, disabled, schema=SCHEMA)
            return b, ans["items"], f"{secs}s"
        except Exception as exc:  # noqa: BLE001
            return b, None, str(exc)[:200]
    st = collections.Counter()
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", "6"))) as pool:
        for b, res, note in pool.map(run, batches):
            if res is None:
                print("FAILED", note, flush=True); continue
            for r in res:
                if 0 <= r["id"] < len(b):
                    it = b[r["id"]]; labs = r["labels"][:len(it["constructs"][:4])]
                    if len(labs) == len(it["constructs"][:4]):
                        done[it["id"]] = dict(zip(it["constructs"][:4], labs)); st.update(labs)
            json.dump(done, open(outp, "w"))
            print(f"batch {len(b)} {note}", flush=True)
    print(json.dumps(dict(st)), flush=True)


#: cue words that put a construct outside the comparison's difference when they govern it in the same clause
NONE_CUES = re.compile(
    r"\b(?:not|no|neither|nor|never|rather than|instead of|independent(?:ly)? of|regardless of|irrespective of|"
    r"unrelated to|unaffected by|control(?:s|led|ling)? for|account(?:s|ed|ing)? for|beyond|over and above|"
    r"cannot be (?:explained|attributed)|rule[sd]? out|ruling out|matched|equated|equal(?:ly)?|similar|comparable|"
    r"subtract(?:s|ed|ing|ion)?|cancel(?:s|led)? out|without|unlikely|did not|does not|do not|was not|were not|"
    r"is not|are not|cannot|no longer|common to|shared by|both conditions)\b", re.I)


def rules_label(text, construct):
    """`none` when a cue governs the construct: the cue sits before it in the same clause (no , ; : between, at most
    12 words apart); else None (no decision)"""
    t = text.lower(); c = construct.lower()
    k = t.find(c)
    if k < 0:
        return None
    clause = re.split(r"[;:,]|\bwhereas\b|\bwhile\b|\bbut\b", t[:k])[-1]
    words = clause.split()
    window = " ".join(words[-12:])
    return "none" if NONE_CUES.search(window) else None


def rules():
    itemsp, outp = sys.argv[2:4]
    out = {}
    for x in open(itemsp):
        it = json.loads(x)
        out[it["id"]] = {c: rules_label(it["text"], c) for c in it["constructs"][:4]}
    json.dump(out, open(outp, "w"))


def nli():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    itemsp, outp = sys.argv[2:4]
    name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tok = AutoTokenizer.from_pretrained(name); model = AutoModelForSequenceClassification.from_pretrained(name).half().cuda().eval()
    lab = {v.lower(): k for k, v in model.config.id2label.items()}
    rows = []
    for x in open(itemsp):
        it = json.loads(x)
        prem = (it["before"] + " " + it["text"]).strip()[-1500:]
        a, b = it["A"] or "the first condition", it["B"] or "baseline"
        for c in it["constructs"][:4]:
            for h, hyp in (("A", f"{a} involves more {c} than {b}."), ("B", f"{b} involves more {c} than {a}."),
                           ("diff", f"The difference between {a} and {b} reflects {c}.")):
                rows.append((it["id"], c, h, prem, hyp))
    out = collections.defaultdict(dict)
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            ch = rows[i:i + 64]
            enc = tok([r[3] for r in ch], [r[4] for r in ch], truncation="only_first", max_length=512, padding=True, return_tensors="pt").to("cuda")
            p = torch.softmax(model(**enc).logits.float(), -1).cpu()
            for r, q in zip(ch, p):
                out[r[0]].setdefault(r[1], {})[r[2]] = {"e": round(float(q[lab["entailment"]]), 4), "c": round(float(q[lab["contradiction"]]), 4)}
            if i % 6400 == 0:
                print(i, len(rows), flush=True)
    json.dump(out, open(outp, "w"))


#: Bespoke-MiniCheck-7B's prompts, from github.com/Liyan06/MiniCheck minicheck/utils.py (read 2026-11-12); its
#: support probability is the next-token probability of "yes" (minicheck/inference.py, any casing)
MC_SYSTEM = ("Determine whether the provided claim is consistent with the corresponding document. Consistency in this "
             "context implies that all information presented in the claim is substantiated by the document. If not, it "
             "should be considered inconsistent. Please assess the claim's consistency with the document by responding "
             "with either \"Yes\" or \"No\".")
MC_USER = "Document: [DOCUMENT]\nClaim: [CLAIM]"


def hypotheses(it, c):
    a, b = it["A"] or "the first condition", it["B"] or "baseline"
    return {"A": f"{a} involves more {c} than {b}.", "B": f"{b} involves more {c} than {a}.",
            "diff": f"The difference between {a} and {b} reflects {c}."}


def minicheck():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    itemsp, outp = sys.argv[2:4]
    only = set(json.load(open(sys.argv[4]))) if len(sys.argv) > 4 else None
    name = "bespokelabs/Bespoke-MiniCheck-7B"
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cfg_cls = get_class_from_dynamic_module("configuration_internlm2.InternLM2Config", name)
    cfg_cls._rope_scaling_validation = lambda self: None   # transformers 5 adds keys to rope_scaling that its check rejects
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True); tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n = torch.cuda.device_count()
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto",
                                                 max_memory={i: os.environ.get("MC_MEM", "5300MiB") for i in range(n)}).eval()
    yes = [i for t, i in tok.get_vocab().items() if tok.convert_tokens_to_string([t]).strip().lower() == "yes"]
    rows = []
    for x in open(itemsp):
        it = json.loads(x)
        if only is not None and it["id"] not in only:
            continue
        doc = (it["before"][-300:] + " " + it["text"][:700]).strip()
        for c in it["constructs"][:4]:
            for h, claim in hypotheses(it, c).items():
                msg = [{"role": "system", "content": MC_SYSTEM}, {"role": "user", "content": MC_USER.replace("[DOCUMENT]", doc).replace("[CLAIM]", claim)}]
                rows.append((it["id"], c, h, tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)))
    out = collections.defaultdict(dict)
    with torch.no_grad():
        rows.sort(key=lambda r: len(r[3]))               # similar lengths per batch: little padding
        for i in range(0, len(rows), 4):
            ch = rows[i:i + 4]
            enc = tok([r[3] for r in ch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            h = model.model(**enc, use_cache=False)[0][:, -1, :]            # the head on the last position only
            logits = model.output(h.to(model.output.weight.device)).float()
            p = torch.softmax(logits, -1)[:, yes].sum(-1).cpu()
            for r, q in zip(ch, p):
                out[r[0]].setdefault(r[1], {})[r[2]] = round(float(q), 4)
            if i % 400 == 0:
                print(i, len(rows), flush=True); json.dump(out, open(outp, "w"))
    json.dump(out, open(outp, "w"))


def minicheck_label(s, t=0.5):
    """none when no hypothesis is supported; A or B by the directional hypothesis supported more; None when only the
    undirected one is"""
    if max(s.values()) < t:
        return "none"
    if max(s["A"], s["B"]) < t:
        return None
    return "A" if s["A"] >= s["B"] else "B"


def nli_label(s, t_e=0.5, t_c=0.5):
    """a label from the three hypotheses' scores: none when the difference hypothesis is contradicted, A or B when its
    own hypothesis is entailed more than the other's, else None"""
    if s["diff"]["c"] >= t_c and s["A"]["e"] < t_e and s["B"]["e"] < t_e:
        return "none"
    if max(s["A"]["e"], s["B"]["e"], s["diff"]["e"]) < t_e:
        return None
    if s["A"]["e"] >= s["B"]["e"]:
        return "A"
    return "B"


def evaluate():
    itemsp, codexp, rulesp, nlip = sys.argv[2:6]
    mcp = sys.argv[6] if len(sys.argv) > 6 else None
    mc = json.load(open(mcp)) if mcp and Path(mcp).exists() else {}
    mcc = collections.Counter(); mcd = collections.Counter()
    items = {json.loads(x)["id"]: json.loads(x) for x in open(itemsp)}
    cx = json.load(open(codexp)); ru = json.load(open(rulesp)) if Path(rulesp).exists() else {}
    nl = json.load(open(nlip)) if Path(nlip).exists() else {}
    keep = lambda lab: lab if lab in ("A", "B") else "drop"  # noqa: E731   what constructs.py does with a label
    R = {"pairs": 0, "codex": collections.Counter()}
    rc = collections.Counter(); nc = collections.Counter(); nd = collections.Counter()
    for iid, labs in cx.items():
        for c, lab in labs.items():
            R["pairs"] += 1; R["codex"][lab] += 1
            r = ru.get(iid, {}).get(c)
            rc[("none" if lab in ("none",) else "other", r or "-")] += 1
            if iid in nl and c in nl[iid]:
                p = nli_label(nl[iid][c])
                nc[(lab, p or "-")] += 1; nd[(keep(lab), keep(p) if p else "drop")] += 1
            if iid in mc and c in mc[iid] and len(mc[iid][c]) == 3:
                p = minicheck_label(mc[iid][c])
                mcc[(lab, p or "-")] += 1; mcd[(keep(lab), keep(p) if p else "drop")] += 1
    tp = rc[("none", "none")]; fp = rc[("other", "none")]; fn = rc[("none", "-")]
    R["rules_none"] = {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3), "flagged": tp + fp}
    R["nli_confusion"] = {f"{a}->{b}": v for (a, b), v in sorted(nc.items())}
    agree = sum(v for (a, b), v in nd.items() if a == b); tot = sum(nd.values())
    R["nli_decision_agreement"] = round(agree / max(tot, 1), 3)
    R["nli_decision_confusion"] = {f"{a}->{b}": v for (a, b), v in sorted(nd.items())}
    if mcd:
        R["minicheck_confusion"] = {f"{a}->{b}": v for (a, b), v in sorted(mcc.items())}
        R["minicheck_decision_agreement"] = round(sum(v for (a, b), v in mcd.items() if a == b) / sum(mcd.values()), 3)
        R["minicheck_decision_confusion"] = {f"{a}->{b}": v for (a, b), v in sorted(mcd.items())}
    R["codex"] = dict(R["codex"])
    by = collections.defaultdict(collections.Counter)
    for iid, labs in cx.items():
        for lab in labs.values():
            by[items[iid]["src"] if iid in items else "?"][lab] += 1
    R["codex_by_source"] = {k: dict(v) for k, v in by.items()}
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    {"prep": prep, "codex": codex, "rules": rules, "nli": nli, "minicheck": minicheck, "eval": evaluate}[sys.argv[1]]()
