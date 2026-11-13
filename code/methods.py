"""Candidate construct-claim detectors and one shared attribution step, over the pilot bundles.

Each detector writes predictions/<method>.jsonl: one line per predicted unit
{"article", "i", "task", "analyses", "score"} — task/analyses from `attribute` unless the method attributes itself.

usage:
  python -I methods.py lexical PILOT_DIR TASK_INVENTORY OUT_DIR     # cue, cue+anchor, ev1, ev1+cue
  python -I methods.py nli PILOT_DIR TASK_INVENTORY OUT_DIR [threads]
"""
import json
import re
import sys
from pathlib import Path

CUE = re.compile(
    r"\b(measur(e|es|ed|ing)|index(es|ed|ing)?|indices|isolat\w*|tap(s|ped|ping)?|assess(es|ed|ing)?|reflect\w*|"
    r"sensitive to|prob(e|es|ed|ing)|engag\w*|elicit\w*|recruit\w*|requir\w*|tax(es|ed|ing)|load(s|ed|ing)? on|"
    r"captur\w*|operationali[sz]\w*|designed to|thought to|believed to|proxy (for|of)|marker of|estimate of|"
    r"used to (study|investigate|examine|assess|measure|probe|elicit|induce)|well[- ]established (measure|paradigm)|"
    r"involv(e|es|ed|ing)|demand(s|ed|ing)?)\b", re.I)
ANCHOR = re.compile(r"\b(task|paradigm|condition|conditions|trials?|contrasts?|blocks?|effect|score|measure|"
                    r"interference|cost|load)\b|>|\bvs\.?\b|\bversus\b|\bminus\b", re.I)
STOP = set("""a an the of and or in on for to with by vs versus minus than task tasks condition conditions trial trials
contrast contrasts activation activity effect effects block blocks baseline greater less more fixation rest
during between all main interaction group groups patients controls healthy session sessions phase run runs
positive negative increase decrease correlation parametric modulation response responses vs. > < - &""".split())
SECTIONS = {"abstract", "introduction", "methods", "results", "discussion", "front"}


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", (s or "").lower()) if w not in STOP and len(w) > 1}


def own_patterns(inventory, canon_ids):
    """regexes for tasks NOT among the paper's canonical tasks (other-task mentions)"""
    pats = []
    for t in inventory["tasks"]:
        if t["id"] in canon_ids or t["family"] == "pseudo_tasks":
            continue
        alts = [re.escape(a) for a in [t["name"], *t.get("aliases", [])] if a and len(a) > 3]
        alts += [p for p in t.get("patterns", []) if p]
        if alts:
            pats.append(re.compile(r"\b(" + "|".join(alts) + r")\b", re.I))
    return pats


def attribute(p, i):
    """task and analyses a unit is about, from the paper's own records: lexical overlap with task names,
    condition names and analysis names (+ contrast-set labels), looking back two units for the task."""
    text = p["units"][i]["text"]
    w = words(text)
    ctx = w | words(p["units"][i - 1]["text"] if i else "") | words(p["units"][i - 2]["text"] if i > 1 else "")
    tasks = p["tasks"]
    def tscore(t):
        return len(ctx & (words(t["name"]) | {x for c in t["conditions"] for x in words(c["name"])}))
    task = max(tasks, key=lambda t: (tscore(t), sum(t["task"] in a["tasks"] for a in p["analyses"])))["task"] if tasks else None
    cond_words = {x for t in tasks if t["task"] == task for c in t["conditions"] for x in words(c["name"])}
    contrastish = bool(w & cond_words) or bool(re.search(r">|\bvs\.?\b|\bversus\b|\bminus\b|\bcontrast|\btrials?\b|"
                                                         r"\beffect\b|\bcondition", text, re.I))
    analyses = []
    if contrastish:
        scored = []
        for a in p["analyses"]:
            if task and a["tasks"] and task not in a["tasks"]:
                continue
            aw = words(a["name"]) | words(a.get("contrast_set_label"))
            scored.append((len(w & aw), a["analysis"]))
        best = max((s for s, _ in scored), default=0)
        if best >= 1:
            analyses = [a for s, a in scored if s == best]
    return task, analyses


def evidence_units(p):
    """unit -> (task, analyses) for every unit the record's ev1 evidence cites, with the slot owner as attribution"""
    out = {}
    for t in p["tasks"]:
        for u in [x for v in t["evidence_units"].values() for x in v] + t["condition_units"]:
            out.setdefault(u, (t["task"], []))
    for a in p["analyses"]:
        for u in [x for v in a["evidence_units"].values() for x in v]:
            task = a["tasks"][0] if a["tasks"] else (p["tasks"][0]["task"] if p["tasks"] else None)
            cur = out.get(u)
            if cur is None or cur[1] == [] and cur[0] == task:
                out[u] = (task, [a["analysis"]])
            elif cur[0] == task:
                out[u] = (task, sorted(set(cur[1]) | {a["analysis"]}))
    for u in p.get("hypothesis_units") or []:
        out.setdefault(u, (p["tasks"][0]["task"] if p["tasks"] else None, []))
    return out


def papers(d):
    for f in sorted(Path(d, "papers").glob("*.json")):
        yield json.loads(f.read_text())


def lexical(pilot, inv_path, out):
    inv = json.load(open(inv_path))
    files = {m: open(Path(out, f"{m}.jsonl"), "w") for m in ("cue", "cue_anchor", "cue_anchor_noother", "ev1", "ev1_cue")}
    for p in papers(pilot):
        others = own_patterns(inv, {t["canonical"] for t in p["tasks"]})
        own_names = [t["name"] for t in p["tasks"] if t["name"]]
        ev = evidence_units(p)
        for u in p["units"]:
            i, text = u["i"], u["text"]
            if u["section"] not in SECTIONS or len(text) < 25:
                continue
            cue = bool(CUE.search(text))
            def emit(m, task, an):
                files[m].write(json.dumps({"article": p["article"], "i": i, "task": task, "analyses": an, "score": 1}) + "\n")
            if cue:
                emit("cue", *attribute(p, i))
                if ANCHOR.search(text) or any(n.lower() in text.lower() for n in own_names):
                    emit("cue_anchor", *attribute(p, i))
                    other = any(rx.search(text) for rx in others)
                    mine = any(n.lower() in text.lower() for n in own_names)
                    if not other or mine:
                        emit("cue_anchor_noother", *attribute(p, i))
            if i in ev:
                emit("ev1", *ev[i])
                if cue:
                    emit("ev1_cue", *ev[i])
    for f in files.values():
        f.close()


HYP = {
    "claim": "This sentence says what psychological process a task, condition or contrast measures.",
    "region": "This sentence says what psychological process a brain region supports.",
    "procedure": "This sentence describes the experimental procedure or stimuli.",
    "result": "This sentence reports a statistical result.",
}


def nli(pilot, inv_path, out, threads=8, only=None, hyps=("claim", "region")):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(threads)
    name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    ent = model.config.label2id.get("entailment", 0)
    H = {h: HYP[h] for h in hyps}
    import time
    t0 = time.time()
    with open(Path(out, "nli_scores.jsonl"), "w") as f:
        for p in papers(pilot):
            if only and p["article"] not in only:
                continue
            units = [u for u in p["units"] if u["section"] in SECTIONS and len(u["text"]) >= 25]
            for k in range(0, len(units), 16):
                batch = units[k:k + 16]
                prem = [u["text"][:1000] for u in batch for _ in H]
                hyp = [h for _ in batch for h in H.values()]
                with torch.no_grad():
                    enc = tok(prem, hyp, truncation=True, max_length=160, padding=True, return_tensors="pt")
                    pr = model(**enc).logits.softmax(-1)[:, ent].reshape(len(batch), len(H))
                for u, row in zip(batch, pr.tolist()):
                    task, an = attribute(p, u["i"])
                    f.write(json.dumps({"article": p["article"], "i": u["i"], "task": task, "analyses": an,
                                        **{f"p_{h}": round(s, 4) for h, s in zip(H, row)}}) + "\n")
            f.flush()
            print(f"{time.time() - t0:7.0f}s {p['article']} {len(units)} units", flush=True)


if __name__ == "__main__":
    cmd, pilot, inv, out = sys.argv[1:5]
    Path(out).mkdir(parents=True, exist_ok=True)
    if cmd == "lexical":
        lexical(pilot, inv, out)
    else:
        only = set(sys.argv[6].split(",")) if len(sys.argv) > 6 else None
        nli(pilot, inv, out, int(sys.argv[5]) if len(sys.argv) > 5 else 8, only)
