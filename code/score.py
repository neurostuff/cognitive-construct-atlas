"""Score construct-claim methods against the pilot gold labels.

Positive = a unit labelled own_task or own_contrast. Detection is scored per unit; attribution on the true
positives a method finds: task right, level right (task vs contrast), contrast hit (predicted analyses share one with
gold), contrast exact. CIs: 2,000 paper-level bootstrap resamples.

usage: python -I score.py [labeller=A]  (run from the cc/ dir)
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
LAB = sys.argv[1] if len(sys.argv) > 1 else "A"
STRICT = "--strict" in sys.argv  # drop units the labeller marked borderline from gold and from predictions
POS = {"own_task", "own_contrast"}


def load_gold(lab):
    gold = {}
    for f in sorted((ROOT / "labels").glob(f"*.{lab}.jsonl")):
        art = f.name.split(".")[0]
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        gold[art] = {r["i"]: r for r in rows}
    return gold


def strict_filter(gold, methods):
    for a, g in gold.items():
        drop = {i for i, r in g.items() if "borderline" in (r.get("note") or "")}
        for i in drop:
            del g[i]
        for pred in methods.values():
            for i in drop:
                pred.get(a, {}).pop(i, None)


def units_meta():
    m = {}
    for f in (ROOT / "pilot/papers").glob("*.json"):
        p = json.loads(f.read_text())
        m[p["article"]] = {u["i"]: u["section"] for u in p["units"]}
    return m


def load_pred(path, arts, keep=None):
    pred = defaultdict(dict)
    for l in open(path):
        r = json.loads(l)
        if r["article"] in arts and (keep is None or keep(r)):
            pred[r["article"]][r["i"]] = r
    return pred


def load_llm(arts, tag):
    pred = defaultdict(dict)
    for a in arts:
        f = ROOT / "llm" / f"{a}.{tag}.jsonl"
        if not f.exists():
            return None
        for l in f.read_text().splitlines():
            l = l.strip().strip(",")
            if not l.startswith("{"):
                continue
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if "i" not in r or r.get("category", "own_task") not in POS:  # v2 header line / negatives
                continue
            i = int(r["i"])  # a unit about several tasks may have one line per task: keep them all
            cur = pred[a].setdefault(i, {"task": r.get("task"), "tasks": [], "analyses": [],
                                         "category": r.get("category")})
            cur["tasks"].append(r.get("task"))
            cur["analyses"] = sorted(set(cur["analyses"]) | set(r.get("analyses") or []))
    return pred


def counts(gold, pred, arts):
    c = Counter()
    for a in arts:
        g = {i for i, r in gold[a].items() if r["category"] in POS}
        p = set(pred.get(a, {}))
        c["tp"] += len(g & p); c["fp"] += len(p - g); c["fn"] += len(g - p)
        for i in g & p:
            gr, pr = gold[a][i], pred[a][i]
            if gr.get("attr") is False:  # added by adjudication: no gold task/contrast to score attribution against
                continue
            c["att_n"] += 1
            c["task_ok"] += gr.get("task") in (pr.get("tasks") or [pr.get("task")])
            glev = "contrast" if gr["category"] == "own_contrast" else "task"
            # the method's own category when it gives one; else (cue, ev1) whether it attached analyses
            plev = ("contrast" if pr.get("category") == "own_contrast" else "task") if pr.get("category") \
                else ("contrast" if pr.get("analyses") else "task")
            c["level_ok"] += glev == plev
            if gr["category"] == "own_contrast" and gr.get("analyses"):
                c["linked"] += bool(pr.get("analyses"))
                c["c_n"] += 1
                inter = set(gr["analyses"]) & set(pr.get("analyses") or [])
                c["c_hit"] += bool(inter)
                c["c_exact"] += set(gr["analyses"]) == set(pr.get("analyses") or [])
        for i in p - g:
            c["fp_" + (gold[a][i]["category"] if i in gold[a] else "none")] += 1
    return c


def prf(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def boot(gold, pred, arts, n=2000, seed=0):
    rng = random.Random(seed); out = []
    per = {a: counts(gold, pred, [a]) for a in arts}
    for _ in range(n):
        c = Counter()
        for a in rng.choices(arts, k=len(arts)):
            c.update(per[a])
        out.append(prf(c))
    q = lambda xs: (sorted(xs)[int(0.025 * n)], sorted(xs)[int(0.975 * n)])
    return [q([o[k] for o in out]) for k in range(3)]


def recall_by(gold, pred, arts, sections, key):
    hit, tot = Counter(), Counter()
    for a in arts:
        for i, r in gold[a].items():
            if r["category"] not in POS:
                continue
            k = key(r, sections[a].get(i)) or "unlabelled"
            tot[k] += 1; hit[k] += i in pred.get(a, {})
    return {k: f"{hit[k]}/{tot[k]}" for k in sorted(tot)}


def main():
    gold = load_gold(LAB)
    arts = sorted(gold)
    sec = units_meta()
    P = ROOT / "preds"
    methods = {}
    for m in ("cue", "cue_anchor", "cue_anchor_noother", "ev1", "ev1_cue"):
        methods[m] = load_pred(P / f"{m}.jsonl", set(arts))
    nli = P / "nli_scores.jsonl"
    if nli.exists():
        hyps = ("claim", "region", "procedure", "result")
        scored = load_pred(nli, set(arts))
        if all(a in scored for a in arts):
            for t in (0.5, 0.7, 0.9):
                methods[f"nli_claim>={t}"] = {a: {i: r for i, r in d.items() if r["p_claim"] >= t} for a, d in scored.items()}
            methods["nli_argmax_claim"] = {a: {i: r for i, r in d.items()
                                               if max(hyps, key=lambda h: r[f"p_{h}"]) == "claim" and r["p_claim"] >= 0.5}
                                           for a, d in scored.items()}
            methods["nli>=0.5_and_cue_anchor"] = {a: {i: r for i, r in d.items() if r["p_claim"] >= 0.5
                                                      and i in methods["cue_anchor"].get(a, {})} for a, d in scored.items()}
    for tag in ("haiku", "haikuguide", "sonnetguide", "sonnetv2", "codex-luna", "codex-terra", "codex-astra", "codex-terra-linked", "codex-astra-linked"):
        llm = load_llm(arts, tag)
        if llm is not None:
            methods[f"llm_{tag}"] = llm
    if LAB != "A":  # the other annotator scored as a method = inter-annotator agreement
        a = load_gold("A")
        methods["labeller_A"] = {k: {i: r for i, r in a[k].items() if r["category"] in POS} for k in arts}
    if STRICT:
        strict_filter(gold, methods)
    n_pos = sum(r["category"] in POS for a in arts for r in gold[a].values())
    cats = Counter(r["category"] for a in arts for r in gold[a].values())
    print(f"gold {LAB}: {len(arts)} papers, {n_pos} positives; categories {dict(cats)}")
    print(f"{'method':28s} {'pred':>5s} {'P':>5s} {'R':>5s} {'F1':>5s}  P95          R95          F1_95        "
          f"task  level c_hit c_exact  fp: none/other/region")
    res = {}
    for m, pred in methods.items():
        c = counts(gold, pred, arts); p, r, f = prf(c); ci = boot(gold, pred, arts)
        an = c["att_n"] or 1; cn = c["c_n"] or 1
        res[m] = {"pred": c["tp"] + c["fp"], "P": p, "R": r, "F1": f, "ci": ci, "task": c["task_ok"] / an,
                  "level": c["level_ok"] / an, "c_hit": c["c_hit"] / cn, "c_exact": c["c_exact"] / cn, "c_n": c["c_n"],
                  "att_n": c["att_n"], "fp": {k[3:]: v for k, v in c.items() if k.startswith("fp_")},
                  "recall_by_section": recall_by(gold, pred, arts, sec, lambda g, s: s),
                  "recall_by_polarity": recall_by(gold, pred, arts, sec, lambda g, s: g.get("polarity")),
                  "recall_by_category": recall_by(gold, pred, arts, sec, lambda g, s: g["category"])}
        fmt = lambda t: f"[{t[0]:.2f},{t[1]:.2f}]"
        print(f"{m:28s} {c['tp']+c['fp']:5d} {p:5.2f} {r:5.2f} {f:5.2f}  {fmt(ci[0])}  {fmt(ci[1])}  {fmt(ci[2])}  "
              f"{c['task_ok']/an:4.2f}  {c['level_ok']/an:4.2f}  {c['c_hit']/cn:4.2f}  {c['c_exact']/cn:4.2f}   "
              f"{c['fp_none']}/{c['fp_other_task']}/{c['fp_region']}")
    for m in res:
        print(m, "by section", res[m]["recall_by_section"], "| polarity", res[m]["recall_by_polarity"],
              "| category", res[m]["recall_by_category"])
    json.dump(res, open(ROOT / f"scores.{LAB}{'.strict' if STRICT else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
