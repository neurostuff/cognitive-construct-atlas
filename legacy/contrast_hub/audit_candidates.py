"""Candidates for the paper -> contrast -> construct audit (James 2026-11-13: "audits of tasks/contrasts appearing in
constructs they do not appear to belong in, and have the evidence checked from the primary literature").

Not a comprehensive review: a screen that ranks placements worth a reader's time. Three kinds:

1. placements: an analysis carries a construct that is rare for its task and for which its task is rare
   (n(construct, task) <= MAX_N contrasts, <= MAX_SHARE of the construct's contrasts and of the task's contrasts),
   and whose task family holds < FAM_SHARE of the construct's contrasts. Per analysis, the flagged constructs are
   grouped: all of its constructs flagged suggests a wrong task label (the cartoon contrast filed under flanker), one
   of several a wrong construct tag. Each carries what the audit needs to weigh it:
   - source of the tag (tier: claims, codex, atlas, paper, paper_phase, task);
   - whether the contrast is normalized (a hub signature) and how many other papers share that signature in the same
     direction: with others, their constructs are evidence too; alone, the paper is the only source;
   - an imaging score: mean log2 P(peak | construct) / P(peak | average contrast) over the analysis's peaks, from the
     most typical fit's own mixture for the construct (in-sample, so a thin construct fits its own contrasts; a
     negative score on a well-supported construct is the informative case);
   - a task-side score, for wrong task labels with right constructs (funny > neutral cartoons filed under flanker
     scores well on humor): task_fit, the same log2 ratio under its task's typical contrast (the model's prediction
     averaged over the task's other contrasts, this one left out); best_task, the task (>= MIN_TASK contrasts) whose
     typical contrast fits its peaks best; task_margin = best_task_fit - task_fit.
2. hierarchy edges: the kind-of edges cogat_connect.py flagged for removal (not applied), with the contrasts the child
   pools into the parent and the correlation of the two constructs' own maps: a wrong edge puts every contrast of the
   child into the parent's map (cognitive load under understanding mental states).
3. signatures: hub signatures (>= MIN_SIG papers) whose analyses disagree on constructs: the share of analyses
   carrying the signature's most common construct is < SIG_AGREE.

usage: (beast, pilot-results/atm; cbmr-gpu venv) python ../code/contrast_hub/audit_candidates.py OUT.json
"""
import collections
import json
import sys

import numpy as np

sys.path.insert(0, "/data/james/agents/dev/task-networks-repro/repo/src")
from task_networks.corpus import load_corpus  # noqa: E402

MAX_N, MAX_SHARE, FAM_SHARE = 3, 0.05, 0.10
MIN_CONSTRUCT = 20   # contrasts a construct needs for "rare within it" to mean something
MIN_SIG, SIG_AGREE = 4, 0.5
MIN_TASK = 10      # contrasts a task needs to be offered as the better-fitting task
CORPUS, HUB, CONS, REL = "corpus_w_B.json", "../contrast_hub/v7", "../contrast_hub/v7/constructs_dir/contrast_constructs_phase.json", "maps_release/K21_w_B"
SPECIAL = ("background", "deactivation")


def main():
    outp = sys.argv[1]
    H = json.load(open(CORPUS)); P = H["parents"]; A = json.load(open(f"{HUB}/corpus_assign.json")); F = json.load(open(CONS))
    C = load_corpus(CORPUS); nodes = C["nodes"]; idx = {n: i for i, n in enumerate(nodes)}; row = {k: i for i, k in enumerate(C["ids"])}
    s = json.load(open(f"{REL}/networks.json"))["most_typical_seed"]
    th = np.load(f"release_wB_K21/seeds/gibbs_s{s}_theta.npy").astype(float); be = np.load(f"release_wB_K21/seeds/gibbs_s{s}_beta.npy").astype(float)
    is_task = lambda i: not i.startswith(("c:", "m:")) and i not in SPECIAL  # noqa: E731
    fams = {p for t, ps in P.items() if is_task(t) for p in ps if is_task(p)}
    fam = lambda t: t if t in fams else next((p for p in P.get(t, []) if p in fams), None)  # noqa: E731
    task_of = {k: next((i for i in d["ids"] if is_task(i)), None) for k, d in H["docs"].items()}
    n_c, n_t, n_ct, n_cf = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    for k, d in H["docs"].items():
        t = task_of[k]; n_t[t] += 1
        for c in d["ids"]:
            if c.startswith("c:"):
                n_c[c] += 1; n_ct[c, t] += 1; n_cf[c, fam(t)] += 1
    sig = collections.defaultdict(set); sig_docs = collections.defaultdict(list)
    for k, d in H["docs"].items():
        a = A.get(k)
        if a:
            sig[a["hub"], a.get("sign", 1)].add(d["group"]); sig_docs[a["hub"], a.get("sign", 1)].append(k)
    wts = np.bincount(np.concatenate([a for a in C["auth"] if len(a)]), minlength=len(nodes)); pall = (wts / wts.sum()) @ (th @ be)

    def imaging(k, c):
        f = C["foci"][row[k]]
        if not len(f):
            return None
        return round(float(np.mean(np.log2((th[idx[c]] @ be)[f] / pall[f]))), 2)
    pa = th @ be; tsum = collections.defaultdict(lambda: np.zeros(be.shape[1])); tn = collections.Counter()
    for k in C["ids"]:
        au = C["auth"][row[k]]
        if len(au) and len(C["toks"][row[k]]):
            tsum[task_of[k]] += pa[au].mean(0); tn[task_of[k]] += 1
    alts = [t for t in tsum if tn[t] >= MIN_TASK and t not in fams]; TP = np.array([tsum[t] / tn[t] for t in alts])

    def task_side(k):
        f = C["foci"][row[k]]; au = C["auth"][row[k]]; t = task_of[k]
        if not len(f) or not len(au):
            return {}
        own = None
        if tn[t] > 1 and len(C["toks"][row[k]]):
            own = float(np.mean(np.log2(((tsum[t] - pa[au].mean(0)) / (tn[t] - 1))[f] / pall[f])))
        fit = np.log2(TP[:, f] / pall[f]).mean(1); fit[[i for i, x in enumerate(alts) if x == t]] = -np.inf
        b = int(np.argmax(fit))
        return {"task_fit": None if own is None else round(own, 2), "best_task": alts[b], "best_task_fit": round(float(fit[b]), 2),
                "task_margin": None if own is None else round(float(fit[b]) - own, 2)}
    placements = []
    for k, d in H["docs"].items():
        t = task_of[k]; cs = [c for c in d["ids"] if c.startswith("c:")]
        flagged = [c for c in cs if n_c[c] >= MIN_CONSTRUCT and n_ct[c, t] <= MAX_N and n_ct[c, t] / n_c[c] <= MAX_SHARE
                   and n_ct[c, t] / n_t[t] <= MAX_SHARE and n_cf[c, fam(t)] / n_c[c] < FAM_SHARE]
        if not flagged:
            continue
        a = A.get(k); key = (a["hub"], a.get("sign", 1)) if a else None
        placements.append({
            "analysis": k, "paper": d["group"], "task": t, "family": fam(t), "task_contrasts": n_t[t],
            "tier": (F.get(k) or {}).get("tier"), "confidence": (F.get(k) or {}).get("confidence"),
            "signature": (a["hub"] + (" (reversed)" if a.get("sign", 1) < 0 else "")) if a else None,
            "other_papers_with_signature": (len(sig[key]) - 1) if key else 0,
            "constructs": [c[2:] for c in cs],
            "flagged": [{"construct": c[2:], "construct_contrasts": n_c[c], "in_this_task": n_ct[c, t],
                         "family_share": round(n_cf[c, fam(t)] / n_c[c], 3), "imaging": imaging(k, c)} for c in flagged],
            "all_flagged": len(flagged) == len(cs), **task_side(k)})
    for p in placements:
        im = [f["imaging"] for f in p["flagged"] if f["imaging"] is not None]
        p["min_imaging"] = min(im) if im else None
    placements.sort(key=lambda p: (not p["all_flagged"], p["min_imaging"] if p["min_imaging"] is not None else 0))
    conn = json.load(open("cogat_connect.json")); edges = []
    kids = collections.defaultdict(list)
    for c, ps in P.items():
        if c.startswith("c:"):
            for q in ps:
                if q.startswith("c:"):
                    kids[q].append(c)

    def below(c, seen=None):
        seen = {c} if seen is None else seen
        for x in kids[c]:
            if x not in seen:
                seen.add(x); below(x, seen)
        return seen
    docs_of = collections.defaultdict(set)
    for k, d in H["docs"].items():
        for c in d["ids"]:
            if c.startswith("c:"):
                docs_of[c].add(k)
    lr = lambda c: np.log2((th[idx[c]] @ be) / pall)  # noqa: E731
    for child, v in conn.items():
        for r in v["remove"]:
            par = "c:" + r["parent"].strip().lower()
            if child not in P or par not in P.get(child, []):
                continue
            pooled = set().union(*(docs_of[x] for x in below(child))); own = docs_of[par]
            edges.append({"child": child[2:], "parent": par[2:], "reason": r["reason"], "child_contrasts_pooled": len(pooled),
                          "parent_own_contrasts": len(own), "pooled_not_tagged_parent": len(pooled - own),
                          "map_r": round(float(np.corrcoef(lr(child), lr(par))[0, 1]), 2) if n_c[child] and n_c[par] else None})
    edges.sort(key=lambda e: -e["pooled_not_tagged_parent"])
    sigs = []
    for key, ks in sig_docs.items():
        if len(sig[key]) < MIN_SIG:
            continue
        cc = collections.Counter(c for k in ks for c in H["docs"][k]["ids"] if c.startswith("c:"))
        if not cc:
            continue
        top, n = cc.most_common(1)[0]
        if n / len(ks) < SIG_AGREE:
            sigs.append({"signature": key[0] + (" (reversed)" if key[1] < 0 else ""), "analyses": len(ks), "papers": len(sig[key]),
                         "top_construct": top[2:], "top_share": round(n / len(ks), 2),
                         "constructs": [[c[2:], m] for c, m in cc.most_common(6)],
                         "tiers": dict(collections.Counter((F.get(k) or {}).get("tier") for k in ks))})
    sigs.sort(key=lambda x: -x["analyses"])
    json.dump({"placements": placements, "edges": edges, "signatures": sigs,
               "params": {"MAX_N": MAX_N, "MAX_SHARE": MAX_SHARE, "FAM_SHARE": FAM_SHARE, "MIN_CONSTRUCT": MIN_CONSTRUCT,
                          "MIN_SIG": MIN_SIG, "SIG_AGREE": SIG_AGREE}}, open(outp, "w"), indent=1)
    af = [p for p in placements if p["all_flagged"]]
    print(json.dumps({"placements": len(placements), "papers": len({p["paper"] for p in placements}),
                      "all constructs flagged (task label suspect)": len(af),
                      "all flagged and another task fits its peaks better by >= 1 log2": sum(1 for p in af if (p.get("task_margin") or 0) >= 1),
                      "negative imaging on a flagged construct": sum(1 for p in placements if (p["min_imaging"] or 0) < 0),
                      "unnormalized": sum(1 for p in placements if not p["signature"]),
                      "normalized, signature shared with other papers": sum(1 for p in placements if p["other_papers_with_signature"] > 0),
                      "by tier": dict(collections.Counter(p["tier"] for p in placements)),
                      "edges flagged": len(edges), "edges pooling >= 50 untagged contrasts": sum(e["pooled_not_tagged_parent"] >= 50 for e in edges),
                      "signatures in disagreement": len(sigs)}, indent=1))
    print("\ntask label suspect (all constructs flagged), lowest imaging first:")
    for p in af[:12]:
        print("  %s | %s | %s | tier %s | %s" % (p["analysis"], p["task"], [f["construct"] for f in p["flagged"]], p["tier"], p["min_imaging"]))
    print("\ntask label suspect, right-looking constructs (all flagged, imaging >= 0), largest task margin first:")
    for p in sorted([p for p in af if (p["min_imaging"] or 0) >= 0 and p.get("task_margin") is not None], key=lambda p: -p["task_margin"])[:12]:
        print("  %s | %s -> %s (margin %.2f) | %s" % (p["analysis"], p["task"], p["best_task"], p["task_margin"], [f["construct"] for f in p["flagged"]]))
    print("\nedges, most contrasts pooled:")
    for e in edges[:10]:
        print("  %s -> %s: %d pooled (%d not tagged with the parent), map r %s" % (e["child"], e["parent"], e["child_contrasts_pooled"], e["pooled_not_tagged_parent"], e["map_r"]))
    print("\nsignatures in disagreement:")
    for x in sigs[:8]:
        print("  %s: %d analyses, %d papers, top %s %.2f" % (x["signature"][:70], x["analyses"], x["papers"], x["top_construct"], x["top_share"]))


if __name__ == "__main__":
    main()
