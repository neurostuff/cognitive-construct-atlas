"""Held-out tests for the nested corpus (nested_corpus.py): 5-fold paper CV and held-out tasks, written in the formats
of cv_abs.py and zeroshot_tasks.py so compare_authors.py compares them with corpus B and R.

Same folds, fit calls and seeds as those scripts: CV fits with restarts 2, iters 1000, seed 0; held-out tasks with
restarts 1, the model's default iterations, seed 0, the same task groups (taken from REFERENCE_CORPUS, the corpus the
other runs used, so the grouping is identical).

Scoring a held-out contrast: its own node has no data, so its mixture is the model's prior, s * mean of its parents'
pooled mixtures + ALPHA (task_networks/model.py). One change: a parent with no training data (a held-out task) uses
its own prior (its family's pooled mixture) where the model would use a flat pooled mixture; in corpus B a held-out
task author likewise gets its parents' mixture. Everything else is as cv_abs.py's rows and zeroshot_tasks.score.

usage: python nested_eval.py cv NESTED.json K OUT.npy S DEVICE
       python nested_eval.py zs NESTED.json REFERENCE_CORPUS K OUT.json S DEVICE
"""
import json
import sys
import time

import numpy as np
import torch

from task_networks.corpus import load_corpus
from task_networks.model import ALPHA, fit

MIN_DOCS = 20  # zeroshot_tasks.MIN_DOCS


def counts(C, th, be, docs, dev):
    """one E-step on docs: expected author-network counts (each contrast's mass sums to 1, as in the model)"""
    T = torch.tensor(th, device=dev, dtype=torch.float64); B = torch.tensor(be, device=dev, dtype=torch.float64)
    n = torch.zeros_like(T)
    for d in docs:
        toks, auth = C["toks"][d], C["auth"][d]
        if len(toks) == 0:
            continue
        a = torch.tensor(auth, device=dev); v = torch.tensor(toks, device=dev)
        q = T[a][:, None, :] * B[:, v].T[None, :, :]
        q = q / q.sum((0, 2), keepdim=True) / len(toks)
        n.index_add_(0, a, q.sum(1))
    return n.cpu().numpy()


def heldout_theta(C, th, be, train, s, dev):
    """theta with every contrast node recomputed from its parents' pooled training mixtures (the model's prior), a
    parent with no training data replaced by its own fitted theta (its prior)"""
    n = counts(C, th, be, train, dev); P = C["parents"]; N = n.copy()
    children = [[] for _ in P]
    for c, ps in enumerate(P):
        for p in ps:
            children[p].append(c)
    order, seen = [], set()

    def visit(i):
        if i in seen:
            return
        seen.add(i)
        for c in children[i]:
            visit(c)
        order.append(i)
    sys.setrecursionlimit(100000)
    for i in range(len(P)):
        visit(i)
    for i in order:                                    # the model's pooling: a child split among its parents
        for p in P[i]:
            N[p] += N[i] / len(P[i])
    bar = N + 1.0; bar /= bar.sum(1, keepdims=True)
    empty = N.sum(1) == 0
    out = th.copy(); trained = np.zeros(len(P), bool)
    for d in train:
        trained[C["auth"][d]] = True
    for i, x in enumerate(C["nodes"]):
        if x.startswith("x:") and not trained[i] and P[i]:
            pr = np.mean([th[p] if empty[p] else bar[p] for p in P[i]], 0)
            t = s * pr + ALPHA; out[i] = t / t.sum()
    return out


def cv(path, K, outp, s, dev):
    C = load_corpus(path); t0 = time.time(); out = []
    for k in range(5):
        tr = [d for d in range(len(C["ids"])) if C["fold"][d] != k]; te = [d for d in range(len(C["ids"])) if C["fold"][d] == k]
        th, be, ll, it = fit([C["toks"][d] for d in tr], [C["auth"][d] for d in tr], len(C["nodes"]), C["parents"], K, s,
                             len(C["xyz"]), iters=1000, restarts=2, seed=0, devices=[dev])
        th = heldout_theta(C, th, be, tr, s, dev); pa = th @ be
        wts = np.bincount(np.concatenate([C["auth"][d] for d in tr]), minlength=len(C["nodes"]))
        marg = (wts / wts.sum()) @ pa
        for d in te:
            f = C["foci"][d]
            if len(C["toks"][d]) == 0:
                continue
            A = C["auth"][d]
            ll_ = float(np.log(pa[A][:, f].mean(0)).sum()) if len(f) else 0.0
            llm = float(np.log(marg[f]).sum()) if len(f) else 0.0
            out.append((C["ids"][d], len(f), ll_ - llm, ll_, llm))
        print(f"K{K} s{s:g} fold {k}: {it} iterations, {time.time() - t0:.0f}s", flush=True)
    np.save(outp, np.array(out, dtype=object), allow_pickle=True)
    n = sum(r[1] for r in out); ll = sum(r[3] for r in out); llm = sum(r[4] for r in out)
    print(f"K{K} s{s:g}: {n} held-out peaks; ll/peak {ll / n:.5f}; training map {llm / n:.5f}; gain {(ll - llm) / n:.5f}")


def zs(path, ref, K, outp, s, dev):
    Hr = json.load(open(ref))
    fams = {p for ps in Hr["parents"].values() for p in ps if not p.startswith(("c:", "m:"))}
    cands = [t for t, ps in Hr["parents"].items() if not t.startswith(("c:", "m:")) and any(p.startswith("c:") for p in ps)
             and Hr["n_contrasts"].get(t, 0) >= MIN_DOCS and t not in fams]
    rng = np.random.default_rng(0); rng.shuffle(cands); groups = [cands[i::5] for i in range(5)]
    C = load_corpus(path); idx = {n: i for i, n in enumerate(C["nodes"])}; P = C["parents"]
    task_of = [{p for a in C["auth"][d] for p in P[a]} for d in range(len(C["ids"]))]   # a contrast's tasks and constructs
    res = {"tasks": {}, "groups": groups}
    for g, held in enumerate(groups):
        hid = {idx[t] for t in held}
        test = [d for d in range(len(C["ids"])) if hid & task_of[d]]; ts = set(test)
        train = [d for d in range(len(C["ids"])) if d not in ts]
        th, be, ll, it = fit([C["toks"][d] for d in train], [C["auth"][d] for d in train], len(C["nodes"]), P, K, s,
                             len(C["xyz"]), restarts=1, seed=0, devices=[dev])
        th = heldout_theta(C, th, be, train, s, dev); pa = th @ be
        wts = np.bincount(np.concatenate([C["auth"][d] for d in train]), minlength=len(C["nodes"]))
        marg = (wts / wts.sum()) @ pa
        for t in held:
            docs = [d for d in test if idx[t] in task_of[d] and len(C["foci"][d])]
            sc = []
            for d in docs:
                f = C["foci"][d]; ll_ = float(np.log(pa[C["auth"][d]][:, f].mean(0)).sum())
                sc.append(((ll_ - np.log(marg[f]).sum()) / len(f), ll_ / len(f)))
            res["tasks"][t] = {"constructs": {
                "contrasts": len(sc), "gain_nats_per_peak": float(np.mean([x[0] for x in sc])) if sc else None,
                "ll_nats_per_peak": float(np.mean([x[1] for x in sc])) if sc else None,
                "per_contrast": {C["ids"][d]: x for d, x in zip(docs, sc)}}}
        print(f"group {g}: {len(held)} tasks held out, {len(test)} contrasts, {it} iterations", flush=True)
    json.dump(res, open(outp, "w"), indent=1)
    gs = [v["constructs"]["gain_nats_per_peak"] for v in res["tasks"].values() if v["constructs"]["gain_nats_per_peak"] is not None]
    print(f"{len(gs)} held-out tasks: mean gain per peak {np.mean(gs):.4f}; above the average map {sum(x > 0 for x in gs)}")


if __name__ == "__main__":
    a = sys.argv
    if a[1] == "cv":
        cv(a[2], int(a[3]), a[4], float(a[5]), a[6])
    else:
        zs(a[2], a[3], int(a[4]), a[5], float(a[6]), a[7])
