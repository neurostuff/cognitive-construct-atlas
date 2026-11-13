"""Predict tasks the model has never seen, from their place in the hierarchy (task-networks repo, unchanged).

Held-out tasks: those with a construct parent in the construct corpus and >= MIN_DOCS contrasts, in the corpus
file's order, split into 5 groups by a shuffle with SEED (default 0, the grouping of every run so far). For each group: every contrast that has a held-out task among its authors is removed, the model is fitted
on the rest (K, s = 100, same corpus documents for both hierarchies), and the removed contrasts are scored with
their authors' mixtures. A held-out task has no counts, so its mixture is its parents' (s > 0): its constructs under
the construct hierarchy, its curated family under the original one. Score per contrast (as heldout_rows):
log P(peaks | authors) - log P(peaks | the author-weighted training map), in nats; > 0 beats the average map. The
absolute log P(peaks | authors) per peak is saved too ("ll_nats_per_peak", and per contrast), so that other
predictors can be scored against one common reference (baselines_heldout.py).

usage: python zeroshot_tasks.py CONSTRUCT_CORPUS FAMILY_CORPUS|- OUT.json K DEVICE [RESTARTS] [SEED]
       ("-": constructs only; SEED, default 0, sets the task grouping and the fits' first seed;
        SIDES=families fits only the families side, on the construct corpus's task groups)
(PYTHONPATH = the task-networks repo src; the cbmr-gpu venv)
"""
import json
import os
import sys

import numpy as np

from task_networks.corpus import load_corpus
from task_networks.model import fit

MIN_DOCS = 20


def score(C, theta, beta, train, test):
    """per contrast: (gain over the author-weighted training map, absolute log-likelihood), both nats per peak"""
    pa = theta @ beta
    wts = np.bincount(np.concatenate([C["auth"][d] for d in train]), minlength=len(C["nodes"]))
    marg = (wts / wts.sum()) @ pa
    out = []
    for d in test:
        f = C["foci"][d]
        if len(f) == 0:
            continue
        A = C["auth"][d]
        ll = float(np.log(pa[A][:, f].mean(0)).sum())
        out.append(((ll - np.log(marg[f]).sum()) / len(f), ll / len(f)))
    return out


def main():
    cc, fc, out, K, dev = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    restarts = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0  # task grouping and the fits' first seed
    Hc = json.load(open(cc))
    fams = {p for ps in json.load(open(fc if fc != "-" else cc))["parents"].values() for p in ps if not p.startswith(("c:", "m:"))}
    cands = [t for t, ps in Hc["parents"].items()  # tasks only: with constructs as authors, constructs have contrasts too
             if not t.startswith(("c:", "m:")) and any(p.startswith("c:") for p in ps)
             and Hc["n_contrasts"].get(t, 0) >= MIN_DOCS and t not in fams]
    rng = np.random.default_rng(seed); rng.shuffle(cands)  # corpus-file order (deterministic), shuffled by SEED
    groups = [cands[i::5] for i in range(5)]
    res = {"tasks": {}, "groups": groups}
    for name, path in (("constructs", cc), ("families", fc)):
        if path == "-" or name not in os.environ.get("SIDES", "constructs,families").split(","):
            continue  # "-": constructs only (a K sweep); SIDES=families: families only
        C = load_corpus(path)
        idx = {n: i for i, n in enumerate(C["nodes"])}
        for g, held in enumerate(groups):
            hid = {idx[t] for t in held}
            test = [d for d in range(len(C["ids"])) if hid & set(C["auth"][d].tolist())]
            train = [d for d in range(len(C["ids"])) if d not in set(test)]
            th, be, ll, it = fit([C["toks"][d] for d in train], [C["auth"][d] for d in train], len(C["nodes"]),
                                 C["parents"], K, 100.0, len(C["xyz"]), restarts=restarts, seed=seed, devices=[dev])
            for t in held:
                docs = [d for d in test if idx[t] in set(C["auth"][d].tolist())]
                s = score(C, th, be, train, docs)
                res["tasks"].setdefault(t, {})[name] = {
                    "contrasts": len(s), "gain_nats_per_peak": float(np.mean([x[0] for x in s])) if s else None,
                    "ll_nats_per_peak": float(np.mean([x[1] for x in s])) if s else None,
                    "per_contrast": {C["ids"][d]: x for d, x in zip([d for d in docs if len(C["foci"][d])], s)}}
            print(f"{name} group {g}: {len(held)} tasks held out, {len(test)} contrasts, {it} iterations", flush=True)
    json.dump(res, open(out, "w"), indent=1)
    sides = {n for v in res["tasks"].values() for n in v}
    if len(sides) == 1:
        side = sides.pop()
        g = [v[side]["gain_nats_per_peak"] for v in res["tasks"].values() if v[side]["gain_nats_per_peak"] is not None]
        print(f"{len(g)} held-out tasks ({side}): mean gain per peak {np.mean(g):.4f}; above the average map {sum(x > 0 for x in g)}")
        return
    both = [v for v in res["tasks"].values() if v.get("constructs", {}).get("gain_nats_per_peak") is not None
            and v.get("families", {}).get("gain_nats_per_peak") is not None]
    c = np.array([v["constructs"]["gain_nats_per_peak"] for v in both]); f = np.array([v["families"]["gain_nats_per_peak"] for v in both])
    print(f"{len(both)} held-out tasks: mean gain per peak, constructs {c.mean():.4f}, families {f.mean():.4f}; "
          f"constructs better on {(c > f).sum()} of {len(both)}; above the average map: constructs {(c > 0).sum()}, families {(f > 0).sum()}")


if __name__ == "__main__":
    main()
