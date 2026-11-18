"""Baselines for the held-out-task test (run list E9, E10): predict a task never seen from its constructs (or its family,
or its name) without the author-topic model, on the same held-out groups, contrasts and peaks as zeroshot_tasks.py.

Every predictor is a probability over the same grid voxels as the model (C["xyz"]), so scores are comparable: the
log-probability of each peak voxel (C["foci"]), averaged per contrast, then over a task's contrasts, then over tasks
(nats per peak; zeroshot_tasks.py aggregates the same way).

Predictors, per held-out group (fit on the contrasts that name no held-out task, as the model's refit is):
- kde_constructs: each construct's map is the peaks of the training contrasts of every task below it in the
  hierarchy (each contrast weight 1, spread over its peaks), smoothed with a Gaussian of FWHM F. A construct with no
  training contrast takes its parents' mean map, then the average map. The task's map is the mean of its
  constructs' maps. This is the model's pooling without its K-network basis.
- kde_families: the same, from the task's curated family.
- nq_constructs: the mean over the task's constructs of NeuroQuery's map for the construct's label (nq_maps.py), as
  a density softmax(tau * z). Labels outside NeuroQuery's vocabulary are skipped, and a task with none takes the
  average map.
- nq_task: NeuroQuery's map for the task's own name, as above.
- average: the smoothed map of all training peaks (the reference).
Every predictor p is shrunk toward the average map a: (1 - lam) p + lam a.

Settings (F in mm, lam, tau) are chosen at their best on the held-out tasks themselves, which is optimistic for
the baselines and so conservative for the model. The other settings are reported too. NeuroQuery was trained on
neuroimaging papers that overlap with this corpus, so it has seen the held-out tasks' papers.

The model's scores come from a zeroshot_tasks.py output with "ll_nats_per_peak" (both its constructs and families
sides), on the same groups.

usage: python baselines_heldout.py CORPUS ZEROSHOT_LL_JSON NQ_NPZ OUT.json   (cbmr-gpu venv; from pilot-results/atm)
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix

sys.path.insert(0, "/data/james/agents/dev/task-networks-repro/repo/src")
from task_networks.corpus import load_corpus  # noqa: E402

FWHMS = (4.0, 6.0, 8.0, 10.0, 14.0)              # tuning grid, mm (the grid is 4 mm); MKDA's usual kernel is a 10 mm sphere
LAMS = (0.0, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95)  # shrinkage toward the average map
TAUS = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)        # NeuroQuery z-map sharpness
FLOOR = 1e-12                       # keeps log finite for a voxel no predictor reaches


def kernel(xyz, fwhm):
    s = fwhm / (2 * np.sqrt(2 * np.log(2)))
    tree = cKDTree(xyz)
    D = tree.sparse_distance_matrix(tree, 3 * s, output_type="coo_matrix")
    w = np.exp(-D.data ** 2 / (2 * s * s))
    diag = np.arange(len(xyz))
    rows = np.r_[D.row, diag]; cols = np.r_[D.col, diag]; vals = np.r_[w, np.ones(len(xyz))]
    return csr_matrix((vals, (rows, cols)), shape=(len(xyz), len(xyz)))


def main():
    corpus, zpath, nqpath, out = sys.argv[1:5]
    C = load_corpus(corpus); H = json.load(open(corpus))
    nodes = C["nodes"]; idx = {n: i for i, n in enumerate(nodes)}; V = len(C["xyz"])
    P = [list(ps) for ps in C["parents"]]
    is_task = [not n.startswith(("c:", "m:")) and n != "background" and n in H["parents"] for n in nodes]
    fams = {p for t, ps in H["parents"].items() for p in ps if not p.startswith(("c:", "m:"))}
    for i, n in enumerate(nodes):
        if n in fams:
            is_task[i] = False
    Z = json.load(open(zpath)); groups = Z["groups"]
    nq = np.load(nqpath, allow_pickle=True)
    nq_map = {str(n): (m, bool(v)) for n, m, v in zip(nq["names"], nq["maps"], nq["in_vocab"])}
    # per-contrast peak histograms (weight 1 per contrast)
    foci = C["foci"]; auth = C["auth"]
    tasks_of = [[a for a in auth[d] if is_task[a]] for d in range(len(foci))]
    children = [[] for _ in nodes]
    for c, ps in enumerate(P):
        for p in ps:
            children[p].append(c)

    def desc_tasks(i, memo={}):
        if i not in memo:
            s = {i} if is_task[i] else set()
            for c in children[i]:
                s |= desc_tasks(c)
            memo[i] = s
        return memo[i]
    kernels = {f: kernel(C["xyz"], f) for f in FWHMS}
    res = {"settings": {"fwhm": FWHMS, "lam": LAMS, "tau": TAUS}, "tasks": {}}
    for g, held in enumerate(groups):
        hid = {idx[t] for t in held if t in idx}
        test = [d for d in range(len(foci)) if hid & set(auth[d].tolist()) and len(foci[d])]
        tset = set(test)
        train = [d for d in range(len(foci)) if d not in tset and len(foci[d])]
        Th = np.zeros((len(nodes), V), np.float64)  # training peak mass per task node
        for d in train:
            for a in tasks_of[d]:
                np.add.at(Th[a], foci[d], 1.0 / len(foci[d]) / len(tasks_of[d]))
        allh = Th.sum(0)
        need = set()
        for t in hid:
            for p in P[t]:
                stack = [p]
                while stack:
                    x = stack.pop()
                    if x not in need:
                        need.add(x); stack.extend(P[x])
        for fwhm, Kf in kernels.items():
            avg = Kf @ allh; avg /= avg.sum()
            cache = {}

            def nmap(i, seen=()):
                if i in cache:
                    return cache[i]
                h = sum((Th[t] for t in desc_tasks(i) if t not in hid), np.zeros(V))
                if h.sum() > 0:
                    m = Kf @ h; m /= m.sum()
                else:
                    ps = [p for p in P[i] if p not in seen]
                    m = np.mean([nmap(p, seen + (i,)) for p in ps], 0) if ps else avg
                cache[i] = m
                return m
            for t in sorted(hid):
                tn = nodes[t]
                docs = [d for d in test if t in auth[d]]
                cons = [p for p in P[t] if nodes[p].startswith("c:")]
                famp = [p for p in P[t] if nodes[p] in fams]
                preds = {"average": avg}
                if cons:
                    preds["kde_constructs"] = np.mean([nmap(c) for c in cons], 0)
                if famp:
                    preds["kde_families"] = np.mean([nmap(f) for f in famp], 0)
                rec =res["tasks"].setdefault(tn, {"contrasts": len(docs), "group": g})
                for name, p in preds.items():
                    for lam in LAMS:
                        q = (1 - lam) * p + lam * avg + FLOOR
                        ll = float(np.mean([np.log(q[foci[d]]).mean() for d in docs]))
                        rec.setdefault(name, {})[f"F{fwhm:g}_lam{lam:g}"] = ll
                # NeuroQuery: densities from z maps, shrunk toward this F's average map
                for name, texts in (("nq_constructs", [nodes[c] for c in cons]), ("nq_task", [tn])):
                    zs = [nq_map[x][0] for x in texts if x in nq_map and nq_map[x][1]]
                    for tau in TAUS:
                        if zs:
                            dens = []
                            for z in zs:
                                e = np.exp(tau * (z - z.max())); dens.append(e / e.sum())
                            p = np.mean(dens, 0)
                        else:
                            p = avg
                        for lam in LAMS:
                            q = (1 - lam) * p + lam * avg + FLOOR
                            ll = float(np.mean([np.log(q[foci[d]]).mean() for d in docs]))
                            rec.setdefault(name, {})[f"F{fwhm:g}_lam{lam:g}_tau{tau:g}"] = ll
        print(f"group {g}: {len(hid)} tasks, {len(test)} test contrasts", flush=True)
    # model scores
    for t, v in Z["tasks"].items():
        if t in res["tasks"]:
            for side in ("constructs", "families"):
                x = v.get(side, {}).get("ll_nats_per_peak")
                if x is not None:
                    res["tasks"][t][f"model_{side}"] = x
    # summary: each method at its best setting (chosen on these tasks: optimistic for the baselines)
    T = res["tasks"]
    methods = ["average", "kde_constructs", "kde_families", "nq_constructs", "nq_task"]
    common = [t for t in T if all(m in T[t] for m in methods)]
    summary = {}
    for m in methods:
        keys = T[common[0]][m].keys()
        best = max(keys, key=lambda k: np.mean([T[t][m][k] for t in common]))
        summary[m] = {"best_setting": best, "ll": float(np.mean([T[t][m][best] for t in common])),
                      "per_task": {t: T[t][m][best] for t in common}}
    for side in ("constructs", "families"):
        ts = [t for t in common if f"model_{side}" in T[t]]
        if ts:
            summary[f"model_{side}"] = {"ll": float(np.mean([T[t][f"model_{side}"] for t in ts])),
                                        "per_task": {t: T[t][f"model_{side}"] for t in ts}}
    res["summary"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_task"} for k, v in summary.items()}
    res["per_task_best"] = {k: v["per_task"] for k, v in summary.items()}
    json.dump(res, open(out, "w"), indent=1)
    ref = summary["average"]["ll"]
    print(f"{len(common)} held-out tasks; log P(peak) per peak, each baseline at its best setting; gain over the "
          f"average map at its best setting ({summary['average']['best_setting']})")
    for k, v in summary.items():
        print(f"  {k:16s} {v['ll']:.4f}  gain {v['ll'] - ref:+.4f}  {v.get('best_setting', '')}")
    if "model_constructs" in summary:
        for m in ("kde_constructs", "nq_constructs", "kde_families"):
            ts = [t for t in summary["model_constructs"]["per_task"] if t in summary[m]["per_task"]]
            d = np.array([summary["model_constructs"]["per_task"][t] - summary[m]["per_task"][t] for t in ts])
            print(f"  model constructs minus {m}: {d.mean():+.4f}; model better on {(d > 0).sum()} of {len(d)}")


if __name__ == "__main__":
    main()
