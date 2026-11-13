"""Release construct maps from many seeds: each construct's map averaged over independent fits, with its reliability
(James 2026-11-11: 100 Gibbs seeds at K = 21, released as seed-averaged construct maps).

Why average: networks from different seeds split some regions differently (only ~12 of 21 recur in 80% of seeds at
K = 21), so network maps cannot be averaged without matching, but a construct's map lives in voxel space and agrees
across seeds (pilot: median cross-seed r 0.95), so it can be averaged directly.

For every fit DIR/gibbs_s<seed> (beta, theta and cached E-step counts _n.npy from construct_stability.py counts):
- each construct's P map exactly as construct_maps_all.py builds it from one fit (construct_map.pooled_mass with
  ALPHA_PRIOR toward the average task g; that fit's shared network, construct_map.shared_network, removed);
- the released P map is the mean over seeds; the released enrichment is log2 (mean P / mean P(g)), the ratio of the
  averages, not the average of log ratios;
- reliability per construct: the mean Pearson r between seeds of its P map (all voxels), and of its enrichment map
  over the voxels where the released P is >= 20% of its maximum (the region the figures and page show);
- credited mass: the mean over seeds.
Networks: the most typical seed (the CBIG ComputeMostTypicalSolution rule: smallest mean, over the other seeds, of
the Hungarian-matched sum of 1 - r between network maps), each of its networks with its recurrence (the share of the
other seeds with a match at r >= 0.8); a construct's "networks" field lists that seed's largest networks for it.

Uncertainty (James 2026-11-11: measure both sources of variance), per construct and voxel, on log2 enrichment:
- seed band: the BAND percentiles over the seeds of each seed's own enrichment, log2 (P_s / P_s(g)): optimizer
  (local-optimum) variability only, on the full data; volumes enrichment_seed_lo / _hi;
- bootstrap band (with BOOT_DIR): the same percentiles over fits to resampled papers (gibbs_init.py fit ... BOOT;
  each resample has its own seed, so this band holds both sampling and optimizer variability); volumes
  enrichment_boot_lo / _hi. This is the one to call a confidence band.
- reliably enriched: the share of the displayed region (P >= 20% of max) whose lower bound is above 0, per band.

Task maps (James 2026-11-11: a brain map for every task): every task author with contrasts gets the same treatment
from its own pooled mass (a task's map is its mixture times the network maps, shared network removed), written to
OUT_DIR/tasks/ in the same format. Curated families that carry contrasts of their own (contrasts labelled only with a
family) are authors too, and get maps there as well.

Writes OUT_DIR as construct_maps_all.py does (nifti/, manifest.json with the extra fields reliability_p,
reliability_enrichment, seeds, and the band summaries; mips.json), plus networks.json (the most typical seed,
recurrences, seed table).

Re-pooling (env FIT_CORPUS): the fits may come from another corpus with the same documents and authors but a different
construct hierarchy (2026-11-11: v1 fits, v2 hierarchy). Each fit's counts are then moved to CORPUS's nodes by name
(construct nodes author no documents, so they carry no counts of their own) and pooled along CORPUS's hierarchy. The
hierarchy entered those fits only through the prior on each task's mixture (s = 100 pseudo-tokens against thousands
of tokens per task), so this approximates a refit on CORPUS.

Construct authors (corpus B, James 2026-11-12: "the tasks are just mixtures of the contrasts"): when constructs author
contrasts, a construct's map is its own authored mass plus its kind-of child constructs', and a task's the content mass
of its own contrasts (construct_map.pooled_mass with task_content). A bootstrap fit's task content is taken on its
resampled contrasts (gibbs_init.py's draw, rebuilt here and checked against the count the fit recorded); cached as
<fit>_tc.npy.

usage: [FIT_CORPUS=...] python construct_maps_ensemble.py CORPUS DIR OUT_DIR [BOOT_DIR]   (cbmr-gpu venv; from atm)
"""
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src"); sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks.corpus import load_corpus  # noqa: E402
import construct_map as cm  # noqa: E402
from construct_maps_all import ENRICH_MASK, write_maps  # noqa: E402

MATCH_R = 0.8
BAND = (2.5, 97.5)  # percentiles of the uncertainty bands


def boot_mult(H, boot):
    """per contrast (load_corpus order), how often bootstrap resample BOOT drew its paper: gibbs_init.py's rule"""
    ids = sorted(H["docs"]); paper = np.array([H["docs"][i]["group"] for i in ids]); papers = np.unique(paper)
    draw = np.random.default_rng(1000003 + boot).choice(len(papers), len(papers), replace=True)
    return np.bincount(draw, minlength=len(papers))[np.searchsorted(papers, paper)]


def task_content_of(C, H, p, m, beta):
    """the fit's task content (construct_map.task_content) on the contrasts it was fitted to; cached"""
    f = f"{p}_tc.npy"
    if os.path.exists(f):
        return np.load(f)
    mult = None
    if m.get("bootstrap") is not None:
        mult = boot_mult(H, m["bootstrap"])
        assert int(mult.sum()) == m["bootstrap_contrasts"], f"{p}: resample does not match the fit's"
    th = np.load(f"{p}_theta.npy").astype(float)
    assert th.shape[0] == len(C["nodes"]), f"{p} was fitted on other nodes: task content needs the fit's own corpus"
    tc = cm.task_content(C, th, beta, mult)
    np.save(f, tc)
    return tc


def load_fits(C, d, cons, H=None):  # cons: the node indices to map (constructs, or constructs then tasks)
    """per fit in d: beta, the constructs' shared-removed mixtures, credited mass, the reference mixture, the shared
    network and its ownership, log-likelihood"""
    nodes = C["nodes"]
    fs = sorted(glob.glob(f"{d}/gibbs_s*_n.npy"), key=lambda f: int(re.search(r"_s(\d+)_n", f).group(1)))
    pre = [f[:-6] for f in fs]
    out = {"seeds": [int(re.search(r"_s(\d+)$", p).group(1)) for p in pre], "B": [], "MIX": [], "MASS": [], "GG": [],
           "SH": [], "OWN": [], "ll": []}
    remap = None; authored = cm.constructs_author(C)
    for p in pre:
        m = json.load(open(f"{p}.json"))
        if m["nodes"] != nodes:   # fits from another hierarchy (FIT_CORPUS): move counts to this corpus's nodes by name
            assert os.environ.get("FIT_CORPUS"), f"{p} was fitted on other nodes; set FIT_CORPUS to re-pool"
            if remap is None:
                src = {x: i for i, x in enumerate(m["nodes"])}
                remap = np.array([src.get(x, -1) for x in nodes])
                lost = [x for x in m["nodes"] if not x.startswith("c:") and x not in set(nodes)]
                assert not lost, f"authors missing from the new corpus: {lost[:5]}"
        out["B"].append(np.load(f"{p}_beta.npy").astype(float)); out["ll"].append(m["ll_tokens"])
        n = np.load(f"{p}_n.npy")
        if m["nodes"] != nodes:
            n2 = np.zeros((len(nodes), n.shape[1])); ok = remap >= 0; n2[ok] = n[remap[ok]]; n = n2
        tc = task_content_of(C, H, p, m, out["B"][-1]) if authored else None
        N, g, children = cm.pooled_mass(C, n, tc)
        bar = N[cons] + cm.ALPHA_PRIOR * g; bar /= bar.sum(1, keepdims=True)
        b = nodes.index("background"); own = n[b] / n.sum(0); sh = int(np.argmax(own))
        mix = bar.copy(); mix[:, sh] = 0; mix /= mix.sum(1, keepdims=True)
        gg = g.copy(); gg[sh] = 0; gg /= gg.sum()
        out["MIX"].append(mix); out["MASS"].append(N[cons].sum(1)); out["GG"].append(gg); out["SH"].append(sh)
        out["OWN"].append(float(own[sh]))
    for k in ("B", "MIX", "MASS", "GG", "ll"):
        out[k] = np.array(out[k])
    out["pall"] = np.einsum("sk,skv->sv", out["GG"], out["B"]); out["children"] = children
    return out


def zrows(X):
    X = X - X.mean(-1, keepdims=True)
    return X / np.linalg.norm(X, axis=-1, keepdims=True)


def main():
    corpus, d, out = sys.argv[1], sys.argv[2], sys.argv[3]
    C = load_corpus(corpus); H = json.load(open(corpus)); nodes = C["nodes"]
    cons = [k for k, x in enumerate(nodes) if x.startswith("c:")]
    authored = {int(a) for au in C["auth"] for a in au}
    tasks = [k for k, x in enumerate(nodes) if k in authored and not x.startswith(("c:", "m:")) and x != "background"]
    if any(x.startswith("x:") for x in nodes):    # nested corpus (nested_corpus.py): contrast nodes author, tasks pool them
        over_x = {p for c, x in enumerate(nodes) if x.startswith("x:") for p in C["parents"][c]}
        tasks = [k for k, x in enumerate(nodes) if (cm.is_task(x) or x == "deactivation") and k in over_x]
    ncons = len(cons); cons = cons + tasks                                      # constructs, then tasks
    F = load_fits(C, d, cons, H)
    seeds, B, MIX, MASS, SH, OWN, pall, children = (F[k] for k in ("seeds", "B", "MIX", "MASS", "SH", "OWN", "pall", "children"))
    S, K, V = B.shape; xyz = np.load(glob.glob(f"{d}/gibbs_s*_xyz.npy")[0])
    pall_mean = pall.mean(0)
    BT = load_fits(C, sys.argv[4], cons, H) if len(sys.argv) > 4 else None
    # most typical seed (CBIG rule) and network recurrence
    Zb = zrows(B); cost = np.zeros((S, S)); match = {}
    for i in range(S):
        for j in range(S):
            if i != j:
                r = Zb[i] @ Zb[j].T; a, b_ = linear_sum_assignment(1 - r)
                cost[i, j] = (1 - r[a, b_]).sum(); match[i, j] = r[a, b_]
    t = int(np.argmin(cost.sum(1) / (S - 1)))
    rec = np.array([np.mean([match[t, j][k] >= MATCH_R for j in range(S) if j != t]) for k in range(K)])
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    r0 = zrows(B[t]) @ zrows(b0).T
    name = {k: f"{names[str(int(np.argmax(r0[k])))].split(':')[0]} ({r0[k].max():.2f})" for k in range(K)}
    name[SH[t]] = "Shared pattern"
    iu = np.triu_indices(S, 1); items = {}
    for ci, k in enumerate(cons):
        P = np.einsum("sk,skv->sv", MIX[:, ci], B)   # S x V: this construct's P map in every seed
        pm = P.mean(0); enr = np.log2(pm / pall_mean)
        Zp = zrows(P); rel_p = float((Zp @ Zp.T)[iu].mean())
        m = pm >= ENRICH_MASK * pm.max()
        Ze = zrows(np.log2(P[:, m] / pall[:, m])); rel_e = float((Ze @ Ze.T)[iu].mean())
        mix_t = MIX[t, ci]
        Es = np.log2(P / pall)                                                   # seed-level enrichment
        lo, hi = np.percentile(Es, BAND, axis=0)
        vols = {"enrichment_seed_lo": lo, "enrichment_seed_hi": hi}
        extra = {"reliability_p": round(rel_p, 3), "reliability_enrichment": round(rel_e, 3), "seeds": S,
                 "seed_band_median_width": round(float(np.median((hi - lo)[m])), 3),
                 "reliably_enriched_seed": round(float(np.mean(lo[m] > 0)), 3)}
        if BT is not None:
            Pb = np.einsum("sk,skv->sv", BT["MIX"][:, ci], BT["B"]); Eb = np.log2(Pb / BT["pall"])
            blo, bhi = np.percentile(Eb, BAND, axis=0)
            vols.update({"enrichment_boot_lo": blo, "enrichment_boot_hi": bhi})
            Zb_ = zrows(Pb); extra.update({"bootstrap_resamples": len(BT["seeds"]),
                                           "reliability_p_boot": round(float((Zb_ @ Zb_.T)[np.triu_indices(len(Pb), 1)].mean()), 3),
                                           "boot_band_median_width": round(float(np.median((bhi - blo)[m])), 3),
                                           "reliably_enriched_boot": round(float(np.mean(blo[m] > 0)), 3)})
        items[k] = {"p": pm, "enrichment": enr, "credited_mass": float(MASS[:, ci].mean()), "volumes": vols,
                    "networks": [{"name": name[q], "weight": round(float(mix_t[q]), 3), "recurrence": round(float(rec[q]), 2)}
                                 for q in np.argsort(-mix_t)[:5]],
                    "extra": extra}
    ll = F["ll"]
    task_items = {k: items.pop(k) for k in tasks}
    write_maps(Path(out) / "tasks", C, H, xyz, children, task_items,
               {"fit": f"{d} ({S} seeds; task maps averaged over seeds)", "corpus": corpus, "K": K}, kind="tasks")
    write_maps(out, C, H, xyz, children, items,
               {"fit": f"{d} ({S} seeds, Gibbs burn-in + EM; construct maps averaged over seeds)", "corpus": corpus,
                "K": K, "seeds": seeds, "most_typical_seed": seeds[t],
                "networks": [f"{name[q]} [recurrence {rec[q]:.2f}]" for q in range(K)],
                "shared_network": [name[SH[t]]]})
    json.dump({"K": K, "seeds": seeds, "most_typical_seed": seeds[t], "best_seed": seeds[int(np.argmax(ll))],
               "ll_tokens": dict(zip(map(str, seeds), ll.tolist())),
               "most_typical_mean_cost": float(cost[t].sum() / (S - 1)),
               "networks": [{"index": int(q), "name": name[q], "recurrence": float(rec[q])} for q in np.argsort(-rec)],
               "shared_owned": dict(zip(map(str, seeds), OWN))},
              open(Path(out) / "networks.json", "w"), indent=1, default=lambda o: o.item())
    np.save(Path(out) / "most_typical_beta.npy", B[t].astype(np.float32))
    relp = np.array([it["extra"]["reliability_p"] for it in items.values()])
    rele = np.array([it["extra"]["reliability_enrichment"] for it in items.values()])
    print(f"{S} seeds; most typical seed {seeds[t]}, best seed {seeds[int(np.argmax(ll))]}; networks recurring >= 80%: "
          f"{(rec >= 0.8).sum()} of {K}; construct reliability (all {ncons} construct nodes) P median {np.median(relp):.3f}, "
          f">= 0.8 {np.mean(relp >= 0.8):.0%}; enrichment median {np.median(rele):.3f}, >= 0.8 {np.mean(rele >= 0.8):.0%}")
    w = np.array([it["extra"]["seed_band_median_width"] for it in items.values()])
    print(f"seed band (log2 enrichment, {BAND[0]}-{BAND[1]} pct) median width over the displayed region: median over "
          f"constructs {np.median(w):.3f}")
    if BT is not None:
        wb = np.array([it["extra"]["boot_band_median_width"] for it in items.values()])
        rb = np.array([it["extra"]["reliably_enriched_boot"] for it in items.values()])
        print(f"bootstrap band ({len(BT['seeds'])} resamples) median width: median over constructs {np.median(wb):.3f}; "
              f"share of displayed region reliably enriched (bootstrap lower bound > 0): median {np.median(rb):.2f}")


if __name__ == "__main__":
    main()
