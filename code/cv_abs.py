"""5-fold held-out CV with absolute log-likelihoods, for perplexity across K (James 2026-11-11).

`python -m task_networks cv` saves each held-out contrast's gain: log P(peaks | its authors) minus log P(peaks | the
fold's author-weighted training map), where that map is built from the same fold's fitted networks, so it differs a
little by K. This runs the same CV (same folds, same fit calls and seeds, so the same fold fits as the cv command) and
saves, per held-out contrast: id, peaks, the gain as before, the absolute log P(peaks | authors), and the absolute
log P(peaks | training map). Perplexity per peak is exp(-sum ll / sum peaks).

usage: python cv_abs.py CORPUS K OUT.npy [task_networks cv options]   (cbmr-gpu venv; PYTHONPATH=task-networks src)
"""
import argparse
import sys
import time

import numpy as np

from task_networks import cli
from task_networks.corpus import load_corpus
from task_networks.model import fit


def rows_abs(C, theta, beta, test, seen):
    """as task_networks.model.heldout_rows, with the two absolute terms of the gain kept"""
    pa = theta @ beta
    train = [d for d in range(len(C["ids"])) if C["fold"][d] != C["fold"][test[0]]]
    wts = np.bincount(np.concatenate([C["auth"][d] for d in train]), minlength=len(C["nodes"]))
    marg = (wts / wts.sum()) @ pa
    out = []
    for d in test:
        v = C["toks"][d]; f = C["foci"][d]
        if len(v) == 0:
            continue
        A = [a for a in C["auth"][d] if seen[a]]
        if A and len(f):
            ll = float(np.log(pa[A][:, f].mean(0)).sum()); llm = float(np.log(marg[f]).sum())
        else:
            ll = llm = 0.0
        out.append((C["ids"][d], len(f), ll - llm, ll, llm))
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("corpus"); p.add_argument("K", type=int); p.add_argument("out")
    p.add_argument("--s", type=float, default=100.0); p.add_argument("--weighted", action="store_true"); cli._common(p)
    a = p.parse_args()
    devs = cli._setup(a); C = load_corpus(a.corpus, a.radius, a.step, a.mask); t0 = time.time(); out = []
    for k in range(5):
        tr = [d for d in range(len(C["ids"])) if C["fold"][d] != k]; te = [d for d in range(len(C["ids"])) if C["fold"][d] == k]
        th, be, ll, it = fit([C["toks"][d] for d in tr], [C["auth"][d] for d in tr], len(C["nodes"]), C["parents"], a.K, a.s,
                             len(C["xyz"]), doc_w=[C["w"][d] for d in tr] if a.weighted else None, **cli._fitargs(a, devs))
        seen = np.zeros(len(C["nodes"]), bool)
        for d in tr:
            seen[C["auth"][d]] = True
        if a.s > 0:
            seen[:] = True
        out += rows_abs(C, th, be, te, seen)
        print(f"K{a.K} fold {k}: {it} iterations, {time.time() - t0:.0f}s", flush=True)
    np.save(a.out, np.array(out, dtype=object), allow_pickle=True)
    n = sum(r[1] for r in out); ll = sum(r[3] for r in out); llm = sum(r[4] for r in out)
    print(f"K{a.K}: {n} held-out peaks; ll/peak {ll / n:.5f} (perplexity {np.exp(-ll / n):.1f}); training map "
          f"{llm / n:.5f}; gain {(ll - llm) / n:.5f}")


if __name__ == "__main__":
    sys.exit(main())
