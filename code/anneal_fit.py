"""Deterministic annealing EM for the task-networks author-topic model, and the seed pilot (James 2026-11-11: a
20-seed pilot with and without deterministic annealing).

Deterministic annealing EM (Ueda & Nakano 1998, Neural Networks 11:271-282, doi:10.1016/S0893-6080(97)00133-0):
the E-step's posterior over (author, network) for each token is raised to a power tau <= 1 and renormalized, so early
iterations make soft, uncommitted assignments; tau rises to 1, where the iterations are plain EM. The M-step is
unchanged. The log-likelihood reported and used for convergence is always the untempered one, and convergence is only
tested once tau = 1, so a fit ends as an ordinary EM fixed point.

This is a copy of task_networks.model._fit_once (split E-step path only) with the E-step tempered; the task-networks
repository is not modified. With TAUS = () it reproduces task_networks.model.fit exactly (checked by `check`).

Schedule (tuning defaults, not from the source, which gives no schedule for a model of this size): tau = TAUS[i] for
STEPS_PER_TAU iterations each, then tau = 1 until convergence (tol 1e-6) or `iters` in total.

Symmetry breaking: when the posterior is softened enough, networks merge into identical copies, and identical copies
are a fixed point that EM never leaves (the first pilot, tau from 0.6: every run ended as 21 copies of one map). So at
each change of tau, beta is multiplied by (1 + EPS * u), u uniform in [-1, 1] from the run's own generator, and
renormalized, as deterministic-annealing clustering perturbs its centroids at each temperature (Rose 1998). After each
level the largest correlation between two networks is printed; a run whose networks end above COLLAPSE_R is flagged.

usage:
  python anneal_fit.py check CORPUS                        # TAUS=() equals task_networks.model.fit (K 4, 6 iterations)
  python anneal_fit.py fit CORPUS K OUT_PREFIX SEED plain|anneal [ITERS [TAUS [EPS]]]
      # one restart; TAUS comma-separated (default the module's); writes beta, theta, json
"""
import json
import sys
import time

import numpy as np
import torch

from task_networks import model as M
from task_networks.corpus import load_corpus

TAUS = (0.6, 0.66, 0.73, 0.8, 0.88, 0.97)  # the first pilot's schedule (collapsed); the CLI takes others
STEPS_PER_TAU = 20
EPS = 0.0          # beta perturbation at each change of tau (0 in the first pilot)
COLLAPSE_R = 0.95  # two networks correlated above this are treated as merged


class TemperedSplit(M._Split):
    def estep(self, theta, beta, tau=1.0):
        """as M._Split.estep, with responsibilities proportional to (theta * beta * w) ** tau; ll is untempered"""
        n = torch.zeros(self.n_nodes, self.K); b = torch.zeros(self.V, self.K); ll = 0.0
        on = {d: (theta.to(d), beta.to(d)) for d in self.devices}
        for i in range(0, len(self.chunks), len(self.devices)):
            parts = []
            for d, T, Au, V, w, nt, dwc in self.chunks[i:i + len(self.devices)]:
                th, be = on[d]
                q = th[Au] * be[:, V].T * w[:, None]
                tok_p = torch.zeros(nt, device=d).index_add_(0, T, q.sum(1))
                if tau != 1.0:
                    q = q ** tau
                    tok_t = torch.zeros(nt, device=d).index_add_(0, T, q.sum(1))
                    r = q / tok_t[T][:, None]; del tok_t
                else:
                    r = q / tok_p[T][:, None]
                del q
                if dwc is not None:
                    r = r * dwc[:, None]
                parts.append((torch.zeros(self.n_nodes, self.K, device=d).index_add_(0, Au, r),
                              torch.zeros(self.V, self.K, device=d).index_add_(0, V, r), torch.log(tok_p).sum()))
                del r, tok_p
            for pn, pb, pl in parts:
                n += pn.cpu(); b += pb.cpu(); ll += pl.item()
        return n.numpy(), b, ll


def max_r(beta):
    b = beta.cpu().numpy(); b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    r = b @ b.T / b.shape[1]; np.fill_diagonal(r, -1)
    return float(r.max())


def fit_once(docs_tok, doc_auth, n_nodes, parents, K, s, n_vox, iters, seed, tol, devices, taus=TAUS,
             steps=STEPS_PER_TAU, cpu_mem=8e9, eps=EPS, verbose=False, init=None):
    """task_networks.model._fit_once (split path, no document weights) with a tempered E-step schedule"""
    g = torch.Generator().manual_seed(seed)
    ntok = np.array([len(t) for t in docs_tok])
    tok_doc = np.repeat(np.arange(len(docs_tok)), ntok)
    tok_vox = np.concatenate([np.asarray(t, int) for t in docs_tok])
    cnt = np.array([len(A) for A in doc_auth])
    pair_tok = np.repeat(np.arange(len(tok_doc)), cnt[tok_doc])
    pair_auth = np.concatenate([np.tile(np.asarray(A, int), n) for A, n in zip(doc_auth, ntok)])
    dev = torch.device("cpu")
    split = TemperedSplit([torch.device(d) for d in devices], pair_tok, pair_auth, tok_vox, (1.0 / cnt[tok_doc])[pair_tok],
                          K, n_nodes, n_vox, None, False, cpu_mem)
    children = [[] for _ in range(n_nodes)]
    for c, ps in enumerate(parents):
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
    for i in range(n_nodes):
        visit(i)
    theta = (torch.rand(n_nodes, K, generator=g) + 0.5).to(dev); theta /= theta.sum(1, keepdim=True)
    beta = torch.rand(K, n_vox, generator=g) + 0.5; beta = (beta / beta.sum(1, keepdim=True)).to(dev)
    if init is not None:  # a given start (gibbs_init.py's Gibbs burn-in) instead of the random one
        theta = torch.tensor(np.asarray(init[0], float), device=dev); beta = torch.tensor(np.asarray(init[1], float), device=dev)
    sched = [t for t in taus for _ in range(steps)]
    prev = -np.inf
    last = None
    for it in range(iters):
        tau = sched[it] if it < len(sched) else 1.0
        if last is not None and tau != last:
            if verbose:
                print(f"  tau {last:g} done at iteration {it}: ll {ll:.0f}, max network r {max_r(beta):.3f}", flush=True)
            if eps > 0:
                u = torch.rand(beta.shape, generator=g, dtype=beta.dtype) * 2 - 1
                beta = beta * (1 + eps * u); beta = beta / beta.sum(1, keepdim=True)
        last = tau
        n, bsum, ll = split.estep(theta, beta, tau)
        if s > 0:
            N = n.copy()
            for i in order:
                for p in parents[i]:
                    N[p] += N[i] / len(parents[i])
            glob = n.sum(0) + M.ALPHA; glob /= glob.sum()
            bar = N + 1.0; bar /= bar.sum(1, keepdims=True)
            prior = np.array([bar[parents[i]].mean(0) if parents[i] else glob for i in range(n_nodes)])
            th = n + s * prior + M.ALPHA
        else:
            th = n + M.ALPHA
        theta = torch.tensor(th / th.sum(1, keepdims=True), device=dev)
        b = bsum.T + M.ETA
        beta = b / b.sum(1, keepdim=True)
        if tau == 1.0 and abs(ll - prev) < tol * abs(ll):
            break
        prev = ll if tau == 1.0 else -np.inf
    if verbose:
        print(f"  end at iteration {it + 1}: ll {ll:.0f}, max network r {max_r(beta):.3f}", flush=True)
    return theta.cpu().numpy(), beta.cpu().numpy(), ll, it + 1


def main():
    cmd, corpus = sys.argv[1], sys.argv[2]
    C = load_corpus(corpus); args = (C["toks"], C["auth"], len(C["nodes"]), C["parents"])
    if cmd == "check":
        a = M.fit(*args, 4, 100.0, len(C["xyz"]), iters=6, seed=3, devices=["cuda:0"])
        b = fit_once(*args, 4, 100.0, len(C["xyz"]), 6, 3, 1e-6, ["cuda:0"], taus=())
        print("theta max abs diff", np.abs(a[0] - b[0]).max(), "beta", np.abs(a[1] - b[1]).max(), "ll", a[2] - b[2],
              "iterations", a[3], b[3])
        return
    K, out, seed, mode = int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), sys.argv[6]
    iters = int(sys.argv[7]) if len(sys.argv) > 7 else 1000
    taus = tuple(float(x) for x in sys.argv[8].split(",")) if len(sys.argv) > 8 else TAUS
    eps = float(sys.argv[9]) if len(sys.argv) > 9 else EPS
    if mode != "anneal":
        taus, eps = (), 0.0
    t0 = time.time()
    th, be, ll, it = fit_once(*args, K, 100.0, len(C["xyz"]), iters, seed, 1e-6, ["cuda:0"], taus=taus, eps=eps,
                              verbose=True)
    np.save(f"{out}_beta.npy", be.astype(np.float32)); np.save(f"{out}_theta.npy", th.astype(np.float32))
    np.save(f"{out}_xyz.npy", C["xyz"])
    zb = (be - be.mean(1, keepdims=True)) / be.std(1, keepdims=True); rr = zb @ zb.T / be.shape[1]
    np.fill_diagonal(rr, -1)
    R = {"K": K, "seed": seed, "mode": mode, "taus": list(taus), "steps_per_tau": STEPS_PER_TAU, "eps": eps,
         "iterations": it, "seconds": round(time.time() - t0), "ll_tokens": ll, "max_network_r": float(rr.max()),
         "collapsed": bool(rr.max() > COLLAPSE_R), "nodes": C["nodes"]}
    json.dump(R, open(f"{out}.json", "w"))
    print(json.dumps({k: v for k, v in R.items() if k != "nodes"}), flush=True)


if __name__ == "__main__":
    main()
