"""Gibbs burn-in, then EM, for the task-networks author-topic model, following the Yeo lab's released procedure
(CBIG stable_projects/meta-analysis/Yeo2015_AuthorTopicEM, commit 634f676; Yeo et al. 2015, Cereb Cortex 25:3654,
doi:10.1093/cercor/bhu217), for the seed pilot (James 2026-11-11: build the Gibbs comparison to be equivalent to Yeo).

What the CBIG code does, and what this does:
1. Collapsed Gibbs over the (author, network) pair of each token jointly (GibbsSamplerAT.cpp, Steyvers & Griffiths
   topic toolbox), from a random start (author uniform among the document's authors, network uniform), for
   BURN_IN = 100 sweeps; only the final state is used. CBIG samples one token at a time in one fixed random order.
   Here tokens are sampled in blocks (at least MIN_BLOCKS per sweep, in a fresh random order every sweep): a block's
   tokens are drawn jointly from the conditional given the current counts of all tokens, and the block's counts are
   then replaced. Within a block the draws do not see each other, and a token's own previous assignment stays in the
   counts it is drawn from (one count among the 12.5M); this is the blocked, parallel approximation to sequential
   Gibbs (as AD-LDA, Newman et al. 2009), needed to sample 12.5M tokens per sweep. (A first version removed the whole
   block's counts before drawing it; with one block that removed every count, and the synthetic check caught it.)
2. The conditional uses this model's own priors, not CBIG's flat ones: P(a, k) ∝ (1 / |A_d|) theta[a, k] beta[k, v]
   with theta[a] ∝ n[a] + s * (parents' pooled mixture) + ALPHA and beta[k] ∝ b[k] + ETA, as in the M-step of
   task_networks.model (CBIG: theta ∝ AT + 100, phi ∝ WP + 0.01; our s = 100 plays the role of their alpha = 100,
   toward the parents instead of toward uniform).
3. Gibbs to EM, as CBIG_AuthorTopicEM_RunGibbsSampler.m and _InitParams.m: theta and beta from the final counts by the
   formulas above; beta smoothed in space and renormalized; theta not smoothed. CBIG smooths 5 times with a masked
   3x3x3 box mean on its 2 mm grid. Our grid is 4 mm, so the same physical smoothing is used instead: 5 passes of a
   2 mm 3-point box have variance 5 * 8/3 mm^2 per axis, a Gaussian of sigma SMOOTH_SIGMA = 3.65 mm, applied with the
   same mask normalization (smoothed map / smoothed mask).
4. EM to convergence from that start: the task-networks EM (anneal_fit.fit_once with no tempering), tol 1e-6 and a
   1000-iteration cap as in our other fits (CBIG: relative tol 1e-5, 10 to 1000 iterations).
5. Seeds and choice of solution: CBIG runs 100 seeds and keeps the most typical (ComputeMostTypicalSolution); the pilot
   runs 20 and pilot_score.py reports the most typical and the best.

usage:
  python gibbs_init.py check-sampler CORPUS        # mean of sampled counts = EM expected counts (fixed theta, beta)
  python gibbs_init.py check-synthetic             # Gibbs + EM recovers known networks
  python gibbs_init.py fit CORPUS K OUT_PREFIX SEED [ITERS [BOOT]]
      # BOOT: fit to a bootstrap resample of papers (resample id BOOT): papers drawn with replacement, each drawn
      # paper's contrasts repeated as many times as it was drawn (folds are grouped by paper, so is the resample).
      # Writes OUT_PREFIX_n.npy, the E-step counts at the fitted parameters on the data it was fitted to.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_networks import model as M  # noqa: E402

import os
BURN_IN = int(os.environ.get("GIBBS_BURN_IN", 100))  # CBIG burn_in; the env override is for quick tests only
MIN_BLOCKS = 10  # Gibbs blocks per sweep (more blocks = closer to sequential Gibbs, slower)
DEV = os.environ.get("GIBBS_DEVICE", "cuda:0")  # checks can run on "cpu"
SMOOTH_SIGMA = 3.65  # mm: 5 passes of a 3x3x3 box at 2 mm (CBIG init_smooth = 5), as a Gaussian


class SampleSplit(M._Split):
    def __init__(self, devices, pair_tok, pair_auth, tok_vox, wpair, K, n_nodes, V, min_chunks=MIN_BLOCKS):
        """as M._Split, with at least min_chunks chunks (the Gibbs blocks)"""
        super().__init__(devices, pair_tok, pair_auth, tok_vox, wpair, K, n_nodes, V)
        if len(self.chunks) >= min_chunks:
            return
        P = len(pair_tok); n_chunks = min_chunks
        starts = np.r_[0, np.flatnonzero(np.diff(pair_tok)) + 1]
        cut = starts[np.minimum(np.searchsorted(starts, np.linspace(0, P, n_chunks + 1)[1:-1]), len(starts) - 1)]
        bounds = np.unique(np.r_[0, cut, P]); self.chunks = []
        for c, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            if b <= a:
                continue
            d = devices[c % len(devices)]; t0 = pair_tok[a]
            self.chunks.append((d, torch.tensor(pair_tok[a:b] - t0, device=d), torch.tensor(pair_auth[a:b], device=d),
                                torch.tensor(tok_vox[pair_tok[a:b]], device=d), torch.tensor(wpair[a:b], device=d),
                                int(pair_tok[b - 1] - t0 + 1), None))

    def sample(self, c, theta, beta, gen):
        """one draw of (author, network) per token of chunk c, from P ∝ theta[a, k] beta[k, v] w; returns the
        chunk's counts (n_nodes x K on the host, V x K on the host)"""
        d, T, Au, V, w, nt, dwc = self.chunks[c]
        q = theta.to(d)[Au] * beta.to(d)[:, V].T * w[:, None]
        g = -torch.log(-torch.log(torch.rand(q.shape, generator=gen[d], device=d)))  # Gumbel-max sampling
        s = torch.log(q) + g
        pmax, kmax = s.max(1)
        tmax = torch.full((nt,), -torch.inf, device=d).scatter_reduce(0, T, pmax, "amax")
        win = pmax == tmax[T]
        idx = torch.arange(len(T), device=d)
        first = torch.full((nt,), len(T), device=d, dtype=torch.long).scatter_reduce(0, T[win], idx[win], "amin")
        n = torch.zeros(self.n_nodes, self.K, device=d); b = torch.zeros(self.V, self.K, device=d)
        n.index_put_((Au[first], kmax[first]), torch.ones(nt, device=d), accumulate=True)
        b.index_put_((V[first], kmax[first]), torch.ones(nt, device=d), accumulate=True)
        return n.cpu(), b.cpu()


def theta_from(n, s, parents, order):
    """the M-step's theta from counts n (task_networks.model._fit_once)"""
    if s > 0:
        N = n.copy()
        for i in order:
            for p in parents[i]:
                N[p] += N[i] / len(parents[i])
        glob = n.sum(0) + M.ALPHA; glob /= glob.sum()
        bar = N + 1.0; bar /= bar.sum(1, keepdims=True)
        prior = np.array([bar[parents[i]].mean(0) if parents[i] else glob for i in range(len(parents))])
        th = n + s * prior + M.ALPHA
    else:
        th = n + M.ALPHA
    return th / th.sum(1, keepdims=True)


def bottom_up(parents):
    children = [[] for _ in parents]
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
    for i in range(len(parents)):
        visit(i)
    return order


def smoother(xyz, sigma=SMOOTH_SIGMA):
    tree = cKDTree(xyz); D = tree.sparse_distance_matrix(tree, 3 * sigma, output_type="coo_matrix")
    rows = np.r_[D.row, np.arange(len(xyz))]; cols = np.r_[D.col, np.arange(len(xyz))]
    vals = np.r_[np.exp(-D.data ** 2 / (2 * sigma ** 2)), np.ones(len(xyz))]
    return csr_matrix((vals, (rows, cols)), shape=(len(xyz), len(xyz)))


def make_split(docs_tok, doc_auth, n_nodes, K, n_vox, devices, cpu_mem=8e9):
    ntok = np.array([len(t) for t in docs_tok])
    tok_doc = np.repeat(np.arange(len(docs_tok)), ntok)
    tok_vox = np.concatenate([np.asarray(t, int) for t in docs_tok])
    cnt = np.array([len(A) for A in doc_auth])
    pair_tok = np.repeat(np.arange(len(tok_doc)), cnt[tok_doc])
    pair_auth = np.concatenate([np.tile(np.asarray(A, int), n) for A, n in zip(doc_auth, ntok)])
    return SampleSplit([torch.device(d) for d in devices], pair_tok, pair_auth, tok_vox, (1.0 / cnt[tok_doc])[pair_tok],
                       K, n_nodes, n_vox)


def gibbs(split, parents, K, s, seed, sweeps=BURN_IN, verbose=False):
    """blocked collapsed Gibbs; returns the final counts (n, b)"""
    order = bottom_up(parents); rng = np.random.default_rng(seed)
    gen = {d: torch.Generator(device=d).manual_seed(seed * 7919 + i) for i, d in enumerate(split.devices)}
    nn, V = split.n_nodes, split.V
    # random start: every token's author uniform among its document's, network uniform (theta, beta uniform)
    th0 = torch.full((nn, K), 1.0 / K); be0 = torch.full((K, V), 1.0 / V)
    parts = [split.sample(c, th0, be0, gen) for c in range(len(split.chunks))]
    n = sum(p[0] for p in parts); b = sum(p[1] for p in parts)
    for sw in range(sweeps):
        for c in rng.permutation(len(split.chunks)):
            th = torch.tensor(theta_from(n.numpy(), s, parents, order))
            bb = b.T + M.ETA; be = bb / bb.sum(1, keepdim=True)
            new = split.sample(c, th, be, gen)
            n += new[0] - parts[c][0]; b += new[1] - parts[c][1]        # replace the block's assignments
            parts[c] = new
        if verbose and (sw + 1) % 20 == 0:
            print(f"  sweep {sw + 1}: tokens {n.sum():.0f}", flush=True)
    return n.numpy(), b.numpy()


def init_from(n, b, s, parents, xyz):
    """CBIG's Gibbs-to-EM start: theta and beta from the final counts, beta smoothed (masked) and renormalized"""
    th = theta_from(n, s, parents, bottom_up(parents))
    be = (b + M.ETA) / (b + M.ETA).sum(0, keepdims=True)              # V x K
    S = smoother(xyz); be = (S @ be) / (S @ np.ones((len(xyz), 1)))
    be = be.T / be.T.sum(1, keepdims=True)
    return th, be


def em_from(docs_tok, doc_auth, n_nodes, parents, K, s, n_vox, theta0, beta0, iters, devices):
    """the task-networks EM from a given start (anneal_fit.fit_once's loop, no tempering)"""
    import anneal_fit as A
    return A.fit_once(docs_tok, doc_auth, n_nodes, parents, K, s, n_vox, iters, 0, 1e-6, devices, taus=(),
                      init=(theta0, beta0))


def main():
    cmd = sys.argv[1]
    if cmd == "check-sampler":
        from task_networks.corpus import load_corpus
        C = load_corpus(sys.argv[2]); D = 300; K = 6; R = 400
        docs, auth = C["toks"][:D], C["auth"][:D]
        sp = make_split(docs, auth, len(C["nodes"]), K, len(C["xyz"]), [DEV])
        g = torch.Generator().manual_seed(1)
        th = torch.rand(len(C["nodes"]), K, generator=g) + 0.1; th /= th.sum(1, keepdim=True)
        be = torch.rand(K, len(C["xyz"]), generator=g) + 0.1; be /= be.sum(1, keepdim=True)
        n_exp, b_exp, _ = M._Split.estep(sp, th, be)
        gen = {d: torch.Generator(device=d).manual_seed(2) for d in sp.devices}
        acc = np.zeros_like(n_exp)
        for _ in range(R):
            acc += sum(sp.sample(c, th, be, gen)[0] for c in range(len(sp.chunks))).numpy()
        m = acc / R; used = n_exp > 1
        z = (m[used] - n_exp[used]) / np.sqrt(n_exp[used] / R)  # Poisson-scale standard error
        print(f"sampled vs expected author-network counts ({used.sum()} cells, {R} draws): max rel err "
              f"{np.max(np.abs(m[used] / n_exp[used] - 1)):.4f}; z mean {z.mean():+.3f}, sd {z.std():.3f} (sd <= ~1 expected)")
        return
    if cmd == "check-synthetic":
        rng = np.random.default_rng(0); K, V, A_, D, L = 4, 300, 12, 800, 60
        xyz = np.c_[np.arange(V) * 4.0, np.zeros(V), np.zeros(V)]
        beta = np.zeros((K, V))
        for k in range(K):
            beta[k] = np.exp(-(np.arange(V) - (k + 0.5) * V / K) ** 2 / (2 * 15.0 ** 2)) + 1e-3
        beta /= beta.sum(1, keepdims=True)
        theta = rng.dirichlet(np.full(K, 0.3), A_)
        docs, auth = [], []
        for _ in range(D):
            a = rng.choice(A_, size=rng.integers(1, 3), replace=False)
            toks = []
            for _ in range(L):
                aa = rng.choice(a); k = rng.choice(K, p=theta[aa]); toks.append(rng.choice(V, p=beta[k]))
            docs.append(np.unique(toks)); auth.append(np.array(a))
        parents = [[] for _ in range(A_)]
        sp = make_split(docs, auth, A_, K, V, [DEV])
        n, b = gibbs(sp, parents, K, 0.0, seed=0, sweeps=100)
        th0, be0 = init_from(n, b, 0.0, parents, xyz)
        th, be, ll, it = em_from(docs, auth, A_, parents, K, 0.0, V, th0, be0, 500, [DEV])
        z = lambda X: (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
        from scipy.optimize import linear_sum_assignment
        for name, est in (("after Gibbs", be0), ("after EM", be)):
            r = z(beta) @ z(est).T / V; i, j = linear_sum_assignment(-r)
            print(f"synthetic, {name}: matched r with the true networks {np.round(r[i, j], 3)}")
        return
    from task_networks.corpus import load_corpus
    corpus, K, out, seed = sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
    iters = int(sys.argv[6]) if len(sys.argv) > 6 else 1000
    boot = int(sys.argv[7]) if len(sys.argv) > 7 else None
    C = load_corpus(corpus); toks, auth = C["toks"], C["auth"]; mult = None
    if boot is not None:
        H = json.load(open(corpus)); ids = sorted(H["docs"])                  # load_corpus document order
        paper = np.array([H["docs"][i]["group"] for i in ids]); papers = np.unique(paper)
        draw = np.random.default_rng(1000003 + boot).choice(len(papers), len(papers), replace=True)
        times = np.bincount(draw, minlength=len(papers)); pidx = np.searchsorted(papers, paper)
        mult = times[pidx]                                                        # per contrast
        keep = np.repeat(np.arange(len(toks)), mult)
        toks = [toks[i] for i in keep]; auth = [auth[i] for i in keep]
    args = (toks, auth, len(C["nodes"]))
    t0 = time.time()
    sp = make_split(*args, K, len(C["xyz"]), [DEV])
    n, b = gibbs(sp, C["parents"], K, 100.0, seed, verbose=True); del sp
    tg = time.time() - t0
    th0, be0 = init_from(n, b, 100.0, C["parents"], C["xyz"])
    th, be, ll, it = em_from(*args, C["parents"], K, 100.0, len(C["xyz"]), th0, be0, iters, [DEV])
    np.save(f"{out}_beta.npy", be.astype(np.float32)); np.save(f"{out}_theta.npy", th.astype(np.float32))
    np.save(f"{out}_xyz.npy", C["xyz"])
    sp = make_split(*args, K, len(C["xyz"]), [DEV])                             # counts on the fitted data
    n_fit, _, _ = M._Split.estep(sp, torch.tensor(th.astype(float)), torch.tensor(be.astype(float))); del sp
    np.save(f"{out}_n.npy", n_fit)
    zb = (be - be.mean(1, keepdims=True)) / be.std(1, keepdims=True); rr = zb @ zb.T / be.shape[1]; np.fill_diagonal(rr, -1)
    R = {"K": K, "seed": seed, "mode": "gibbs", "bootstrap": boot,
         "bootstrap_contrasts": None if mult is None else int(mult.sum()),
         "bootstrap_papers_drawn": None if mult is None else int((times > 0).sum()), "burn_in": BURN_IN, "smooth_sigma_mm": SMOOTH_SIGMA, "iterations": it,
         "gibbs_seconds": round(tg), "seconds": round(time.time() - t0), "ll_tokens": ll, "max_network_r": float(rr.max()),
         "nodes": C["nodes"]}
    json.dump(R, open(f"{out}.json", "w"))
    print(json.dumps({k: v for k, v in R.items() if k != "nodes"}), flush=True)


if __name__ == "__main__":
    main()
