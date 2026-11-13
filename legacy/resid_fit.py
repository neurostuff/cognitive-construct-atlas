"""Task-networks author-topic model with hierarchical residual maps (James 2026-11-11: try private task maps, shrunk
toward their constructs' residuals, to see if they help prediction and representation).

Model. A token of contrast d picks one of its authors a uniformly. Network-only authors (modality, material, the
shared pattern) then pick a network k ~ theta[a] and a voxel ~ beta[k], as in task_networks.model. A task author
(any author that is not m:* or background: tasks, and families that carry contrasts) does that with probability
1 - lam[a], and with probability lam[a] draws the voxel from its own residual map rho[a]:

    P(v | a) = (1 - lam[a]) sum_k theta[a, k] beta[k, v] + lam[a] rho[a, v]

Priors (MAP, all closed form):
- theta, beta: as task_networks.model (theta shrunk toward its parents' pooled mixtures with s; beta + ETA);
- lam[a] ~ Beta(A0, B0): lam = (private mass + A0) / (author mass + A0 + B0);
- rho[a] ~ Dirichlet(c * m[a] + ETA) on spatially smoothed residual counts: rho[a] ∝ smooth(R[a]) + c m[a] + ETA,
  where smooth is a masked Gaussian (RESID_SIGMA mm) and m[a] is the mean over a's parents of the parent's pooled
  residual: the normalized sum of the smoothed residual counts of the other task authors in its subtree (each once,
  as construct_map.pooled_mass pools task mass; a's own counts left out, so a task is shrunk toward what its
  siblings share, not toward itself). c (pseudo-counts, in tokens) is the shrinkage strength.
  (First run, 2026-11-11: no smoothing, self included, c 1e2-1e5: residuals took over (lam ~1) and held-out
  prediction fell by 0.02-0.09 nats/peak; tokens are 10 mm spheres, so a task's ~1e5 tokens are far fewer
  independent observations, and an unsmoothed private map copies its training peaks.) A node with no parent residual uses the corpus residual. A task with no contrasts (held out) gets
  rho = m, lam = A0 / (A0 + B0), theta from its parents: its prediction comes from its constructs and family.
Fitting: EM (the split E-step of task_networks.model with one more option per task author: its residual map),
started from a fitted network model (theta, beta) with lam = LAM0 and rho[a] = a's own network map, so lam = 0
reproduces the starting fit.

usage:
  python resid_fit.py synthetic
  python resid_fit.py job CORPUS K OUT.json cv|zeroshot INDEX C[,C...] [SEED]   (one CV fold or held-out-task group)
  python resid_fit.py full CORPUS FIT_PREFIX C OUT_PREFIX                        (continue a full-data fit)
(cbmr-gpu venv; PYTHONPATH = task-networks src; from pilot-results/atm)
"""
import json
import os
import sys
import time

import numpy as np
import torch

from task_networks import model as M

A0, B0 = 1.0, 9.0   # lam prior, Beta(1, 9): mean 0.1 (a tuning default)
RESID_SIGMA = 4.25  # mm, residual smoothing: a Gaussian of 10 mm FWHM, the MKDA sphere radius of the tokens (a tuning default)
LAM0 = 0.05         # lam at the start of the continuation
DEV = os.environ.get("RESID_DEVICE", "cuda:0")


def private_mask(nodes):
    return np.array([not x.startswith("m:") and x != "background" and not x.startswith("c:") for x in nodes])


def subtree_tasks(parents, is_priv):
    """per node, the private-author (task) nodes in its subtree, itself included"""
    children = [[] for _ in parents]
    for c, ps in enumerate(parents):
        for p in ps:
            children[p].append(c)
    memo = {}

    def below(i):
        if i not in memo:
            s = {i} if is_priv[i] else set()
            for c in children[i]:
                s |= below(c)
            memo[i] = s
        return memo[i]
    sys.setrecursionlimit(10000)
    return [sorted(below(i)) for i in range(len(parents))]


class ResidSplit(M._Split):
    def estep(self, theta, beta, lam, rho, prow):
        """expected network counts n (nodes x K), voxel counts b (V x K), residual counts R (n_priv x V), private
        mass per node, and the log-likelihood"""
        n = torch.zeros(self.n_nodes, self.K); b = torch.zeros(self.V, self.K)
        R = torch.zeros(rho.shape); pm = torch.zeros(self.n_nodes); ll = 0.0
        on = {d: (theta.to(d), beta.to(d), lam.to(d), rho.to(d), prow.to(d)) for d in self.devices}
        for i in range(0, len(self.chunks), len(self.devices)):
            parts = []
            for d, T, Au, V, w, nt, dwc in self.chunks[i:i + len(self.devices)]:
                th, be, la, rh, pr = on[d]
                q = th[Au] * be[:, V].T * (w * (1 - la[Au]))[:, None]
                ra = pr[Au]; has = ra >= 0
                qp = torch.zeros(len(T), device=d)
                qp[has] = w[has] * la[Au[has]] * rh[ra[has], V[has]]
                tok = torch.zeros(nt, device=d).index_add_(0, T, q.sum(1) + qp)
                r = q / tok[T][:, None]; rp = qp / tok[T]
                Rd = torch.zeros(rh.shape, device=d)
                Rd.index_put_((ra[has], V[has]), rp[has], accumulate=True)
                parts.append((torch.zeros(self.n_nodes, self.K, device=d).index_add_(0, Au, r),
                              torch.zeros(self.V, self.K, device=d).index_add_(0, V, r), Rd,
                              torch.zeros(self.n_nodes, device=d).index_add_(0, Au, rp), torch.log(tok).sum()))
                del q, r, qp, rp, tok
            for pn, pb, pR, pp, pl in parts:
                n += pn.cpu(); b += pb.cpu(); R += pR.cpu(); pm += pp.cpu(); ll += pl.item()
        return n.numpy(), b, R.numpy(), pm.numpy(), ll


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


def theta_step(n, s, parents, order):
    N = n.copy()
    for i in order:
        for p in parents[i]:
            N[p] += N[i] / len(parents[i])
    glob = n.sum(0) + M.ALPHA; glob /= glob.sum()
    bar = N + 1.0; bar /= bar.sum(1, keepdims=True)
    prior = np.array([bar[parents[i]].mean(0) if parents[i] else glob for i in range(len(parents))])
    th = n + s * prior + M.ALPHA
    return th / th.sum(1, keepdims=True)


def residual_centres(R, pidx, parents, sub):
    """m per private author: the mean over its parents of the parent's pooled residual (normalized), the author's
    own counts left out"""
    glob = R.sum(0) + 1e-12; glob /= glob.sum()
    sums = {}

    def pool(p):
        if p not in sums:
            rows = [pidx[t] for t in sub[p] if pidx[t] >= 0]
            sums[p] = R[rows].sum(0) if rows else None
        return sums[p]
    m = np.zeros_like(R)
    for a, row in enumerate(pidx):
        if row < 0:
            continue
        cs = []
        for p in parents[a]:
            v = pool(p)
            if v is None:
                continue
            v = v - R[row] if a in sub[p] else v
            if v.sum() > 1e-9:
                cs.append(v / v.sum())
        m[row] = np.mean(cs, 0) if cs else glob
    return m


def smoother(xyz, sigma=RESID_SIGMA):
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz); D = tree.sparse_distance_matrix(tree, 3 * sigma, output_type="coo_matrix")
    rows = np.r_[D.row, np.arange(len(xyz))]; cols = np.r_[D.col, np.arange(len(xyz))]
    vals = np.r_[np.exp(-D.data ** 2 / (2 * sigma ** 2)), np.ones(len(xyz))]
    S = csr_matrix((vals, (rows, cols)), shape=(len(xyz), len(xyz)))
    return S.multiply(1.0 / np.asarray(S.sum(1))).tocsr()   # rows sum to 1: masked smoothing that keeps mass


def fit_resid(docs_tok, doc_auth, nodes, parents, K, s, n_vox, theta0, beta0, c, iters=1000, tol=1e-6, verbose=False,
              xyz=None):
    nn = len(nodes); priv = private_mask(nodes)
    pidx = -np.ones(nn, int); pidx[priv] = np.arange(priv.sum())
    ntok = np.array([len(t) for t in docs_tok]); tok_doc = np.repeat(np.arange(len(docs_tok)), ntok)
    tok_vox = np.concatenate([np.asarray(t, int) for t in docs_tok]); cnt = np.array([len(A) for A in doc_auth])
    pair_tok = np.repeat(np.arange(len(tok_doc)), cnt[tok_doc])
    pair_auth = np.concatenate([np.tile(np.asarray(A, int), n) for A, n in zip(doc_auth, ntok)])
    sp = ResidSplit([torch.device(DEV)], pair_tok, pair_auth, tok_vox, (1.0 / cnt[tok_doc])[pair_tok], K, nn, n_vox)
    order = bottom_up(parents); sub = subtree_tasks(parents, priv)
    theta = np.asarray(theta0, float); beta = np.asarray(beta0, float)
    lam = np.where(priv, LAM0, 0.0)
    rho = (theta @ beta)[priv]; rho /= rho.sum(1, keepdims=True)
    prow = torch.tensor(pidx)
    S = smoother(xyz) if xyz is not None else None
    prev = -np.inf
    for it in range(iters):
        n, bsum, R, pm, ll = sp.estep(torch.tensor(theta), torch.tensor(beta), torch.tensor(lam), torch.tensor(rho), prow)
        theta = theta_step(n, s, parents, order)
        bb = bsum.numpy().T + M.ETA; beta = bb / bb.sum(1, keepdims=True)
        mass = n.sum(1) + pm
        lam = np.where(priv, (pm + A0) / (mass + A0 + B0), 0.0)
        Rs = (S @ R.T).T if S is not None else R
        m = residual_centres(Rs, pidx, parents, sub)
        rho = Rs + c * m + M.ETA; rho /= rho.sum(1, keepdims=True)
        if verbose and it % 20 == 0:
            print(f"  it {it}: ll {ll:.0f}, mean lam {lam[priv].mean():.3f}", flush=True)
        if abs(ll - prev) < tol * abs(ll):
            break
        prev = ll
    return {"theta": theta, "beta": beta, "lam": lam, "rho": rho, "pidx": pidx, "ll": ll, "iters": it + 1, "R": R}


def author_maps(F):
    pa = (F["theta"] @ F["beta"]) * (1 - F["lam"])[:, None]
    rows = F["pidx"] >= 0
    pa[rows] += F["lam"][rows][:, None] * F["rho"][F["pidx"][rows]]
    return pa


def score(C, pa, test):
    """per contrast: (absolute log P(peaks | authors), peaks), as task_networks.model.heldout_rows / zeroshot_tasks"""
    out = []
    for d in test:
        f = C["foci"][d]
        if len(f) == 0:
            continue
        out.append((float(np.log(pa[C["auth"][d]][:, f].mean(0)).sum()), len(f)))
    return out


def job(corpus, K, out, kind, index, cs, seed):
    from task_networks.corpus import load_corpus
    C = load_corpus(corpus); H = json.load(open(corpus)); nodes = C["nodes"]; idx = {x: i for i, x in enumerate(nodes)}
    if kind == "cv":
        test = [d for d in range(len(C["ids"])) if C["fold"][d] == index]
        held = []
    else:
        fams = {p for ps in H["parents"].values() for p in ps if not p.startswith(("c:", "m:"))}
        cands = [t for t, ps in H["parents"].items() if not t.startswith(("c:", "m:")) and any(p.startswith("c:") for p in ps)
                 and H["n_contrasts"].get(t, 0) >= 20 and t not in fams]
        rng = np.random.default_rng(0); rng.shuffle(cands); held = cands[index::5]   # zeroshot_tasks.py's groups
        hid = {idx[t] for t in held}
        test = [d for d in range(len(C["ids"])) if hid & set(C["auth"][d].tolist())]
    tset = set(test); train = [d for d in range(len(C["ids"])) if d not in tset]
    dt, da = [C["toks"][d] for d in train], [C["auth"][d] for d in train]
    t0 = time.time()
    th, be, ll0, it0 = M.fit(dt, da, len(nodes), C["parents"], K, 100.0, len(C["xyz"]), iters=1000, seed=seed, devices=[DEV])
    res = {"kind": kind, "index": index, "K": K, "seed": seed, "held": held, "baseline": {"ll_train": ll0, "iters": it0,
           "scores": score(C, th @ be, test)}, "resid": {}}
    for c in cs:
        F = fit_resid(dt, da, nodes, C["parents"], K, 100.0, len(C["xyz"]), th, be, c, xyz=C["xyz"])
        res["resid"][str(c)] = {"ll_train": F["ll"], "iters": F["iters"], "mean_lam": float(F["lam"][F["pidx"] >= 0].mean()),
                                "scores": score(C, author_maps(F), test)}
    if kind == "zeroshot":   # per held-out task, which test contrasts carry it
        res["test_tasks"] = [[nodes[a] for a in C["auth"][d] if nodes[a] in set(held)] for d in test if len(C["foci"][d])]
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(out, "w"))
    b = res["baseline"]["scores"]; nb = sum(x[1] for x in b); lb = sum(x[0] for x in b) / nb
    print(f"{kind} {index}: baseline {lb:.5f} nats/peak; " + "; ".join(
        f"c={c}: {sum(x[0] for x in r['scores']) / nb - lb:+.5f} (lam {r['mean_lam']:.3f}, {r['iters']} it)" for c, r in res["resid"].items()))


def synthetic():
    """3 networks; tasks 0, 2 and 4 each also activate their own blob (30% of their tokens), more patterns than the
    3 fitted networks can hold. Scored on held-out documents, and by how much of each blob task's map lands on its blob"""
    rng = np.random.default_rng(0); K, V, A_, D, L = 3, 300, 6, 1200, 40
    beta = np.zeros((K, V))
    for k in range(K):
        beta[k] = np.exp(-(np.arange(V) - (k + 0.5) * 60) ** 2 / (2 * 10.0 ** 2)) + 1e-3
    beta /= beta.sum(1, keepdims=True)
    centres = {0: 200, 2: 240, 4: 280}
    blobs = {a: np.exp(-(np.arange(V) - c) ** 2 / (2 * 5.0 ** 2)) + 1e-5 for a, c in centres.items()}
    blobs = {a: v / v.sum() for a, v in blobs.items()}
    theta = rng.dirichlet(np.full(K, 0.5), A_)
    nodes = [f"task{a}" for a in range(A_)] + ["c:shared"]; parents = [[A_] for _ in range(A_)] + [[]]

    def make(n):
        docs, auth = [], []
        for _ in range(n):
            a = int(rng.integers(A_)); toks = []
            for _ in range(L):
                if a in blobs and rng.random() < 0.3:
                    toks.append(rng.choice(V, p=blobs[a]))
                else:
                    toks.append(rng.choice(V, p=beta[rng.choice(K, p=theta[a])]))
            docs.append(np.unique(toks)); auth.append(np.array([a]))
        return docs, auth
    docs, auth = make(D); tdocs, tauth = make(400)
    th, be, ll0, it0 = M.fit(docs, auth, len(nodes), parents, K, 0.0, V, iters=500, seed=1, devices=[DEV])
    def heldout(pa):
        return sum(float(np.log(pa[a[0], d]).sum()) for d, a in zip(tdocs, tauth)) / sum(len(d) for d in tdocs)
    def on_blob(pa):
        return {a: round(float(pa[a][max(0, c - 12):c + 13].sum()), 3) for a, c in centres.items()}
    pa0 = th @ be
    print(f"network-only: held-out {heldout(pa0):.4f} nats/token; map mass on own blob {on_blob(pa0)} (true ~0.3)")
    for c in (10.0, 100.0, 1000.0):
        F = fit_resid(docs, auth, nodes, parents, K, 0.0, V, th, be, c, iters=500)
        pa = author_maps(F)
        print(f"c={c:g}: held-out {heldout(pa):.4f} nats/token; map mass on own blob {on_blob(pa)}; lam {np.round(F['lam'][:A_], 2)}")


def main():
    cmd = sys.argv[1]
    if cmd == "synthetic":
        synthetic()
    elif cmd == "job":
        corpus, K, out, kind, index, cs = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7]
        job(corpus, K, out, kind, index, [float(x) for x in cs.split(",")], int(sys.argv[8]) if len(sys.argv) > 8 else 0)
    elif cmd == "full":
        from task_networks.corpus import load_corpus
        corpus, prefix, c, outp = sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5]
        C = load_corpus(corpus)
        th = np.load(f"{prefix}_theta.npy").astype(float); be = np.load(f"{prefix}_beta.npy").astype(float)
        F = fit_resid(C["toks"], C["auth"], C["nodes"], C["parents"], len(be), 100.0, len(C["xyz"]), th, be, c, verbose=True,
                      xyz=C["xyz"])
        for k in ("theta", "beta", "lam", "rho", "pidx", "R"):
            np.save(f"{outp}_{k}.npy", np.asarray(F[k], np.float32 if k not in ("pidx",) else np.int64))
        np.save(f"{outp}_xyz.npy", C["xyz"])
        json.dump({"c": c, "ll": F["ll"], "iters": F["iters"], "nodes": C["nodes"]}, open(f"{outp}.json", "w"))
        print(f"full fit: ll {F['ll']:.0f}, {F['iters']} iterations")


if __name__ == "__main__":
    main()
