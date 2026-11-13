"""A construct's brain map from a construct-hierarchy fit of the task-networks model.

The construct's pooled mixture thetabar (task_networks/model.py: one E-step with the fitted theta and beta gives
every node's expected counts n, a contrast's unit mass shared among its authors) is the summed counts of every task
in its subtree, each task counted once and in full (the subtree is a set, so a task reached by two paths counts once;
a task with several constructs counts fully for each), plus ALPHA_PRIOR contrast-equivalents of the reference g,
normalized. g is the corpus mixture of the same kind of mass: every task author's counts (modality, material and
shared-pattern authors excluded, since a construct map holds task mass only). So a construct with little data shrinks
toward the average task, not toward a flat mixture. Times the network maps beta it gives P(voxel | construct). Two
maps, on the fit's 4 mm grid, written as NIfTI (MNI152) with a PNG:
- <label>_p.nii.gz: P(voxel | construct) with the shared-pattern network removed (renormalized);
- <label>_enrichment.nii.gz: log2 P(voxel | construct) / P(voxel | g), both with the shared network
  removed: where the construct's activation is above (> 0) or below the average task. The PNG shows it only where
  P(voxel | construct) is >= 20% of its maximum: elsewhere it is a ratio of two tiny numbers.
(Before 2026-11-11 each child was split equally among its parents and a flat +1 per network was added: a task with
11 parents gave each 1/11 of its mass, so thin constructs were mostly the flat prior; selective attention had 10.3
contrast-equivalents against 20.)
Prints the tasks and constructs below it and its top networks (named by match to the original K=16 maps).

usage: python construct_map.py FIT_PREFIX CORPUS_JSON "construct label" OUT_DIR   (cbmr-gpu venv)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

R = "/data/james/agents/dev/task-networks-repro"
sys.path.insert(0, f"{R}/repo/src")
from task_networks.corpus import load_corpus  # noqa: E402


SHARED_MIN_OWNED = 0.6  # below this the fit has no clean shared network (K=20: 0.71; K=21: 0.57, insula-cingulate)
ALPHA_PRIOR = 10.0  # contrast-equivalents of the reference g added to every pooled mixture (a tuning default)


def is_task_author(node):
    """content authors: tasks and constructs (modality and material, shared-pattern and deactivation authors excluded:
    a map holds what tasks measure, and the deactivation author has its own map)"""
    return not node.startswith("m:") and node not in ("background", "deactivation")


def is_task(node):
    return not node.startswith(("c:", "m:", "x:")) and node not in ("background", "deactivation")


def constructs_author(C):
    """whether construct nodes author contrasts (contrast-level construct authors, corpus B) rather than only sit above
    tasks (the release corpus R)"""
    return any(C["nodes"][a].startswith("c:") for au in C["auth"] for a in au)


def task_content(C, th, be, mult=None):
    """with construct authors (James 2026-11-12: "the tasks are just mixtures of the contrasts"): per task node, the
    content mass of its own contrasts (each contrast's expected counts of its task and construct authors, split
    equally among the contrast's tasks), times mult per contrast (a bootstrap fit's multiplicities); other rows zero.
    Counted in tokens, like the fits' E-step counts (gibbs_init.py's _n.npy) that pooled_mass adds them to, not in
    expected_counts' unit mass per contrast"""
    nodes = C["nodes"]; content = np.array([is_task_author(x) for x in nodes]); task = np.array([is_task(x) for x in nodes])
    out = np.zeros_like(th, dtype=float)
    for d, (toks, auth) in enumerate(zip(C["toks"], C["auth"])):
        w = 1.0 if mult is None else float(mult[d])
        if len(toks) == 0 or len(auth) == 0 or w == 0:
            continue
        a = np.asarray(auth); q = th[a][:, None, :] * be[:, toks].T[None, :, :]
        q = q / q.sum((0, 2), keepdims=True)
        share = q[content[a]].sum((0, 1)); ts = a[task[a]]
        if len(ts):
            out[ts] += w * share / len(ts)
    return out


def expected_counts(C, th, be):
    """one E-step: expected peak mass per node x network from its own documents (each contrast's mass sums to 1)"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.tensor(th, device=dev, dtype=torch.float64); B = torch.tensor(be, device=dev, dtype=torch.float64)
    n = torch.zeros_like(T)
    for toks, auth in zip(C["toks"], C["auth"]):
        if len(toks) == 0 or len(auth) == 0:
            continue
        a = torch.tensor(auth, device=dev); v = torch.tensor(toks, device=dev)
        q = T[a][:, None, :] * B[:, v].T[None, :, :]
        q = q / q.sum((0, 2), keepdim=True) / len(toks)
        n.index_add_(0, a, q.sum(1))
    return n.cpu().numpy()


def pooled_mass(C, n, tc=None):
    """N (node x network): the summed counts of the task authors in each node's subtree (itself included), each
    once; g: the corpus mixture of task-author mass; children lists.
    With construct authors (tc, task_content's rows; James 2026-11-12): a construct's N is its own counts plus those of
    the constructs below it along construct-to-construct (kind-of) edges, never a task's (a task's constructs already
    author its task-vs-baseline contrasts, and a task under a construct would bring all its contrasts: simple choice
    RT's feedback contrasts into processing speed); a task's N is its contrasts' content mass tc; any other node's its
    own counts. g is the same in both: all content-author mass"""
    P = C["parents"]; children = [[] for _ in P]
    for c, ps in enumerate(P):
        for p in ps:
            children[p].append(c)
    task = np.array([is_task_author(x) for x in C["nodes"]])
    memo = {}

    def below(i):
        if i not in memo:
            s = {i} if task[i] else set()
            for c in children[i]:
                s |= below(c)
            memo[i] = frozenset(s)
        return memo[i]
    sys.setrecursionlimit(10000)
    g = n[task].sum(0); g = g / g.sum()
    if tc is not None:
        con = np.array([x.startswith("c:") for x in C["nodes"]]); cm_ = {}

        def cbelow(i):
            if i not in cm_:
                s = {i}
                for c in children[i]:
                    if con[c]:
                        s |= cbelow(c)
                cm_[i] = frozenset(s)
            return cm_[i]
        N = np.array([n[list(cbelow(i))].sum(0) if con[i] else tc[i] if is_task(C["nodes"][i]) else n[i]
                      for i in range(len(P))])
        return N, g, children
    N = np.array([n[list(below(i))].sum(0) if below(i) else np.zeros(n.shape[1]) for i in range(len(P))])
    return N, g, children


def pooled(C, th, be):
    """thetabar (node x network), the own counts n, children; see the module docstring"""
    n = expected_counts(C, th, be)
    N, g, children = pooled_mass(C, n)
    bar = N + ALPHA_PRIOR * g
    return bar / bar.sum(1, keepdims=True), n, children


def shared_network(C, n):
    """the shared-pattern network: the one the shared-pattern author ("background") owns the largest share of. The
    design defines that author as the pattern common to every contrast, so its network is found from the fit itself;
    a match to the original K=16 shared map picks the wrong network when the fit's shared map differs (K=21: r 0.30)"""
    b = C["nodes"].index("background"); own = n[b] / n.sum(0); k = int(np.argmax(own))
    if own[k] < SHARED_MIN_OWNED:
        print(f"WARNING: the shared-pattern author owns only {own[k]:.2f} of its network {k}: no network is the shared "
              "pattern alone, and removing it removes task signal too", flush=True)
    return k


def reference(C, n):
    """g: the corpus mixture of task-author mass, the reference for enrichment"""
    task = np.array([is_task_author(x) for x in C["nodes"]])
    g = n[task].sum(0)
    return g / g.sum()


def main():
    prefix, corpus, label, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    th = np.load(f"{prefix}_theta.npy").astype(float); be = np.load(f"{prefix}_beta.npy").astype(float)
    xyz = np.load(f"{prefix}_xyz.npy"); nodes = json.load(open(f"{prefix}.json"))["nodes"]
    H = json.load(open(corpus)); lab = H["labels"]
    C = load_corpus(corpus); assert C["nodes"] == nodes
    b0 = np.load(f"{R}/K16_seed4_beta.npy"); names = json.load(open(f"{R}/repo/results/k16_names.json"))
    z = lambda M: (M - M.mean(1, keepdims=True)) / M.std(1, keepdims=True)
    from scipy.optimize import linear_sum_assignment
    r = z(be) @ z(b0).T / be.shape[1]; i, j = linear_sum_assignment(-r)
    name = {a: names[str(b)] for a, b in zip(i, j)}; rr = {a: r[a, b] for a, b in zip(i, j)}
    node = next(n for n in nodes if n.startswith("c:") and lab.get(n) == label)
    k_ = nodes.index(node)
    bar, n, children = pooled(C, th, be)
    shared = [shared_network(C, n)]
    keep = [k for k in range(len(be)) if k not in shared]
    mix = bar[k_].copy(); mix[shared] = 0; mix /= mix.sum()
    glob = reference(C, n); glob[shared] = 0; glob /= glob.sum()
    p = mix @ be; p_all = glob @ be
    enr = np.log2(p / p_all)
    # 4 mm grid image (the fit's voxels are the 2 mm MNI152 voxels with even indices)
    import nibabel as nib
    from nimare.utils import get_template
    tmpl = get_template("mni152_2mm", mask="brain"); aff2 = tmpl.affine
    ijk = np.round((np.c_[xyz, np.ones(len(xyz))] @ np.linalg.inv(aff2).T)[:, :3]).astype(int) // 2
    shape = tuple(np.array(tmpl.shape) // 2 + 1); aff4 = aff2 @ np.diag([2, 2, 2, 1])
    safe = label.replace(" ", "_").replace("/", "-")
    for tag, vals in (("p", p), ("enrichment", enr)):
        vol = np.zeros(shape, np.float32); vol[tuple(ijk.T)] = vals
        nib.save(nib.Nifti1Image(vol, aff4), out / f"{safe}_{tag}.nii.gz")
    try:  # pictures: P scaled to its maximum (> 0.3); enrichment only where P >= 20% of its maximum (a ratio of two
        # tiny probabilities is noise), positive part (> 0.5, i.e. > 1.4x the average contrast)
        from nilearn import plotting
        pv = np.zeros(shape, np.float32); pv[tuple(ijk.T)] = p / p.max()
        ev = np.zeros(shape, np.float32); ev[tuple(ijk.T)] = np.where(p >= 0.2 * p.max(), np.clip(enr, 0, None), 0)
        for tag, vol, thr, title in (("p", pv, 0.3, f"{label}: P(voxel | construct), scaled to max"),
                                     ("enrichment", ev, 0.5, f"{label}: log2 enrichment over all contrasts, where P >= 20% of max")):
            disp = plotting.plot_glass_brain(nib.Nifti1Image(vol, aff4), threshold=thr, colorbar=True, cmap="hot",
                                             plot_abs=False, vmin=0, title=title)
            disp.savefig(str(out / f"{safe}_{tag}.png"), dpi=130); disp.close()
    except Exception as e:
        print("no figure:", e)
    below = [nodes[c] for c in children[k_]]
    print(f"{label} ({node}): parents {[lab.get(nodes[q], nodes[q]) for q in C['parents'][k_]]}; "
          f"below: {[lab.get(b, b) for b in below]}")
    print("  networks (shared pattern removed): " + "; ".join(
        f"{name[k].split(':')[0]} {mix[k]:.2f} (match r {rr[k]:.2f})" for k in np.argsort(-mix)[:5]))
    top = np.argsort(-enr)[:8]
    print("  most enriched voxels (MNI mm, log2):", "; ".join(f"({x:.0f}, {y:.0f}, {zz:.0f}) {enr[t]:.1f}" for t, (x, y, zz) in zip(top, xyz[top])))
    print(f"  wrote {out}/{safe}_p.nii.gz, {safe}_enrichment.nii.gz and PNGs")


if __name__ == "__main__":
    main()
