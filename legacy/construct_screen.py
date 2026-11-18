"""Screen well-estimated constructs for gross model errors (James 2026-11-13: "which well estimated constructs should
I look at to find gross errors"). For constructs with credited mass >= 30 contrast-equivalents, map reliability >= 0.9,
bootstrap band <= 2 log2 units and >= 500 significant MKDAChi2 association voxels, three numbers, each over the
contrasts the construct or a narrower construct authors (the set its map and MKDAChi2 use):
- r_spec: the release's enrichment map against the MKDAChi2 association z (on the model's 4 mm grid);
- r_full: the model's full prediction for those contrasts (each contrast's mixture over all its authors, in the most
  typical fit, averaged; log2 over the average contrast) against the same z: whether the model reproduces its data,
  apart from the construct map's specific-contribution reading;
- gain: B's held-out log-likelihood per peak over the training map on those contrasts (k21_within CV), and B - R.
Writes atm/construct_screen.json.

usage: (beast, pilot-results/atm; cbmr-gpu venv) python ../code/construct_screen.py
"""
import json, sys, collections, numpy as np, nibabel as nib
sys.path.insert(0, "/data/james/agents/dev/task-networks-repro/repo/src")
from task_networks.corpus import load_corpus
R = "maps_release/K21_w_B"; TPC = 426.15
C = load_corpus("corpus_w_B.json"); nodes = C["nodes"]; idx = {n: i for i, n in enumerate(nodes)}; xyz = C["xyz"]
man = json.load(open(f"{R}/manifest.json"))["constructs"]; summ = json.load(open(f"{R}/mkda/summary.json"))["constructs"]
T = json.load(open("artifact_data_w_B/taxonomy.json")); tid = {c["label"]: c["id"] for c in T["constructs"]}
Nw = json.load(open(f"{R}/networks.json")); s = Nw["most_typical_seed"]
th = np.load(f"release_wB_K21/seeds/gibbs_s{s}_theta.npy").astype(np.float32); be = np.load(f"release_wB_K21/seeds/gibbs_s{s}_beta.npy").astype(np.float32)
pa = th @ be                                                     # P(voxel | author), every author
ok = [m for m in man if m["credited_mass"] / TPC >= 30 and (m.get("reliability_p") or 0) >= 0.9
      and (m.get("boot_band_median_width") or 9) <= 2.0 and summ.get(tid[m["label"]], {}).get("association_voxels", 0) >= 500]
print(len(ok), "well-estimated constructs of", len(man))
# a construct's contrasts: those it or a narrower construct authors, as for its specific map and MKDAChi2
kids = collections.defaultdict(list)
for c, ps in enumerate(C["parents"]):
    if nodes[c].startswith("c:"):
        for p in ps:
            if nodes[p].startswith("c:"): kids[p].append(c)
def below(k, seen=None):
    seen = {k} if seen is None else seen
    for c in kids[k]:
        if c not in seen: seen.add(c); below(c, seen)
    return seen
want = {idx[m["node"]]: below(idx[m["node"]]) for m in ok}
member = collections.defaultdict(set)
for k, S in want.items():
    for c in S: member[c].add(k)
acc = {k: np.zeros(len(xyz), np.float32) for k in want}; n = collections.Counter(); allacc = np.zeros(len(xyz), np.float64)
for toks, auth in zip(C["toks"], C["auth"]):
    if len(toks) == 0: continue
    dm = pa[auth].mean(0); allacc += dm
    for k in set().union(*(member.get(int(a), set()) for a in auth)):
        acc[k] += dm; n[k] += 1
pall = allacc / len(C["toks"])
# held-out gain per contrast (B's CV, same analysis ids) and R's, for the construct's contrasts
rows = lambda f: {x[0]: (float(x[1]), float(x[2]), float(x[3])) for x in np.load(f, allow_pickle=True)}
cvB = rows("k21_within/cv_abs_K21_B.npy"); cvR = rows("k21_within/cv_abs_K21_R.npy")
docs_of = collections.defaultdict(list)
for k, d in json.load(open("corpus_w_B.json"))["docs"].items():
    for i in d["ids"]:
        if i.startswith("c:"): docs_of[i].append(k)
def sample(img_path):
    img = nib.load(img_path); inv = np.linalg.inv(img.affine); X = img.get_fdata()
    ijk = np.round(np.c_[xyz, np.ones(len(xyz))] @ inv.T)[:, :3].astype(int)
    inside = np.all((ijk >= 0) & (ijk < X.shape), 1); out = np.full(len(xyz), np.nan); out[inside] = X[tuple(ijk[inside].T)]; return out
res = []
for m in ok:
    k = idx[m["node"]]; mk = sample(f"{R}/mkda/constructs/{tid[m['label']]}_association_z.nii.gz")
    enr = sample(f"{R}/{m['files']['enrichment']}"); full = np.log2((acc[k] / n[k]) / pall)
    good = np.isfinite(mk) & np.isfinite(enr)
    r_spec = np.corrcoef(enr[good], mk[good])[0, 1]; r_full = np.corrcoef(full[good], mk[good])[0, 1]
    ks = sorted({d for c in want[k] for d in docs_of[nodes[c]]}); ks = [d for d in ks if d in cvB and d in cvR and cvB[d][0] > 0]
    pk = sum(cvB[d][0] for d in ks)
    gB = sum(cvB[d][1] for d in ks) / pk if pk else np.nan; dBR = (sum(cvB[d][2] for d in ks) - sum(cvR[d][2] for d in ks)) / pk if pk else np.nan
    res.append(dict(label=m["label"], contrasts=m["contrasts"], mass=round(m["credited_mass"] / TPC), rel=m["reliability_p"], band=m["boot_band_median_width"],
                    r_spec=round(float(r_spec), 3), r_full=round(float(r_full), 3), gain=round(float(gB), 3), b_minus_r=round(float(dBR), 3)))
json.dump(res, open("construct_screen.json", "w"), indent=1)
rs = np.array([x["r_spec"] for x in res]); rf = np.array([x["r_full"] for x in res])
print("median r(specific map, MKDA) %.2f; median r(full prediction, MKDA) %.2f; correlation of the two r's %.2f" % (np.median(rs), np.median(rf), np.corrcoef(rs, rf)[0, 1]))
print("\nlowest r(full prediction, MKDA): the model cannot reproduce its own contrasts")
for x in sorted(res, key=lambda x: x["r_full"])[:15]: print("  %-34s r_full %.2f  r_spec %.2f  gain %+.3f  B-R %+.3f  (%d contrasts, %d c-e, rel %.2f, band %.1f)" % (x["label"][:34], x["r_full"], x["r_spec"], x["gain"], x["b_minus_r"], x["contrasts"], x["mass"], x["rel"], x["band"]))
print("\nlowest held-out gain over the average map:")
for x in sorted(res, key=lambda x: x["gain"])[:10]: print("  %-34s gain %+.3f  B-R %+.3f  r_full %.2f  (%d contrasts)" % (x["label"][:34], x["gain"], x["b_minus_r"], x["r_full"], x["contrasts"]))
print("\nlow specific, high full (the specific-contribution effect, not an error):")
for x in sorted([x for x in res if x["r_full"] > 0.6], key=lambda x: x["r_spec"])[:8]: print("  %-34s r_spec %.2f  r_full %.2f" % (x["label"][:34], x["r_spec"], x["r_full"]))
