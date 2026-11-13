"""Figures and a printed summary of the author-topic results (run list F1), from atm/ after atm_queue.sh.

- fig1_designs.png: held-out contrasts, each K=16 design minus families + modality (nats per peak, 95% paper
  bootstrap, cv_compare.compare), and held-out tasks, each design's mean gain over the average map;
- fig2_ksweep.png: held-out contrasts and held-out tasks by K, for the chosen construct design and the families
  design (every K evaluated, including the refinement);
- fig3_tasks.png: held-out tasks, constructs vs families per task (the K=16 chosen design);
- fig4_maps.png: enrichment glass brains (where P >= 20% of max) of a panel of constructs, at the chosen K;
- results.md: the same numbers as tables.

usage: python results_figures.py DESIGN_CORPUS DESIGN_TAG16 K_MAPS OUT_DIR   (cbmr-gpu venv; from pilot-results/atm)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_compare import compare  # noqa: E402

DESIGNS = [  # (CV file, name, held-out json, side of that json)
    ("cv/K16fm.npy", "families + modality", "zeroshot_K16_both_mod.json", "families"),
    ("cv/K16am.npy", "constructs + modality", "zeroshot_K16_all_mod.json", "constructs"),
    ("cv/K16bm.npy", "constructs + family + modality", "zeroshot_K16_both_mod.json", "constructs"),
    ("cv/K16ca.npy", "constructs as authors (+ family, modality)", "zeroshot_K16_ca.json", "constructs"),
    ("cv/K16fx.npy", "families + modality + material", "zeroshot_K16_both_mat.json", "families"),
    ("cv/K16ax.npy", "constructs + modality + material", "zeroshot_K16_all_mat.json", "constructs"),
    ("cv/K16bx.npy", "constructs + family + modality + material", "zeroshot_K16_both_mat.json", "constructs")]
PANEL = ["response inhibition", "working memory", "memory", "episodic memory", "theory of mind", "fear",
         "reward processing", "selective attention", "decision making", "language", "pain", "motor control",
         "emotion regulation", "face perception", "mental rotation", "attention"]


def zs(path, side="constructs"):
    T = json.load(open(path))["tasks"]
    return {t: v[side]["gain_nats_per_peak"] for t, v in T.items()
            if v.get(side, {}).get("gain_nats_per_peak") is not None}


def main():
    design, tag16, kmaps, out = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Author-topic results\n"]
    # 1. designs at K=16
    have = [d for d in DESIGNS if Path(d[0]).exists()]
    rows = compare("corpus_family_mod.json", have[0][0], [d[0] for d in have[1:]])
    md += ["## Designs at K = 16\n", "Held-out contrasts: nats per peak over the average map, and the difference from "
           "families + modality (95% interval, papers resampled). Held-out tasks: mean gain of a task the fit never saw, "
           "predicted from its parents.\n",
           "| design | held-out contrasts | minus families + modality | held-out tasks |", "|---|---|---|---|"]
    zrow = []
    for (cvf, name, zf, side), r in zip(have, rows):
        z = zs(zf, side) if Path(zf).exists() else None
        zrow.append((name, np.mean(list(z.values())) if z else None, len(z) if z else 0))
        md.append(f"| {name} | {r['score']:.4f} | {r['diff']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                  + (f"{np.mean(list(z.values())):.4f} ({len(z)} tasks)" if z else "—") + " |")
    md.append("")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    y = np.arange(len(rows))[::-1]
    ax[0].errorbar([r["diff"] for r in rows], y, xerr=[[r["diff"] - r["lo"] for r in rows], [r["hi"] - r["diff"] for r in rows]],
                   fmt="o", color="#1f5fa8", capsize=3)
    ax[0].axvline(0, color="0.6", lw=1); ax[0].set_yticks(y, [d[1] for d in have]); ax[0].set_xlabel("held-out contrasts: minus families + modality (nats/peak)")
    zz = [(n, v) for n, v, _ in zrow if v is not None]
    ax[1].barh(np.arange(len(zz))[::-1], [v for _, v in zz],
               color=["#999999" if n.startswith("families") else "#1f5fa8" for n, _ in zz])
    ax[1].set_yticks(np.arange(len(zz))[::-1], [n for n, _ in zz]); ax[1].set_xlabel("held-out tasks: mean gain over the average map (nats/peak)")
    fig.suptitle("K = 16 designs"); fig.tight_layout(); fig.savefig(out / "fig1_designs.png", dpi=140); plt.close(fig)
    # 2. K sweep
    import glob, re
    Ks = sorted({16} | {int(re.search(r"K(\d+)", f).group(1)) for f in glob.glob("cv/sweep_K*.npy")
                        if Path(f"zeroshot_sweep_K{re.search(r'K(\d+)', f).group(1)}.json").exists()})
    cvs = [f"cv/{tag16}.npy" if k == 16 else f"cv/sweep_K{k}.npy" for k in Ks]
    sw = compare(design, f"cv/{tag16}.npy", [c for c in cvs if "sweep" in c])
    swd = {16: sw[0]}; swd.update({k: r for k, r in zip([k for k in Ks if k != 16], sw[1:])})
    zf16 = {d[0]: d[2] for d in DESIGNS}.get(f"cv/{tag16}.npy")  # the design's K=16 held-out json
    zk = {k: (np.mean(list(zs(f"zeroshot_sweep_K{k}.json").values())) if k != 16 else
              (np.mean(list(zs(zf16).values())) if zf16 else np.nan)) for k in Ks}
    md += [f"## K sweep ({design})\n", "| K | held-out contrasts | minus K = 16 | held-out tasks |", "|---|---|---|---|"]
    md += [f"| {k} | {swd[k]['score']:.4f} | {swd[k]['diff']:+.4f} [{swd[k]['lo']:+.4f}, {swd[k]['hi']:+.4f}] | {zk[k]:.4f} |" for k in Ks]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    cvs_abs = {k: swd[k]["score"] for k in Ks}
    ax[0].plot(Ks, [cvs_abs[k] for k in Ks], "o-", color="#1f5fa8", label="constructs + family (+ modality, material)")
    ax[1].plot(Ks, [zk[k] for k in Ks], "o-", color="#1f5fa8", label="constructs + family")
    fK = sorted({16} | {int(re.search(r"K(\d+)", f).group(1)) for f in glob.glob("cv/famsweep_K*.npy")
                        if Path(f"zeroshot_famsweep_K{re.search(r'K(\d+)', f).group(1)}.json").exists()})
    def cvs(path):
        a = np.load(path, allow_pickle=True); return sum(float(r[2]) for r in a) / sum(int(r[1]) for r in a)
    fcv = {k: cvs("cv/K16fx.npy" if k == 16 else f"cv/famsweep_K{k}.npy") for k in fK}
    fzs = {k: np.mean(list(zs("zeroshot_K16_both_mat.json" if k == 16 else f"zeroshot_famsweep_K{k}.json", "families").values())) for k in fK}
    ax[0].plot(fK, [fcv[k] for k in fK], "s--", color="#999999", label="families (+ modality, material)")
    ax[1].plot(fK, [fzs[k] for k in fK], "s--", color="#999999", label="families")
    ax[0].set_xlabel("K"); ax[0].set_ylabel("held-out contrasts (nats/peak)"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("K"); ax[1].set_ylabel("held-out tasks: mean gain (nats/peak)"); ax[1].legend(fontsize=8)
    md += ["", "Families design: " + ", ".join(f"K{k} CV {fcv[k]:.4f} / tasks {fzs[k]:.4f}" for k in fK)]
    fig.suptitle(f"K sweep: {design}"); fig.tight_layout(); fig.savefig(out / "fig2_ksweep.png", dpi=140); plt.close(fig)
    # 3. held-out tasks, constructs vs families (K=16, the chosen design)
    if zf16:
        T = json.load(open(zf16))["tasks"]
        famT = T  # the families side of the same run (same tasks, groups and contrasts)
        pairs = [(t, v["constructs"]["gain_nats_per_peak"], famT[t]["families"]["gain_nats_per_peak"]) for t, v in T.items()
                 if t in famT and v.get("constructs", {}).get("gain_nats_per_peak") is not None
                 and famT[t].get("families", {}).get("gain_nats_per_peak") is not None]
        H = json.load(open(design)); lab = H["labels"]
        fig, ax = plt.subplots(figsize=(6.5, 6.5))
        c = np.array([p[1] for p in pairs]); f = np.array([p[2] for p in pairs])
        ax.scatter(f, c, s=14, color="#1f5fa8"); lim = [min(c.min(), f.min()) - 0.02, max(c.max(), f.max()) + 0.02]
        ax.plot(lim, lim, color="0.6", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
        for t, cc, ff in sorted(pairs, key=lambda p: -abs(p[1] - p[2]))[:12]:
            ax.annotate(lab.get(t, t)[:28], (ff, cc), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("families: gain (nats/peak)"); ax.set_ylabel("constructs: gain (nats/peak)")
        ax.set_title(f"held-out tasks: constructs better on {(c > f).sum()} of {len(c)}")
        fig.tight_layout(); fig.savefig(out / "fig3_tasks.png", dpi=140); plt.close(fig)
        md += ["\n## Held-out tasks, largest differences (constructs minus families)\n", "| task | constructs | families |", "|---|---|---|"]
        md += [f"| {lab.get(t, t)} | {cc:.3f} | {ff:.3f} |" for t, cc, ff in sorted(pairs, key=lambda p: -(p[1] - p[2]))[:8]]
        md += [f"| {lab.get(t, t)} | {cc:.3f} | {ff:.3f} |" for t, cc, ff in sorted(pairs, key=lambda p: (p[1] - p[2]))[:8]]
    # 4. maps panel
    M = json.load(open(f"maps_all/K{kmaps}/manifest.json"))
    byl = {m["label"].lower(): m for m in M["constructs"]}
    panel = [byl[x] for x in PANEL if x in byl][:12]
    import nibabel as nib
    from nilearn import plotting
    fig = plt.figure(figsize=(16, 3.0 * ((len(panel) + 2) // 3)))
    for n, m in enumerate(panel):
        ax = fig.add_subplot((len(panel) + 2) // 3, 3, n + 1)
        p = nib.load(f"maps_all/K{kmaps}/{m['files']['p']}"); e = nib.load(f"maps_all/K{kmaps}/{m['files']['enrichment']}")
        pv, ev = p.get_fdata(), e.get_fdata()
        show = np.where(pv >= 0.2 * pv.max(), np.clip(ev, 0, None), 0)
        plotting.plot_glass_brain(nib.Nifti1Image(show, e.affine), threshold=0.5, vmin=0, vmax=3, cmap="hot",
                                  plot_abs=False, colorbar=False, axes=ax, display_mode="lzr",
                                  title=f"{m['label']} ({m['contrasts']} contrasts)")
    fig.suptitle(f"log2 enrichment over the average task, where P >= 20% of max (K = {kmaps})")
    fig.savefig(out / "fig4_maps.png", dpi=110); plt.close(fig)
    md += [f"\n## Maps (K = {kmaps})\n", f"{len(M['constructs'])} constructs with data, "
           f"{sum(m['cogat_id'] is not None for m in M['constructs'])} linked to Cognitive Atlas.\n",
           "| construct | Cognitive Atlas | contrasts | papers | top networks |", "|---|---|---|---|---|"]
    md += [f"| {m['label']} | {m['cogat_name'] or 'new'} | {m['contrasts']} | {m['papers']} | "
           + "; ".join(f"{x['name']} {x['weight']:.2f}" for x in m["networks"][:3]) + " |" for m in panel]
    (out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
