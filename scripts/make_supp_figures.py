#!/usr/bin/env python
"""Supplementary / response-letter figures that do not depend on the seed sweeps:

  S1  axial coordinate diagram (entrance / centroid / exit conventions, delta_z)
  S2  dBPM discretisation convergence versus N_sub (paper seed-0 height maps)
  S3  FDTD mesh convergence (lambda/10, lambda/20, lambda/30) and single-layer
      same-structure fidelity (thin vs dBPM prediction against FDTD)
  S4  dBPM training-N_sub convergence (if available)
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

p = argparse.ArgumentParser()
p.add_argument("--exp_root", default="experiments")
p.add_argument("--out_dir", default="results")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(args.out_dir, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(args.out_dir, name + ".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------- S1 axial diagram
def axial_diagram():
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5.6), sharex=True)
    z_l, z_next, hmax = 0.0, 10.0, 2.4          # in arbitrary units (h_max exaggerated for clarity)
    conventions = [("(a) thin-layer training: zero-thickness phase mask at the nominal plane $z_\\ell$", None),
                   ("(b) relief entrance at $z_\\ell$ (thin mask sits $h_{\\max}/2$ before the volume centroid)", 0.0),
                   ("(c) current convention: relief centroid at $z_\\ell$ (+ optional global offset $\\delta z$, fitted on validation data)", 0.5)]
    rng = np.random.default_rng(0)
    prof = rng.uniform(0.15, 1.0, 14)
    for ax, (title, anchor) in zip(axes, conventions):
        ax.set_xlim(-2.5, 12.5); ax.set_ylim(-2.0, 1.9); ax.axis("off")
        ax.plot([-2.5, 12.5], [0, 0], color="0.6", lw=0.8, ls=":")
        ax.text(-2.4, 1.55, title, fontsize=8.5, ha="left")
        for zp, lab in ((z_l, "$z_\\ell$"), (z_next, "$z_{\\ell+1}$")):
            ax.plot([zp, zp], [-1.7, 1.25], color="k", lw=1.0, ls="--")
            ax.text(zp, -1.95, lab, ha="center", fontsize=9)
        ax.annotate("", xy=(z_next, 1.05), xytext=(z_l, 1.05), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text((z_l + z_next) / 2, 1.12, "$z_{\\mathrm{diff}}$ (unchanged)", ha="center", fontsize=8)
        if anchor is None:
            ax.plot([z_l, z_l], [-1.0, 1.0], color="tab:red", lw=3)
            ax.text(z_l + 0.25, -0.9, "$t_\\ell=e^{i\\phi_\\ell}$", color="tab:red", fontsize=9)
        else:
            z0 = z_l - anchor * hmax
            ax.add_patch(Rectangle((z0, -1.0), hmax, 2.0, facecolor="tab:blue", alpha=0.12, edgecolor="none"))
            ys = np.linspace(-1.0, 1.0, len(prof) + 1)
            for y0, y1, hr in zip(ys[:-1], ys[1:], prof):
                ax.add_patch(Rectangle((z0, y0), hr * hmax, y1 - y0, facecolor="tab:blue", alpha=0.65, edgecolor="k", lw=0.3))
            ax.plot([z0, z0], [-1.0, 1.0], color="k", lw=0.8)
            ax.plot([z0 + hmax, z0 + hmax], [-1.0, 1.0], color="k", lw=0.8, ls=":")
            ax.plot([z0 + hmax / 2] * 2, [-1.0, 1.0], color="tab:orange", lw=1.6)
            ax.annotate("", xy=(z0 + hmax, -1.2), xytext=(z0, -1.2), arrowprops=dict(arrowstyle="<->", lw=0.8))
            ax.text(z0 + hmax + 0.15, -1.3, "$h_{\\max}=\\lambda/\\Delta n$", ha="left", fontsize=8)
            ax.text(z0 - 0.1, 0.55, "entrance", ha="right", fontsize=7.5)
            ax.text(z0 + hmax / 2 + 0.12, 1.02, "centroid", ha="left", va="bottom", fontsize=7.5, color="tab:orange")
            ax.text(z0 + hmax + 0.1, 0.55, "exit", ha="left", fontsize=7.5)
            ax.annotate("", xy=(z_next - (anchor * hmax if anchor else 0), 0.45), xytext=(z0 + hmax, 0.45), arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.3"))
            ax.text((z0 + hmax + z_next) / 2, 0.52, "free space to next relief: $z_{\\mathrm{diff}}-h_{\\max}$", ha="center", fontsize=7.5, color="0.3")
            if anchor == 0.5:   # ghost of the next layer's relief (entrance at z_{l+1} - h_max/2)
                ax.add_patch(Rectangle((z_next - hmax / 2, -1.0), hmax / 2, 2.0, facecolor="tab:blue", alpha=0.10, edgecolor="k", lw=0.5, ls=":"))
            if anchor == 0.5:
                ax.annotate("", xy=(z_l + 1.1, -0.55), xytext=(z_l, -0.55), arrowprops=dict(arrowstyle="->", lw=1.0, color="tab:green"))
                ax.text(z_l + 1.25, -0.62, "$\\delta z$ (one scalar for all layers)", fontsize=7.5, color="tab:green")
            if anchor == 0.0:
                ax.annotate("", xy=(z_l + hmax / 2, -0.55), xytext=(z_l, -0.55), arrowprops=dict(arrowstyle="->", lw=1.0, color="tab:red"))
                ax.text(z_l + hmax / 2 + 0.15, -0.62, "implicit defocus $h_{\\max}/2$", fontsize=7.5, color="tab:red")
    fig.suptitle("Axial coordinate conventions for a volumetric layer (light propagates left to right)", fontsize=10)
    save(fig, "figS1_axial_coordinates")


# ----------------------------------------------------------------------------- S2 N_sub convergence
def nsub_convergence():
    files = sorted(f for f in os.listdir(os.path.join(args.exp_root, "nsub_convergence")) if f.endswith(".json"))
    if not files:
        return
    d = json.load(open(os.path.join(args.exp_root, "nsub_convergence", files[0])))
    res = d["results"]
    ns = sorted({r["n"] for r in res})
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(ns):
        for kind, ls, mk in (("thin", "--", "s"), ("bpm", "-", "o")):
            rr = sorted([r for r in res if r["n"] == n and r["kind"] == kind], key=lambda r: r["n_sub"])
            if not rr:
                continue
            x = [r["n_sub"] for r in rr]
            col = cmap(i / max(1, len(ns) - 1))
            lab = f"n={n:.1f} ({'thin' if kind == 'thin' else 'dBPM'}-trained map)"
            axes[0].loglog(x, [max(r["field_nmse_vs_ref"], 1e-6) for r in rr], ls=ls, marker=mk, ms=4, color=col, label=lab)
            axes[1].semilogx(x, [r["acc"] - r["acc_ref"] for r in rr], ls=ls, marker=mk, ms=4, color=col)
            axes[2].semilogx(x, [100 * r["top1_agreement_vs_ref"] for r in rr], ls=ls, marker=mk, ms=4, color=col)
    xx = np.array([15, 240]); axes[0].loglog(xx, 0.12 * (15 / xx) ** 2, color="0.5", lw=1, ls=":"); axes[0].text(60, 0.03, "$\\propto N_{\\rm sub}^{-2}$", color="0.4", fontsize=8)
    axes[0].set_ylabel("output-field NMSE vs. $N_{\\rm sub}=480$ reference"); axes[0].set_xlabel("$N_{\\rm sub}$ (slices per $h_{\\max}$)")
    axes[1].set_ylabel("accuracy $-$ accuracy at $N_{\\rm sub}=480$ (points)"); axes[1].set_xlabel("$N_{\\rm sub}$"); axes[1].axhline(0, color="0.5", lw=0.8)
    axes[2].set_ylabel("top-1 agreement with reference (%)"); axes[2].set_xlabel("$N_{\\rm sub}$")
    for ax in axes:
        ax.grid(alpha=0.3, which="both"); ax.axvline(15, color="tab:red", lw=0.8, ls=":")
        ax.set_xticks([15, 30, 60, 120, 240]); ax.set_xticklabels(["15", "30", "60", "120", "240"]); ax.minorticks_off()
    axes[0].text(15.5, 2e-4, "submitted\n$N_{\\rm sub}=15$", color="tab:red", fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"dBPM discretisation convergence ({d['args']['dataset'].upper()}, {d['args']['encoding']} encoding, submitted seed-0 height maps, {d['args']['axial_mode']} convention)", fontsize=9)
    fig.tight_layout()
    save(fig, "figS2_nsub_convergence")
    # table
    rows = ["| n | h_max/lambda | map | N_sub=15 | 30 | 60 | 120 | 240 | acc(15) - acc(480) | acc(30) - acc(480) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for n in ns:
        for kind in ("thin", "bpm"):
            rr = {r["n_sub"]: r for r in res if r["n"] == n and r["kind"] == kind}
            if not rr:
                continue
            cells = [f"{rr[k]['field_nmse_vs_ref']:.1e}" if k in rr else "-" for k in (15, 30, 60, 120, 240)]
            rows.append(f"| {n:.1f} | {1 / (n - 1):.2f} | {kind} | " + " | ".join(cells) +
                        f" | {rr[15]['acc'] - rr[15]['acc_ref']:+.2f} | {rr[30]['acc'] - rr[30]['acc_ref']:+.2f} |" if 15 in rr and 30 in rr else "")
    open(os.path.join(args.out_dir, "tableS_nsub_convergence.md"), "w").write(
        "Output-field NMSE of the dBPM forward model at N_sub slices relative to N_sub=480, and accuracy differences.\n\n" + "\n".join(rows) + "\n")


# ----------------------------------------------------------------------------- S3 FDTD mesh + single layer fidelity
def fdtd_single_layer():
    runs = [("single_layer", "n = 1.5 ($T=2\\lambda$)"), ("single_layer_n1.2", "n = 1.2 ($T=5\\lambda$)")]
    data = []
    for sub, lab in runs:
        f = os.path.join(args.exp_root, "fdtd", sub, "results.json")
        if os.path.exists(f):
            data.append((lab, json.load(open(f))))
    if not data:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    # panel A: mesh convergence (relative change of detector intensity), panel B: fidelity vs mesh
    for lab, d in data:
        for kind, mk in (("thin", "s"), ("bpm", "o")):
            r = d.get(kind)
            if r is None:
                continue
            conv = r.get("mesh_convergence", {})
            xs, ys = [], []
            for key, x in (("10_vs_20", 20), ("20_vs_30", 30)):
                if key in conv:
                    xs.append(x); ys.append(100 * conv[key]["rel_l2_intensity"])
            if xs:
                axes[0].plot(xs, ys, marker=mk, label=f"{lab}, {kind}-trained map")
            meshes = [m for m in (10, 20, 30) if f"mesh{m}" in r and "thin_vs_fdtd" in r[f"mesh{m}"]]
            axes[1].plot(meshes, [r[f"mesh{m}"]["thin_vs_fdtd"]["field_nmse"] for m in meshes], marker=mk, ls="--", color="tab:red", alpha=0.8)
            axes[1].plot(meshes, [r[f"mesh{m}"]["bpm_vs_fdtd"]["field_nmse"] for m in meshes], marker=mk, ls="-", color="tab:blue", alpha=0.8)
    axes[0].set_xlabel("finer mesh of the pair (cells per $\\lambda$ in each medium)"); axes[0].set_ylabel("rel. $L_2$ change of detector intensity (%)")
    axes[0].set_xticks([20, 30]); axes[0].set_xticklabels(["$\\lambda/10 \\to \\lambda/20$", "$\\lambda/20 \\to \\lambda/30$"]); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=7)
    axes[0].set_title("FDTD mesh convergence (single layer)", fontsize=9)
    axes[1].set_xlabel("FDTD mesh (cells per $\\lambda$)"); axes[1].set_ylabel("field NMSE, model vs. FDTD"); axes[1].grid(alpha=0.3)
    axes[1].plot([], [], color="tab:red", ls="--", label="thin-layer prediction vs FDTD"); axes[1].plot([], [], color="tab:blue", ls="-", label="dBPM prediction vs FDTD")
    axes[1].plot([], [], color="k", marker="s", ls="", label="thin-trained map"); axes[1].plot([], [], color="k", marker="o", ls="", label="dBPM-trained map")
    axes[1].legend(fontsize=7); axes[1].set_title("Same-structure fidelity (single layer)", fontsize=9)
    fig.tight_layout(); save(fig, "figS3_fdtd_single_layer")
    rows = ["| case | height map | mesh | thin–FDTD field NMSE | dBPM–FDTD field NMSE | thin–FDTD int. NMSE | dBPM–FDTD int. NMSE | thin–FDTD Pearson | dBPM–FDTD Pearson | thin–FDTD SSIM | dBPM–FDTD SSIM |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for lab, d in data:
        for kind in ("thin", "bpm"):
            r = d.get(kind)
            if r is None:
                continue
            for m in (10, 20, 30):
                x = r.get(f"mesh{m}")
                if x and "thin_vs_fdtd" in x:
                    t, b = x["thin_vs_fdtd"], x["bpm_vs_fdtd"]
                    rows.append(f"| {lab.replace('$', '')} | {kind}-trained | λ/{m} | {t['field_nmse']:.4f} | {b['field_nmse']:.4f} | {t['int_nmse']:.4f} | {b['int_nmse']:.4f} | {t['pearson']:.4f} | {b['pearson']:.4f} | {t['ssim']:.4f} | {b['ssim']:.4f} |")
    conv_rows = ["", "| case | height map | pair | rel. L2 change of detector intensity | field NMSE between meshes | Pearson |", "|---|---|---|---|---|---|"]
    for lab, d in data:
        for kind in ("thin", "bpm"):
            r = d.get(kind)
            for key in ("10_vs_20", "20_vs_30"):
                if r and key in r.get("mesh_convergence", {}):
                    c = r["mesh_convergence"][key]
                    conv_rows.append(f"| {lab.replace('$', '')} | {kind}-trained | λ/{key.replace('_vs_', ' vs λ/')} | {100 * c['rel_l2_intensity']:.2f}% | {c['field_nmse']:.2e} | {c['pearson']:.5f} |")
    open(os.path.join(args.out_dir, "tableS_fdtd_single_layer.md"), "w").write(
        "Single-layer same-structure fidelity: thin-layer and dBPM predictions of the detector field for the SAME exported height map, compared with FDTD (complex field, one global complex scale). Scalar models are evaluated on a zero-padded 128x128 window (open boundaries).\n\n" +
        "\n".join(rows + conv_rows) + "\n")


# ----------------------------------------------------------------------------- S4 training-N_sub convergence
def nsub_training():
    f = os.path.join(args.exp_root, "nsub_training", "results.json")
    if not os.path.exists(f):
        return
    res = json.load(open(f))
    if not res:
        return
    ns = sorted({r["n"] for r in res})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    for n in ns:
        rr = sorted([r for r in res if r["n"] == n], key=lambda r: r["train_n_sub"])
        axes[0].plot([r["train_n_sub"] for r in rr], [r["acc_own"] for r in rr], marker="o", ls="--", label=f"n={n:.1f}: evaluated at own $N_{{\\rm sub}}$")
        axes[0].plot([r["train_n_sub"] for r in rr], [r["acc_ref"] for r in rr], marker="s", ls="-", label=f"n={n:.1f}: evaluated at reference $N_{{\\rm sub}}$={rr[0]["ref_n_sub"]}")
        axes[1].semilogy([r["train_n_sub"] for r in rr], [max(r["field_nmse_own_vs_ref"], 1e-7) for r in rr], marker="o", label=f"n={n:.1f}")
    axes[0].set_xlabel("training $N_{\\rm sub}$"); axes[0].set_ylabel("test accuracy (%)"); axes[0].grid(alpha=0.3); axes[0].legend(fontsize=6.5)
    axes[1].set_xlabel("training $N_{\\rm sub}$"); axes[1].set_ylabel("field NMSE: own $N_{\\rm sub}$ vs reference"); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=7)
    fig.suptitle("dBPM trained at different $N_{\\rm sub}$ (MNIST, phase encoding)", fontsize=9)
    fig.tight_layout(); save(fig, "figS4_nsub_training")
    rows = ["| n | training N_sub | dz/lambda | acc at own N_sub | acc at reference N_sub | field NMSE own vs ref | train time (min) |", "|---|---|---|---|---|---|---|"]
    for r in sorted(res, key=lambda r: (r["n"], r["train_n_sub"])):
        rows.append(f"| {r['n']:.1f} | {r['train_n_sub']} | {r['dz_over_lambda']:.3f} | {r['acc_own']:.2f} | {r['acc_ref']:.2f} (ref N_sub={r["ref_n_sub"]}) | {r['field_nmse_own_vs_ref']:.2e} | {r['train_time_s'] / 60:.1f} |")
    # imaging: dBPM (from scratch and fine-tuned) trained at different slice thicknesses, evaluated at dz <= lambda/12
    img_rows = []
    for d in sorted(glob.glob(os.path.join(args.exp_root, "img_train_dz", "*", "*", "results.json"))) + \
             sorted(glob.glob(os.path.join(args.exp_root, "img_train_dz", "*", "results.json"))):
        for r in json.load(open(d)):
            if r.get("task") != "imaging" or "bpm_vol" not in r:
                continue
            ft = r.get("finetune", {}).get("final_fine", {})
            img_rows.append((r["n"], r["train_n_sub"], r["h_max_over_lambda"] / r["train_n_sub"], r["eval_n_sub"], r["dataset"], r["seed"],
                             r["bpm_vol"]["train"]["ssim"], r["bpm_vol"]["fine"]["ssim"], ft.get("ssim", float("nan"))))
    if img_rows:
        rows += ["", "Imaging (corrected geometry, seed 0): dBPM trained at different discretisations; SSIM on the test split "
                 "(experiments/img_train_dz/; protocol_dz3 = the submitted N_sub = 15 protocol, evaluated at dz <= lambda/12 only)",
                 "", "| dataset | n | training N_sub | training dz/lambda | eval N_sub | dBPM from scratch: SSIM at own N_sub | at eval N_sub | thin + 10 ep. fine-tuning at eval N_sub |",
                 "|---|---|---|---|---|---|---|---|"]
        for n, tn, dz, en, ds, sd, own, fine, ftv in sorted(set(img_rows)):
            rows.append(f"| {ds} | {n:.1f} | {tn} | {dz:.3f} | {en} | {own:.3f} | {fine:.3f} | {ftv:.3f} |")
    open(os.path.join(args.out_dir, "tableS_nsub_training.md"), "w").write("\n".join(rows) + "\n")


# ----------------------------------------------------------------------------- S5 fabrication statistics + slope penalty
def fabrication():
    import glob
    rows = ["| task | dataset/encoding | n | h_max (nm) | map | median |dh| (nm) | p95 |dh| (nm) | max |dh| (nm) | median slope | p95 slope | max slope | frac > 45 deg | frac > 80 deg | wrap edges (|dh| > 0.5 h_max) | AR_max = h_max/d | AR_p95 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in ("cls", "img"):
        for path in sorted(glob.glob(os.path.join(args.exp_root, task, "*", "results.json"))):
            cfg = os.path.basename(os.path.dirname(path))
            recs = json.load(open(path))
            for n in sorted({r["n"] for r in recs}):
                rn = [r for r in recs if r["n"] == n and "fab_thin" in r]
                if not rn:
                    continue
                for key, lab in (("fab_thin", "thin-trained"), ("fab_bpm", "dBPM-trained")):
                    agg = lambda k: np.mean([np.mean([lay[k] for lay in r[key]]) for r in rn])
                    mx = lambda k: np.max([np.max([lay[k] for lay in r[key]]) for r in rn])
                    rows.append(f"| {task} | {cfg} | {n:.1f} | {rn[0][key][0]['h_max_nm']:.0f} | {lab} | {agg('dh_median_nm'):.0f} | {agg('dh_p95_nm'):.0f} | {mx('dh_max_nm'):.0f} | "
                                f"{agg('slope_median_deg'):.1f} | {agg('slope_p95_deg'):.1f} | {mx('slope_max_deg'):.1f} | {100 * agg('frac_slope_gt_45deg'):.1f}% | {100 * agg('frac_slope_gt_80deg'):.2f}% | "
                                f"{100 * agg('wrap_edge_fraction'):.2f}% | {rn[0][key][0]['aspect_ratio_max']:.1f} | {agg('aspect_ratio_p95'):.2f} |")
    if len(rows) > 2:
        open(os.path.join(args.out_dir, "tableS_fabrication.md"), "w").write(
            "Fabrication-geometry statistics of the trained height maps (pixel pitch d = 0.535 lambda = 284.6 nm at 532 nm; nearest-neighbour differences |dh| pooled over x and y and over the five layers; mean over seeds; max = maximum over layers and seeds).\n\n" + "\n".join(rows) + "\n")
        print("wrote tableS_fabrication.md")
    f = os.path.join(args.exp_root, "slope_penalty", "results.json")
    if not os.path.exists(f):
        return
    res = sorted(json.load(open(f)), key=lambda r: r["tv_weight"])
    if len(res) < 2:
        return
    w = [r["tv_weight"] for r in res]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    axes[0].plot(w, [r["acc_ref"] for r in res], marker="o"); axes[0].set_ylabel("test accuracy (%) at $\\Delta z\\leq\\lambda/12$")
    axes[1].plot(w, [r["fab"]["slope_p95_deg"] for r in res], marker="o", label="95th percentile"); axes[1].plot(w, [r["fab"]["slope_max_deg"] for r in res], marker="s", label="maximum")
    axes[1].set_ylabel("local slope angle (deg)"); axes[1].legend(fontsize=8)
    axes[2].plot(w, [100 * r["fab"]["wrap_edge_fraction"] for r in res], marker="o", label="$|\\Delta h|>0.5h_{\\max}$"); axes[2].plot(w, [100 * r["fab"]["frac_slope_gt_45deg"] for r in res], marker="s", label="slope > 45°")
    axes[2].set_ylabel("fraction of adjacent pixel pairs (%)"); axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xscale("symlog", linthresh=0.1); ax.set_xlabel("total-variation weight $\\lambda_{TV}$"); ax.grid(alpha=0.3)
    fig.suptitle(f"Slope (total-variation) penalty during dBPM training (MNIST-{res[0]['phase_param']}, n = {res[0]['n']}, seed {res[0]['seed']})", fontsize=9)
    fig.tight_layout(); save(fig, "figS5_slope_penalty")
    rows = ["| lambda_TV | acc (own N_sub) | acc (dz<=lambda/12) | median |dh| (nm) | p95 |dh| (nm) | max |dh| (nm) | p95 slope (deg) | max slope (deg) | frac > 45 deg | wrap edges | AR_p95 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        fb = r["fab"]
        rows.append(f"| {r['tv_weight']} | {r['acc_own']:.2f} | {r['acc_ref']:.2f} | {fb['dh_median_nm']:.0f} | {fb['dh_p95_nm']:.0f} | {fb['dh_max_nm']:.0f} | {fb['slope_p95_deg']:.1f} | {fb['slope_max_deg']:.1f} | {100 * fb['frac_slope_gt_45deg']:.1f}% | {100 * fb['wrap_edge_fraction']:.2f}% | {fb['aspect_ratio_p95']:.2f} |")
    open(os.path.join(args.out_dir, "tableS_slope_penalty.md"), "w").write("\n".join(rows) + "\n")


axial_diagram()
nsub_convergence()
fdtd_single_layer()
nsub_training()
fabrication()
