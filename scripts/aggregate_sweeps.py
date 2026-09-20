#!/usr/bin/env python
"""Aggregate the classification / imaging sweeps (results.json per configuration)
into seed-averaged tables (CSV + Markdown) and the revised figures.

Conditions (all volumetric numbers at the converged discretisation: dz <= lambda/12 for classification,
N_sub >= 120 and dz <= lambda/12 for imaging):
  thin->thin, thin->vol(entrance, submitted convention), thin->vol(centroid),
  thin->vol + delta_z, thin->vol + s, thin->vol + s + delta_z, thin + k-epoch dBPM
  fine-tuning -> vol, dBPM->vol.
"""
import argparse, glob, json, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--exp_root", default="experiments")
p.add_argument("--out_dir", default="results")
p.add_argument("--no_figs", action="store_true")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

CLS_CONDS = [("thin_thin", "Thin train -> thin test"),
             ("thin_vol_entrance", "Thin -> vol., entrance-anchored (submitted)"),
             ("thin_vol_centroid", "Thin -> vol., centroid-anchored"),
             ("fit_dz", "Thin -> vol., centroid + fitted delta_z"),
             ("fit_s", "Thin -> vol., centroid + fitted s"),
             ("fit_joint", "Thin -> vol., centroid + fitted s and delta_z"),
             ("finetune", "Thin + dBPM fine-tuning (5 ep. cls / 10 ep. img) -> vol."),
             ("bpm_vol", "dBPM train -> vol. test")]


def cls_value(rec, cond):
    """Test accuracy (%) of a condition at the converged discretisation."""
    if cond == "thin_thin":
        return rec["acc_thin_thin"]
    if cond in ("thin_vol_entrance", "thin_vol_centroid", "thin_vol_exit", "thin_vol_entrance_anchored", "bpm_vol"):
        return rec.get(cond, {}).get("fine")
    if cond in ("fit_dz", "fit_s", "fit_joint"):
        return rec.get(cond, {}).get("test_fine")
    if cond == "finetune":
        return rec.get("finetune", {}).get("final_fine")
    return None


def img_value(rec, cond, metric):
    if cond == "thin_thin":
        return rec["thin_thin"][metric]
    if cond in ("thin_vol_entrance", "thin_vol_centroid", "thin_vol_exit", "thin_vol_entrance_anchored", "bpm_vol"):
        v = rec.get(cond, {}).get("fine"); return None if v is None else v[metric]
    if cond in ("fit_dz", "fit_s", "fit_joint"):
        v = rec.get(cond, {}).get("test_fine"); return None if v is None else v[metric]
    if cond == "finetune":
        v = rec.get("finetune", {}).get("final_fine"); return None if v is None else v[metric]
    return None


def ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    if v.size == 0:
        return (np.nan, np.nan, 0)
    return (float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size))


summary = {"classification": {}, "imaging": {}}
# ------------------------------------------------------------------ classification
for path in sorted(glob.glob(os.path.join(args.exp_root, "cls", "*", "results.json"))):
    cfg = os.path.basename(os.path.dirname(path))
    recs = json.load(open(path))
    ns = sorted({r["n"] for r in recs})
    table = {}
    for n in ns:
        rn = [r for r in recs if r["n"] == n]
        row = {"seeds": sorted(r["seed"] for r in rn), "h_max_over_lambda": rn[0]["h_max_over_lambda"],
               "train_n_sub": rn[0].get("train_n_sub"), "eval_n_sub": rn[0]["eval_n_sub"]}
        for cond, _ in CLS_CONDS:
            row[cond] = ms([cls_value(r, cond) for r in rn])
        row["thin_vol_exit"] = ms([cls_value(r, "thin_vol_exit") for r in rn])
        row["thin_vol_entrance_anchored"] = ms([cls_value(r, "thin_vol_entrance_anchored") for r in rn])
        row["s_star"] = ms([r.get("fit_s", {}).get("height_scale") for r in rn])
        row["dz_star_over_lambda"] = ms([r.get("fit_dz", {}).get("axial_shift_over_lambda") for r in rn])
        row["joint_s"] = ms([r.get("fit_joint", {}).get("height_scale") for r in rn])
        row["joint_dz_over_lambda"] = ms([r.get("fit_joint", {}).get("axial_shift_over_lambda") for r in rn])
        ft = [r["finetune"]["test_acc"] for r in rn if "finetune" in r]
        row["finetune_curve"] = [ms([f[k] for f in ft if len(f) > k]) for k in range(max(len(f) for f in ft))] if ft else []
        row["bpm_thin"] = ms([r.get("bpm_thin") for r in rn])
        for fk in ("fab_thin", "fab_bpm"):
            if fk in rn[0]:
                keys = rn[0][fk][0].keys()
                row[fk] = {k: ms([np.mean([lay[k] for lay in r[fk]]) for r in rn if fk in r]) for k in keys}
        table[f"{n:.2f}"] = row
    summary["classification"][cfg] = table

# ------------------------------------------------------------------ imaging
for path in sorted(glob.glob(os.path.join(args.exp_root, "img", "*", "results.json"))):
    cfg = os.path.basename(os.path.dirname(path))
    recs = json.load(open(path))
    ns = sorted({r["n"] for r in recs})
    table = {}
    for n in ns:
        rn = [r for r in recs if r["n"] == n]
        row = {"seeds": sorted(r["seed"] for r in rn), "h_max_over_lambda": rn[0]["h_max_over_lambda"],
               "train_n_sub": rn[0].get("train_n_sub"), "eval_n_sub": rn[0]["eval_n_sub"]}
        for metric in ("mse", "ssim", "ssim_crop", "mse_crop", "pearson", "mse_scaled", "ssim_scaled"):
            for cond, _ in CLS_CONDS:
                row[f"{cond}__{metric}"] = ms([img_value(r, cond, metric) for r in rn])
        row["s_star"] = ms([r.get("fit_s", {}).get("height_scale") for r in rn])
        row["dz_star_over_lambda"] = ms([r.get("fit_dz", {}).get("axial_shift_over_lambda") for r in rn])
        ft = [[e["ssim"] for e in r["finetune"]["test"]] for r in rn if "finetune" in r]
        row["finetune_curve_ssim"] = [ms([f[k] for f in ft if len(f) > k]) for k in range(max(len(f) for f in ft))] if ft else []
        for fk in ("fab_thin", "fab_bpm"):
            if fk in rn[0]:
                keys = rn[0][fk][0].keys()
                row[fk] = {k: ms([np.mean([lay[k] for lay in r[fk]]) for r in rn if fk in r]) for k in keys}
        table[f"{n:.2f}"] = row
    summary["imaging"][cfg] = table

json.dump(summary, open(os.path.join(args.out_dir, "sweep_summary.json"), "w"), indent=1)

# ------------------------------------------------------------------ markdown tables
md = ["# Seed-averaged sweep results (mean +- sd over seeds; volumetric numbers at the converged discretisation: "
      "dz <= lambda/12 for classification, N_sub >= 120 and dz <= lambda/12 for imaging)", ""]
for cfg, table in summary["classification"].items():
    md += [f"## Classification: {cfg}", "", "| n | h_max/lambda | seeds | " + " | ".join(l for _, l in CLS_CONDS) + " | s* | dz*/lambda |",
           "|---|---|---|" + "---|" * (len(CLS_CONDS) + 2)]
    for n, row in table.items():
        cells = [f"{row[c][0]:.2f} +- {row[c][1]:.2f}" if row[c][2] else "-" for c, _ in CLS_CONDS]
        md.append(f"| {n} | {row['h_max_over_lambda']:.2f} | {len(row['seeds'])} | " + " | ".join(cells) +
                  f" | {row['s_star'][0]:.3f} +- {row['s_star'][1]:.3f} | {row['dz_star_over_lambda'][0]:+.3f} +- {row['dz_star_over_lambda'][1]:.3f} |")
    md.append("")
for cfg, table in summary["imaging"].items():
    for metric in ("ssim", "mse"):
        md += [f"## Imaging ({metric.upper()}): {cfg}", "", "| n | h_max/lambda | seeds | " + " | ".join(l for _, l in CLS_CONDS) + " | s* |",
               "|---|---|---|" + "---|" * (len(CLS_CONDS) + 1)]
        for n, row in table.items():
            fmt = "{:.3f}" if metric == "ssim" else "{:.5f}"
            cells = [(fmt.format(row[f"{c}__{metric}"][0]) + " +- " + fmt.format(row[f"{c}__{metric}"][1])) if row[f"{c}__{metric}"][2] else "-" for c, _ in CLS_CONDS]
            md.append(f"| {n} | {row['h_max_over_lambda']:.2f} | {len(row['seeds'])} | " + " | ".join(cells) + f" | {row['s_star'][0]:.3f} |")
        md.append("")
open(os.path.join(args.out_dir, "sweep_summary.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))

if args.no_figs:
    raise SystemExit
# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"thin_thin": dict(color="k", ls="--", marker="o", label="Thin train → thin test"),
         "thin_vol_entrance": dict(color="tab:red", ls=":", marker="v", label="Thin → volumetric (entrance-anchored, submitted)"),
         "thin_vol_centroid": dict(color="tab:orange", ls="-", marker="s", label="Thin → volumetric (centroid-anchored)"),
         "fit_joint": dict(color="tab:purple", ls="-", marker="D", label="Thin → volumetric + fitted global $s$, $\\delta z$"),
         "finetune": dict(color="tab:green", ls="-", marker="^", label="Thin + few-epoch dBPM fine-tuning → volumetric (5 cls / 10 img)"),
         "bpm_vol": dict(color="tab:blue", ls="-", marker="o", label="dBPM train → volumetric test")}
ORDER = ["thin_thin", "thin_vol_entrance", "thin_vol_centroid", "fit_joint", "finetune", "bpm_vol"]
TITLES = {"mnist_amp": "MNIST, amplitude", "mnist_phase": "MNIST, phase", "fmnist_amp": "Fashion-MNIST, amplitude",
          "fmnist_phase": "Fashion-MNIST, phase", "mnist": "MNIST", "fmnist": "Fashion-MNIST", "cifar100": "CIFAR-100"}


def plot_panel(ax, table, getter, ylabel, title):
    ns = sorted(float(k) for k in table)
    for cond in ORDER:
        m = np.array([getter(table[f"{n:.2f}"], cond)[0] for n in ns], float)
        s = np.array([getter(table[f"{n:.2f}"], cond)[1] for n in ns], float)
        if np.all(np.isnan(m)):
            continue
        st = dict(STYLE[cond]); ax.plot(ns, m, ms=4, lw=1.4, **st)
        ax.fill_between(ns, m - s, m + s, color=st["color"], alpha=0.15, lw=0)
    ax.axvspan(1.05, 1.2, color="tab:blue", alpha=0.07, lw=0)  # h_max >= 5 lambda (low index contrast)
    ax.text(1.125, 0.985, "$h_{max}\\geq5\\lambda$", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7, color="tab:blue")
    ax.set_xlabel("refractive index $n$"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


if summary["classification"]:
    cfgs = [c for c in ("mnist_amp", "mnist_phase", "fmnist_amp", "fmnist_phase") if c in summary["classification"]]
    fig, axes = plt.subplots(1, len(cfgs), figsize=(4.2 * len(cfgs), 3.6), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        plot_panel(ax, summary["classification"][cfg], lambda row, c: row[c], "test accuracy (%)", TITLES.get(cfg, cfg))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(3, len(cfgs)), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(os.path.join(args.out_dir, "fig2_classification_revised.pdf")); fig.savefig(os.path.join(args.out_dir, "fig2_classification_revised.png"), dpi=200)
if summary["imaging"]:
    cfgs = [c for c in ("mnist", "fmnist", "cifar100") if c in summary["imaging"]]
    fig, axes = plt.subplots(2, len(cfgs), figsize=(4.2 * len(cfgs), 6.6), squeeze=False)
    for j, cfg in enumerate(cfgs):
        plot_panel(axes[0][j], summary["imaging"][cfg], lambda row, c: row[f"{c}__mse"], "MSE", TITLES.get(cfg, cfg))
        plot_panel(axes[1][j], summary["imaging"][cfg], lambda row, c: row[f"{c}__ssim"], "SSIM", TITLES.get(cfg, cfg))
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(3, len(cfgs)), fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(args.out_dir, "fig3_imaging_revised.pdf")); fig.savefig(os.path.join(args.out_dir, "fig3_imaging_revised.png"), dpi=200)
print("figures written to", args.out_dir)
