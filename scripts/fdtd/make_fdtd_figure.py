"""Revised FDTD figure (main text): same-structure fidelity of the two scalar forward models against
FDTD for every simulated five-layer structure, the paired design comparison, and one qualitative example.

    python scripts/fdtd/make_fdtd_figure.py --exp_root experiments --out_dir results

Reads the analysis_*.json files written by analyze_fdtd.py and the per-run .npz files.
"""
import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--exp_root", default="experiments")
p.add_argument("--out_dir", default="results")
p.add_argument("--example", default="multilayer_cls_n1.2:designs_cls_n1.2:3", help="run_dir:design_dir:sample for the qualitative panel")
args = p.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

STUDIES = [("multilayer_cls_n1.5", "MNIST cls.\n$n=1.5$ ($T=2\\lambda$)", "classification"),
           ("multilayer_cls_n1.2", "MNIST cls.\n$n=1.2$ ($T=5\\lambda$)", "classification"),
           ("multilayer_cls_n1.1", "MNIST cls.\n$n=1.1$ ($T=10\\lambda$)", "classification"),
           ("multilayer_img_n1.5", "F-MNIST img.\n$n=1.5$ ($T=2\\lambda$)", "imaging")]
COL = {"multilayer_cls_n1.5": "tab:blue", "multilayer_cls_n1.2": "tab:orange", "multilayer_cls_n1.1": "tab:red", "multilayer_img_n1.5": "tab:green"}

ana = {}
for run, _, task in STUDIES:
    f = os.path.join(args.exp_root, "fdtd", run, f"analysis_{task}_mesh10.json")
    if os.path.exists(f):
        ana[run] = json.load(open(f))
if not ana:
    raise SystemExit("no analysis files found; run analyze_fdtd.py first")

fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.38)

# (A) per-structure scatter: thin-model NMSE vs dBPM-model NMSE
ax = fig.add_subplot(gs[0, 0:2])
lim = 0.0
for run, label, task in STUDIES:
    if run not in ana:
        continue
    per = ana[run]["per_sample"]
    for design, mk in (("thin", "s"), ("bpm", "o")):
        xs = [v["fid_thin_model"]["field_nmse"] for v in per[design].values()]
        ys = [v["fid_bpm_model"]["field_nmse"] for v in per[design].values()]
        lim = max(lim, max(xs + ys))
        ax.scatter(xs, ys, s=22, marker=mk, color=COL[run], alpha=0.75, lw=0.5, edgecolor="k",
                   label=f"{label.replace(chr(10), ' ')}, {'thin' if design == 'thin' else 'dBPM'}-trained" )
lim *= 1.08
ax.plot([0, lim], [0, lim], "k--", lw=1)
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("thin-layer prediction vs FDTD: field NMSE")
ax.set_ylabel("dBPM prediction vs FDTD: field NMSE")
ax.set_title("(A) same structure, same input: which scalar model predicts FDTD?", fontsize=10, loc="left")
ax.legend(fontsize=6.5, ncol=2, loc="upper left", frameon=False)
ax.text(0.97, 0.05, "below the diagonal: dBPM closer to FDTD", transform=ax.transAxes, ha="right", fontsize=8)
ax.grid(alpha=0.3)

# (B) bars: mean field NMSE per study and model
ax = fig.add_subplot(gs[0, 2:4])
xt, xl = [], []
k = 0
for run, label, task in STUDIES:
    if run not in ana:
        continue
    fid = ana[run]["fidelity"]
    for design in ("thin", "bpm"):
        t = fid[f"{design}_design__fid_thin_model"]["field_nmse"]; b = fid[f"{design}_design__fid_bpm_model"]["field_nmse"]
        ax.bar(k - 0.2, t["mean"], 0.38, yerr=t["std"], color="tab:red", alpha=0.75, capsize=2, label="thin-layer model" if k == 0 else None)
        ax.bar(k + 0.2, b["mean"], 0.38, yerr=b["std"], color="tab:blue", alpha=0.75, capsize=2, label="dBPM model" if k == 0 else None)
        pt = fid[f"{design}_design__paired_field_nmse_thin_vs_bpm"]  # x = thin model, y = dBPM model
        n_closer = pt.get("n_y_smaller", pt.get("n_x_better"))
        ax.text(k, max(t["mean"] + t["std"], b["mean"] + b["std"]) + 0.01, f"{int(n_closer)}/{pt['n']}",
                ha="center", fontsize=7)
        short = label.split("\n")[1].split(" (")[0] + (" img" if task == "imaging" else "")
        xt.append(k); xl.append(f"{short}\n{'thin' if design == 'thin' else 'dBPM'}-tr.")
        k += 1
ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=6.5)
ax.set_ylabel("field NMSE vs FDTD (mean $\\pm$ s.d., 20 structures)")
ax.set_title("(B) forward-model fidelity per study (label: structures with dBPM closer)", fontsize=10, loc="left")
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3, axis="y")

# (C) paired design comparison (classification accuracies with bootstrap CI of the difference)
ax = fig.add_subplot(gs[1, 0:2])
rows = [(run, label) for run, label, task in STUDIES if run in ana and task == "classification"]
for i, (run, label) in enumerate(rows):
    a = ana[run]["accuracy"]
    acc_t, acc_b = a["thin_design"] * 100, a["bpm_design"] * 100
    ax.bar(i - 0.2, acc_t, 0.38, color="tab:red", alpha=0.75, label="thin-trained design" if i == 0 else None)
    ax.bar(i + 0.2, acc_b, 0.38, color="tab:blue", alpha=0.75, label="dBPM-trained design" if i == 0 else None)
    mc = a["mcnemar"]; ci = a["diff_ci"]
    ax.text(i, max(acc_t, acc_b) + 4, f"McNemar p = {mc['p_value_exact']:.2g}\n$\\Delta$acc 95% CI [{ci['ci_low']:+.2f}, {ci['ci_high']:+.2f}]",
            ha="center", fontsize=7)
ax.set_xticks(range(len(rows))); ax.set_xticklabels([l for _, l in rows], fontsize=7)
ax.set_ylim(0, 118); ax.set_ylabel("FDTD accuracy (%) on 20 paired samples")
ax.set_title("(C) paired design comparison in FDTD (same 20 samples through both designs)", fontsize=10, loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper right")
ax.grid(alpha=0.3, axis="y")

# (D) qualitative example: FDTD vs thin vs dBPM prediction for one structure
run, ddir, s = args.example.split(":"); s = int(s)
res = json.load(open(os.path.join(args.exp_root, "fdtd", run, "results.json")))
key = [k for k, r in res.items() if r.get("done") and r["sample"] == s and r["design"] == "thin" and r["mesh"] == 10]
if key:
    z = np.load(res[key[0]]["file"])
    ims = [("FDTD", z["I"]), ("thin-layer prediction", z["I_thinmodel"]), ("dBPM prediction", z["I_bpm_finemodel"])]
    vmax = max(im.max() / im.sum() for _, im in ims)
    # three small panels inside the right half
    sub = gs[1, 2:4].subgridspec(1, 3, wspace=0.08)
    from dbpm_d2nn.metrics import nmse_field
    for j, (name, im) in enumerate(ims):
        axi = fig.add_subplot(sub[0, j])
        axi.imshow(im / im.sum(), cmap="inferno", vmin=0, vmax=vmax)
        axi.set_xticks([]); axi.set_yticks([])
        ttl = name
        if j == 1:
            ttl += f"\nNMSE {nmse_field(z['U_thinmodel'], z['Ey']):.2f}"
        if j == 2:
            ttl += f"\nNMSE {nmse_field(z['U_bpm_finemodel'], z['Ey']):.2f}"
        axi.set_title(ttl, fontsize=8)
        if j == 0:
            axi.set_ylabel(f"(D) thin-trained structure,\n{run.split('_')[1]} $n$={run.split('_n')[1]}, sample {s}", fontsize=8)

fig.savefig(os.path.join(args.out_dir, "fig5_fdtd_fidelity_revised.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(args.out_dir, "fig5_fdtd_fidelity_revised.png"), dpi=200, bbox_inches="tight")
print("wrote fig5_fdtd_fidelity_revised")
