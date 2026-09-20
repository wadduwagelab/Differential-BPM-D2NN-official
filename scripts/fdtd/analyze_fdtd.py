#!/usr/bin/env python
"""Analyse the paired FDTD runs produced by run_multilayer.py.

Outputs (JSON + Markdown):
  * task metrics per design (classification: accuracy, p_y, margin ratio, efficiency,
    CE; imaging: Pearson r, rel. L2, NRMSE vs target) with paired statistics
    (exact McNemar for accuracy, paired bootstrap CI + Wilcoxon / permutation for
    continuous metrics),
  * same-structure fidelity: thin-model-vs-FDTD and dBPM-model-vs-FDTD discrepancy
    (complex-field NMSE, intensity NMSE, Pearson r, SSIM, top-1 agreement) for every
    simulated structure.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dbpm_d2nn.metrics import nmse_field, nmse_intensity, pearson, ssim_np, rel_l2, nrmse
from dbpm_d2nn.stats import mcnemar_exact, paired_tests, accuracy_diff_ci, mean_std

p = argparse.ArgumentParser()
p.add_argument("--run_dir", required=True)
p.add_argument("--export", required=True)
p.add_argument("--task", required=True, choices=["classification", "imaging"])
p.add_argument("--mesh", type=int, default=10)
p.add_argument("--out", default=None)
args = p.parse_args()

d = np.load(args.export)
summ = json.load(open(os.path.join(os.path.dirname(args.export), "summary.json")))
res = json.load(open(os.path.join(args.run_dir, "results.json")))
labels = d["labels"]; x_in = d["x_in"]
EPS = 1e-12


def patch_energies(I, patches):
    return np.array([I[y0:y1, x0:x1].sum() for (y0, y1, x0, x1) in patches])


def cls_metrics(I, y, patches):
    E = patch_energies(I, patches)
    p_ = E / (E.sum() + EPS)
    pred = int(E.argmax())
    others = np.delete(E, y)
    return {"pred": pred, "correct": pred == y, "p_y": float(p_[y]), "margin_ratio": float(E[y] / (others.max() + EPS)),
            "eta_correct": float(E[y] / (I.sum() + EPS)), "ce": float(-np.log(p_[y] + EPS))}


def img_metrics(I, target):
    return {"pearson": pearson(I, target), "rel_l2": rel_l2(I, target), "nrmse": nrmse(I, target)}


def fidelity(I_model, U_model, I_f, Ey_f, patches=None):
    out = {"field_nmse": nmse_field(U_model, Ey_f), "int_nmse": nmse_intensity(I_model, I_f),
           "pearson": pearson(I_model, I_f), "ssim": ssim_np(I_model, I_f)}
    if patches is not None:
        out["top1_agree"] = bool(patch_energies(I_model, patches).argmax() == patch_energies(I_f, patches).argmax())
        Em, Ef = patch_energies(I_model, patches), patch_energies(I_f, patches)
        out["patch_cosine"] = float(np.dot(Em, Ef) / (np.linalg.norm(Em) * np.linalg.norm(Ef) + EPS))
    return out


patches = [tuple(pp) for pp in summ["meta"]["patches"]] if args.task == "classification" else None
per = {"thin": {}, "bpm": {}}
for key, r in res.items():
    if not r.get("done") or r["mesh"] != args.mesh:
        continue
    z = np.load(r["file"])
    s, design = r["sample"], r["design"]
    I_f, Ey_f = z["I"], z["Ey"]
    rec = {"sample": s, "label": int(labels[s]), "real_cost": r.get("real_cost")}
    if args.task == "classification":
        rec["fdtd"] = cls_metrics(I_f, int(labels[s]), patches)
        rec["model_thin"] = cls_metrics(z["I_thinmodel"], int(labels[s]), patches)
        rec["model_bpm"] = cls_metrics(z["I_bpm_finemodel"], int(labels[s]), patches)
    else:
        target = x_in[s] ** 2
        rec["fdtd"] = img_metrics(I_f, target)
        rec["model_thin"] = img_metrics(z["I_thinmodel"], target)
        rec["model_bpm"] = img_metrics(z["I_bpm_finemodel"], target)
    rec["fid_thin_model"] = fidelity(z["I_thinmodel"], z["U_thinmodel"], I_f, Ey_f, patches)
    rec["fid_bpm_model"] = fidelity(z["I_bpm_finemodel"], z["U_bpm_finemodel"], I_f, Ey_f, patches)
    rec["fid_bpm_train_model"] = fidelity(z["I_bpmmodel"], z["U_bpmmodel"], I_f, Ey_f, patches)
    per[design][s] = rec

common = sorted(set(per["thin"]) & set(per["bpm"]))
out = {"task": args.task, "mesh": args.mesh, "n_paired": len(common), "samples": common,
       "n_thin": len(per["thin"]), "n_bpm": len(per["bpm"])}
lines = [f"# FDTD analysis: {args.task} (mesh lambda/{args.mesh}, {len(common)} paired samples)", ""]

if args.task == "classification":
    ct = np.array([per["thin"][s]["fdtd"]["correct"] for s in common]); cb = np.array([per["bpm"][s]["fdtd"]["correct"] for s in common])
    out["accuracy"] = {"thin_design": float(ct.mean()), "bpm_design": float(cb.mean()),
                       "mcnemar": mcnemar_exact(ct, cb), "diff_ci": accuracy_diff_ci(ct, cb)}
    out["accuracy_scalar_models"] = {
        "thin_design_thinmodel": float(np.mean([per["thin"][s]["model_thin"]["correct"] for s in common])),
        "thin_design_bpmmodel": float(np.mean([per["thin"][s]["model_bpm"]["correct"] for s in common])),
        "bpm_design_bpmmodel": float(np.mean([per["bpm"][s]["model_bpm"]["correct"] for s in common])),
        "bpm_design_thinmodel": float(np.mean([per["bpm"][s]["model_thin"]["correct"] for s in common]))}
    lines += [f"FDTD accuracy: thin-trained {ct.mean()*100:.1f}% ({ct.sum()}/{len(ct)}), dBPM-trained {cb.mean()*100:.1f}% ({cb.sum()}/{len(cb)})",
              f"exact McNemar: {out['accuracy']['mcnemar']}", f"paired bootstrap CI of accuracy difference: {out['accuracy']['diff_ci']}", ""]
    out["metrics"] = {}
    lines += ["| metric | thin-trained (mean +- sd) | dBPM-trained (mean +- sd) | paired diff [95% CI] | Wilcoxon p | perm. p |", "|---|---|---|---|---|---|"]
    for m in ("p_y", "margin_ratio", "eta_correct", "ce"):
        a = [per["thin"][s]["fdtd"][m] for s in common]; b = [per["bpm"][s]["fdtd"][m] for s in common]
        t = paired_tests(a, b)
        out["metrics"][m] = {"thin": mean_std(a), "bpm": mean_std(b), "tests": t}
        lines.append(f"| {m} | {np.mean(a):.4f} +- {np.std(a, ddof=1):.4f} | {np.mean(b):.4f} +- {np.std(b, ddof=1):.4f} | "
                     f"{t['mean_diff']:+.4f} [{t['boot_ci_low']:+.4f}, {t['boot_ci_high']:+.4f}] | {t['wilcoxon_p']:.3g} | {t['perm_p']:.3g} |")
else:
    out["metrics"] = {}
    lines += ["| metric | thin-trained (mean +- sd) | dBPM-trained (mean +- sd) | paired diff [95% CI] | Wilcoxon p | perm. p |", "|---|---|---|---|---|---|"]
    for m in ("pearson", "rel_l2", "nrmse"):
        a = [per["thin"][s]["fdtd"][m] for s in common]; b = [per["bpm"][s]["fdtd"][m] for s in common]
        t = paired_tests(a, b)
        out["metrics"][m] = {"thin": mean_std(a), "bpm": mean_std(b), "tests": t}
        lines.append(f"| {m} | {np.mean(a):.4f} +- {np.std(a, ddof=1):.4f} | {np.mean(b):.4f} +- {np.std(b, ddof=1):.4f} | "
                     f"{t['mean_diff']:+.4f} [{t['boot_ci_low']:+.4f}, {t['boot_ci_high']:+.4f}] | {t['wilcoxon_p']:.3g} | {t['perm_p']:.3g} |")
    out["metrics_scalar_models"] = {k: {m: float(np.mean([per[dsg][s][mk][m] for s in common])) for m in ("pearson", "rel_l2", "nrmse")}
                                    for k, (dsg, mk) in {"thin_design_thinmodel": ("thin", "model_thin"), "thin_design_bpmmodel": ("thin", "model_bpm"),
                                                         "bpm_design_bpmmodel": ("bpm", "model_bpm"), "bpm_design_thinmodel": ("bpm", "model_thin")}.items()}

# ---- same-structure fidelity
lines += ["", "## Same-structure fidelity (model prediction vs FDTD for the identical exported structure)", "",
          "| structure | model | field NMSE | intensity NMSE | Pearson r | SSIM |" + (" top-1 agreement |" if patches else ""),
          "|---|---|---|---|---|---|" + ("---|" if patches else "")]
out["fidelity"] = {}
paired_lines = []
for design in ("thin", "bpm"):
    for mk, mname in (("fid_thin_model", "thin"), ("fid_bpm_model", "dBPM (dz<=lambda/12)"), ("fid_bpm_train_model", "dBPM (training N_sub)")):
        recs = [per[design][s][mk] for s in common]
        agg = {m: mean_std([r[m] for r in recs]) for m in ("field_nmse", "int_nmse", "pearson", "ssim")}
        if patches:
            agg["top1_agree"] = float(np.mean([r["top1_agree"] for r in recs]))
        out["fidelity"][f"{design}_design__{mk}"] = agg
        lines.append(f"| {design}-trained | {mname} | {agg['field_nmse']['mean']:.3f} +- {agg['field_nmse']['std']:.3f} | "
                     f"{agg['int_nmse']['mean']:.3f} +- {agg['int_nmse']['std']:.3f} | {agg['pearson']['mean']:.3f} | {agg['ssim']['mean']:.3f} |"
                     + (f" {agg['top1_agree']*100:.0f}% |" if patches else ""))
    # paired test thin-model vs bpm-model fidelity on the same structures
    fa = np.array([per[design][s]["fid_thin_model"]["field_nmse"] for s in common]); fb = np.array([per[design][s]["fid_bpm_model"]["field_nmse"] for s in common])
    out["fidelity"][f"{design}_design__paired_field_nmse_thin_vs_bpm"] = paired_tests(fa, fb)
    ia = [per[design][s]["fid_thin_model"]["int_nmse"] for s in common]; ib = [per[design][s]["fid_bpm_model"]["int_nmse"] for s in common]
    out["fidelity"][f"{design}_design__paired_int_nmse_thin_vs_bpm"] = paired_tests(ia, ib)
    pf = out["fidelity"][f"{design}_design__paired_field_nmse_thin_vs_bpm"]
    paired_lines.append(f"- {design}-trained structures, field NMSE (dBPM model minus thin model): {pf['mean_diff']:+.4f} "
                        f"[95% CI {pf['boot_ci_low']:+.4f}, {pf['boot_ci_high']:+.4f}], Wilcoxon p = {pf['wilcoxon_p']:.2e}; "
                        f"dBPM closer to FDTD in {int(np.sum(fb < fa))}/{len(fa)} structures")
lines += ["", "Paired same-structure fidelity tests (thin model vs dBPM model as predictors of FDTD):"] + paired_lines
costs = [r.get("real_cost") for r in res.values() if r.get("real_cost") is not None and r.get("done") and r["mesh"] == args.mesh]
out["total_real_cost"] = float(np.sum(costs)) if costs else None
out["per_sample"] = {k: {str(s): v for s, v in per[k].items()} for k in per}
lines += ["", f"Total FlexCredits (real) for the mesh lambda/{args.mesh} runs in this directory: {out['total_real_cost']}"]
base = args.out or os.path.join(args.run_dir, f"analysis_{args.task}_mesh{args.mesh}")
json.dump(out, open(base + ".json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
open(base + ".md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
