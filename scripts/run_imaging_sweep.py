#!/usr/bin/env python
"""Imaging sweep over refractive index (default seeds 0, 1, 2).

Geometry matches classification: 200 x 200 grid,
pixel pitch 0.535 lambda (aperture 107 lambda), inter-layer spacing 40 lambda and
40 lambda from the last layer to the detector.  The image occupies the central
``--img`` x ``--img`` pixels of the grid.

For every (seed, n) the script produces the same set of conditions as the
classification sweep (thin->thin, thin->vol at several axial conventions,
one-parameter delta_z / s / joint calibrations fitted on the validation split,
k-epoch dBPM fine-tuning from the thin solution, dBPM from scratch) with
MSE / SSIM (paper protocol, full grid) and scale-invariant / central-crop variants.
"""
import argparse, json, math, os, sys, time, random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dbpm_d2nn.model import D2NN, save_checkpoint
from dbpm_d2nn.data import imaging_datasets, make_loaders
from dbpm_d2nn.evaluate import evaluate_imaging
from dbpm_d2nn.train import train_imaging
from dbpm_d2nn.calibrate import fit_axial_shift, fit_height_scale
from dbpm_d2nn.metrics import fabrication_statistics

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="mnist", choices=["mnist", "fmnist", "cifar100"])
p.add_argument("--n_values", default="1.1,1.2,1.3,1.5,1.7,1.9")
p.add_argument("--seeds", default="0,1,2")
p.add_argument("--grid", type=int, default=200); p.add_argument("--img", type=int, default=128)
p.add_argument("--wavelength", type=float, default=532e-9)
p.add_argument("--z_lam", type=float, default=40.0); p.add_argument("--z_out_lam", type=float, default=40.0)
p.add_argument("--aperture_lam", type=float, default=107.0)
p.add_argument("--L", type=int, default=5)
p.add_argument("--epochs", type=int, default=50); p.add_argument("--ft_epochs", type=int, default=10)
p.add_argument("--batch", type=int, default=4); p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--phase_param", default="sigmoid", choices=["sigmoid", "wrap"])
# Imaging protocol (see configs/imaging.yaml and Supplement Table S3): a design trained with N_sub = 15 slices
# over-fits the 15-level soft staircase of Eq. (11) (SSIM 0.55-0.56 at its own N_sub, 0.47-0.49 at N_sub = 18,
# 0.51-0.53 converged), whereas designs trained with >= 30 slices are converged to within 0.005 SSIM; the
# evaluation therefore uses at least 120 slices (height-quantisation error <= 5e-4 in field NMSE at every index).
p.add_argument("--n_sub", type=int, default=30, help="minimum number of sub-slices used for training")
p.add_argument("--train_dz_lambda", type=float, default=6.0,
               help="training discretisation: N_sub = max(n_sub, ceil(this * h_max / lambda))")
p.add_argument("--eval_dz_lambda", type=float, default=12.0, help="converged evaluation: dz <= lambda/this")
p.add_argument("--min_eval_n_sub", type=int, default=120, help="minimum number of sub-slices used for evaluation")
p.add_argument("--n_train", type=int, default=1500); p.add_argument("--n_val", type=int, default=200)
p.add_argument("--n_test", type=int, default=300)
p.add_argument("--skip_bpm", action="store_true"); p.add_argument("--skip_fits", action="store_true")
p.add_argument("--skip_ft", action="store_true")
p.add_argument("--out_dir", required=True)
p.add_argument("--device", default="cuda:0"); p.add_argument("--workers", type=int, default=4)
args = p.parse_args()

dev = args.device
os.makedirs(args.out_dir, exist_ok=True)
ckpt_dir = os.path.join(args.out_dir, "checkpoints"); os.makedirs(ckpt_dir, exist_ok=True)
lam = args.wavelength
dx = args.aperture_lam * lam / args.grid
z = args.z_lam * lam
z_out = args.z_out_lam * lam
logf = open(os.path.join(args.out_dir, "log.txt"), "a")


def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


train, val, test = imaging_datasets(args.dataset, args.grid, args.img, n_train=args.n_train, n_val=args.n_val,
                                    n_test=args.n_test, split_seed=0)
u_max = math.degrees(math.asin(min(1.0, lam / (2 * dx))))
R_over_w = (z * math.tan(math.radians(u_max))) / (args.grid * dx)
log(f"== imaging {args.dataset}: grid {args.grid}, img {args.img}, dx={dx/lam:.3f}lam, z={args.z_lam}lam, "
    f"z_out={args.z_out_lam}lam, u_max={u_max:.1f}deg, R/w={R_over_w:.2f}")
common = dict(H=args.grid, W=args.grid, L=args.L, wavelength=lam, dx=dx, z=z, z_output=z_out,
              phase_param=args.phase_param)


def train_n_sub(n):
    return max(args.n_sub, int(math.ceil(args.train_dz_lambda / (n - 1.0) - 1e-9)))


def eval_n_sub(n):
    return max(args.n_sub, args.min_eval_n_sub, int(math.ceil(args.eval_dz_lambda / (n - 1.0) - 1e-9)))


def metrics_of(model, loader):
    return evaluate_imaging(model, loader, dev, crop=args.img)


def vol_eval(design: D2NN, n, *, mode="anchor", anchor=0.5, shift=0.0, scale=1.0, loader=None):
    out = {}
    for tag, ns in (("train", train_n_sub(n)), ("fine", eval_n_sub(n))):
        m = D2NN(**common, n_material=n, n_sub=ns, axial_mode=mode, anchor=anchor, axial_shift=shift,
                 height_scale=scale).to(dev)
        m.load_layers_from(design)
        out[tag] = metrics_of(m, loader)
        out[f"{tag}_n_sub"] = m.n_sub
    return out


def brief(d):
    return f"mse={d['mse']:.4g} ssim={d['ssim']:.3f} ssim_crop={d['ssim_crop']:.3f}"


results_path = os.path.join(args.out_dir, "results.json")
all_results = json.load(open(results_path)) if os.path.exists(results_path) else []
done = {(r["seed"], r["n"]) for r in all_results}

for seed in [int(s) for s in args.seeds.split(",")]:
    seed_all(seed)
    tl, vl, te = make_loaders(train, val, test, batch_size=args.batch, seed=seed, num_workers=args.workers, test_batch=50)
    thin_path = os.path.join(ckpt_dir, f"thin_seed{seed}.pt")
    thin = D2NN(**common, n_material=1.5, n_sub=1, seed=seed).to(dev)
    if os.path.exists(thin_path):
        thin.load_state_dict(torch.load(thin_path, map_location=dev)["state_dict"]); thin.eval()
        log(f"[seed {seed}] loaded thin model")
    else:
        log(f"[seed {seed}] training thin model")
        train_imaging(thin, tl, dev, epochs=args.epochs, lr=args.lr,
                      epoch_callback=lambda ep, r: {"val_mse": metrics_of(thin, vl)["mse"]}, log=log)
        save_checkpoint(thin, thin_path, seed=seed, kind="thin", dataset=args.dataset, task="imaging")
    m_thin_thin = metrics_of(thin, te); m_thin_val = metrics_of(thin, vl)
    log(f"[seed {seed}] thin->thin test {brief(m_thin_thin)}")

    for n in [float(v) for v in args.n_values.split(",")]:
        if (seed, n) in done:
            log(f"[seed {seed} n={n}] already done"); continue
        t_start = time.time()
        rec = {"seed": seed, "n": n, "dataset": args.dataset, "task": "imaging",
               "thin_thin": m_thin_thin, "val_thin_thin": m_thin_val,
               "h_max_over_lambda": 1.0 / (n - 1.0), "T_over_z": 1.0 / ((n - 1.0) * args.z_lam),
               "eval_n_sub": eval_n_sub(n), "train_n_sub": train_n_sub(n)}
        rec["thin_vol_entrance"] = vol_eval(thin, n, mode="legacy", loader=te)
        rec["thin_vol_centroid"] = vol_eval(thin, n, mode="anchor", anchor=0.5, loader=te)
        rec["thin_vol_exit"] = vol_eval(thin, n, mode="anchor", anchor=1.0, loader=te)
        rec["thin_vol_entrance_anchored"] = vol_eval(thin, n, mode="anchor", anchor=0.0, loader=te)
        log(f"[seed {seed} n={n}] thin->vol entrance(legacy) {brief(rec['thin_vol_entrance']['fine'])} | "
            f"centroid {brief(rec['thin_vol_centroid']['fine'])}")
        base = D2NN(**common, n_material=n, n_sub=train_n_sub(n), axial_mode="anchor", anchor=0.5).to(dev)
        base.load_layers_from(thin)
        if not args.skip_fits:
            obj = lambda m: metrics_of(m, vl)["mse"]
            t0 = time.time()
            fdz = fit_axial_shift(base, obj)
            rec["fit_dz"] = {"axial_shift_m": fdz["axial_shift"], "axial_shift_over_lambda": fdz["axial_shift"] / lam,
                             "axial_shift_over_T": fdz["axial_shift"] / fdz["T"], "val_mse": fdz["objective"],
                             "grid": [(v / lam, o) for v, o in fdz["grid"]]}
            rec["fit_dz"].update({"test_" + k: v for k, v in vol_eval(thin, n, shift=fdz["axial_shift"], loader=te).items()})
            fs = fit_height_scale(base, obj)
            rec["fit_s"] = {"height_scale": fs["height_scale"], "val_mse": fs["objective"],
                            "grid": [(v, o) for v, o in fs["grid"]]}
            rec["fit_s"].update({"test_" + k: v for k, v in vol_eval(thin, n, scale=fs["height_scale"], loader=te).items()})
            b2 = base.clone(height_scale=fs["height_scale"])
            fj1 = fit_axial_shift(b2, obj, height_scale=fs["height_scale"])
            b3 = base.clone(axial_shift=fj1["axial_shift"])
            fj2 = fit_height_scale(b3, obj, axial_shift=fj1["axial_shift"])
            s_j, dz_j = fj2["height_scale"], fj1["axial_shift"]
            rec["fit_joint"] = {"height_scale": s_j, "axial_shift_m": dz_j, "axial_shift_over_lambda": dz_j / lam,
                                "val_mse": fj2["objective"]}
            rec["fit_joint"].update({"test_" + k: v for k, v in vol_eval(thin, n, shift=dz_j, scale=s_j, loader=te).items()})
            log(f"[seed {seed} n={n}] fits ({time.time()-t0:.0f}s): dz*={rec['fit_dz']['axial_shift_over_lambda']:+.3f}lam -> "
                f"{brief(rec['fit_dz']['test_fine'])} | s*={fs['height_scale']:.3f} -> {brief(rec['fit_s']['test_fine'])} | "
                f"joint s={s_j:.3f} dz={dz_j/lam:+.3f}lam -> {brief(rec['fit_joint']['test_fine'])}")
        if not args.skip_ft:
            ft = D2NN(**common, n_material=n, n_sub=train_n_sub(n), axial_mode="anchor", anchor=0.5).to(dev)
            ft.load_layers_from(thin)
            hist = train_imaging(ft, tl, dev, epochs=args.ft_epochs, lr=args.lr, anneal=False,
                                 epoch_callback=lambda ep, r: {"test": metrics_of(ft, te), "val_mse": metrics_of(ft, vl)["mse"]},
                                 log=None)
            rec["finetune"] = {"epochs": [h["epoch"] for h in hist], "test": [h["test"] for h in hist],
                               "val_mse": [h["val_mse"] for h in hist], "train_loss": [h["train_loss"] for h in hist],
                               "final_fine": vol_eval(ft, n, loader=te)["fine"]}
            save_checkpoint(ft, os.path.join(ckpt_dir, f"ft_seed{seed}_n{n:.2f}.pt"), seed=seed, n=n, kind="thin_finetuned")
            log(f"[seed {seed} n={n}] fine-tune test ssim per epoch: {['%.3f' % h['test']['ssim'] for h in hist]}")
        if not args.skip_bpm:
            seed_all(seed)
            bpm = D2NN(**common, n_material=n, n_sub=train_n_sub(n), axial_mode="anchor", anchor=0.5, seed=seed).to(dev)
            hist = train_imaging(bpm, tl, dev, epochs=args.epochs, lr=args.lr, anneal=True,
                                 epoch_callback=lambda ep, r: {"val_mse": metrics_of(bpm, vl)["mse"]}, log=log)
            rec["bpm_vol"] = vol_eval(bpm, n, loader=te)
            rec["bpm_vol"]["val"] = metrics_of(bpm, vl)
            rec["bpm_vol"]["train_loss"] = [h["train_loss"] for h in hist]
            thin_of_bpm = D2NN(**common, n_material=n, n_sub=1).to(dev); thin_of_bpm.load_layers_from(bpm)
            rec["bpm_thin"] = metrics_of(thin_of_bpm, te)
            save_checkpoint(bpm, os.path.join(ckpt_dir, f"bpm_seed{seed}_n{n:.2f}.pt"), seed=seed, n=n, kind="bpm")
            log(f"[seed {seed} n={n}] dBPM->vol {brief(rec['bpm_vol']['fine'])} | dBPM->thin {brief(rec['bpm_thin'])}")
            rec["fab_bpm"] = [fabrication_statistics(h.detach().cpu().numpy(), dx, lam, bpm.h_max) for h in bpm.heights()]
        hthin = D2NN(**common, n_material=n, n_sub=1).to(dev); hthin.load_layers_from(thin)
        rec["fab_thin"] = [fabrication_statistics(h.detach().cpu().numpy(), dx, lam, hthin.h_max) for h in hthin.heights()]
        rec["time_s"] = time.time() - t_start
        all_results.append(rec)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=1)
        log(f"[seed {seed} n={n}] done in {rec['time_s']/60:.1f} min")
log("sweep finished")
