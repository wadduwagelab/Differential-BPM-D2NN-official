#!/usr/bin/env python
"""Train the small (default 32 x 32) designs used for the full-wave (FDTD) validation and export
everything the FDTD scripts need (height maps, input fields of the selected test
samples, scalar-model predictions).

Geometry (identical design rule as the main experiments): pixel pitch 0.535 lambda,
aperture 32 * 0.535 = 17.1 lambda, inter-layer spacing 8 lambda (R/w = 1.2, T/z = 0.25 at n = 1.5), the
same distance from the last layer to the detector, n = 1.5, lambda = 532 nm.

Tasks
  classification : MNIST, amplitude encoding, digit 18 px, 3-4-3 detector layout
                   with 2 px patches (paper FDTD layout scaled), 10 epochs, batch 128
  imaging        : Fashion-MNIST, amplitude, image 20 px, 1500/200/300 split,
                   50 epochs, batch 4 (paper protocol)

Sample selection is *stratified random* (2 per class for classification, 20 random
test images for imaging) with a fixed seed and is independent of any model output.
"""
import argparse, json, math, os, sys, random
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dbpm_d2nn.model import D2NN, save_checkpoint
from dbpm_d2nn.data import (classification_datasets, imaging_datasets, make_loaders, make_detector_patches, encode)
from dbpm_d2nn.evaluate import evaluate_classification, evaluate_imaging, predict_classification
from dbpm_d2nn.train import train_classification, train_imaging

p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["classification", "imaging"])
p.add_argument("--out_dir", required=True)
p.add_argument("--n", type=float, default=1.5)
p.add_argument("--grid", type=int, default=32, help="aperture (trainable region) in pixels")
p.add_argument("--window", type=int, default=128,
               help="simulation window in pixels; the region outside the central aperture is transparent "
                    "(zero padding, no periodic wrap-around)")
p.add_argument("--pitch_lam", type=float, default=0.535)
p.add_argument("--z_lam", type=float, default=8.0)
p.add_argument("--digit", type=int, default=None); p.add_argument("--img", type=int, default=None)
p.add_argument("--patch", type=int, default=None)
p.add_argument("--wavelength", type=float, default=532e-9)
p.add_argument("--n_sub", type=int, default=15)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_samples", type=int, default=20)
p.add_argument("--select_seed", type=int, default=2026)
p.add_argument("--device", default="cuda:0")
args = p.parse_args()

dev = args.device
os.makedirs(args.out_dir, exist_ok=True)
lam = args.wavelength
dx = args.pitch_lam * lam
z = args.z_lam * lam
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
common = dict(H=args.window, W=args.window, L=5, wavelength=lam, dx=dx, z=z, z_output=z, aperture=args.grid)
n_sub = max(args.n_sub, int(math.ceil(3.0 / (args.n - 1.0) - 1e-9)))
log_lines = []


def log(s):
    print(s, flush=True); log_lines.append(s)


if args.task == "classification":
    digit = args.digit or int(round(0.5625 * args.grid))
    patch = args.patch or max(2, args.grid // 16)
    m3, m4, mv = 3 * patch // 2, patch, 3 * patch // 2
    train, val, test = classification_datasets("mnist", args.window, digit)
    tl, vl, te = make_loaders(train, val, test, batch_size=128, seed=args.seed, num_workers=4, test_batch=500)
    patches = make_detector_patches(args.window, args.window, patch=patch, margin_3=m3, margin_4=m4, margin_v=mv)
    o = (args.window - args.grid) // 2
    patches_aperture = [(y0 - o, y1 - o, x0 - o, x1 - o) for (y0, y1, x0, x1) in patches]
    enc = "amplitude"
    common["phase_param"] = "wrap"
    thin = D2NN(**common, n_material=args.n, n_sub=1, seed=args.seed).to(dev)
    train_classification(thin, tl, patches, enc, dev, epochs=10, lr=1e-3, log=log)
    bpm = D2NN(**common, n_material=args.n, n_sub=n_sub, axial_mode="anchor", anchor=0.5, seed=args.seed).to(dev)
    train_classification(bpm, tl, patches, enc, dev, epochs=10, lr=1e-3, anneal=True, log=log)
    res = {"thin_thin": evaluate_classification(thin, te, patches, enc, dev)["acc"],
           "bpm_vol": evaluate_classification(bpm, te, patches, enc, dev)["acc"]}
    thin_vol = D2NN(**common, n_material=args.n, n_sub=n_sub, axial_mode="anchor", anchor=0.5).to(dev)
    thin_vol.load_layers_from(thin)
    res["thin_vol_centroid"] = evaluate_classification(thin_vol, te, patches, enc, dev)["acc"]
    thin_vol_legacy = D2NN(**common, n_material=args.n, n_sub=n_sub, axial_mode="legacy").to(dev)
    thin_vol_legacy.load_layers_from(thin)
    res["thin_vol_entrance_legacy"] = evaluate_classification(thin_vol_legacy, te, patches, enc, dev)["acc"]
    bpm_thin = D2NN(**common, n_material=args.n, n_sub=1).to(dev); bpm_thin.load_layers_from(bpm)
    res["bpm_thin"] = evaluate_classification(bpm_thin, te, patches, enc, dev)["acc"]
    log(f"test accuracies: {res}")
    # ---- stratified random selection: n_samples/10 per class from the test set
    labels = torch.tensor([test[i][1] for i in range(len(test))])
    g = torch.Generator().manual_seed(args.select_seed)
    per_class = args.n_samples // 10
    sel = []
    for c in range(10):
        idx = torch.nonzero(labels == c).flatten()
        sel += idx[torch.randperm(len(idx), generator=g)[:per_class]].tolist()
    sel = sorted(sel)
    xs = torch.stack([test[i][0] for i in sel]); ys = torch.tensor([test[i][1] for i in sel])
    meta = {"patches": patches_aperture, "patches_window": patches, "digit": digit, "encoding": enc, "dataset": "mnist"}
else:
    img = args.img or int(round(0.625 * args.grid))
    train, val, test = imaging_datasets("fmnist", args.window, img, split_seed=0)
    tl, vl, te = make_loaders(train, val, test, batch_size=4, seed=args.seed, num_workers=4, test_batch=50)
    enc = "amplitude"
    common["phase_param"] = "sigmoid"
    thin = D2NN(**common, n_material=args.n, n_sub=1, seed=args.seed).to(dev)
    train_imaging(thin, tl, dev, epochs=50, lr=1e-3, log=log)
    bpm = D2NN(**common, n_material=args.n, n_sub=n_sub, axial_mode="anchor", anchor=0.5, seed=args.seed).to(dev)
    train_imaging(bpm, tl, dev, epochs=50, lr=1e-3, anneal=True, log=log)
    res = {"thin_thin": evaluate_imaging(thin, te, dev, crop=img), "bpm_vol": evaluate_imaging(bpm, te, dev, crop=img)}  # crop = image region
    thin_vol = D2NN(**common, n_material=args.n, n_sub=n_sub, axial_mode="anchor", anchor=0.5).to(dev)
    thin_vol.load_layers_from(thin)
    res["thin_vol_centroid"] = evaluate_imaging(thin_vol, te, dev, crop=img)
    bpm_thin = D2NN(**common, n_material=args.n, n_sub=1).to(dev); bpm_thin.load_layers_from(bpm)
    res["bpm_thin"] = evaluate_imaging(bpm_thin, te, dev, crop=img)
    log(f"test metrics: {json.dumps(res)}")
    g = torch.Generator().manual_seed(args.select_seed)
    sel = sorted(torch.randperm(len(test), generator=g)[: args.n_samples].tolist())
    xs = torch.stack([test[i][0] for i in sel]); ys = torch.tensor([test[i][1] for i in sel])
    meta = {"img": img, "encoding": enc, "dataset": "fmnist"}

save_checkpoint(thin, os.path.join(args.out_dir, "thin.pt"), task=args.task, kind="thin")
save_checkpoint(bpm, os.path.join(args.out_dir, "bpm.pt"), task=args.task, kind="bpm")

# ---- scalar-model predictions for the selected samples (both designs, both forward models, centroid convention)
u_in = encode(xs, enc).to(dev)
pred = {}
with torch.no_grad():
    for dname, design in (("thin", thin), ("bpm", bpm)):
        for mname, ns in (("thin", 1), ("bpm", n_sub), ("bpm_fine", max(n_sub, int(math.ceil(12.0 / (args.n - 1.0)))))):
            m = D2NN(**common, n_material=args.n, n_sub=ns, axial_mode="anchor", anchor=0.5).to(dev)
            m.load_layers_from(design)
            I, u = m(u_in, return_field=True)
            pred[f"I_{dname}design_{mname}model"] = m.crop(I).cpu().numpy()
            pred[f"U_{dname}design_{mname}model"] = m.crop(u).cpu().numpy()
        # field just behind layer 1 for the single-layer study (BPM model, centroid)
heights = {f"h_{d}_um": np.stack([m.crop(h).detach().cpu().numpy() for h in m.heights()]) * 1e6
           for d, m in (("thin", thin), ("bpm", bpm))}
np.savez(os.path.join(args.out_dir, "fdtd_export.npz"),
         u_in=thin.crop(u_in).cpu().numpy(), x_in=thin.crop(xs.squeeze(1)).numpy(), labels=ys.numpy(), test_indices=np.array(sel),
         window=args.window, aperture=args.grid,
         dx_um=dx * 1e6, wavelength_um=lam * 1e6, z_um=z * 1e6, z_out_um=z * 1e6, n_material=args.n,
         h_max_um=thin.h_max * 1e6, n_sub=n_sub, **heights, **pred)
with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
    json.dump({"args": vars(args), "results": res, "meta": meta, "selected_test_indices": sel,
               "labels": ys.tolist(), "n_sub": n_sub, "log": log_lines}, f, indent=1)
log(f"exported {len(sel)} samples to {args.out_dir}")
