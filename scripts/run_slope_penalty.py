#!/usr/bin/env python
"""Slope (total-variation) penalty sweep for dBPM training at fixed n.

L = L_task + w_tv * mean_l TV(h_l / h_max),  TV(h) = mean nearest-neighbour |dh|.
For every weight we report test accuracy (own N_sub and converged N_sub) and the
fabrication statistics of the resulting height maps (|dh| percentiles, slope-angle
percentiles, fraction of 2pi-wrap edges, aspect ratio).
"""
import argparse, json, math, os, sys, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dbpm_d2nn.model import D2NN, save_checkpoint
from dbpm_d2nn.data import classification_datasets, make_loaders, detector_patches_from_geometry
from dbpm_d2nn.evaluate import evaluate_classification
from dbpm_d2nn.train import train_classification
from dbpm_d2nn.metrics import fabrication_statistics

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="mnist"); p.add_argument("--encoding", default="phase")
p.add_argument("--n", type=float, default=1.5); p.add_argument("--weights", default="0,0.1,0.3,1,3")
p.add_argument("--phase_param", default="wrap", choices=["wrap", "sigmoid"])
p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=10)
p.add_argument("--grid", type=int, default=200); p.add_argument("--digit", type=int, default=112)
p.add_argument("--z_lam", type=float, default=40.0); p.add_argument("--aperture_lam", type=float, default=107.0)
p.add_argument("--out_dir", required=True); p.add_argument("--device", default="cuda:0")
args = p.parse_args()

dev = args.device
os.makedirs(args.out_dir, exist_ok=True)
lam = 532e-9; dx = args.aperture_lam * lam / args.grid; z = args.z_lam * lam
n = args.n
n_sub = max(15, int(math.ceil(3.0 / (n - 1.0) - 1e-9)))
ref_ns = max(15, int(math.ceil(12.0 / (n - 1.0) - 1e-9)))
train, val, test = classification_datasets(args.dataset, args.grid, args.digit)
patches = detector_patches_from_geometry(args.grid, args.grid, dx, lam)
common = dict(H=args.grid, W=args.grid, L=5, wavelength=lam, dx=dx, z=z, z_output=z, phase_param=args.phase_param,
              axial_mode="anchor", anchor=0.5, n_material=n)
results_path = os.path.join(args.out_dir, "results.json")
results = json.load(open(results_path)) if os.path.exists(results_path) else []
done = {r["tv_weight"] for r in results}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def fab(model):
    st = [fabrication_statistics(h.detach().cpu().numpy(), dx, lam, model.h_max) for h in model.heights()]
    return {k: float(np.mean([s[k] for s in st])) for k in st[0]}


for w in [float(v) for v in args.weights.split(",")]:
    if w in done:
        continue
    seed_all(args.seed)
    tl, vl, te = make_loaders(train, val, test, batch_size=128, seed=args.seed, num_workers=4, test_batch=250)
    t0 = time.time()
    m = D2NN(**common, n_sub=n_sub, seed=args.seed).to(dev)
    hist = train_classification(m, tl, patches, args.encoding, dev, epochs=args.epochs, lr=1e-3, anneal=True, tv_weight=w, log=print)
    rec = {"tv_weight": w, "n": n, "train_n_sub": n_sub, "seed": args.seed, "phase_param": args.phase_param,
           "acc_own": evaluate_classification(m, te, patches, args.encoding, dev)["acc"],
           "acc_ref": evaluate_classification(m.clone(n_sub=ref_ns), te, patches, args.encoding, dev)["acc"],
           "fab": fab(m), "train_loss": [h["train_loss"] for h in hist], "time_s": time.time() - t0}
    save_checkpoint(m, os.path.join(args.out_dir, f"bpm_n{n:.2f}_tv{w:g}.pt"), n=n, tv_weight=w, seed=args.seed)
    results.append(rec)
    json.dump(results, open(results_path, "w"), indent=1)
    print(f"tv_weight={w:g}: acc {rec['acc_own']:.2f} (ref {rec['acc_ref']:.2f}) | slope p95 {rec['fab']['slope_p95_deg']:.1f} deg, "
          f"|dh| median {rec['fab']['dh_median_nm']:.0f} nm, wrap edges {rec['fab']['wrap_edge_fraction']*100:.2f}%", flush=True)
print("done")
