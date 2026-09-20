#!/usr/bin/env python
"""Convergence of the dBPM forward model with respect to the number of axial
sub-slices N_sub (Reviewer 1, comment 5).

For fixed physical height maps (thin-trained and dBPM-trained designs) the
detector-plane field is evaluated with N_sub = 15 ... N_max and compared with
the finest discretisation.  Reported per refractive index:

* test accuracy vs N_sub (full test set),
* complex-field NMSE and intensity NMSE vs the N_max reference (subset),
* top-1 agreement with the N_max reference.

The slice thickness is dz = h_max / N_sub, and the soft-occupancy temperature
follows the paper convention tau = 0.05 dz, so both the axial step and the
height quantisation are refined together.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dbpm_d2nn.model import load_checkpoint
from dbpm_d2nn.data import classification_datasets, make_loaders, detector_patches_from_geometry, encode, patch_energies
from dbpm_d2nn.evaluate import evaluate_classification
from dbpm_d2nn.metrics import nmse_field, nmse_intensity

p = argparse.ArgumentParser()
p.add_argument("--ckpt_pattern", required=True, help="e.g. .../models/pixel_v3_{kind}_n{n:.2f}.pt")
p.add_argument("--kinds", default="thin,bpm")
p.add_argument("--n_values", default="1.1,1.2,1.3,1.5,1.7,1.9")
p.add_argument("--n_subs", default="15,30,60,120,240")
p.add_argument("--n_ref", type=int, default=480)
p.add_argument("--dataset", default="mnist"); p.add_argument("--encoding", default="phase")
p.add_argument("--grid", type=int, default=200); p.add_argument("--digit", type=int, default=112)
p.add_argument("--axial_mode", default="legacy"); p.add_argument("--anchor", type=float, default=0.5)
p.add_argument("--n_field", type=int, default=256, help="test samples for field NMSE")
p.add_argument("--out", required=True)
p.add_argument("--device", default="cuda:0")
args = p.parse_args()

dev = args.device
train, val, test = classification_datasets(args.dataset, args.grid, args.digit)
_, _, te = make_loaders(train, val, test, test_batch=250, num_workers=2)
xs = torch.stack([test[i][0] for i in range(args.n_field)]).to(dev)
n_subs = [int(v) for v in args.n_subs.split(",")]
results = []
os.makedirs(os.path.dirname(args.out), exist_ok=True)
for n in [float(v) for v in args.n_values.split(",")]:
    for kind in args.kinds.split(","):
        path = args.ckpt_pattern.format(kind=kind, n=n)
        if not os.path.exists(path):
            print("missing", path); continue
        base = load_checkpoint(path, dev)
        patches = detector_patches_from_geometry(args.grid, args.grid, base.dx, base.wavelength)
        # reference: finest discretisation
        ref = base.clone(n_sub=args.n_ref, axial_mode=args.axial_mode, anchor=args.anchor)
        with torch.no_grad():
            I_ref, u_ref = ref(encode(xs, args.encoding), return_field=True)
        E_ref = patch_energies(I_ref, patches).argmax(1)
        t0 = time.time(); acc_ref = evaluate_classification(ref, te, patches, args.encoding, dev)["acc"]
        print(f"n={n} {kind}: reference N_sub={args.n_ref} acc={acc_ref:.2f} ({time.time()-t0:.0f}s)", flush=True)
        for ns in n_subs:
            m = base.clone(n_sub=ns, axial_mode=args.axial_mode, anchor=args.anchor)
            with torch.no_grad():
                I, u = m(encode(xs, args.encoding), return_field=True)
            nm_f = np.mean([nmse_field(u[i].cpu().numpy(), u_ref[i].cpu().numpy()) for i in range(xs.shape[0])])
            nm_i = np.mean([nmse_intensity(I[i].cpu().numpy(), I_ref[i].cpu().numpy()) for i in range(xs.shape[0])])
            agree = (patch_energies(I, patches).argmax(1) == E_ref).float().mean().item()
            acc = evaluate_classification(m, te, patches, args.encoding, dev)["acc"]
            rec = {"n": n, "kind": kind, "n_sub": ns, "dz_over_lambda": m.dz_sub / m.wavelength,
                   "sigmoid_temp_over_lambda": m.sigmoid_temp / m.wavelength, "acc": acc, "acc_ref": acc_ref,
                   "n_ref": args.n_ref, "field_nmse_vs_ref": float(nm_f), "intensity_nmse_vs_ref": float(nm_i),
                   "top1_agreement_vs_ref": agree, "axial_mode": args.axial_mode}
            results.append(rec)
            print(f"   N_sub={ns:4d} dz={rec['dz_over_lambda']:.3f}lam acc={acc:.2f} fieldNMSE={nm_f:.2e} "
                  f"intNMSE={nm_i:.2e} agree={agree:.4f}", flush=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "results": results}, f, indent=1)
print("saved", args.out)
