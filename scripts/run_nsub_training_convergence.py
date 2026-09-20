#!/usr/bin/env python
"""Training-side N_sub convergence: train dBPM designs at several axial
discretisations (N_sub = 15, 30, 60 at fixed n) and evaluate every design both at
its own N_sub and at a common converged discretisation (dz <= lambda/12).

If the designs trained with coarse and fine slicing reach the same converged
accuracy, the training results do not depend on the split-step resolution.
"""
import argparse, json, math, os, sys, random, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dbpm_d2nn.model import D2NN, save_checkpoint
from dbpm_d2nn.data import classification_datasets, make_loaders, detector_patches_from_geometry
from dbpm_d2nn.evaluate import evaluate_classification, output_fields
from dbpm_d2nn.train import train_classification
from dbpm_d2nn.metrics import nmse_field, nmse_intensity

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="mnist"); p.add_argument("--encoding", default="phase")
p.add_argument("--n_values", default="1.1,1.5"); p.add_argument("--n_subs", default="15,30,60")
p.add_argument("--seed", type=int, default=0); p.add_argument("--epochs", type=int, default=10)
p.add_argument("--grid", type=int, default=200); p.add_argument("--digit", type=int, default=112)
p.add_argument("--z_lam", type=float, default=40.0); p.add_argument("--aperture_lam", type=float, default=107.0)
p.add_argument("--eval_dz_lambda", type=float, default=12.0)
p.add_argument("--min_ref_n_sub", type=int, default=120,
               help="reference discretisation: N_sub = max(this, ceil(eval_dz_lambda * h_max / lambda)) (converged in slice "
                    "thickness and in the number of height levels)")
p.add_argument("--out_dir", required=True); p.add_argument("--device", default="cuda:0")
args = p.parse_args()

dev = args.device
os.makedirs(args.out_dir, exist_ok=True)
lam = 532e-9; dx = args.aperture_lam * lam / args.grid; z = args.z_lam * lam
train, val, test = classification_datasets(args.dataset, args.grid, args.digit)
patches = detector_patches_from_geometry(args.grid, args.grid, dx, lam)
common = dict(H=args.grid, W=args.grid, L=5, wavelength=lam, dx=dx, z=z, z_output=z, phase_param="wrap",
              axial_mode="anchor", anchor=0.5)
results_path = os.path.join(args.out_dir, "results.json")
results = json.load(open(results_path)) if os.path.exists(results_path) else []
done = {(r["n"], r["train_n_sub"]) for r in results}
# fixed batch of 256 test images for field-level comparisons
xb = torch.stack([test[i][0] for i in range(256)]).to(dev)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


from dbpm_d2nn.model import load_checkpoint

for n in [float(v) for v in args.n_values.split(",")]:
    ref_ns = max(args.min_ref_n_sub, int(math.ceil(args.eval_dz_lambda / (n - 1.0) - 1e-9)))
    for ns in [int(v) for v in args.n_subs.split(",")]:
        ckpt_path = os.path.join(args.out_dir, f"bpm_n{n:.2f}_nsub{ns}.pt")
        tl, vl, te = make_loaders(train, val, test, batch_size=128, seed=args.seed, num_workers=4, test_batch=250)
        if (n, ns) in done:
            rec = next(r for r in results if (r["n"], r["train_n_sub"]) == (n, ns))
            if rec.get("ref_n_sub") == ref_ns or not os.path.exists(ckpt_path):
                continue
            # re-evaluate an existing design at the (new) reference discretisation
            m = load_checkpoint(ckpt_path, dev)
            results.remove(rec)
        else:
            seed_all(args.seed)
            t0 = time.time()
            m = D2NN(**common, n_material=n, n_sub=ns, seed=args.seed).to(dev)
            hist = train_classification(m, tl, patches, args.encoding, dev, epochs=args.epochs, lr=1e-3, anneal=True, log=print)
            rec = {"n": n, "train_n_sub": ns, "dz_over_lambda": m.dz_sub / lam, "seed": args.seed,
                   "acc_own": evaluate_classification(m, te, patches, args.encoding, dev)["acc"],
                   "train_time_s": time.time() - t0, "train_loss": [h["train_loss"] for h in hist]}
        ref = m.clone(n_sub=ref_ns)
        rec["acc_ref"] = evaluate_classification(ref, te, patches, args.encoding, dev)["acc"]
        rec["ref_n_sub"] = ref.n_sub
        # field-level self-consistency: own discretisation vs converged reference on 256 test images
        from dbpm_d2nn.data import encode
        with torch.no_grad():
            I_own, U_own = m(encode(xb, args.encoding), return_field=True)
            I_ref, U_ref = ref(encode(xb, args.encoding), return_field=True)
        rec["field_nmse_own_vs_ref"] = float(np.mean([nmse_field(U_own[i].cpu().numpy(), U_ref[i].cpu().numpy()) for i in range(xb.size(0))]))
        rec["int_nmse_own_vs_ref"] = float(np.mean([nmse_intensity(I_own[i].cpu().numpy(), I_ref[i].cpu().numpy()) for i in range(xb.size(0))]))
        if not os.path.exists(ckpt_path):
            save_checkpoint(m, ckpt_path, n=n, n_sub=ns, seed=args.seed)
        results.append(rec)
        results.sort(key=lambda r: (r["n"], r["train_n_sub"]))
        json.dump(results, open(results_path, "w"), indent=1)
        print(f"n={n} N_sub={ns} (dz={rec['dz_over_lambda']:.3f}lam): acc own {rec['acc_own']:.2f} | at ref N_sub={ref.n_sub}: "
              f"{rec['acc_ref']:.2f} | field NMSE own-vs-ref {rec['field_nmse_own_vs_ref']:.2e} ({rec['train_time_s']/60:.1f} min)", flush=True)
print("done")
