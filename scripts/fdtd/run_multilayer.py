#!/usr/bin/env python
"""Paired full-wave (Tidy3D FDTD) evaluation of the thin-trained and dBPM-trained
five-layer 64 x 64 designs on the *same* stratified-random test samples.

Every (sample, design) pair is one FDTD simulation.  The detector field is
area-averaged onto the design grid and stored together with the scalar-model
predictions (thin model and dBPM model applied to the same exported structure) so
that both the task metrics and the same-structure model-vs-FDTD fidelity can be
analysed afterwards (analyze_fdtd.py).
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dbpm_d2nn.fdtd import build_simulation, bin_field

p = argparse.ArgumentParser()
p.add_argument("--export", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--designs", default="thin,bpm")
p.add_argument("--samples", default="all", help="'all' or comma separated indices into the exported set")
p.add_argument("--mesh", type=int, default=10, help="min. grid steps per wavelength in each medium")
p.add_argument("--layer_monitors_for", default="0", help="sample indices that also get after-layer monitors")
p.add_argument("--yz_for", default="0", help="sample indices that also get a YZ cross-section monitor")
p.add_argument("--estimate_only", action="store_true")
p.add_argument("--chunk", type=int, default=8, help="simulations per Tidy3D batch")
p.add_argument("--tag", default="")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
d = np.load(args.export)
lam_um, dx_um, n = float(d["wavelength_um"]), float(d["dx_um"]), float(d["n_material"])
z_um, z_out_um, T_um = float(d["z_um"]), float(d["z_out_um"]), float(d["h_max_um"])
u_in_all = d["u_in"]
N = u_in_all.shape[0]
H, W = u_in_all.shape[1:]
samples = list(range(N)) if args.samples == "all" else [int(s) for s in args.samples.split(",")]
lm_for = {int(s) for s in args.layer_monitors_for.split(",") if s != ""}
yz_for = {int(s) for s in args.yz_for.split(",") if s != ""}
results_path = os.path.join(args.out_dir, "results.json")
results = json.load(open(results_path)) if os.path.exists(results_path) else {}
mesh_tag = f"mesh{args.mesh}"

from tidy3d import web

jobs = []
for s in samples:
    for design in args.designs.split(","):
        key = f"s{s:02d}_{design}_{mesh_tag}{args.tag}"
        if key in results and results[key].get("done"):
            continue
        b = build_simulation(list(d[f"h_{design}_um"]), dx_um, lam_um, n, z_um, z_out_um, u_in_all[s], T_um=T_um,
                             anchor=0.5, steps_per_wvl=args.mesh, layer_monitors=(s in lm_for), yz_monitor=(s in yz_for))
        jobs.append((key, s, design, b))
print(f"{len(jobs)} simulations to run ({len(results)} already in results.json)", flush=True)

for c0 in range(0, len(jobs), args.chunk):
    chunk = jobs[c0:c0 + args.chunk]
    sims = {key: b["sim"] for key, _, _, b in chunk}
    batch = web.Batch(simulations=sims, folder_name="dbpm_d2nn_fdtd", verbose=False)
    t0 = time.time()
    est = batch.estimate_cost(verbose=False)
    print(f"chunk {c0//args.chunk}: {len(chunk)} sims, estimated {est:.2f} FlexCredits", flush=True)
    for key, s, design, b in chunk:
        results[key] = {"sample": s, "design": design, "mesh": args.mesh, "n_boxes": b["n_boxes"], "z": b["z"], "done": False}
    if args.estimate_only:
        continue
    bd = batch.run(path_dir=os.path.join(args.out_dir, "data"))
    for key, s, design, b in chunk:
        sd = bd[key]
        I_f, Ey_f = bin_field(sd["detector"], b["aperture_um"], H, W)
        out = {"I": I_f, "Ey": Ey_f}
        for mon in sd.simulation.monitors:
            if mon.name.startswith("after_layer") or mon.name == "after_source":
                Ii, Ei = bin_field(sd[mon.name], b["aperture_um"], H, W)
                out[f"I_{mon.name}"] = Ii; out[f"Ey_{mon.name}"] = Ei
            if mon.name == "yz":
                fd = sd["yz"]
                Iyz = sum(np.abs(getattr(fd, c).isel(f=0).squeeze(drop=True).values) ** 2 for c in ("Ex", "Ey", "Ez"))
                out["I_yz"] = Iyz; out["yz_y"] = fd.Ey.y.values; out["yz_z"] = fd.Ey.z.values
        for mname in ("thin", "bpm", "bpm_fine"):
            out[f"I_{mname}model"] = d[f"I_{design}design_{mname}model"][s]
            out[f"U_{mname}model"] = d[f"U_{design}design_{mname}model"][s]
        fpath = os.path.join(args.out_dir, f"{key}.npz")
        np.savez(fpath, **out)
        real_cost = None
        try:
            real_cost = float(web.real_cost(batch.jobs[key].task_id, verbose=False))
        except Exception:
            pass
        results[key].update({"done": True, "file": fpath, "real_cost": real_cost})
    json.dump(results, open(results_path, "w"), indent=1)
    print(f"chunk done in {(time.time()-t0)/60:.1f} min", flush=True)
json.dump(results, open(results_path, "w"), indent=1)
print("finished")
