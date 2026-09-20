#!/usr/bin/env python
"""Single-layer FDTD study: same structure through thin / dBPM / FDTD + mesh convergence.

For one exported relief layer h(x, y) (layer 1 of the dBPM-trained and of the
thin-trained 64 x 64 designs) and one input field we compute the field 13 lambda
behind the nominal layer plane with

  * the thin-layer model            (phase mask at the nominal plane),
  * the dBPM model                  (slab centroid at the nominal plane, dz <= lambda/12),
  * full-wave FDTD (Tidy3D)         at lambda/10, lambda/20 and lambda/30 (steps per
                                     wavelength inside each medium).

The FDTD detector field is area-averaged onto the design grid.  We report the
model-vs-FDTD discrepancy (complex-field NMSE with one global complex scale,
intensity NMSE with one global scale, Pearson r, SSIM) and the mesh-convergence
error between successive FDTD resolutions.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dbpm_d2nn.model import D2NN
from dbpm_d2nn.fdtd import build_simulation, bin_field, estimate_cost, run
from dbpm_d2nn.metrics import nmse_field, nmse_intensity, pearson, ssim_np

p = argparse.ArgumentParser()
p.add_argument("--export", default="experiments/fdtd/designs_cls/fdtd_export.npz")
p.add_argument("--out_dir", default="experiments/fdtd/single_layer")
p.add_argument("--designs", default="bpm,thin")
p.add_argument("--layer", type=int, default=0)
p.add_argument("--sample", type=int, default=0)
p.add_argument("--src_gap_lam", type=float, default=3.0)
p.add_argument("--z_out_lam", type=float, default=13.0)
p.add_argument("--meshes", default="10,20,30")
p.add_argument("--estimate_only", action="store_true")
p.add_argument("--max_cost", type=float, default=6.0, help="skip simulations above this FlexCredit estimate")
p.add_argument("--window", type=int, default=128,
               help="zero-padded window (pixels) of the scalar models: no periodic wrap-around")
p.add_argument("--n_eval", type=float, default=None,
               help="evaluate the same phase profile as a relief of a different index (heights rescaled by dn_design/dn_eval)")
p.add_argument("--device", default="cuda:0")
args = p.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
d = np.load(args.export)
lam_um, dx_um, n_design = float(d["wavelength_um"]), float(d["dx_um"]), float(d["n_material"])
n = args.n_eval if args.n_eval is not None else n_design
h_rescale = (n_design - 1.0) / (n - 1.0)
lam, dx = lam_um * 1e-6, dx_um * 1e-6
u_in = d["u_in"][args.sample]
H, W = u_in.shape
src_gap, z_out = args.src_gap_lam * lam, args.z_out_lam * lam
results_path = os.path.join(args.out_dir, "results.json")
results = json.load(open(results_path)) if os.path.exists(results_path) else {}
dev = args.device if torch.cuda.is_available() else "cpu"


def scalar_fields(h_m: np.ndarray):
    """Thin and dBPM predictions (complex field at the detector plane) for the given heights.

    The scalar models run on a zero-padded window (transparent outside the aperture) so
    that light leaving the aperture is not wrapped around by the FFT."""
    out = {}
    Wd = max(args.window, W)
    o = (Wd - W) // 2
    for name, ns in (("thin", 1), ("bpm", int(math.ceil(12.0 / (n - 1.0))))):
        m = D2NN(H=Wd, W=Wd, L=1, wavelength=lam, dx=dx, z=src_gap, z_output=z_out, n_material=n, n_sub=ns,
                 axial_mode="anchor", anchor=0.5, aperture=W if Wd > W else None).to(dev)
        # set the layer phase from the height map: phi = 2 pi dn h / lambda
        phi = torch.zeros(Wd, Wd)
        phi[o:o + H, o:o + W] = torch.tensor(2 * math.pi * (n - 1.0) * h_m / lam, dtype=torch.float32)
        m.layers[0].set_phase(phi.to(dev))
        u0 = torch.zeros(Wd, Wd, dtype=torch.complex64)
        u0[o:o + H, o:o + W] = torch.tensor(u_in, dtype=torch.complex64)
        with torch.no_grad():
            I, u = m(u0[None].to(dev), return_field=True)
        out[name] = {"I": I[0, o:o + H, o:o + W].cpu().numpy(), "U": u[0, o:o + H, o:o + W].cpu().numpy(), "n_sub": m.n_sub}
    return out


def compare(I_model, U_model, I_fdtd, Ey_fdtd):
    return {"field_nmse": nmse_field(U_model, Ey_fdtd), "int_nmse": nmse_intensity(I_model, I_fdtd),
            "pearson": pearson(I_model, I_fdtd), "ssim": ssim_np(I_model / I_model.max(), I_fdtd / I_fdtd.max())}


for design in args.designs.split(","):
    h_um = d[f"h_{design}_um"][args.layer] * h_rescale
    T_um = float(d["h_max_um"]) * h_rescale
    sc = scalar_fields(h_um * 1e-6)
    rec = results.get(design, {})
    rec.update({"layer": args.layer, "sample": args.sample, "n": n, "n_design": n_design, "T_um": T_um, "h_mean_um": float(h_um.mean()),
                "bpm_n_sub": sc["bpm"]["n_sub"],
                "thin_vs_bpm": compare(sc["thin"]["I"], sc["thin"]["U"], sc["bpm"]["I"], sc["bpm"]["U"])})
    for mesh in [int(v) for v in args.meshes.split(",")]:
        tag = f"mesh{mesh}"
        if tag in rec and "I_fdtd_file" in rec[tag]:
            zz = np.load(rec[tag]["I_fdtd_file"]); I_f, Ey_f = zz["I"], zz["Ey"]
            np.savez(rec[tag]["I_fdtd_file"], I=I_f, Ey=Ey_f, I_thin=sc["thin"]["I"], U_thin=sc["thin"]["U"],
                     I_bpm=sc["bpm"]["I"], U_bpm=sc["bpm"]["U"])
            rec[tag].update({"thin_vs_fdtd": compare(sc["thin"]["I"], sc["thin"]["U"], I_f, Ey_f),
                             "bpm_vs_fdtd": compare(sc["bpm"]["I"], sc["bpm"]["U"], I_f, Ey_f), "scalar_window": args.window})
            print(f"[{design}] {tag} (FDTD reused): thin-vs-FDTD {rec[tag]['thin_vs_fdtd']} | bpm-vs-FDTD {rec[tag]['bpm_vs_fdtd']}", flush=True)
            results[design] = rec; json.dump(results, open(results_path, "w"), indent=1)
            continue
        b = build_simulation([h_um], dx_um, lam_um, n, z_um=args.z_out_lam * lam_um, z_out_um=args.z_out_lam * lam_um,
                             u_in=u_in, T_um=T_um, anchor=0.5, steps_per_wvl=mesh, source_offset_um=args.src_gap_lam * lam_um,
                             layer_monitors=False)
        sim = b["sim"]
        ncells = int(np.prod([len(c) for c in (sim.grid.boundaries.x, sim.grid.boundaries.y, sim.grid.boundaries.z)]))
        cost = estimate_cost(sim, task_name=f"dbpm_single_{design}_{tag}_est")
        print(f"[{design}] {tag}: grid cells ~{ncells:.3e}, estimated cost {cost:.3f} FlexCredits", flush=True)
        rec[tag] = {"steps_per_wvl": mesh, "n_cells": ncells, "est_cost": cost, "z": b["z"]}
        if args.estimate_only or cost > args.max_cost:
            print(f"[{design}] {tag}: skipped (estimate_only={args.estimate_only}, cost {cost:.2f} > {args.max_cost})")
            continue
        t0 = time.time()
        sd = run(sim, path=os.path.join(args.out_dir, f"{design}_{tag}.hdf5"), task_name=f"dbpm_single_{design}_{tag}")
        I_f, Ey_f = bin_field(sd["detector"], b["aperture_um"], H, W)
        fpath = os.path.join(args.out_dir, f"{design}_{tag}_detector.npz")
        np.savez(fpath, I=I_f, Ey=Ey_f, I_thin=sc["thin"]["I"], U_thin=sc["thin"]["U"], I_bpm=sc["bpm"]["I"],
                 U_bpm=sc["bpm"]["U"])
        rec[tag].update({"I_fdtd_file": fpath, "wall_s": time.time() - t0, "scalar_window": args.window,
                         "thin_vs_fdtd": compare(sc["thin"]["I"], sc["thin"]["U"], I_f, Ey_f),
                         "bpm_vs_fdtd": compare(sc["bpm"]["I"], sc["bpm"]["U"], I_f, Ey_f)})
        try:
            rec[tag]["real_cost"] = float(sd.log.split("FlexCredits")[0].split()[-1]) if hasattr(sd, "log") else None
        except Exception:
            pass
        print(f"[{design}] {tag}: thin-vs-FDTD {rec[tag]['thin_vs_fdtd']} | bpm-vs-FDTD {rec[tag]['bpm_vs_fdtd']}", flush=True)
        results[design] = rec
        json.dump(results, open(results_path, "w"), indent=1)
    # mesh convergence between successive resolutions
    meshes = sorted(int(k[4:]) for k in rec if k.startswith("mesh") and "I_fdtd_file" in rec[k])
    conv = {}
    for a, b_ in zip(meshes[:-1], meshes[1:]):
        Ia = np.load(rec[f"mesh{a}"]["I_fdtd_file"])["I"]; Ib = np.load(rec[f"mesh{b_}"]["I_fdtd_file"])["I"]
        Ea = np.load(rec[f"mesh{a}"]["I_fdtd_file"])["Ey"]; Eb = np.load(rec[f"mesh{b_}"]["I_fdtd_file"])["Ey"]
        conv[f"{a}_vs_{b_}"] = {"rel_l2_intensity": float(np.linalg.norm(Ia - Ib) / np.linalg.norm(Ib)),
                                "int_nmse_scaled": nmse_intensity(Ia, Ib), "field_nmse": nmse_field(Ea, Eb),
                                "pearson": pearson(Ia, Ib)}
    rec["mesh_convergence"] = conv
    results[design] = rec
    json.dump(results, open(results_path, "w"), indent=1)
    print(f"[{design}] mesh convergence: {conv}")
print("done")
