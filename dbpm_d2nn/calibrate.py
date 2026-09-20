"""One-parameter post-hoc calibrations of a thin-layer-trained design that is
evaluated with the volumetric (dBPM) forward model.

Two scalar corrections are considered (both fitted on the *validation* split
only, one value for the whole network, all layers, all images):

* global axial offset  delta_z : rigid shift of all slabs relative to the input
                                  and detector planes (centroid-anchored slabs),
* global height scale  s       : h_corrected(x,y) = s * h_thin(x,y)  (Eq. 3).

``objective(model) -> float`` must return a scalar that is *lower is better*
(e.g. -accuracy or MSE on the validation split).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .model import D2NN


def _eval_grid(base: D2NN, objective: Callable[[D2NN], float], param: str, values, fixed: Dict) -> List[Tuple[float, float]]:
    out = []
    for v in values:
        kw = dict(fixed)
        kw[param] = float(v)
        try:
            m = base.clone(**kw)
        except ValueError:  # negative gap -> infeasible
            out.append((float(v), float("inf")))
            continue
        out.append((float(v), float(objective(m))))
    return out


def _search(base, objective, param, lo, hi, fixed, n_coarse=11, n_fine=7, refine_frac=0.12):
    coarse = _eval_grid(base, objective, param, np.linspace(lo, hi, n_coarse), fixed)
    best_v, best_o = min(coarse, key=lambda t: t[1])
    half = refine_frac * (hi - lo)
    fine = _eval_grid(base, objective, param, np.linspace(max(lo, best_v - half), min(hi, best_v + half), n_fine), fixed)
    allpts = coarse + fine
    best_v, best_o = min(allpts, key=lambda t: t[1])
    return best_v, best_o, allpts


def fit_axial_shift(base: D2NN, objective, shift_range_frac: float = 1.0, **fixed):
    """Fit delta_z in [-f*T, +f*T] (T = slab thickness) for a centroid-anchored model."""
    T = base.thickness
    lo, hi = -shift_range_frac * T, shift_range_frac * T
    # feasibility: first gap = z - a T + shift >= 0 ; last gap = z_out - (1-a) T - shift >= 0
    a = base.anchor
    lo = max(lo, -(base.z - a * T) + 1e-12)
    hi = min(hi, base.z_output - (1 - a) * T - 1e-12)
    v, o, pts = _search(base, objective, "axial_shift", lo, hi, fixed)
    return {"axial_shift": v, "objective": o, "grid": pts, "T": T}


def fit_height_scale(base: D2NN, objective, s_min: float = 0.5, s_max: float = 1.5, **fixed):
    v, o, pts = _search(base, objective, "height_scale", s_min, s_max, fixed)
    return {"height_scale": v, "objective": o, "grid": pts}


def fit_joint(base: D2NN, objective, rounds: int = 2, s_min=0.5, s_max=1.5, shift_range_frac=1.0):
    """Coordinate descent over (height_scale, axial_shift)."""
    s, dz = base.height_scale, base.axial_shift
    trace = []
    for r in range(rounds):
        rs = fit_height_scale(base, objective, s_min, s_max, axial_shift=dz)
        s = rs["height_scale"]
        # shift range must follow the (possibly larger) thickness s*h_max
        tmp = base.clone(height_scale=s, axial_shift=0.0)
        rd = fit_axial_shift(tmp, objective, shift_range_frac, height_scale=s)
        dz = rd["axial_shift"]
        trace.append({"round": r + 1, "height_scale": s, "axial_shift": dz, "objective": rd["objective"]})
    return {"height_scale": s, "axial_shift": dz, "objective": trace[-1]["objective"], "trace": trace}
