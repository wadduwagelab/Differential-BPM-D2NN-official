"""Image / field metrics and fabrication-geometry statistics."""
from __future__ import annotations

import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- image metrics
def ssim_1ch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5,
             size_average: bool = True) -> torch.Tensor:
    """SSIM for single-channel batches [B,H,W] with data range 1 (same as the paper)."""
    min_dim = min(x.shape[-2], x.shape[-1])
    if window_size > min_dim:
        window_size = min_dim if min_dim % 2 == 1 else min_dim - 1
    g = torch.tensor([math.exp(-(t - window_size // 2) ** 2 / (2 * sigma ** 2)) for t in range(window_size)],
                     device=x.device, dtype=torch.float32)
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    x_, y_ = x.unsqueeze(1).float(), y.unsqueeze(1).float()
    p = window_size // 2
    mu_x, mu_y = F.conv2d(x_, w, padding=p), F.conv2d(y_, w, padding=p)
    sxx = F.conv2d(x_ * x_, w, padding=p) - mu_x ** 2
    syy = F.conv2d(y_ * y_, w, padding=p) - mu_y ** 2
    sxy = F.conv2d(x_ * y_, w, padding=p) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return m.mean() if size_average else m.mean(dim=(1, 2, 3))


def best_linear_rescale(I: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-image optimal global gain alpha = <I,T>/<I,I>."""
    If, Tf = I.flatten(1), target.flatten(1)
    alpha = (If * Tf).sum(1) / (If * If).sum(1).clamp(min=eps)
    return I * alpha.view(-1, 1, 1)


def nmse_intensity(I_model: np.ndarray, I_ref: np.ndarray, fit_scale: bool = True) -> float:
    """||alpha I_model - I_ref||^2 / ||I_ref||^2 with optional global alpha."""
    a, b = I_model.ravel().astype(np.float64), I_ref.ravel().astype(np.float64)
    alpha = float(np.dot(a, b) / (np.dot(a, a) + 1e-30)) if fit_scale else 1.0
    return float(np.sum((alpha * a - b) ** 2) / (np.sum(b ** 2) + 1e-30))


def nmse_field(E_model: np.ndarray, E_ref: np.ndarray) -> float:
    """Complex-field NMSE with one global complex scalar alpha (amplitude + phase)."""
    a, b = E_model.ravel().astype(np.complex128), E_ref.ravel().astype(np.complex128)
    alpha = np.vdot(a, b) / (np.vdot(a, a) + 1e-30)
    return float(np.sum(np.abs(alpha * a - b) ** 2) / (np.sum(np.abs(b) ** 2) + 1e-30))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() < 1e-30 or b.std() < 1e-30:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rel_l2(I_model: np.ndarray, I_ref: np.ndarray) -> float:
    return math.sqrt(nmse_intensity(I_model, I_ref, fit_scale=True))


def nrmse(I_model: np.ndarray, I_ref: np.ndarray, eps: float = 1e-12) -> float:
    a, b = I_model.ravel().astype(np.float64), I_ref.ravel().astype(np.float64)
    alpha = float(np.dot(a, b) / (np.dot(a, a) + 1e-30))
    rmse = math.sqrt(np.mean((alpha * a - b) ** 2))
    return float(rmse / (b.max() - b.min() + eps))


def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    ta = torch.as_tensor(a, dtype=torch.float32)[None]
    tb = torch.as_tensor(b, dtype=torch.float32)[None]
    ta = ta / (ta.max() + 1e-30); tb = tb / (tb.max() + 1e-30)
    return float(ssim_1ch(ta, tb))


# ----------------------------------------------------------------------------- fabrication geometry
def fabrication_statistics(height_m: np.ndarray, dx_m: float, wavelength_m: float,
                           h_max_m: float, wrap_frac: float = 0.5) -> Dict[str, float]:
    """Geometry statistics of one height map (metres).

    * nearest-neighbour height differences |dh| (x and y), reported in nm and in lambda,
    * local sidewall slope angle theta = atan(|dh| / dx),
    * fraction of neighbour pairs that are 2pi-wrap edges (|dh| > wrap_frac * h_max),
    * maximum aspect ratio h_max / dx (one-pixel-wide full-height feature),
    * minimum lateral feature size = one pixel pitch.
    """
    h = np.asarray(height_m, dtype=np.float64)
    dhx = np.abs(np.diff(h, axis=1)).ravel()
    dhy = np.abs(np.diff(h, axis=0)).ravel()
    dh = np.concatenate([dhx, dhy])
    theta = np.degrees(np.arctan(dh / dx_m))
    q = lambda v, p: float(np.percentile(v, p))
    return {
        "dh_median_nm": q(dh, 50) * 1e9, "dh_p95_nm": q(dh, 95) * 1e9, "dh_max_nm": float(dh.max()) * 1e9,
        "dh_median_lam": q(dh, 50) / wavelength_m, "dh_p95_lam": q(dh, 95) / wavelength_m,
        "dh_max_lam": float(dh.max()) / wavelength_m,
        "dh_mean_nm": float(dh.mean()) * 1e9,
        "slope_median_deg": q(theta, 50), "slope_p95_deg": q(theta, 95), "slope_max_deg": float(theta.max()),
        "frac_slope_gt_45deg": float(np.mean(theta > 45.0)),
        "frac_slope_gt_80deg": float(np.mean(theta > 80.0)),
        "wrap_edge_fraction": float(np.mean(dh > wrap_frac * h_max_m)),
        "aspect_ratio_max": float(h_max_m / dx_m),
        "aspect_ratio_p95": q(dh, 95) / dx_m,
        "min_lateral_feature_nm": float(dx_m) * 1e9,
        "h_max_nm": float(h_max_m) * 1e9, "height_mean_nm": float(h.mean()) * 1e9,
        "height_std_nm": float(h.std()) * 1e9,
    }


def total_variation(h: torch.Tensor) -> torch.Tensor:
    """Mean absolute nearest-neighbour height difference (slope penalty)."""
    return (torch.abs(h[:, 1:] - h[:, :-1]).mean() + torch.abs(h[1:, :] - h[:-1, :]).mean()) / 2.0
