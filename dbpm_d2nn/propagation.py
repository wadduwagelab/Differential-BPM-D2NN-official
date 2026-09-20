"""Angular-spectrum-method (ASM) free-space propagation (Eqs. 4-7 of the paper).

The transfer function H(fx, fy; z) = exp(i k0 z gamma) with
gamma = sqrt(1 - (lambda fx)^2 - (lambda fy)^2) (analytically continued to
i*sqrt(...) for evanescent components) is cached per (H, W, dx, lambda, z,
device) because the dBPM layer re-uses the same sub-slice propagator N_sub
times per layer.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

_TF_CACHE: Dict[Tuple, torch.Tensor] = {}
_MAX_CACHE = 1024


def asm_transfer_function(H: int, W: int, dx: float, wavelength: float, z: float,
                          device, dtype=torch.complex64) -> torch.Tensor:
    """Return the (cached) ASM transfer function for a propagation distance z."""
    key = (H, W, float(dx), float(wavelength), float(z), str(device), dtype)
    tf = _TF_CACHE.get(key)
    if tf is None:
        fx = torch.fft.fftfreq(W, d=dx, device=device, dtype=torch.float64)
        fy = torch.fft.fftfreq(H, d=dx, device=device, dtype=torch.float64)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        root = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
        gamma = torch.where(
            root >= 0,
            torch.sqrt(root.clamp(min=0.0)).to(torch.complex128),
            1j * torch.sqrt((-root).clamp(min=0.0)).to(torch.complex128),
        )
        k0 = 2.0 * math.pi / wavelength
        tf = torch.exp(1j * k0 * z * gamma).to(dtype)
        if len(_TF_CACHE) >= _MAX_CACHE:
            _TF_CACHE.clear()
        _TF_CACHE[key] = tf
    return tf


def asm_propagate(u: torch.Tensor, wavelength: float, dx: float, z: float) -> torch.Tensor:
    """Propagate a batch of complex fields u[B, H, W] by a distance z (metres).

    z must be >= 0 (negative distances would amplify evanescent components).
    """
    if z < -1e-15:
        raise ValueError(f"negative propagation distance z={z:.3e} m")
    if abs(z) < 1e-15:
        return u
    B, H, W = u.shape
    tf = asm_transfer_function(H, W, dx, wavelength, z, u.device, u.dtype)
    return torch.fft.ifft2(torch.fft.fft2(u) * tf[None, :, :])


def clear_cache():
    _TF_CACHE.clear()
