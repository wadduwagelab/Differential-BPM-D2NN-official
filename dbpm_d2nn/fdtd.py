"""Tidy3D FDTD helpers used for the full-wave validation.

* :func:`scalar_to_em_source` converts the scalar input field of the design model
  into a Maxwell-consistent forward-propagating (E, H) dataset for a
  ``CustomFieldSource`` (linear y polarisation, evanescent components removed).
* :func:`build_simulation` builds the 3-D dielectric structure from exported height
  maps (one box per pixel), places every slab so that the plane at fraction
  ``anchor`` of the nominal slab thickness coincides with the nominal layer plane
  (anchor = 0.5: centroid) and adds
  field monitors at the detector plane (and optionally behind every layer).
* :func:`bin_field` area-averages a monitor onto the design pixel grid and returns
  the intensity |E|^2 and the dominant complex component Ey.

Units: Tidy3D works in micrometres; all arguments of this module are in micrometres.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import tidy3d as td
from tidy3d import (Box, CustomFieldSource, FieldDataset, FieldMonitor, GaussianPulse, GeometryGroup, Medium,
                    ScalarFieldDataArray, Simulation, Structure)


def pixel_centers(aperture_um: float, n: int) -> np.ndarray:
    d = aperture_um / n
    return np.linspace(-aperture_um / 2 + d / 2, aperture_um / 2 - d / 2, n)


def scalar_to_em_source(u_yx: np.ndarray, xs: np.ndarray, ys: np.ndarray, z_source: float, freq0: float,
                        wavelength_um: float) -> FieldDataset:
    """Scalar field u(y, x) -> forward (+z) plane-wave-consistent (E, H) with E along y."""
    eta0 = td.ETA_0
    Ey_xy = np.transpose(u_yx).astype(np.complex128)  # (nx, ny)
    nx, ny = Ey_xy.shape
    dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])
    FX, FY = np.meshgrid(np.fft.fftfreq(nx, d=dx), np.fft.fftfreq(ny, d=dy), indexing="ij")
    alpha, beta = wavelength_um * FX, wavelength_um * FY
    gamma2 = 1.0 - alpha ** 2 - beta ** 2
    prop = gamma2 > 1e-8
    gamma = np.zeros_like(gamma2, dtype=np.complex128)
    gamma[prop] = np.sqrt(gamma2[prop])
    Ey_k = np.fft.fft2(Ey_xy)
    Ey_k[~prop] = 0.0
    Ex_k = np.zeros_like(Ey_k)
    Ez_k = np.zeros_like(Ey_k)
    Ez_k[prop] = -(beta[prop] / gamma[prop]) * Ey_k[prop]
    Hx_k, Hy_k, Hz_k = (np.zeros_like(Ey_k) for _ in range(3))
    Hx_k[prop] = (beta[prop] * Ez_k[prop] - gamma[prop] * Ey_k[prop]) / eta0
    Hy_k[prop] = (gamma[prop] * Ex_k[prop] - alpha[prop] * Ez_k[prop]) / eta0
    Hz_k[prop] = (alpha[prop] * Ey_k[prop] - beta[prop] * Ex_k[prop]) / eta0
    comps = {k: np.fft.ifft2(v) for k, v in dict(Ex=Ex_k, Ey=Ey_k, Ez=Ez_k, Hx=Hx_k, Hy=Hy_k, Hz=Hz_k).items()}
    coords = {"x": xs, "y": ys, "z": [z_source], "f": [freq0]}
    return FieldDataset(**{k: ScalarFieldDataArray(v[:, :, None, None], coords=coords) for k, v in comps.items()})


def build_simulation(heights_um: Sequence[np.ndarray], dx_um: float, wavelength_um: float, n_material: float,
                     z_um: float, z_out_um: float, u_in: np.ndarray, *, T_um: Optional[float] = None,
                     anchor: float = 0.5, steps_per_wvl: int = 10, pad_um: Optional[float] = None,
                     layer_monitors: bool = False, yz_monitor: bool = False, run_time: float = 1.0e-12,
                     shutoff: float = 1e-5, source_offset_um: Optional[float] = None) -> Dict:
    """Build the Tidy3D simulation for a stack of relief layers.

    heights_um : list of L arrays (H, W) with the relief height of every pixel [um]
    T_um       : nominal slab thickness (defaults to the maximum height over all layers)
    anchor     : fraction of T at which the nominal plane z_l lies inside the slab (0.5 = centroid)
    steps_per_wvl : minimum grid steps per wavelength *inside each medium* (Tidy3D AutoGrid)
    Returns dict(sim=..., z=dict(source, planes, detector), aperture_um=...)
    """
    L = len(heights_um)
    H, W = heights_um[0].shape
    aperture_um = W * dx_um
    T = float(T_um) if T_um is not None else float(max(np.max(h) for h in heights_um))
    pad = pad_um if pad_um is not None else 2.0 * wavelength_um
    z_source = pad / 2
    src_gap = source_offset_um if source_offset_um is not None else z_um  # source -> nominal plane 1
    planes = [z_source + src_gap + l * z_um for l in range(L)]
    entrances = [zp - anchor * T for zp in planes]
    z_detector = planes[-1] + z_out_um
    for l in range(1, L):
        if entrances[l] < entrances[l - 1] + T - 1e-9:
            raise ValueError("slabs overlap: z spacing smaller than slab thickness")
    if entrances[0] <= z_source + 0.25 * wavelength_um:
        raise ValueError("first slab too close to the source plane")
    xs, ys = pixel_centers(aperture_um, W), pixel_centers(aperture_um, H)
    medium = Medium(permittivity=n_material ** 2)
    boxes = []
    for l in range(L):
        h = np.asarray(heights_um[l], dtype=float)
        for i in range(H):
            for j in range(W):
                if h[i, j] <= 1e-6:
                    continue
                boxes.append(Box(center=(xs[j], ys[i], entrances[l] + h[i, j] / 2), size=(dx_um, dx_um, h[i, j])))
    structures = [Structure(geometry=GeometryGroup(geometries=tuple(boxes)), medium=medium)]
    freq0 = td.C_0 / wavelength_um
    dataset = scalar_to_em_source(u_in, xs, ys, z_source, freq0, wavelength_um)
    source = CustomFieldSource(source_time=GaussianPulse(freq0=freq0, fwidth=freq0 / 10),
                               center=(0, 0, z_source), size=(aperture_um, aperture_um, 0), field_dataset=dataset)
    sx = sy = aperture_um + 2 * pad
    sz = z_detector + pad
    monitors = [FieldMonitor(center=(0, 0, z_detector), size=(sx, sy, 0), freqs=[freq0], name="detector"),
                FieldMonitor(center=(0, 0, z_source + 0.5 * wavelength_um), size=(sx, sy, 0), freqs=[freq0],
                             name="after_source")]
    if layer_monitors:
        for l in range(L):
            z_after = entrances[l] + T + 0.5 * wavelength_um
            monitors.append(FieldMonitor(center=(0, 0, z_after), size=(sx, sy, 0), freqs=[freq0], name=f"after_layer_{l+1}"))
    if yz_monitor:
        monitors.append(FieldMonitor(center=(0, 0, sz / 2), size=(0, sy, sz), freqs=[freq0], name="yz"))
    grid_spec = td.GridSpec.auto(min_steps_per_wvl=steps_per_wvl, wavelength=wavelength_um)
    sim = Simulation(center=(0, 0, sz / 2), size=(sx, sy, sz), grid_spec=grid_spec, structures=structures,
                     sources=[source], monitors=monitors, run_time=run_time, shutoff=shutoff)
    return {"sim": sim, "z": {"source": z_source, "planes": planes, "entrances": entrances, "T": T,
                              "detector": z_detector, "after_layers": [e + T + 0.5 * wavelength_um for e in entrances]},
            "aperture_um": aperture_um, "n_boxes": len(boxes), "freq0": freq0}


def bin_field(field_data, aperture_um: float, H: int, W: int, oversample: int = 6):
    """Area-average a planar FieldMonitor onto the (H, W) design pixel grid.

    Returns (intensity[H, W], Ey[H, W] complex).  Row index = y, column index = x
    (image convention used by the design model)."""
    d = aperture_um / W
    fine_x = np.linspace(-aperture_um / 2 + d / (2 * oversample), aperture_um / 2 - d / (2 * oversample), W * oversample)
    fine_y = np.linspace(-aperture_um / 2 + d / (2 * oversample), aperture_um / 2 - d / (2 * oversample), H * oversample)
    comps = {}
    for c in ("Ex", "Ey", "Ez"):
        arr = getattr(field_data, c).isel(f=0)
        arr = arr.squeeze(drop=True)
        arr = arr.interp(x=fine_x, y=fine_y, method="linear")
        comps[c] = np.asarray(arr.values).T  # -> (y, x)
    I_fine = sum(np.abs(v) ** 2 for v in comps.values())
    I = I_fine.reshape(H, oversample, W, oversample).mean(axis=(1, 3))
    Ey = comps["Ey"].reshape(H, oversample, W, oversample).mean(axis=(1, 3))
    return I, Ey


def estimate_cost(sim: Simulation, task_name: str = "estimate") -> float:
    from tidy3d import web
    tid = web.upload(sim, task_name=task_name, verbose=False)
    cost = web.estimate_cost(tid, verbose=False)
    try:
        web.delete(tid)
    except Exception:
        pass
    return float(cost)


def run(sim: Simulation, path: str, task_name: str, verbose: bool = False):
    from tidy3d import web
    return web.run(sim, task_name=task_name, path=path, verbose=verbose)
