"""D2NN forward model with thin-layer and differentiable-BPM (dBPM) layers.

Geometry conventions
--------------------
Nominal layer planes are at z_l = l * z  (l = 1..L) behind the input plane and the
detector is at z_L + z_output.  In the thin-layer model every layer is an
infinitesimal phase mask located exactly at z_l.

A dBPM layer occupies a slab of thickness T = height_scale * h_max, where
h_max = lambda * phase_bound / (2 pi delta_n) is the relief height for the full
phase range.  How the slab is positioned relative to the nominal plane is
controlled by ``axial_mode``:

* ``"legacy"``   - the slab *entrance* is at z_l, the free-space gap between
                   slabs is z - T, and the detector is z_output behind the
                   exit of the last slab (i.e. the whole stack behind layer 1
                   is T longer than in the thin model).
* ``"anchor"``   - the slab is placed such that the plane at fraction
                   ``anchor`` of its thickness coincides with z_l
                   (anchor = 0: entrance, 0.5: centroid, 1: exit).  Input->
                   layer-1 and layer-L->detector distances are shortened so that
                   the nominal planes and the detector stay exactly where they
                   are in the thin model.  A global rigid shift ``axial_shift``
                   (metres, +z = towards the detector) of all slabs can be added.

``height_scale`` multiplies the exported height map h(x,y) (Eq. 3) by a single
global factor s (control for an effective-index error of the phase-to-height
conversion).  The slab thickness scales accordingly and the sub-slice thickness
dz is kept fixed (n_sub = ceil(s * n_sub_nominal)).
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .propagation import asm_propagate


class PhaseLayer(nn.Module):
    """Trainable phase profile of one diffractive layer.

    ``param="wrap"``    : phi = mod(2 pi beta, 2 pi)            (Eq. 1; classification)
    ``param="sigmoid"`` : phi = phase_bound * sigmoid(beta)     (imaging notebooks)
    """

    def __init__(self, H: int, W: int, phase_bound: float = 2 * math.pi,
                 param: str = "wrap", init_scale: float = 0.01, generator=None):
        super().__init__()
        self.phase_bound = float(phase_bound)
        self.param = param
        noise = torch.randn(H, W, generator=generator)
        if param == "wrap":
            # paper-style init: beta ~ 0.5 -> phi ~ pi
            self.raw = nn.Parameter(0.5 + init_scale * noise)
        elif param == "sigmoid":
            self.raw = nn.Parameter(init_scale * noise)
        else:
            raise ValueError(param)

    def phase(self) -> torch.Tensor:
        if self.param == "wrap":
            return torch.remainder(2.0 * math.pi * self.raw, 2.0 * math.pi)
        return self.phase_bound * torch.sigmoid(self.raw)

    @torch.no_grad()
    def set_phase(self, phi: torch.Tensor):
        """Set the layer so that ``phase()`` returns phi (phi in [0, phase_bound))."""
        phi = phi.to(self.raw.device, torch.float32)
        if self.param == "wrap":
            self.raw.data = phi / (2.0 * math.pi)
        else:
            t = torch.clamp(phi / self.phase_bound, 1e-6, 1.0 - 1e-6)
            self.raw.data = torch.logit(t)


class D2NN(nn.Module):
    def __init__(self, H: int, W: int, L: int, wavelength: float, dx: float, z: float,
                 z_output: Optional[float] = None, n_material: float = 1.5,
                 n_sub: int = 15, sigmoid_temp: Optional[float] = None,
                 temp_factor: float = 0.05, phase_bound: float = 2 * math.pi,
                 phase_param: str = "wrap", axial_mode: str = "legacy",
                 anchor: float = 0.5, axial_shift: float = 0.0,
                 height_scale: float = 1.0, dz: Optional[float] = None,
                 aperture: Optional[int] = None, seed: Optional[int] = None):
        super().__init__()
        self.H, self.W, self.L = H, W, L
        # optional square aperture (pixels): the phase is trainable only inside the central
        # aperture x aperture region; outside it is 0 (transparent).  This removes the
        # periodic wrap-around of the FFT propagation (used for the FDTD-matched designs).
        self.aperture = int(aperture) if aperture else None
        if self.aperture:
            mask = torch.zeros(H, W)
            o_h, o_w = (H - self.aperture) // 2, (W - self.aperture) // 2
            mask[o_h:o_h + self.aperture, o_w:o_w + self.aperture] = 1.0
            self.register_buffer("phase_mask", mask)
        else:
            self.phase_mask = None
        self.wavelength, self.dx, self.z = float(wavelength), float(dx), float(z)
        self.z_output = float(z_output) if z_output is not None else float(z)
        self.n_material = float(n_material)
        self.delta_n = self.n_material - 1.0
        self.phase_bound = float(phase_bound)
        self.phase_param = phase_param
        self.h_max = self.wavelength * self.phase_bound / (2.0 * math.pi * self.delta_n)
        self.height_scale = float(height_scale)
        self.axial_mode = axial_mode
        self.anchor = float(anchor)
        self.axial_shift = float(axial_shift)
        self.temp_factor = float(temp_factor)
        self.is_thin = (n_sub is None) or (n_sub <= 1)
        self.n_sub_nominal = 1 if self.is_thin else int(n_sub)
        self.dz_nominal = dz
        # sub-slice discretisation
        if not self.is_thin:
            T = self.thickness
            if dz is not None:
                self.n_sub = max(1, int(math.ceil(T / dz - 1e-9)))
            else:
                self.n_sub = max(1, int(math.ceil(n_sub * self.height_scale - 1e-9)))
            self.dz_sub = T / self.n_sub
            self.sigmoid_temp = float(sigmoid_temp) if sigmoid_temp is not None else self.temp_factor * self.dz_sub
        else:
            self.n_sub, self.dz_sub, self.sigmoid_temp = 1, 0.0, 0.0
        gen = None
        if seed is not None:
            gen = torch.Generator().manual_seed(int(seed))
        self.layers = nn.ModuleList(
            [PhaseLayer(H, W, phase_bound=phase_bound, param=phase_param, generator=gen) for _ in range(L)]
        )
        self._check_gaps()

    # ------------------------------------------------------------------ geometry
    @property
    def thickness(self) -> float:
        """Axial thickness of the volumetric slab (0 for the thin model)."""
        return 0.0 if self.is_thin else self.height_scale * self.h_max

    def gaps(self):
        """Return (input->layer1, between slabs, last slab->detector) distances."""
        if self.is_thin:
            return self.z, self.z, self.z_output
        T = self.thickness
        if self.axial_mode == "legacy":
            return self.z, self.z - T, self.z_output
        if self.axial_mode == "anchor":
            a = self.anchor
            first = self.z - a * T + self.axial_shift
            last = self.z_output - (1.0 - a) * T - self.axial_shift
            return first, self.z - T, last
        raise ValueError(self.axial_mode)

    def _check_gaps(self):
        for g in self.gaps():
            if g < -1e-12:
                raise ValueError(
                    f"negative free-space gap {g:.3e} m (z={self.z:.3e}, z_out={self.z_output:.3e}, "
                    f"T={self.thickness:.3e}, shift={self.axial_shift:.3e}, mode={self.axial_mode})")

    def plane_positions(self):
        """Axial coordinates (m) of nominal planes, slab entrance/exit and detector."""
        first, mid, last = self.gaps()
        T = self.thickness
        out = []
        zc = first
        for l in range(self.L):
            out.append({"layer": l + 1, "entrance": zc, "exit": zc + T, "centroid": zc + T / 2,
                        "nominal": (l + 1) * self.z})
            zc += T + (mid if l < self.L - 1 else last)
        return out, zc  # detector position

    # ------------------------------------------------------------------ heights
    def phases(self) -> List[torch.Tensor]:
        if self.phase_mask is not None:
            return [layer.phase() * self.phase_mask for layer in self.layers]
        return [layer.phase() for layer in self.layers]

    def crop(self, x: torch.Tensor) -> torch.Tensor:
        """Central aperture x aperture crop of a [..., H, W] tensor (identity without aperture)."""
        if not self.aperture:
            return x
        o_h, o_w = (self.H - self.aperture) // 2, (self.W - self.aperture) // 2
        return x[..., o_h:o_h + self.aperture, o_w:o_w + self.aperture]

    def heights(self) -> List[torch.Tensor]:
        """Physical height maps h_l(x,y) in metres (Eq. 3, times height_scale)."""
        c = self.height_scale * self.wavelength / (2.0 * math.pi * self.delta_n)
        return [c * ph for ph in self.phases()]

    @torch.no_grad()
    def set_phases(self, phases):
        for layer, ph in zip(self.layers, phases):
            layer.set_phase(torch.as_tensor(ph))

    @torch.no_grad()
    def load_layers_from(self, other: "D2NN"):
        """Copy the trainable phase parameters of another model (same H, W, L, param)."""
        for a, b in zip(self.layers, other.layers):
            if a.param == b.param:
                a.raw.data.copy_(b.raw.data)
            else:
                a.set_phase(b.phase())

    # ------------------------------------------------------------------ forward
    def _layer(self, u: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        if self.is_thin:
            return u * torch.exp(1j * phase)[None]
        height = self.height_scale * self.wavelength / (2.0 * math.pi * self.delta_n) * phase
        k0 = 2.0 * math.pi / self.wavelength
        base = k0 * self.delta_n * self.dz_sub  # phase of one fully occupied sub-slice
        for s in range(self.n_sub):
            z_s = (s + 0.5) * self.dz_sub
            m = torch.sigmoid((height - z_s) / self.sigmoid_temp)  # soft occupancy (Eq. 11)
            u = u * torch.exp(1j * (base * m))[None]
            u = asm_propagate(u, self.wavelength, self.dx, self.dz_sub)
        return u

    def forward(self, u0: torch.Tensor, return_field: bool = False):
        """u0: complex input field [B, H, W]. Returns detector intensity (and field)."""
        if not u0.is_complex():
            u0 = torch.exp(1j * u0.to(torch.float32))
        first, mid, last = self.gaps()
        u = asm_propagate(u0, self.wavelength, self.dx, first)
        for l, phase in enumerate(self.phases()):
            u = self._layer(u, phase)
            u = asm_propagate(u, self.wavelength, self.dx, last if l == self.L - 1 else mid)
        I = u.real ** 2 + u.imag ** 2
        return (I, u) if return_field else I

    # ------------------------------------------------------------------ io
    def config(self) -> dict:
        return {
            "H": self.H, "W": self.W, "L": self.L, "wavelength": self.wavelength, "dx": self.dx,
            "z": self.z, "z_output": self.z_output, "n_material": self.n_material,
            "n_sub": self.n_sub_nominal, "dz": self.dz_nominal,
            "sigmoid_temp": self.sigmoid_temp if not self.is_thin else None,
            "temp_factor": self.temp_factor, "phase_bound": self.phase_bound,
            "phase_param": self.phase_param, "axial_mode": self.axial_mode, "anchor": self.anchor,
            "axial_shift": self.axial_shift, "height_scale": self.height_scale, "aperture": self.aperture,
        }

    @classmethod
    def from_config(cls, cfg: dict, **overrides) -> "D2NN":
        cfg = dict(cfg)
        # a change of the slice discretisation invalidates the stored absolute temperature
        if any(k in overrides for k in ("n_sub", "dz", "temp_factor")) and "sigmoid_temp" not in overrides:
            cfg["sigmoid_temp"] = None
        if "dz" in overrides and overrides["dz"] is not None:
            cfg["n_sub"] = 2  # any value > 1: dz takes precedence
        if "n_sub" in overrides and "dz" not in overrides:
            cfg["dz"] = None
        cfg.update(overrides)
        cfg.pop("seed", None)
        return cls(**cfg)

    def clone(self, **overrides) -> "D2NN":
        """New model with the same phases but (possibly) different forward settings."""
        m = D2NN.from_config(self.config(), **overrides).to(next(self.parameters()).device)
        m.load_layers_from(self)
        return m


def save_checkpoint(model: D2NN, path: str, **meta):
    torch.save({"format": "dbpm_d2nn.v1", "config": model.config(),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "meta": meta}, path)


def load_checkpoint(path: str, device="cpu", **overrides) -> D2NN:
    """Load a checkpoint written by ``save_checkpoint`` or by the original notebooks."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "dbpm_d2nn.v1":
        return load_paper_checkpoint(path, device, **overrides)
    m = D2NN.from_config(ckpt["config"], **overrides).to(device)
    m.load_state_dict(ckpt["state_dict"], strict=True)
    m.eval()
    return m


def load_paper_checkpoint(path: str, device="cpu", **overrides) -> D2NN:
    """Load a checkpoint written by the original notebooks (config + state_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    n_sub = int(c["n_sub"])
    m = D2NN(H=c["H"], W=c["W"], L=c["L"], wavelength=c["wavelength"], dx=c["dx"], z=c["z"],
             z_output=c.get("z_output", c["z"]), n_material=c["n_material"], n_sub=n_sub,
             sigmoid_temp=c["sigmoid_temp"] if n_sub > 1 else None,
             phase_bound=c["phase_bound"], phase_param=overrides.pop("phase_param", "wrap"),
             axial_mode="legacy", **overrides).to(device)
    sd = {k: v for k, v in ckpt["state_dict"].items() if k.startswith("layers.")}
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m
