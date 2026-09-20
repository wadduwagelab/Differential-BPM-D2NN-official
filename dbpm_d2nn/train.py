"""Training and fine-tuning loops (paper protocol).

Classification: Adam, lr 1e-3, batch 128, 10 epochs, scaled-logit cross entropy.
Imaging       : Adam, lr 1e-3, batch 4, 50 epochs, MSE(I_out, A^2), grad-norm clip 1.
dBPM training : sigmoid temperature annealed geometrically from 0.5 dz to 0.05 dz.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .data import encode, patch_energies
from .evaluate import sce_logits
from .metrics import total_variation


def _anneal(model, ep, epochs, temp_start, temp_end):
    if model.is_thin or temp_start is None:
        return
    frac = (ep - 1) / max(epochs - 1, 1)
    model.sigmoid_temp = temp_start * (temp_end / temp_start) ** frac


def train_classification(model, train_loader, patches, encoding: str, device, epochs: int = 10,
                         lr: float = 1e-3, anneal: bool = True, temp_start_factor: float = 0.5,
                         temp_end_factor: float = 0.05, tv_weight: float = 0.0,
                         epoch_callback: Optional[Callable[[int, Dict], None]] = None,
                         log: Optional[Callable[[str], None]] = print) -> List[Dict]:
    """Train (or fine-tune) a D2NN classifier.  Returns per-epoch history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ts = temp_start_factor * model.dz_sub if (anneal and not model.is_thin) else None
    te = temp_end_factor * model.dz_sub if (anneal and not model.is_thin) else None
    history = []
    for ep in range(1, epochs + 1):
        _anneal(model, ep, epochs, ts, te)
        model.train()
        run, nb, t0 = 0.0, 0, time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            I = model(encode(x, encoding))
            E = patch_energies(I, patches)
            loss = F.cross_entropy(sce_logits(E), y)
            if tv_weight > 0:
                loss = loss + tv_weight * sum(total_variation(h / model.h_max) for h in model.heights()) / model.L
            loss.backward()
            opt.step()
            run += loss.item(); nb += 1
        rec = {"epoch": ep, "train_loss": run / max(nb, 1), "sigmoid_temp": model.sigmoid_temp,
               "time_s": time.time() - t0}
        if epoch_callback is not None:
            extra = epoch_callback(ep, rec)
            if extra:
                rec.update(extra)
        history.append(rec)
        if log:
            log(f"  epoch {ep:02d} loss={rec['train_loss']:.4f} " +
                " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k.startswith(("val", "test"))) +
                f" ({rec['time_s']:.0f}s)")
    return history


def train_imaging(model, train_loader, device, epochs: int = 50, lr: float = 1e-3, anneal: bool = True,
                  temp_start_factor: float = 0.5, temp_end_factor: float = 0.05, clip: float = 1.0,
                  tv_weight: float = 0.0, epoch_callback: Optional[Callable[[int, Dict], None]] = None,
                  log: Optional[Callable[[str], None]] = print) -> List[Dict]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ts = temp_start_factor * model.dz_sub if (anneal and not model.is_thin) else None
    te = temp_end_factor * model.dz_sub if (anneal and not model.is_thin) else None
    history = []
    for ep in range(1, epochs + 1):
        _anneal(model, ep, epochs, ts, te)
        model.train()
        run, nb, t0 = 0.0, 0, time.time()
        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)
            target = x.squeeze(1) ** 2
            opt.zero_grad(set_to_none=True)
            I = model(encode(x, "amplitude"))
            loss = F.mse_loss(I, target)
            if tv_weight > 0:
                loss = loss + tv_weight * sum(total_variation(h / model.h_max) for h in model.heights()) / model.L
            if torch.isnan(loss):
                raise RuntimeError("NaN loss")
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            opt.step()
            run += loss.item(); nb += 1
        rec = {"epoch": ep, "train_loss": run / max(nb, 1), "sigmoid_temp": model.sigmoid_temp,
               "time_s": time.time() - t0}
        if epoch_callback is not None:
            extra = epoch_callback(ep, rec)
            if extra:
                rec.update(extra)
        history.append(rec)
        if log and (ep % 5 == 0 or ep == 1 or ep == epochs):
            log(f"  epoch {ep:02d} loss={rec['train_loss']:.3e} " +
                " ".join(f"{k}={v:.4g}" for k, v in rec.items() if k.startswith(("val", "test"))) +
                f" ({rec['time_s']:.0f}s)")
    return history
