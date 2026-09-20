"""Evaluation helpers."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .data import encode, patch_energies
from .metrics import ssim_1ch, best_linear_rescale


def sce_logits(energies: torch.Tensor) -> torch.Tensor:
    """Scaled logits used by the paper: 10 * p_c / max_j p_j (max detached)."""
    m = energies.max(dim=1, keepdim=True)[0].detach().clamp(min=1e-8)
    return 10.0 * energies / m


@torch.no_grad()
def evaluate_classification(model, loader, patches, encoding: str, device,
                            max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for b, (x, y) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        I = model(encode(x, encoding))
        E = patch_energies(I, patches)
        loss_sum += F.cross_entropy(sce_logits(E), y, reduction="sum").item()
        correct += (E.argmax(1) == y).sum().item()
        total += y.numel()
    return {"acc": 100.0 * correct / total, "loss": loss_sum / total, "n": total}


@torch.no_grad()
def predict_classification(model, loader, patches, encoding: str, device):
    """Return per-sample (pred, label, patch energies) for a whole loader."""
    model.eval()
    preds, labels, energies = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        I = model(encode(x, encoding))
        E = patch_energies(I, patches)
        preds.append(E.argmax(1).cpu()); labels.append(y); energies.append(E.cpu())
    return torch.cat(preds), torch.cat(labels), torch.cat(energies)


@torch.no_grad()
def evaluate_imaging(model, loader, device, crop: Optional[int] = None) -> Dict[str, float]:
    """MSE / SSIM between output intensity and target A^2 (paper protocol) plus
    scale-invariant variants.  ``crop`` = side of a central crop for extra metrics."""
    model.eval()
    n = 0
    acc = {"mse": 0.0, "ssim": 0.0, "mse_scaled": 0.0, "ssim_scaled": 0.0, "pearson": 0.0}
    if crop:
        acc.update({"mse_crop": 0.0, "ssim_crop": 0.0})
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        amp = x.squeeze(1)
        target = amp ** 2
        I = model(encode(x, "amplitude"))
        B = target.size(0)
        acc["mse"] += F.mse_loss(I, target, reduction="sum").item() / target[0].numel()
        acc["ssim"] += ssim_1ch(I.clamp(0, 1), target.clamp(0, 1), size_average=False).sum().item()
        Is = best_linear_rescale(I, target)
        acc["mse_scaled"] += F.mse_loss(Is, target, reduction="sum").item() / target[0].numel()
        acc["ssim_scaled"] += ssim_1ch(Is.clamp(0, 1), target.clamp(0, 1), size_average=False).sum().item()
        a = I.flatten(1) - I.flatten(1).mean(1, keepdim=True)
        b = target.flatten(1) - target.flatten(1).mean(1, keepdim=True)
        acc["pearson"] += ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)).sum().item()
        if crop:
            H, W = target.shape[-2:]
            y0, x0 = (H - crop) // 2, (W - crop) // 2
            Ic, Tc = I[:, y0:y0 + crop, x0:x0 + crop], target[:, y0:y0 + crop, x0:x0 + crop]
            acc["mse_crop"] += F.mse_loss(Ic, Tc, reduction="sum").item() / Tc[0].numel()
            acc["ssim_crop"] += ssim_1ch(Ic.clamp(0, 1), Tc.clamp(0, 1), size_average=False).sum().item()
        n += B
    out = {k: v / n for k, v in acc.items()}
    out["n"] = n
    return out


@torch.no_grad()
def output_fields(model, x: torch.Tensor, encoding: str):
    """Complex detector-plane field for an image batch (for model-vs-model NMSE)."""
    model.eval()
    I, u = model(encode(x, encoding), return_field=True)
    return I, u
