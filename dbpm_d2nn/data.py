"""Datasets, input encodings and detector layout."""
from __future__ import annotations

import math
import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

# torchvision downloads the datasets here on first use; override with DBPM_DATA_ROOT
DEFAULT_DATA_ROOT = os.environ.get(
    "DBPM_DATA_ROOT", os.path.join(os.path.expanduser("~"), ".dbpm_d2nn_data")
)


# ----------------------------------------------------------------------------- encodings
def encode(x: torch.Tensor, encoding: str) -> torch.Tensor:
    """x: image batch [B,1,H,W] in [0,1] -> complex field [B,H,W]."""
    img = x.squeeze(1).to(torch.float32)
    if encoding == "phase":          # U0 = exp(i 2 pi X)
        return torch.exp(1j * 2.0 * math.pi * img)
    if encoding == "amplitude":      # U0 = X
        return torch.complex(img, torch.zeros_like(img))
    raise ValueError(encoding)


# ----------------------------------------------------------------------------- detectors
def make_detector_patches(H: int, W: int, patch: int, margin_3: int, margin_4: int,
                          margin_v: int, num_classes: int = 10) -> List[Tuple[int, int, int, int]]:
    """3-4-3 layout of square detector patches (pixel indices y0,y1,x0,x1)."""
    assert num_classes == 10
    patches = []
    rows = [3, 4, 3]
    total_h = len(rows) * patch + (len(rows) - 1) * margin_v
    y_start = (H - total_h) // 2
    for r, n_in_row in enumerate(rows):
        margin_h = margin_4 if n_in_row == 4 else margin_3
        row_w = n_in_row * patch + (n_in_row - 1) * margin_h
        x_start = W // 2 - row_w // 2
        y0 = y_start + r * (patch + margin_v)
        for c in range(n_in_row):
            x0 = x_start + c * (patch + margin_h)
            patches.append((y0, y0 + patch, x0, x0 + patch))
    return patches[:num_classes]


def detector_patches_from_geometry(H: int, W: int, dx: float, wavelength: float,
                                   patch_lam: float = 6.4, margin_tb_lam: float = 9.6,
                                   margin_mid_lam: float = 20.0 / 3.0, margin_v_lam: float = 9.6):
    """Paper detector layout expressed in wavelengths (6.4 lambda patches)."""
    px = lambda v: max(1, int(round(v * wavelength / dx)))
    return make_detector_patches(H, W, px(patch_lam), px(margin_tb_lam), px(margin_mid_lam), px(margin_v_lam))


def patch_energies(intensity: torch.Tensor, patches) -> torch.Tensor:
    return torch.stack([intensity[:, y0:y1, x0:x1].sum(dim=(1, 2)) for (y0, y1, x0, x1) in patches], dim=1)


# ----------------------------------------------------------------------------- classification data
def classification_transform(grid: int, digit: int):
    pad = (grid - digit) // 2
    return transforms.Compose([
        transforms.Resize((digit, digit)),
        transforms.ToTensor(),
        transforms.Pad((pad, pad, grid - digit - pad, grid - digit - pad), fill=0),
    ])


def classification_datasets(name: str, grid: int, digit: int, root: str = DEFAULT_DATA_ROOT,
                            n_val: int = 5000, split_seed: int = 0):
    tfm = classification_transform(grid, digit)
    cls = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST}[name]
    full = cls(root=root, train=True, download=True, transform=tfm)
    test = cls(root=root, train=False, download=True, transform=tfm)
    g = torch.Generator().manual_seed(split_seed)
    train, val = random_split(full, [len(full) - n_val, n_val], generator=g)
    return train, val, test


def make_loaders(train, val, test, batch_size: int = 128, seed: int = 0, num_workers: int = 4,
                 test_batch: int = 256):
    g = torch.Generator().manual_seed(seed)
    tl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=True, generator=g, drop_last=False, persistent_workers=num_workers > 0)
    vl = DataLoader(val, batch_size=test_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    te = DataLoader(test, batch_size=test_batch, shuffle=False, num_workers=num_workers, pin_memory=True)
    return tl, vl, te


# ----------------------------------------------------------------------------- imaging data
def imaging_transform(grid: int, img: int, grayscale: bool):
    pad = (grid - img) // 2
    t = [transforms.Resize((img, img))]
    if grayscale:
        t.append(transforms.Grayscale(num_output_channels=1))
    t += [transforms.ToTensor(), transforms.Pad((pad, pad, grid - img - pad, grid - img - pad), fill=0)]
    return transforms.Compose(t)


def imaging_datasets(name: str, grid: int, img: int, root: str = DEFAULT_DATA_ROOT,
                     n_train: int = 1500, n_val: int = 200, n_test: int = 300, split_seed: int = 0):
    """Random subsets (paper protocol: 1500 / 200 / 300) of MNIST, Fashion-MNIST or CIFAR-100."""
    gray = name == "cifar100"
    tfm = imaging_transform(grid, img, gray)
    if name == "mnist":
        full, test_full = (datasets.MNIST(root=root, train=t, download=True, transform=tfm) for t in (True, False))
    elif name == "fmnist":
        full, test_full = (datasets.FashionMNIST(root=root, train=t, download=True, transform=tfm) for t in (True, False))
    elif name == "cifar100":
        full, test_full = (datasets.CIFAR100(root=root, train=t, download=True, transform=tfm) for t in (True, False))
    else:
        raise ValueError(name)
    g = torch.Generator().manual_seed(split_seed)
    tv = torch.randperm(len(full), generator=g)[: n_train + n_val]
    te = torch.randperm(len(test_full), generator=g)[:n_test]
    return Subset(full, tv[:n_train].tolist()), Subset(full, tv[n_train:].tolist()), Subset(test_full, te.tolist())
