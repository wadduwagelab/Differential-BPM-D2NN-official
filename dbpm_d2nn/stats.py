"""Paired statistics for the FDTD validation and seed-averaged sweeps."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import stats as sps


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict:
    """Exact (binomial) McNemar test on paired binary outcomes.

    Returns the 2x2 contingency table and the two-sided exact p-value
    P(X <= min(b,c) | n = b + c, p = 0.5) * 2 (clipped to 1).
    """
    a = np.asarray(correct_a, bool); b = np.asarray(correct_b, bool)
    both = int(np.sum(a & b)); a_only = int(np.sum(a & ~b)); b_only = int(np.sum(~a & b)); neither = int(np.sum(~a & ~b))
    n_disc = a_only + b_only
    if n_disc == 0:
        p = 1.0
    else:
        k = min(a_only, b_only)
        p = min(1.0, 2.0 * sps.binom.cdf(k, n_disc, 0.5))
    return {"table": {"both_correct": both, "A_only": a_only, "B_only": b_only, "neither": neither},
            "n_discordant": n_disc, "p_value_exact": float(p),
            "acc_A": float(a.mean()), "acc_B": float(b.mean()), "n": int(a.size)}


def paired_bootstrap_ci(x: Sequence[float], y: Sequence[float], n_boot: int = 20000, seed: int = 0,
                        stat=np.mean, alpha: float = 0.05) -> Dict:
    """Percentile bootstrap CI of stat(y - x) over paired samples."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    boots = stat(d[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(stat(d)), "ci_low": float(lo), "ci_high": float(hi), "n": int(d.size)}


def paired_tests(x: Sequence[float], y: Sequence[float], n_perm: int = 20000, seed: int = 0) -> Dict:
    """Wilcoxon signed-rank (exact when possible), paired t and paired sign-flip permutation test on y - x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = y - x
    out = {"mean_x": float(x.mean()), "mean_y": float(y.mean()), "mean_diff": float(d.mean()),
           "median_diff": float(np.median(d)), "n": int(d.size), "n_y_greater": int(np.sum(d > 0)),
           "n_y_smaller": int(np.sum(d < 0))}
    nz = d[d != 0]
    if nz.size >= 5:
        try:
            w = sps.wilcoxon(nz, alternative="two-sided", method="exact" if nz.size <= 25 else "auto")
            out["wilcoxon_p"] = float(w.pvalue)
        except Exception as e:  # pragma: no cover
            out["wilcoxon_p"] = float("nan"); out["wilcoxon_err"] = str(e)
    else:
        out["wilcoxon_p"] = float("nan")
    if d.size >= 2 and d.std() > 0:
        out["ttest_p"] = float(sps.ttest_rel(y, x).pvalue)
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, d.size))
    perm = (signs * d[None, :]).mean(1)
    out["perm_p"] = float(np.mean(np.abs(perm) >= abs(d.mean()) - 1e-15))
    out.update({f"boot_{k}": v for k, v in paired_bootstrap_ci(x, y, seed=seed).items() if k != "n"})
    return out


def accuracy_diff_ci(correct_a, correct_b, n_boot=20000, seed=0) -> Dict:
    """Paired bootstrap CI for the accuracy difference acc(B) - acc(A)."""
    return paired_bootstrap_ci(np.asarray(correct_a, float), np.asarray(correct_b, float), n_boot=n_boot, seed=seed)


def mean_std(values: Sequence[float]) -> Dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}
