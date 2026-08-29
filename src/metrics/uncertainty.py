from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


def mean_and_std(samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """samples: (K, ...) independent conditional-diffusion reconstructions."""
    mean = samples.mean(dim=0)
    std = samples.std(dim=0, unbiased=True)
    return mean, std


def pearson_r(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.flatten()
    b = b.flatten()
    a = a - a.mean()
    b = b - b.mean()
    r = (a * b).mean() / (a.std(unbiased=False) * b.std(unbiased=False) + eps)
    return float(r.item())


@dataclass
class CoverageResult:
    coverage95: float
    # Decile-binned, regression-style calibration error (manuscript Sec.
    # "Uncertainty estimation": |mean(|error|) - mean(sigma)| averaged over
    # uncertainty deciles, weighted by bin occupancy). NOT the
    # classification-style Expected Calibration Error the "ECE" abbreviation
    # usually denotes elsewhere -- named cal_err here to avoid that collision.
    cal_err: float


def coverage_and_ece(error: torch.Tensor, sigma: torch.Tensor, z: float = 1.96, eps: float = 1e-8) -> CoverageResult:

    err = error.abs()
    bound = z * sigma.abs().clamp(min=eps)
    covered = (err <= bound).float()
    coverage95 = float(covered.mean().item())

    sig = sigma.flatten().detach().cpu().numpy()
    errn = err.flatten().detach().cpu().numpy()
    bins = np.quantile(sig, np.linspace(0, 1, 11))
    cal_err = 0.0
    total = len(sig)
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        m = (sig >= lo) & (sig <= hi if i == 9 else sig < hi)
        if m.sum() == 0:
            continue
        avg_sig = sig[m].mean()
        avg_err = errn[m].mean()
        cal_err += (m.sum() / total) * abs(avg_err - avg_sig)
    return CoverageResult(coverage95=coverage95, cal_err=float(cal_err))


def error_gap(error: torch.Tensor, sigma: torch.Tensor, quantile: float = 0.2) -> float:
    """Error Gap (Table: uncertainty-calibration metrics): the difference in
    mean absolute reconstruction error between the highest- and lowest-
    uncertainty pixel quantiles (top `quantile` fraction of sigma vs. bottom
    `quantile` fraction). Positive and larger values indicate sigma actually
    separates more- from less-reliable pixels."""
    if not 0.0 < quantile < 0.5:
        raise ValueError(f"quantile must be in (0, 0.5); got {quantile}")
    err = error.abs().flatten().detach().cpu().numpy()
    sig = sigma.flatten().detach().cpu().numpy()
    lo_cut, hi_cut = np.quantile(sig, [quantile, 1.0 - quantile])
    low_mask = sig <= lo_cut
    high_mask = sig >= hi_cut
    if low_mask.sum() == 0 or high_mask.sum() == 0:
        return 0.0
    return float(err[high_mask].mean() - err[low_mask].mean())
