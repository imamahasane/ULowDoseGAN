from __future__ import annotations

import torch


def to_physical_units(x_norm: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
    """Affine map from the network's [-1,1] image domain to the physical
    attenuation units the sinogram y and forward operator A live in
    (manuscript Eq. for x_hat0^phys, Sec. "Training objective"):

        x_phys = 0.5 * (x_norm + 1) * (x_max - x_min) + x_min

    x_min/x_max are per-sample scalars (shape (B,) or broadcastable to
    (B,1,1,1)) describing the physical attenuation range each image was
    normalized from -- fixed dataset-wide constants for LoDoPaB-CT, per-slice
    values for Mayo-AAPM (see src/data/lodopab.py, src/data/mayo_aapm.py).
    """
    if x_norm.ndim != 4:
        raise ValueError(f"x_norm must be (B,1,H,W); got {tuple(x_norm.shape)}")
    x_min = x_min.to(device=x_norm.device, dtype=x_norm.dtype).view(-1, 1, 1, 1)
    x_max = x_max.to(device=x_norm.device, dtype=x_norm.dtype).view(-1, 1, 1, 1)
    return 0.5 * (x_norm + 1.0) * (x_max - x_min) + x_min


def to_normalized_units(x_phys: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
    """Inverse of `to_physical_units`: physical attenuation units -> the
    network's [-1,1] image domain. Used for baselines that reconstruct
    directly in physical units (e.g. FBP, which operates on the sinogram
    itself) so their output is comparable to every learned method's PSNR/SSIM,
    which are computed in the shared [-1,1] domain (data_range=2.0)."""
    if x_phys.ndim != 4:
        raise ValueError(f"x_phys must be (B,1,H,W); got {tuple(x_phys.shape)}")
    x_min = x_min.to(device=x_phys.device, dtype=x_phys.dtype).view(-1, 1, 1, 1)
    x_max = x_max.to(device=x_phys.device, dtype=x_phys.dtype).view(-1, 1, 1, 1)
    return 2.0 * (x_phys - x_min) / (x_max - x_min) - 1.0
