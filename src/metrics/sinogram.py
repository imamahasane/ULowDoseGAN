from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SinogramResidual:
    l2: float
    rmse: float
    # ||A(x_hat)-y||_2 / ||y||_2 -- the relative sinogram residual actually
    # reported in the manuscript's results table (Table: reconstruction
    # quality and sinogram-domain consistency), as opposed to the raw,
    # scale-dependent l2/rmse above.
    relative: float


def sinogram_residual(projector, x_hat0: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> SinogramResidual:

    sino_hat = projector.A(x_hat0)
    diff = sino_hat - y
    l2 = torch.linalg.vector_norm(diff).item()
    rmse = torch.sqrt(F.mse_loss(sino_hat, y)).item()
    y_norm = torch.linalg.vector_norm(y).item()
    relative = l2 / max(y_norm, eps)
    return SinogramResidual(l2=l2, rmse=rmse, relative=relative)
