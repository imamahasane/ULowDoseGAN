from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ct.units import to_physical_units


class PhysicsConsistencyLoss(nn.Module):
    """L_phys = ||A(x_hat0^phys) - y||^2 (manuscript Eq. for the total loss).

    x_hat0 is expected in the network's [-1,1] domain; it is affinely
    rescaled to physical attenuation units via x_min/x_max (src/ct/units.py)
    before the forward operator is applied, so the residual is computed in
    units commensurate with the measured sinogram y. Defaults of
    x_min=-1, x_max=1 make the rescale the identity, for callers (and the
    existing unit tests) that already operate directly in A's native domain.
    """

    def forward(
        self,
        projector,
        x_hat0: torch.Tensor,
        y: torch.Tensor,
        x_min: torch.Tensor | float = -1.0,
        x_max: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        if not torch.is_tensor(x_min):
            x_min = torch.full((x_hat0.shape[0],), float(x_min), device=x_hat0.device, dtype=x_hat0.dtype)
        if not torch.is_tensor(x_max):
            x_max = torch.full((x_hat0.shape[0],), float(x_max), device=x_hat0.device, dtype=x_hat0.dtype)
        x_hat0_phys = to_physical_units(x_hat0, x_min, x_max)
        sino_hat = projector.A(x_hat0_phys)
        return F.mse_loss(sino_hat, y)
