from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.ct.units import to_physical_units

from .ddpm import DDPMOutput, extract
from .schedule import DiffusionSchedule


class UnconditionalDDPM(nn.Module):
    """The DPS prior (manuscript Sec. "Head-to-head comparison with
    inference-time physics correction"): the same DDPM family as
    PhysicsConditionedDDPM, but trained -- and sampled from, absent the
    correction in `dps_sample` -- with NO measurement conditioning (no c
    channel) and no physics-consistency loss. Physics is reintroduced only at
    inference time, via a per-step data-fidelity gradient (Eq. for DPS)."""

    def __init__(self, denoiser: nn.Module, schedule: DiffusionSchedule):
        super().__init__()
        self.denoiser = denoiser
        self.schedule = schedule

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> torch.Tensor:
        if eps is None:
            eps = torch.randn_like(x0)
        a_bar = extract(self.schedule.alpha_bars, t, x0.shape)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps

    def predict_eps_and_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> DDPMOutput:
        eps_pred = self.denoiser(x_t, t)
        a_bar = extract(self.schedule.alpha_bars, t, x_t.shape)
        x0_pred = (x_t - torch.sqrt(1.0 - a_bar) * eps_pred) / torch.sqrt(a_bar)
        return DDPMOutput(eps_pred=eps_pred, x0_pred=x0_pred)


def dps_sample(
    model: UnconditionalDDPM,
    projector,
    y: torch.Tensor,
    shape: Tuple[int, int, int, int],
    eta: float,
    x_min: torch.Tensor,
    x_max: torch.Tensor,
) -> torch.Tensor:
    """Diffusion posterior sampling (Chung et al. 2022; manuscript Eq. for
    DPS): at every reverse step, perturb the unconditional DDPM mean by the
    gradient of the data-fidelity term ||A(x0_hat(x_t)) - y||^2 w.r.t. x_t,
    computed by backpropagating through the network's own x0 prediction.

    Unlike `PhysicsConditionedDDPM.sample`, this cannot run fully under
    `torch.no_grad()`: the correction gradient is taken with respect to the
    sampling trajectory itself, so each step briefly re-enables autograd for
    the network forward pass and detaches immediately after.
    """
    device = y.device
    T = model.schedule.betas.shape[0]
    x_t = torch.randn(shape, device=device)

    for ti in reversed(range(T)):
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        x_t = x_t.detach().requires_grad_(True)

        with torch.enable_grad():
            out = model.predict_eps_and_x0(x_t, t)
            x0_phys = to_physical_units(out.x0_pred, x_min, x_max)
            data_fidelity = torch.sum((projector.A(x0_phys) - y) ** 2)
            (grad,) = torch.autograd.grad(data_fidelity, x_t)

        with torch.no_grad():
            alpha_t = extract(model.schedule.alphas, t, x_t.shape)
            a_bar_t = extract(model.schedule.alpha_bars, t, x_t.shape)
            mu = (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - a_bar_t) * out.eps_pred.detach()) / torch.sqrt(alpha_t)
            mu = mu - eta * grad

            if ti == 0:
                x_t = mu
            else:
                var = extract(model.schedule.posterior_variance, t, x_t.shape)
                noise = torch.randn_like(x_t)
                x_t = mu + torch.sqrt(var) * noise

    return x_t.detach()
