from __future__ import annotations

import torch

from src.ct.operators import ParallelBeamGeometry
from src.ct.operators_torch import TorchRadonProjector
from src.ct.units import to_physical_units
from src.diffusion.dps import UnconditionalDDPM, dps_sample
from src.diffusion.schedule import make_ddpm_schedule
from src.models.unet import UNetConfig, UNetModel


def _make_unconditional_model(image_size=16, T=6):
    schedule = make_ddpm_schedule(T=T, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, device=torch.device("cpu"))
    cfg = UNetConfig(
        in_channels=1, out_channels=1, base_channels=8,
        channel_mult=(1, 2), num_res_blocks=1, attention_resolutions=(), num_heads=2, dropout=0.0,
    )
    denoiser = UNetModel(cfg, image_size=image_size)
    return UnconditionalDDPM(denoiser, schedule)


def test_dps_sample_shape_and_finite():
    torch.manual_seed(0)
    model = _make_unconditional_model()
    geom = ParallelBeamGeometry(image_size=16, angles=8, det_count=16)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))
    y = torch.randn(1, 8, 16)
    x_min = torch.tensor([-1.0])
    x_max = torch.tensor([1.0])

    out = dps_sample(model, proj, y, shape=(1, 1, 16, 16), eta=0.1, x_min=x_min, x_max=x_max)
    assert out.shape == (1, 1, 16, 16)
    assert torch.isfinite(out).all()
    assert not out.requires_grad


def test_dps_correction_gradient_locally_decreases_data_fidelity():
    # Direct check of the DPS correction term (Eq. for DPS) in isolation from
    # a full noisy reverse trajectory: -grad_{x_t} of the data-fidelity term,
    # taken through the network's own x0 estimate and the forward operator,
    # should be a local descent direction for that same term -- i.e. the
    # autograd-computed gradient used by dps_sample actually points the way
    # Eq. for DPS says it should.
    torch.manual_seed(0)
    model = _make_unconditional_model()
    geom = ParallelBeamGeometry(image_size=16, angles=8, det_count=16)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))
    x_min = torch.tensor([-1.0])
    x_max = torch.tensor([1.0])
    y = torch.randn(1, 8, 16)
    t = torch.tensor([3])

    x_t = torch.randn(1, 1, 16, 16, requires_grad=True)
    with torch.enable_grad():
        out = model.predict_eps_and_x0(x_t, t)
        x0_phys = to_physical_units(out.x0_pred, x_min, x_max)
        loss0 = torch.sum((proj.A(x0_phys) - y) ** 2)
        (grad,) = torch.autograd.grad(loss0, x_t)

    step = 1e-4
    x_t2 = (x_t - step * grad).detach()
    with torch.no_grad():
        out2 = model.predict_eps_and_x0(x_t2, t)
        x0_phys2 = to_physical_units(out2.x0_pred, x_min, x_max)
        loss1 = torch.sum((proj.A(x0_phys2) - y) ** 2)

    assert loss1.item() < loss0.item()


def test_dps_sample_eta_zero_matches_unconditional_ancestral_sampling():
    # eta=0 should reduce dps_sample exactly to plain unconditional DDPM
    # ancestral sampling (no data-fidelity correction applied at all).
    model = _make_unconditional_model()
    geom = ParallelBeamGeometry(image_size=16, angles=8, det_count=16)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))
    y = torch.randn(1, 8, 16)
    x_min = torch.tensor([-1.0])
    x_max = torch.tensor([1.0])

    torch.manual_seed(3)
    out_dps = dps_sample(model, proj, y, shape=(1, 1, 16, 16), eta=0.0, x_min=x_min, x_max=x_max)

    torch.manual_seed(3)
    T = model.schedule.betas.shape[0]
    x_t = torch.randn(1, 1, 16, 16)
    from src.diffusion.ddpm import extract
    with torch.no_grad():
        for ti in reversed(range(T)):
            t = torch.full((1,), ti, dtype=torch.long)
            out = model.predict_eps_and_x0(x_t, t)
            alpha_t = extract(model.schedule.alphas, t, x_t.shape)
            a_bar_t = extract(model.schedule.alpha_bars, t, x_t.shape)
            mu = (x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - a_bar_t) * out.eps_pred) / torch.sqrt(alpha_t)
            if ti == 0:
                x_t = mu
            else:
                var = extract(model.schedule.posterior_variance, t, x_t.shape)
                x_t = mu + torch.sqrt(var) * torch.randn_like(x_t)

    assert torch.allclose(out_dps, x_t, atol=1e-5)
