import torch

from src.diffusion.ddpm import PhysicsConditionedDDPM
from src.diffusion.schedule import make_ddpm_schedule
from src.models.unet import UNetConfig, UNetModel


def _make_ddpm(image_size=16):
    schedule = make_ddpm_schedule(T=10, beta_schedule="linear", beta_start=1e-4, beta_end=2e-2, device=torch.device("cpu"))
    cfg = UNetConfig(
        in_channels=2, out_channels=1, base_channels=8,
        channel_mult=(1, 2), num_res_blocks=1, attention_resolutions=(), num_heads=2, dropout=0.0,
    )
    denoiser = UNetModel(cfg, image_size=image_size)
    return PhysicsConditionedDDPM(denoiser, schedule)


def test_unet_forward_shape():
    cfg = UNetConfig(in_channels=2, out_channels=1, base_channels=8, channel_mult=(1, 2), num_res_blocks=1, attention_resolutions=(8,), num_heads=2, dropout=0.0)
    model = UNetModel(cfg, image_size=16)
    x = torch.randn(2, 2, 16, 16)
    t = torch.randint(0, 10, (2,))
    out = model(x, t)
    assert out.shape == (2, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_q_sample_shape_and_range():
    ddpm = _make_ddpm()
    x0 = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 10, (2,))
    x_t = ddpm.q_sample(x0, t)
    assert x_t.shape == x0.shape
    assert torch.isfinite(x_t).all()


def test_predict_eps_and_x0_shapes():
    ddpm = _make_ddpm()
    x_t = torch.randn(2, 1, 16, 16)
    c = torch.randn(2, 1, 16, 16)
    t = torch.randint(0, 10, (2,))
    out = ddpm.predict_eps_and_x0(x_t, c, t)
    assert out.eps_pred.shape == (2, 1, 16, 16)
    assert out.x0_pred.shape == (2, 1, 16, 16)


def test_sample_produces_finite_output():
    ddpm = _make_ddpm()
    c = torch.randn(1, 1, 16, 16)
    out = ddpm.sample(c, shape=(1, 1, 16, 16))
    assert out.shape == (1, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_p_sample_deterministic_at_t0():

    ddpm = _make_ddpm()
    x_t = torch.randn(1, 1, 16, 16)
    c = torch.randn(1, 1, 16, 16)
    t = torch.zeros(1, dtype=torch.long)
    out1 = ddpm.p_sample(x_t.clone(), c, t)
    out2 = ddpm.p_sample(x_t.clone(), c, t)
    assert torch.allclose(out1, out2)


def test_dpm_solver_sample_produces_finite_output():
    ddpm = _make_ddpm()
    c = torch.randn(1, 1, 16, 16)
    out = ddpm.dpm_solver_sample(c, shape=(1, 1, 16, 16), steps=4)
    assert out.shape == (1, 1, 16, 16)
    assert torch.isfinite(out).all()


def test_dpm_solver_sample_deterministic():
    # Algorithm 2 is explicitly deterministic (no injected noise); two runs
    # from the same seed must match exactly.
    torch.manual_seed(0)
    ddpm = _make_ddpm()
    c = torch.randn(1, 1, 16, 16)

    torch.manual_seed(42)
    out1 = ddpm.dpm_solver_sample(c, shape=(1, 1, 16, 16), steps=4)
    torch.manual_seed(42)
    out2 = ddpm.dpm_solver_sample(c, shape=(1, 1, 16, 16), steps=4)
    assert torch.allclose(out1, out2)


def test_dpm_solver_sample_single_step_matches_direct_x0_estimate():
    # With steps=1, the only iteration runs at t=T-1 and, per Algorithm 2,
    # returns the network's x0 estimate directly (t_0=0 branch).
    ddpm = _make_ddpm()
    torch.manual_seed(7)
    x_T = torch.randn(1, 1, 16, 16)
    c = torch.randn(1, 1, 16, 16)
    t = torch.full((1,), 9, dtype=torch.long)  # T=10 -> last index is 9
    expected = ddpm.predict_eps_and_x0(x_T, c, t).x0_pred

    torch.manual_seed(7)
    out = ddpm.dpm_solver_sample(c, shape=(1, 1, 16, 16), steps=1)
    assert torch.allclose(out, expected, atol=1e-5)


def test_dpm_solver_sample_rejects_nonpositive_steps():
    ddpm = _make_ddpm()
    c = torch.randn(1, 1, 16, 16)
    try:
        ddpm.dpm_solver_sample(c, shape=(1, 1, 16, 16), steps=0)
        assert False, "expected ValueError for steps=0"
    except ValueError:
        pass
