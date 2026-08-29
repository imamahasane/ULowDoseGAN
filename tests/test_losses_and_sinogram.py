import torch

from src.ct.operators import ParallelBeamGeometry
from src.ct.operators_torch import TorchRadonProjector
from src.ct.units import to_normalized_units, to_physical_units
from src.losses.physics import PhysicsConsistencyLoss
from src.metrics.sinogram import sinogram_residual


def _projector():
    geom = ParallelBeamGeometry(image_size=16, angles=8, det_count=16)
    return TorchRadonProjector(geom, device=torch.device("cpu"))


def test_physics_consistency_loss_zero_for_matching_sinogram():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y = proj.A(x)
    loss_fn = PhysicsConsistencyLoss()
    loss = loss_fn(proj, x, y)
    assert loss.item() < 1e-8


def test_physics_consistency_loss_nonzero_for_mismatched_sinogram():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y_wrong = torch.randn(1, 8, 16)
    loss_fn = PhysicsConsistencyLoss()
    loss = loss_fn(proj, x, y_wrong)
    assert loss.item() > 0


def test_to_physical_units_identity_at_default_range():
    x = torch.randn(2, 1, 8, 8)
    x_min = torch.full((2,), -1.0)
    x_max = torch.full((2,), 1.0)
    out = to_physical_units(x, x_min, x_max)
    assert torch.allclose(out, x, atol=1e-6)


def test_to_physical_units_maps_endpoints():
    x = torch.tensor([[[[-1.0, 1.0]]]])  # (1,1,1,2)
    x_min = torch.tensor([2.0])
    x_max = torch.tensor([5.0])
    out = to_physical_units(x, x_min, x_max)
    assert torch.allclose(out, torch.tensor([[[[2.0, 5.0]]]]), atol=1e-6)


def test_to_normalized_units_is_inverse_of_to_physical_units():
    x_norm = torch.randn(3, 1, 8, 8).clamp(-1, 1)
    x_min = torch.tensor([0.0, -2.0, 5.0])
    x_max = torch.tensor([1.0, 3.0, 9.0])
    x_phys = to_physical_units(x_norm, x_min, x_max)
    x_roundtrip = to_normalized_units(x_phys, x_min, x_max)
    assert torch.allclose(x_roundtrip, x_norm, atol=1e-5)


def test_physics_consistency_loss_uses_physical_rescale():
    # x_hat0 is in [-1,1]; A is only self-consistent with y once rescaled to
    # the [x_min,x_max] physical range the sinogram was generated in -- the
    # loss should be near zero after rescaling and large without it.
    proj = _projector()
    x_min = torch.tensor([0.0])
    x_max = torch.tensor([2.0])
    x_norm = torch.randn(1, 1, 16, 16).clamp(-1, 1)
    x_phys = to_physical_units(x_norm, x_min, x_max)
    y = proj.A(x_phys)

    loss_fn = PhysicsConsistencyLoss()
    loss_rescaled = loss_fn(proj, x_norm, y, x_min, x_max)
    loss_raw = loss_fn(proj, x_norm, y)  # default x_min=-1,x_max=1: no rescale
    assert loss_rescaled.item() < 1e-6
    assert loss_raw.item() > loss_rescaled.item()


def test_sinogram_residual_zero_for_self_consistent_input():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y = proj.A(x)
    res = sinogram_residual(proj, x, y)
    assert res.l2 < 1e-3
    assert res.rmse < 1e-3
    assert res.relative < 1e-3


def test_sinogram_residual_positive_for_inconsistent_input():
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y_wrong = torch.randn(1, 8, 16) * 10
    res = sinogram_residual(proj, x, y_wrong)
    assert res.l2 > 0
    assert res.rmse > 0
    assert res.relative > 0


def test_sinogram_residual_relative_is_scale_invariant():
    # Scaling both the reconstruction error and the measurement by the same
    # factor should leave the *relative* residual unchanged, unlike l2/rmse.
    proj = _projector()
    x = torch.randn(1, 1, 16, 16)
    y_wrong = torch.randn(1, 8, 16)
    res_a = sinogram_residual(proj, x, y_wrong)
    res_b = sinogram_residual(proj, x * 5.0, y_wrong * 5.0)
    assert abs(res_a.relative - res_b.relative) < 1e-4
    assert res_b.l2 > res_a.l2 * 4  # raw l2 is NOT scale invariant
