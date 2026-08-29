from __future__ import annotations

import torch

from src.ct.fbp import filtered_backprojection
from src.ct.operators import ParallelBeamGeometry
from src.ct.operators_torch import TorchRadonProjector


def _shepp_logan_like(size: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij"
    )
    img = torch.zeros(size, size)
    img[(xx / 0.9) ** 2 + (yy / 0.65) ** 2 <= 1] = 1.0
    img[((xx - 0.2) / 0.2) ** 2 + (yy / 0.2) ** 2 <= 1] = 0.6
    img[((xx + 0.2) / 0.15) ** 2 + ((yy - 0.1) / 0.15) ** 2 <= 1] = 0.3
    return img


def test_fbp_output_shape_and_finite():
    geom = ParallelBeamGeometry(image_size=32, angles=60, det_count=45)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))
    x = _shepp_logan_like(32)[None, None]
    y = proj.A(x)

    x_fbp = filtered_backprojection(proj, y)
    assert x_fbp.shape == (1, 1, 32, 32)
    assert torch.isfinite(x_fbp).all()


def test_fbp_closer_to_phantom_than_unfiltered_adjoint():
    # The whole point of the ramp filter: FBP should recover the phantom's
    # scale/structure noticeably better than the raw (blurry) unfiltered
    # backprojection A*(y) that PCDiff uses only as conditioning, not as a
    # reconstruction in its own right.
    geom = ParallelBeamGeometry(image_size=48, angles=90, det_count=68)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))
    x = _shepp_logan_like(48)[None, None]
    y = proj.A(x)

    x_fbp = filtered_backprojection(proj, y)
    x_unfiltered = proj.AT(y)

    err_fbp = (x_fbp - x).pow(2).mean().item()
    err_unfiltered = (x_unfiltered - x).pow(2).mean().item()
    assert err_fbp < err_unfiltered


def test_fbp_zero_input_gives_zero_output():
    geom = ParallelBeamGeometry(image_size=16, angles=8, det_count=16)
    proj = TorchRadonProjector(geom, device=torch.device("cpu"))
    y = torch.zeros(1, 8, 16)
    x_fbp = filtered_backprojection(proj, y)
    assert torch.allclose(x_fbp, torch.zeros_like(x_fbp), atol=1e-6)
