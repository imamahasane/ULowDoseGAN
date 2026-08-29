from __future__ import annotations

import math

import torch


def _ramp_filter(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Ram-Lak filter in the frequency domain for a length-n (padded, power-
    of-two) detector axis: |omega|, normalized to Nyquist = 1.0 and windowed
    with a Hann taper to control the high-frequency noise amplification that
    a pure ramp filter is known to have on noisy, low-dose sinograms."""
    freqs = torch.fft.fftfreq(n, d=1.0, device=device).to(dtype)
    ramp = freqs.abs() * 2.0  # Nyquist frequency (|f|=0.5) maps to 1.0
    hann = 0.5 + 0.5 * torch.cos(math.pi * freqs * 2.0)
    return ramp * hann


def filtered_backprojection(projector, y: torch.Tensor) -> torch.Tensor:
    """Standard (Ram-Lak, Hann-windowed) filtered backprojection: ramp-filter
    the sinogram along the detector axis, then backproject with the same
    unfiltered adjoint A* used throughout this codebase (manuscript Sec.
    "Conditioning mechanism and architecture": FBP filters, PCDiff's
    conditioning does not).

    y: (B, angles, det). Returns x_fbp: (B, 1, H, W), on projector.geom's
    image_size grid, in the physical units of y / A.
    """
    if y.ndim != 3:
        raise ValueError(f"y must be (B,angles,det); got {tuple(y.shape)}")
    b, num_angles, det = y.shape
    n_pad = 1 << (2 * det - 1).bit_length()  # next power of two >= 2*det, avoids circular-convolution wraparound

    y_pad = torch.zeros(b, num_angles, n_pad, device=y.device, dtype=y.dtype)
    y_pad[..., :det] = y

    filt = _ramp_filter(n_pad, y.device, y.dtype)  # (n_pad,)
    y_fft = torch.fft.fft(y_pad, dim=-1)
    y_filtered = torch.fft.ifft(y_fft * filt[None, None, :], dim=-1).real
    y_filtered = y_filtered[..., :det]

    x_bp = projector.AT(y_filtered)
    # Standard parallel-beam FBP normalization for evenly-spaced angles over
    # [0, pi): scale the (unnormalized-sum) adjoint backprojection by
    # pi / num_angles so it approximates the continuous filtered-
    # backprojection integral rather than a raw angular sum.
    return x_bp * (math.pi / num_angles)
