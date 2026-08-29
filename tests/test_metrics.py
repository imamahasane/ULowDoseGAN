import numpy as np
import torch

from src.metrics.image import psnr, ssim_torch
from src.metrics.stats import bootstrap_ci, bootstrap_ci_diff, wilcoxon_signed_rank
from src.metrics.uncertainty import coverage_and_ece, error_gap, mean_and_std, pearson_r


def test_psnr_identical_is_infinite():
    x = torch.randn(1, 8, 8)
    assert psnr(x, x, data_range=2.0) == float("inf")


def test_psnr_known_value():
    # MSE of exactly 1.0 with data_range=2.0 -> PSNR = 20*log10(2) - 10*log10(1) = 6.0206 dB
    x = torch.zeros(1, 4, 4)
    y = torch.ones(1, 4, 4)
    val = psnr(x, y, data_range=2.0)
    assert abs(val - 6.0206) < 1e-2


def test_ssim_identical_is_one():
    x = torch.rand(1, 16, 16)
    val = ssim_torch(x, x, data_range=1.0)
    assert abs(val - 1.0) < 1e-6


def test_ssim_rejects_wrong_shape():
    x = torch.rand(2, 16, 16)  # wrong: batch dim where channel dim expected
    try:
        ssim_torch(x, x, data_range=1.0)
        assert False, "expected ValueError for malformed shape"
    except ValueError:
        pass


def test_mean_and_std_shapes():
    samples = torch.randn(8, 1, 1, 16, 16)
    mean, std = mean_and_std(samples)
    assert mean.shape == (1, 1, 16, 16)
    assert std.shape == (1, 1, 16, 16)
    assert (std >= 0).all()


def test_pearson_r_perfect_correlation():
    a = torch.arange(10).float()
    b = 2 * a + 1
    r = pearson_r(a, b)
    assert abs(r - 1.0) < 1e-4


def test_coverage_and_ece_reasonable_range():
    torch.manual_seed(0)
    err = torch.rand(1000) * 0.1
    sigma = err + 0.01  # sigma tracks error well -> should calibrate reasonably
    result = coverage_and_ece(err, sigma)
    assert 0.0 <= result.coverage95 <= 1.0
    assert result.cal_err >= 0.0


def test_error_gap_positive_when_sigma_tracks_error():
    torch.manual_seed(0)
    err = torch.rand(1000) * 0.1
    sigma = err + 0.01  # sigma tracks error well -> high-sigma pixels should have higher error
    gap = error_gap(err, sigma)
    assert gap > 0.0


def test_error_gap_near_zero_when_sigma_uninformative():
    torch.manual_seed(0)
    err = torch.rand(2000) * 0.1
    sigma = torch.rand(2000) * 0.1  # independent of error -> gap should be small
    gap = error_gap(err, sigma)
    assert abs(gap) < 0.03


def test_wilcoxon_matches_scipy_reference():
    from scipy import stats as scipy_stats

    rng = np.random.default_rng(0)
    a = rng.normal(35.0, 1.0, size=50)
    b = rng.normal(34.0, 1.0, size=50)
    ours = wilcoxon_signed_rank(a, b)
    ref = scipy_stats.wilcoxon(a, b)
    assert abs(ours.p_value - ref.pvalue) < 1e-9
    assert abs(ours.statistic - ref.statistic) < 1e-9


def test_bootstrap_ci_contains_true_mean_most_of_the_time():
    rng = np.random.default_rng(1)
    x = rng.normal(10.0, 2.0, size=200)
    ci = bootstrap_ci(x, n_boot=2000, seed=1)
    assert ci.lo < ci.mean < ci.hi


def test_bootstrap_ci_diff_paired():
    rng = np.random.default_rng(2)
    a = rng.normal(35.0, 1.0, size=100)
    b = a - 0.5 + rng.normal(0, 0.1, size=100)  # a consistently ~0.5 higher than b
    diff = bootstrap_ci_diff(a, b, n_boot=2000, seed=2)
    assert diff.lo > 0, "paired diff CI should exclude zero given the constructed consistent gap"
