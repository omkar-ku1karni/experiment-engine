import numpy as np
import pytest

from expengine import cuped


def _data(rng, n=4000, rho=0.7, tau=0.0):
    x = rng.standard_normal(2 * n)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(2 * n)
    y[n:] += tau
    return y[:n], x[:n], y[n:], x[n:]


def test_variance_reduction_close_to_theory():
    r = cuped.cuped_effect(*_data(np.random.default_rng(0)))
    assert r.variance_ratio == pytest.approx(1 - 0.7**2, abs=0.03)
    assert r.cuped_se < r.raw_se


def test_unbiased_under_null_and_effect():
    rng = np.random.default_rng(1)
    for tau in (0.0, 0.1):
        est = [cuped.cuped_effect(*_data(rng, 500, 0.6, tau)).cuped_diff for _ in range(400)]
        assert abs(np.mean(est) - tau) < 4 * np.std(est) / np.sqrt(len(est))


def test_zero_variance_covariate_falls_back_to_raw():
    y0, _, y1, _ = _data(np.random.default_rng(2))
    r = cuped.cuped_effect(y0, np.ones_like(y0), y1, np.ones_like(y1))
    assert r.theta == 0 and "COVARIATE_ZERO_VARIANCE" in r.flags
    assert r.cuped_diff == pytest.approx(r.raw_diff)


def test_missing_covariates_are_flagged_and_handled():
    y0, x0, y1, x1 = _data(np.random.default_rng(3))
    x0 = x0.copy(); x0[::3] = np.nan
    r = cuped.cuped_effect(y0, x0, y1, x1)
    assert "COVARIATE_MISSING_IMPUTED_MEAN" in r.flags and np.isfinite(r.cuped_diff)
    allmiss = cuped.cuped_effect(y0, np.full_like(y0, np.nan), y1, np.full_like(y1, np.nan))
    assert "COVARIATE_ALL_MISSING" in allmiss.flags and allmiss.cuped_diff == pytest.approx(allmiss.raw_diff)


def test_theta_is_treatment_blind():
    """theta is fitted on the pooled sample: relabelling arms does not change it."""
    y0, x0, y1, x1 = _data(np.random.default_rng(4), tau=0.5)
    a = cuped.cuped_effect(y0, x0, y1, x1).theta
    b = cuped.cuped_effect(y1, x1, y0, x0).theta
    assert a == pytest.approx(b)
