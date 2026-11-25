import numpy as np
import pytest

from expengine import power


def test_textbook_sample_size():
    n0, n1 = power.sample_size_per_arm(0.05, 0.10, relative=True)
    assert n0 == n1 == 31234          # 5% -> 5.5%, alpha .05, power .8


def test_statsmodels_crosscheck_within_two_percent():
    for p0, mde in [(0.01, 0.2), (0.05, 0.1), (0.2, 0.05)]:
        n0, _ = power.sample_size_per_arm(p0, mde, relative=True)
        sm = power.statsmodels_sample_size(p0, p0 * (1 + mde))
        assert abs(n0 / sm - 1) < 0.02


def test_power_at_planned_n_is_target():
    n0, n1 = power.sample_size_per_arm(0.05, 0.10, relative=True, treatment_share=0.3)
    assert power.power_two_proportions(0.05, 0.055, n0, n1) == pytest.approx(0.8, abs=0.002)


def test_unequal_allocation_needs_more_units():
    eq = sum(power.sample_size_per_arm(0.05, 0.1, relative=True))
    uneq = sum(power.sample_size_per_arm(0.05, 0.1, relative=True, treatment_share=0.3))
    assert uneq > eq


def test_mde_inverts_power():
    d = power.mde_at_sample_size(0.05, 31234, 31234)
    assert d == pytest.approx(0.005, rel=1e-3)


def test_simulated_power_matches_formula():
    rng = np.random.default_rng(3)
    n = 31234
    x0, x1 = rng.binomial(n, 0.05, 4000), rng.binomial(n, 0.055, 4000)
    pp = (x0 + x1) / (2 * n)
    z = (x1 - x0) / n / np.sqrt(pp * (1 - pp) * 2 / n)
    emp = np.mean(np.abs(z) > 1.959964)
    assert abs(emp - 0.8) < 3 * np.sqrt(0.8 * 0.2 / 4000)


def test_plan_duration_and_invalid_inputs():
    p = power.plan(0.05, 0.10, daily_traffic=10000)
    assert p.n_total == 62468 and p.expected_duration_days == pytest.approx(6.3)
    with pytest.raises(ValueError):
        power.plan(0.95, 0.10)        # treatment rate would exceed 1
