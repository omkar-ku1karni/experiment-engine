import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from expengine import inference as inf


def test_wilson_matches_statsmodels():
    for x, n in [(0, 50), (3, 50), (25, 50), (50, 50), (120, 10000)]:
        lo, hi = inf.wilson_interval(x, n)
        slo, shi = proportion_confint(x, n, method="wilson")
        assert lo == pytest.approx(slo, abs=1e-10) and hi == pytest.approx(shi, abs=1e-10)


def test_newcombe_published_example():
    # Newcombe (1998), Stat Med 17:873, example (a): 56/70 vs 48/80 -> 0.2000 (0.0524, 0.3339)
    lo, hi = inf.newcombe_diff_interval(48, 80, 56, 70)
    assert lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)


def test_zero_clicks_no_division_errors():
    r = inf.two_proportion(0, 400, 0, 410)
    assert r.p_value == 1.0 and "ZERO_EVENTS" in r.flags and r.rel_lift is None
    assert np.isfinite(r.ci_low) and np.isfinite(r.ci_high)


def test_sparse_counts_use_fisher_and_flag_undefined_lift():
    r = inf.two_proportion(0, 400, 3, 410)
    assert r.test == "fisher_exact" and "SPARSE_COUNTS" in r.flags
    assert "LIFT_UNDEFINED_ZERO_CONTROL" in r.flags and r.rel_lift is None
    assert r.p_value == pytest.approx(stats.fisher_exact([[3, 407], [0, 400]]).pvalue)


def test_large_counts_use_pooled_z():
    r = inf.two_proportion(500, 10000, 560, 10000)
    assert r.test == "pooled_z"
    pp = 1060 / 20000
    z = 0.006 / np.sqrt(pp * (1 - pp) * 2 / 10000)
    assert r.p_value == pytest.approx(2 * stats.norm.sf(z))
    assert r.rel_lift == pytest.approx(0.12)


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        inf.two_proportion(11, 10, 1, 10)
    with pytest.raises(ValueError):
        inf.two_proportion(0, 0, 1, 10)


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = rng.random(rng.integers(1, 8)) ** 3
        assert np.allclose(inf.holm(p), multipletests(p, method="holm")[1])


def test_welch_matches_scipy_and_includes_zeros():
    rng = np.random.default_rng(1)
    y0 = np.where(rng.random(500) < 0.9, 0, rng.lognormal(3, 1, 500))
    y1 = np.where(rng.random(520) < 0.88, 0, rng.lognormal(3, 1, 520))
    r = inf.welch(y0, y1)
    ref = stats.ttest_ind(y1, y0, equal_var=False)
    assert r.p_value == pytest.approx(ref.pvalue) and r.n_control == 500
    lo, hi = inf.bootstrap_mean_diff(y0, y1, n_boot=500)
    assert lo < r.diff < hi


def test_homogeneity_matches_scipy():
    c, n = np.array([25, 40, 26, 26]), np.array([2500, 2490, 2510, 2505])
    stat, p, dof = inf.homogeneity_test(c, n)
    ref = stats.chi2_contingency(np.vstack([c, n - c]), correction=False)
    assert p == pytest.approx(ref.pvalue) and dof == 3
