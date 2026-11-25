import numpy as np
import pandas as pd
import pytest

from expengine import integrity


def test_srm_flags_60_40_under_50_50():
    assert integrity.srm_test([6000, 4000])["srm"]


def test_valid_70_30_is_not_flagged():
    r = integrity.srm_test([7012, 2988], [0.7, 0.3])
    assert not r["srm"] and r["method"] == "chi_square"


def test_small_expected_counts_use_monte_carlo():
    r = integrity.srm_test([3, 4], min_expected=5)
    assert r["method"] == "monte_carlo_multinomial" and not r["srm"]


def test_vectorized_matches_scalar():
    arms = pd.DataFrame({"experiment_id": ["a", "a", "b", "b", "b"], "impressions": [500, 520, 100, 130, 95]})
    v = integrity.srm_equal_vectorized(arms)
    for eid, g in arms.groupby("experiment_id"):
        assert v.loc[eid, "srm_p_value"] == pytest.approx(integrity.srm_test(g["impressions"].to_numpy())["p_value"])


def test_srm_calibration_under_valid_design():
    rng = np.random.default_rng(5)
    flags = [integrity.srm_test(rng.multinomial(10000, [0.5, 0.5]))["srm"] for _ in range(3000)]
    assert np.mean(flags) < 0.004   # gate is 0.001
