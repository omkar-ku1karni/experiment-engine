import numpy as np
import pytest

from expengine import sequential as seq


def test_gsdesign_regression_values():
    # gsDesign::gsDesign(k=5, test.type=2, alpha=0.025, sfu=sfLDOF)$upper$bound
    c = seq.boundaries(np.linspace(0.2, 1, 5), 0.05, "obf_per_side")
    assert c == pytest.approx([4.8769, 3.3569, 2.6803, 2.2898, 2.0310], abs=2e-4)


def test_first_look_closed_form_and_monotone():
    c = seq.boundaries(np.linspace(0.1, 1, 10), 0.05, "obf")
    assert c[0] == pytest.approx(1.959964 / np.sqrt(0.1), rel=1e-6)
    assert np.all(np.diff(c) < 0)


@pytest.mark.parametrize("kind", ["obf", "pocock"])
def test_monte_carlo_type_one_error(kind):
    rng = np.random.default_rng(7)
    K, R = 10, 400_000
    t = np.linspace(1 / K, 1, K)
    z = np.cumsum(rng.standard_normal((R, K)) * np.sqrt(np.diff(np.concatenate([[0], t]))), axis=1) / np.sqrt(t)
    rate = (seq.first_crossing(z, seq.boundaries(t, 0.05, kind)) >= 0).mean()
    assert abs(rate - 0.05) < 4 * np.sqrt(0.05 * 0.95 / R)


def test_naive_peeking_inflates_type_one_error():
    rng = np.random.default_rng(8)
    K, R = 10, 100_000
    t = np.linspace(1 / K, 1, K)
    z = np.cumsum(rng.standard_normal((R, K)) * np.sqrt(np.diff(np.concatenate([[0], t]))), axis=1) / np.sqrt(t)
    assert (seq.first_crossing(z, np.full(K, 1.959964)) >= 0).mean() > 0.15


def test_monitor_stops_at_first_crossing():
    looks = [{"information_fraction": f, "estimate": e, "se": 1.0} for f, e in [(0.25, 1.0), (0.5, 4.0), (0.75, 5.0), (1, 5.0)]]
    out = seq.monitor(looks)
    assert out["rejected"] and out["stop_look"] == 2 and out["rci_low"] > 0
