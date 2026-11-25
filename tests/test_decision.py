import numpy as np
import pytest

from expengine import decision as dec

GUARDS = [dec.Guardrail("refund_rate", 0.003), dec.Guardrail("support_rate", 0.005)]
POLICY = dec.Policy(delta=0.05, guardrails=GUARDS)
SAFE = {"refund_rate": dec.Estimate(0.0005, -0.001, 0.002), "support_rate": dec.Estimate(0.0, -0.002, 0.002)}


def test_ship_requires_delta_and_safe_guardrails():
    d = dec.decide([], dec.Estimate(0.20, 0.10, 0.30), POLICY, SAFE)
    assert d.recommendation == "ship" and d.reason_codes == ["SHIP_PRIMARY_CLEARS_DELTA", "SHIP_GUARDRAILS_NONINFERIOR"]


def test_positive_but_tiny_lift_is_not_shipped():
    d = dec.decide([], dec.Estimate(0.010, 0.004, 0.016), POLICY, SAFE)   # significant, below delta
    assert d.recommendation == "no_ship" and "NS_UPPER_BELOW_DELTA" in d.reason_codes


def test_meaningful_lift_with_harmful_guardrail_is_no_ship():
    g = dict(SAFE, refund_rate=dec.Estimate(0.008, 0.006, 0.010))
    d = dec.decide([], dec.Estimate(0.30, 0.20, 0.40), POLICY, g)
    assert d.recommendation == "no_ship" and "NS_GUARDRAIL_HARM:refund_rate" in d.reason_codes


def test_missing_guardrail_is_inconclusive_not_ship():
    d = dec.decide([], dec.Estimate(0.30, 0.20, 0.40), POLICY, {"refund_rate": SAFE["refund_rate"]})
    assert d.recommendation == "inconclusive" and "INC_MISSING_GUARDRAIL:support_rate" in d.reason_codes


def test_missing_delta_is_inconclusive():
    d = dec.decide([], dec.Estimate(0.30, 0.20, 0.40), dec.Policy(delta=None, guardrails=GUARDS), SAFE)
    assert d.recommendation == "inconclusive" and "INC_MISSING_DELTA" in d.reason_codes


def test_stricter_evidence_rule():
    e = dec.Estimate(0.08, 0.02, 0.14)
    assert dec.decide([], e, POLICY, SAFE).recommendation == "ship"
    strict = dec.Policy(0.05, "lcb_ge_delta", GUARDS)
    assert dec.decide([], e, strict, SAFE).recommendation == "inconclusive"


def test_invalid_experiments_never_ship():
    """Property: any critical integrity failure yields BLOCKED, whatever the evidence says."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        est = rng.normal(0.5, 0.5)
        half = abs(rng.normal(0.05, 0.05))
        crit = list(rng.choice(["R001_SCHEMA_VALID", "R003_SRM", "R009_DUPLICATE_ASSIGNMENT",
                                "R005_DOCUMENTED_RANDOMIZATION_PROBLEM"], size=rng.integers(1, 3), replace=False))
        d = dec.decide(crit, dec.Estimate(est, est - half, est + half), POLICY, SAFE)
        assert d.recommendation == "blocked" and not d.actionable


def test_nan_estimate_is_blocked():
    d = dec.decide([], dec.Estimate(float("nan"), 0, 1), POLICY, SAFE)
    assert d.recommendation == "blocked" and d.reason_codes == ["BLK_ANALYSIS_INVALID"]


def test_historical_readout_is_never_actionable():
    d = dec.decide([], dec.Estimate(0.02, 0.01, 0.03), None, historical=True, historical_evidence="EVID_DIFFERENCE_DETECTED")
    assert d.recommendation == "inconclusive" and not d.actionable and "HIST_NO_CONTROL" in d.reason_codes


def test_trace_is_deterministic_and_complete():
    a = dec.decide([], dec.Estimate(0.20, 0.10, 0.30), POLICY, SAFE)
    b = dec.decide([], dec.Estimate(0.20, 0.10, 0.30), POLICY, SAFE)
    assert a.to_dict() == b.to_dict()
    assert [s["rule"] for s in a.trace][:3] == ["integrity_gate", "guardrail:refund_rate", "guardrail:support_rate"]
    assert all(dec.describe(c) != c for c in a.reason_codes)
