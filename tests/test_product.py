"""Synthetic product scenarios: each demonstrates one recommendation state."""

import pytest

from expengine.config import load_yaml
from expengine.memo import product_context, render
from expengine.product import analyse
from expengine.simulation import ScenarioSpec, generate

SC, POL = load_yaml("scenarios.yaml"), load_yaml("policy.yaml")


@pytest.fixture(scope="module")
def results():
    out = {}
    for s in SC["scenarios"]:
        spec = ScenarioSpec.from_dict(s)
        out[spec.experiment_id] = analyse(spec, generate(spec), POL, "test-run")
    return out


def test_every_scenario_reaches_its_documented_recommendation(results):
    for s in SC["scenarios"]:
        assert results[s["experiment_id"]]["decision"]["recommendation"] == s["expected_recommendation"], s["experiment_id"]


def test_all_four_states_are_demonstrated(results):
    assert {r["decision"]["recommendation"] for r in results.values()} == {"ship", "no_ship", "inconclusive", "blocked"}


def test_valid_unequal_allocation_is_not_flagged(results):
    srm = [a for a in results["aa-70-30"]["audit"] if a["rule_id"] == "R003_SRM"][0]
    assert srm["status"] == "pass"


def test_blocked_experiments_never_ship(results):
    for r in results.values():
        if r["integrity_status"] == "blocked":
            assert r["decision"]["recommendation"] == "blocked"


def test_generation_and_readout_are_deterministic(results):
    spec = ScenarioSpec.from_dict(SC["scenarios"][0])
    again = analyse(spec, generate(spec), POL, "test-run")
    assert again["decision"] == results[spec.experiment_id]["decision"]
    assert render(product_context(again, "2026-01-01")) == render(product_context(results[spec.experiment_id], "2026-01-01"))


def test_memo_contains_required_sections(results):
    html = render(product_context(results["recs-carousel"], "2026-01-01"))
    for s in ["Hypothesis and design", "Integrity verdict", "Primary metric", "Guardrails", "Recommendation",
              "Next steps", "Missing evidence", "Decision trace", "NS_GUARDRAIL_HARM:refund_rate", "SYNTHETIC SCENARIO"]:
        assert s in html
