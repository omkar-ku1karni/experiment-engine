"""Deterministic decision policy. Health is evaluated before impact, impact before value.

Priority (first match wins; every evaluated rule is written to the trace):
  1. BLOCKED       critical integrity failure, invalid schema or unusable analysis
  2. NO_SHIP       evidence of unacceptable guardrail harm, or the primary interval rules
                   out a worthwhile improvement (upper bound below delta)
  3. SHIP          primary effect clears delta under the evidence rule AND every required
                   guardrail is shown non-inferior (harm interval upper bound below its margin)
  4. INCONCLUSIVE  everything else (insufficient evidence, missing inputs, uncertain guardrails)

Historical archive records have no commercial threshold, guardrails or control, so they
get a statistical readout with a non-actionable INCONCLUSIVE recommendation (unless BLOCKED).

A p-value below 0.05 is never sufficient on its own: SHIP needs integrity, a predeclared
delta, an interval-based evidence rule, and guardrail safety.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

BLOCKED, NO_SHIP, SHIP, INCONCLUSIVE = "blocked", "no_ship", "ship", "inconclusive"

REASONS = {
    # blocked
    "BLK_SCHEMA_INVALID": "Source records failed schema validation.",
    "BLK_MIN_TWO_ARMS": "Fewer than two analysable arms.",
    "BLK_SRM": "Sample ratio mismatch: observed allocation is inconsistent with the design.",
    "BLK_DOCUMENTED_RANDOMIZATION_PROBLEM": "Archive documents unreliable randomization for this period.",
    "BLK_DUPLICATE_ASSIGNMENT": "Units were exposed to more than one arm.",
    "BLK_ANALYSIS_INVALID": "Primary metric estimate or interval could not be computed.",
    # no ship
    "NS_GUARDRAIL_HARM": "A guardrail shows statistically significant harm beyond its margin.",
    "NS_UPPER_BELOW_DELTA": "Primary interval upper bound is below the minimum worthwhile effect.",
    # ship
    "SHIP_PRIMARY_CLEARS_DELTA": "Primary effect meets the evidence rule against delta.",
    "SHIP_GUARDRAILS_NONINFERIOR": "Every required guardrail is non-inferior within its margin.",
    # inconclusive
    "INC_INSUFFICIENT_EVIDENCE": "Primary interval is compatible with both worthwhile and non-worthwhile effects.",
    "INC_MISSING_DELTA": "No minimum worthwhile effect was predeclared.",
    "INC_MISSING_GUARDRAIL": "A required guardrail metric is missing.",
    "INC_GUARDRAIL_UNCERTAIN": "Guardrail interval cannot rule out harm beyond its margin.",
    # historical readouts
    "HIST_NO_COMMERCIAL_THRESHOLD": "Archive records carry no predeclared minimum worthwhile effect.",
    "HIST_NO_GUARDRAILS": "Archive records carry no guardrail metrics.",
    "HIST_NO_CONTROL": "No control arm is recorded; comparisons use a predeclared exploratory reference arm.",
    "EVID_DIFFERENCE_DETECTED": "At least one arm differs from the reference after Holm adjustment.",
    "EVID_NO_DIFFERENCE_DETECTED": "No arm differs from the reference after Holm adjustment.",
}

RULE_TO_BLOCK = {
    "R001_SCHEMA_VALID": "BLK_SCHEMA_INVALID",
    "R002_MIN_TWO_ARMS": "BLK_MIN_TWO_ARMS",
    "R003_SRM": "BLK_SRM",
    "R005_DOCUMENTED_RANDOMIZATION_PROBLEM": "BLK_DOCUMENTED_RANDOMIZATION_PROBLEM",
    "R009_DUPLICATE_ASSIGNMENT": "BLK_DUPLICATE_ASSIGNMENT",
}


@dataclass
class Guardrail:
    name: str
    margin: float                      # largest acceptable harm (absolute, in harm direction)
    harm_direction: int = 1            # +1: an increase is harmful; -1: a decrease is harmful
    required: bool = True


@dataclass
class Policy:
    delta: float | None                # minimum worthwhile absolute effect on the primary metric
    evidence_rule: str = "lcb_gt_zero_and_estimate_ge_delta"
    guardrails: list[Guardrail] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        return cls(d.get("delta"), d.get("evidence_rule", "lcb_gt_zero_and_estimate_ge_delta"),
                   [Guardrail(**g) for g in d.get("guardrails", [])])


@dataclass
class Estimate:
    estimate: float
    ci_low: float
    ci_high: float
    interval: str = "fixed"            # fixed | sequential_rci


@dataclass
class Decision:
    recommendation: str
    actionable: bool
    reason_codes: list[str]
    trace: list[dict]

    def to_dict(self) -> dict:
        return asdict(self)


def _step(trace, rule, outcome, detail):
    trace.append({"step": len(trace) + 1, "rule": rule, "outcome": outcome, "detail": detail})


def classify_guardrail(g: Guardrail, est: Estimate | None) -> tuple[str, str]:
    """harm | non_inferior | uncertain | missing, plus a human-readable detail string.

    harm:          interval excludes zero in the harmful direction AND the estimate exceeds the margin
    non_inferior:  the whole interval lies below the margin (harm ruled out)
    uncertain:     neither
    """
    if est is None:
        return "missing", "required" if g.required else "optional"
    s = g.harm_direction
    harm = s * est.estimate
    lo, hi = sorted((s * est.ci_low, s * est.ci_high))
    if lo > 0 and harm > g.margin:
        outcome = "harm"
    elif hi < g.margin:
        outcome = "non_inferior"
    else:
        outcome = "uncertain"
    return outcome, f"harm {harm:+.5f} [{lo:+.5f}, {hi:+.5f}] vs margin {g.margin:+.5f}"


def _primary_clears(e: Estimate, delta: float, rule: str) -> bool:
    if rule == "lcb_ge_delta":
        return e.ci_low >= delta
    if rule == "lcb_gt_zero_and_estimate_ge_delta":
        return e.ci_low > 0 and e.estimate >= delta
    raise ValueError(f"unknown evidence rule {rule!r}")


def decide(
    critical_failures: list[str],
    primary: Estimate | None,
    policy: Policy | None,
    guardrails: dict[str, Estimate | None] | None = None,
    historical: bool = False,
    historical_evidence: str | None = None,
) -> Decision:
    trace: list[dict] = []
    guardrails = guardrails or {}

    # 1. Blocked ---------------------------------------------------------------
    blocks = [RULE_TO_BLOCK.get(r, "BLK_SCHEMA_INVALID") for r in critical_failures]
    if primary is None or any(v != v for v in (primary.estimate, primary.ci_low, primary.ci_high)):
        blocks.append("BLK_ANALYSIS_INVALID")
    _step(trace, "integrity_gate", "fail" if blocks else "pass",
          ", ".join(critical_failures) if critical_failures else "no critical integrity failures")
    if blocks:
        codes = list(dict.fromkeys(blocks))
        return Decision(BLOCKED, False, codes, trace)

    # Historical readout --------------------------------------------------------
    if historical:
        codes = ["HIST_NO_COMMERCIAL_THRESHOLD", "HIST_NO_GUARDRAILS", "HIST_NO_CONTROL"]
        if historical_evidence:
            codes.append(historical_evidence)
        _step(trace, "historical_readout", "non_actionable",
              "archive record: statistical readout only; commercial threshold, guardrails and control are absent")
        return Decision(INCONCLUSIVE, False, codes, trace)

    assert policy is not None, "product decisions need a policy"

    # 2. No ship -----------------------------------------------------------------
    harm_codes, uncertain, missing, safe = [], [], [], []
    for g in policy.guardrails:
        outcome, detail = classify_guardrail(g, guardrails.get(g.name))
        {"missing": missing, "non_inferior": safe, "uncertain": uncertain}.get(outcome, []).append(g.name)
        if outcome == "harm":
            harm_codes.append(f"NS_GUARDRAIL_HARM:{g.name}")
        _step(trace, f"guardrail:{g.name}", outcome, detail)

    upper_below = policy.delta is not None and primary.ci_high < policy.delta
    _step(trace, "primary_upper_bound_vs_delta", "below" if upper_below else ("not_evaluated" if policy.delta is None else "not_below"),
          f"upper {primary.ci_high:+.5f} vs delta {policy.delta}")
    if harm_codes or upper_below:
        codes = harm_codes + (["NS_UPPER_BELOW_DELTA"] if upper_below else [])
        return Decision(NO_SHIP, True, codes, trace)

    # 3. Ship --------------------------------------------------------------------
    clears = policy.delta is not None and _primary_clears(primary, policy.delta, policy.evidence_rule)
    _step(trace, f"primary_evidence:{policy.evidence_rule}", "pass" if clears else "fail",
          f"estimate {primary.estimate:+.5f} [{primary.ci_low:+.5f}, {primary.ci_high:+.5f}] ({primary.interval}) vs delta {policy.delta}")
    required_missing = [g.name for g in policy.guardrails if g.required and g.name in missing]
    required_uncertain = [g.name for g in policy.guardrails if g.required and g.name in uncertain]
    if clears and not required_missing and not required_uncertain:
        return Decision(SHIP, True, ["SHIP_PRIMARY_CLEARS_DELTA", "SHIP_GUARDRAILS_NONINFERIOR"], trace)

    # 4. Inconclusive -------------------------------------------------------------
    codes = []
    if policy.delta is None:
        codes.append("INC_MISSING_DELTA")
    elif not clears:
        codes.append("INC_INSUFFICIENT_EVIDENCE")
    codes += [f"INC_MISSING_GUARDRAIL:{n}" for n in required_missing]
    codes += [f"INC_GUARDRAIL_UNCERTAIN:{n}" for n in required_uncertain]
    return Decision(INCONCLUSIVE, True, codes, trace)


def describe(code: str) -> str:
    base = code.split(":", 1)[0]
    suffix = f" ({code.split(':', 1)[1]})" if ":" in code else ""
    return REASONS.get(base, code) + suffix
