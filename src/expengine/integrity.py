"""Experiment health checks. Each rule emits one auditable record:

    run_id, experiment_id, rule_id, severity, status, observed, threshold, explanation

severity: critical (blocks any recommendation) | warning | info
status:   pass | fail | not_applicable

Sample ratio mismatch uses the chi-square goodness-of-fit test against the expected
allocation (df = k - 1). When the smallest expected count is below `min_expected`, the
p-value comes from a Monte Carlo multinomial test instead of the chi-square approximation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

CRITICAL, WARNING, INFO = "critical", "warning", "info"
PASS, FAIL, NA = "pass", "fail", "not_applicable"

RULES = {
    "R001_SCHEMA_VALID": (CRITICAL, "All arm records parse: counts present, integers, non-negative, clicks <= impressions, valid timestamps, unique keys."),
    "R002_MIN_TWO_ARMS": (CRITICAL, "An experiment needs at least two arms."),
    "R003_SRM": (CRITICAL, "Observed allocation is consistent with the expected allocation (chi-square / multinomial test)."),
    "R004_SRM_SMALL_EXPECTED": (INFO, "Smallest expected arm count is large enough for the chi-square approximation."),
    "R005_DOCUMENTED_RANDOMIZATION_PROBLEM": (CRITICAL, "Experiment is outside the archive's documented randomization-problem window (Jun 25 2013 - Jan 10 2014)."),
    "R006_ZERO_EVENT_ARM": (WARNING, "Every arm has at least one click/conversion (otherwise exact methods are used)."),
    "R007_DUPLICATE_STIMULUS": (INFO, "No two arms show an identical headline + image package."),
    "R008_CONTROL_IDENTIFIED": (WARNING, "A control arm is recorded in the design (otherwise comparisons are exploratory)."),
    "R009_DUPLICATE_ASSIGNMENT": (CRITICAL, "Units are assigned to exactly one arm."),
    "R010_GUARDRAILS_PRESENT": (WARNING, "All guardrail metrics required by the policy are present."),
    "R011_COVARIATE_COMPLETE": (INFO, "Pre-period covariate is available for CUPED."),
}


@dataclass
class AuditRecord:
    run_id: str
    experiment_id: str
    rule_id: str
    severity: str
    status: str
    observed: str | None
    threshold: str | None
    explanation: str

    def to_dict(self) -> dict:
        return asdict(self)


def record(run_id, experiment_id, rule_id, status, observed=None, threshold=None, note=None) -> AuditRecord:
    severity, text = RULES[rule_id]
    return AuditRecord(run_id, str(experiment_id), rule_id, severity, status,
                       None if observed is None else str(observed),
                       None if threshold is None else str(threshold),
                       note or text)


# --------------------------------------------------------------------------- SRM
def srm_test(
    observed, expected_shares=None, alpha: float = 0.001, min_expected: float = 5.0,
    mc_draws: int = 20000, seed: int = 0,
) -> dict:
    obs = np.asarray(observed, float)
    k = len(obs)
    shares = np.full(k, 1 / k) if expected_shares is None else np.asarray(expected_shares, float)
    shares = shares / shares.sum()
    n = obs.sum()
    exp = n * shares
    stat = float(np.sum((obs - exp) ** 2 / exp)) if n > 0 else 0.0
    if n == 0:
        return {"statistic": 0.0, "df": k - 1, "p_value": 1.0, "method": "no_units", "min_expected": 0.0,
                "srm": False, "observed_shares": [0.0] * k, "expected_shares": shares.tolist()}
    if exp.min() >= min_expected:
        p, method = float(stats.chi2.sf(stat, k - 1)), "chi_square"
    else:
        rng = np.random.default_rng(seed)
        sims = rng.multinomial(int(n), shares, size=mc_draws)
        sim_stat = ((sims - exp) ** 2 / exp).sum(axis=1)
        p, method = float((np.sum(sim_stat >= stat - 1e-12) + 1) / (mc_draws + 1)), "monte_carlo_multinomial"
    return {"statistic": stat, "df": k - 1, "p_value": p, "method": method, "min_expected": float(exp.min()),
            "srm": p < alpha, "observed_shares": (obs / n).tolist(), "expected_shares": shares.tolist()}


def srm_equal_vectorized(arms: pd.DataFrame, group="experiment_id", count="impressions") -> pd.DataFrame:
    """Chi-square SRM for many experiments at once under equal allocation."""
    g = arms.groupby(group)[count]
    n, k = g.transform("sum").astype(float), g.transform("size").astype(float)
    exp = n / k
    contrib = (arms[count].astype(float) - exp) ** 2 / exp
    out = pd.DataFrame({"srm_statistic": contrib.groupby(arms[group]).sum(), "n_arms": g.size(), "n_units": g.sum()})
    out["srm_df"] = out["n_arms"] - 1
    out["srm_min_expected"] = out["n_units"] / out["n_arms"]
    out["srm_p_value"] = stats.chi2.sf(out["srm_statistic"], out["srm_df"])
    out["srm_method"] = "chi_square"
    return out


# --------------------------------------------------------------------------- archive audit
def audit_archive(experiments: pd.DataFrame, arms: pd.DataFrame, cfg: dict, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rule-level records for every deployed experiment, plus a per-experiment SRM table."""
    a_cfg = cfg["analysis"]
    deployed = experiments[experiments["split"].isin(cfg["deployed_splits"])].set_index("experiment_id")
    use = arms[arms["arm_included"]].copy()
    use["impressions"] = use["impressions"].astype("int64")
    use["clicks"] = use["clicks"].astype("int64")

    srm = srm_equal_vectorized(use)
    small = srm.index[srm["srm_min_expected"] < a_cfg["min_expected_count"]]
    for i, eid in enumerate(small):  # exact-ish fallback where the approximation is poor
        counts = use.loc[use["experiment_id"] == eid, "impressions"].to_numpy()
        r = srm_test(counts, alpha=a_cfg["srm_alpha"], min_expected=a_cfg["min_expected_count"],
                     mc_draws=a_cfg["srm_mc_draws"], seed=i)
        srm.loc[eid, ["srm_p_value", "srm_method"]] = [r["p_value"], r["method"]]
    srm["srm_flag"] = srm["srm_p_value"] < a_cfg["srm_alpha"]

    zero_arm = use.groupby("experiment_id")["clicks"].min().eq(0)
    stim = use.assign(_k=use["headline"].fillna("") + "|" + use["eyecatcher_id"].fillna(""))
    dup_stim = stim.groupby("experiment_id")["_k"].agg(lambda s: s.duplicated().any())

    recs: list[AuditRecord] = []
    for eid, e in deployed.iterrows():
        if not e["included"]:
            recs.append(record(run_id, eid, "R001_SCHEMA_VALID" if e["exclusion_reason"] != "fewer_than_two_arms" else "R002_MIN_TWO_ARMS",
                               FAIL, e["exclusion_reason"], None,
                               f"Excluded at ingestion: {e['exclusion_reason']}."))
            continue
        recs.append(record(run_id, eid, "R001_SCHEMA_VALID", PASS))
        recs.append(record(run_id, eid, "R002_MIN_TWO_ARMS", PASS, int(e["n_arms_valid"]), 2))
        s = srm.loc[eid]
        recs.append(record(
            run_id, eid, "R003_SRM", FAIL if s["srm_flag"] else PASS, f"p={s['srm_p_value']:.3g} ({s['srm_method']})",
            f"p<{a_cfg['srm_alpha']}",
            "Impression counts are inconsistent with equal allocation (assumed; not recorded in source)."
            if s["srm_flag"] else None))
        recs.append(record(run_id, eid, "R004_SRM_SMALL_EXPECTED",
                           FAIL if s["srm_min_expected"] < a_cfg["min_expected_count"] else PASS,
                           round(float(s["srm_min_expected"]), 1), a_cfg["min_expected_count"]))
        recs.append(record(run_id, eid, "R005_DOCUMENTED_RANDOMIZATION_PROBLEM",
                           FAIL if e["problem_reconstructed"] else PASS, str(e["experiment_date"]),
                           f"{cfg['problem_window']['start']}..{cfg['problem_window']['end']}"))
        recs.append(record(run_id, eid, "R006_ZERO_EVENT_ARM", FAIL if zero_arm.get(eid, False) else PASS))
        recs.append(record(run_id, eid, "R007_DUPLICATE_STIMULUS", FAIL if dup_stim.get(eid, False) else PASS))
        recs.append(record(run_id, eid, "R008_CONTROL_IDENTIFIED", FAIL, "none recorded", None,
                           "No control is recorded in the archive; the earliest-created package is the predeclared "
                           "exploratory reference arm."))
    audit = pd.DataFrame([r.to_dict() for r in recs])
    return audit, srm.reset_index().rename(columns={"index": "experiment_id"})


def integrity_status(audit: pd.DataFrame) -> pd.DataFrame:
    """Per-experiment summary: blocked if any critical rule fails, else warning/pass."""
    fails = audit[audit["status"] == FAIL]
    crit = fails[fails["severity"] == CRITICAL].groupby("experiment_id")["rule_id"].agg(list)
    warn = fails[fails["severity"] == WARNING].groupby("experiment_id")["rule_id"].agg(list)
    ids = audit["experiment_id"].unique()
    out = pd.DataFrame(index=pd.Index(ids, name="experiment_id"))
    out["critical_failures"] = crit.reindex(out.index)
    out["warnings"] = warn.reindex(out.index)
    out["integrity_status"] = np.where(out["critical_failures"].notna(), "blocked",
                                       np.where(out["warnings"].notna(), "warning", "pass"))
    for c in ("critical_failures", "warnings"):
        out[c] = out[c].map(lambda v: ",".join(v) if isinstance(v, list) else "")
    return out.reset_index()
