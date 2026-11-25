"""Analyse synthetic product experiments end to end:

    user-level outcomes -> integrity rules -> group-sequential monitoring (primary metric,
    CUPED-adjusted for revenue) -> guardrails at the stopping look -> decision -> trace

Primary intervals are Lan-DeMets O'Brien-Fleming repeated confidence intervals at the
stopping look, so peeking at the planned looks does not invalidate them. Guardrails use
the same boundary at the same look.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from expengine import decision as dec
from expengine.audit import _append_run
from expengine.config import DATA, ENGINE_VERSION, load_yaml, stable_hash
from expengine.cuped import adjust
from expengine.inference import two_proportion, welch
from expengine.integrity import FAIL, NA, PASS, integrity_status, record, srm_test
from expengine.sequential import boundaries
from expengine.simulation import ScenarioSpec, generate

REQUIRED_COLUMNS = ["experiment_id", "unit_id", "arm_id", "assigned_at", "conversion", "revenue"]


def check_integrity(df: pd.DataFrame, spec: ScenarioSpec, policy_cfg: dict, run_id: str) -> tuple[list, pd.DataFrame]:
    recs = []
    eid = spec.experiment_id
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    problems = []
    if missing_cols:
        problems.append(f"missing columns {missing_cols}")
    else:
        if df["unit_id"].isna().any():
            problems.append("null unit_id")
        if not df["arm_id"].isin(spec.arms).all():
            problems.append("unknown arm_id")
        if not df["conversion"].isin([0, 1]).all():
            problems.append("conversion not in {0,1}")
        if (df["revenue"] < 0).any():
            problems.append("negative revenue")
        if pd.to_datetime(df["assigned_at"], errors="coerce").isna().any():
            problems.append("invalid assigned_at")
    recs.append(record(run_id, eid, "R001_SCHEMA_VALID", FAIL if problems else PASS, "; ".join(problems) or "ok"))
    if problems:
        return recs, df

    arms_seen = df["arm_id"].nunique()
    recs.append(record(run_id, eid, "R002_MIN_TWO_ARMS", PASS if arms_seen >= 2 else FAIL, arms_seen, 2))

    per_unit = df.groupby("unit_id")["arm_id"].nunique()
    dup_rate = float((per_unit > 1).mean())
    max_dup = policy_cfg["max_duplicate_assignment_rate"]
    recs.append(record(run_id, eid, "R009_DUPLICATE_ASSIGNMENT", FAIL if dup_rate > max_dup else PASS,
                       f"{dup_rate:.4%} of units", f"<= {max_dup:.2%}"))
    clean = df[df["unit_id"].map(per_unit) == 1]  # analyse units with a single assignment only

    counts = clean.groupby("arm_id").size().reindex(spec.arms, fill_value=0).to_numpy()
    s = srm_test(counts, spec.allocation, alpha=policy_cfg["srm_alpha"])
    recs.append(record(run_id, eid, "R003_SRM", FAIL if s["srm"] else PASS,
                       f"p={s['p_value']:.3g}; observed {np.round(s['observed_shares'], 4).tolist()}",
                       f"expected {spec.allocation}; p<{policy_cfg['srm_alpha']} fails"))
    recs.append(record(run_id, eid, "R004_SRM_SMALL_EXPECTED", PASS if s["min_expected"] >= 5 else FAIL,
                       round(s["min_expected"], 1), 5))

    required = [g["name"] for g in spec.policy.get("guardrails", []) if g.get("required", True)]
    absent = [g for g in required if g not in df.columns]
    recs.append(record(run_id, eid, "R010_GUARDRAILS_PRESENT", FAIL if absent else PASS,
                       ",".join(absent) or "all present"))
    if "pre_period_metric" in df.columns:
        miss = float(df["pre_period_metric"].isna().mean())
        recs.append(record(run_id, eid, "R011_COVARIATE_COMPLETE", PASS if miss == 0 else FAIL, f"{miss:.1%} missing"))
    else:
        recs.append(record(run_id, eid, "R011_COVARIATE_COMPLETE", NA, "no covariate"))
    recs.append(record(run_id, eid, "R008_CONTROL_IDENTIFIED", PASS, spec.arms[0]))
    return recs, clean


def _primary_at(d: pd.DataFrame, metric: str, control: str, alpha: float) -> dict:
    c, t = d[d["arm_id"] == control], d[d["arm_id"] != control]
    if metric == "conversion":
        r = two_proportion(int(c["conversion"].sum()), len(c), int(t["conversion"].sum()), len(t), alpha)
        se = np.sqrt(r.rate_control * (1 - r.rate_control) / len(c) + r.rate_treatment * (1 - r.rate_treatment) / len(t))
        return {"estimate": r.diff, "se": float(se), "control_mean": r.rate_control, "treatment_mean": r.rate_treatment,
                "method": "difference in proportions", "n_control": len(c), "n_treatment": len(t)}
    y = d["revenue"].to_numpy(float)
    x = d["pre_period_metric"].to_numpy(float) if "pre_period_metric" in d else np.full(len(d), np.nan)
    ystar, theta, flags = adjust(y, x)
    is_c = (d["arm_id"] == control).to_numpy()
    raw = welch(y[is_c], y[~is_c], alpha)
    adj = welch(ystar[is_c], ystar[~is_c], alpha)
    return {"estimate": adj.diff, "se": adj.se, "control_mean": raw.mean_control, "treatment_mean": raw.mean_treatment,
            "raw_estimate": raw.diff, "raw_se": raw.se, "theta": theta, "variance_ratio": (adj.se / raw.se) ** 2,
            "method": "CUPED-adjusted Welch (pre-period revenue, treatment-blind theta)", "flags": flags,
            "n_control": int(is_c.sum()), "n_treatment": int((~is_c).sum())}


def _guardrails_at(d: pd.DataFrame, spec: ScenarioSpec, control: str, boundary: float):
    """Guardrail differences with a repeated CI at the same boundary as the primary look."""
    est, rows = {}, []
    for g in spec.policy.get("guardrails", []):
        name = g["name"]
        if name not in d.columns:
            est[name] = None
            rows.append({"name": name, "status": "missing", "margin": g["margin"]})
            continue
        c, t = d[d["arm_id"] == control][name], d[d["arm_id"] != control][name]
        pc, pt = float(c.mean()), float(t.mean())
        se = float(np.sqrt(pc * (1 - pc) / len(c) + pt * (1 - pt) / len(t)))
        e = dec.Estimate(pt - pc, pt - pc - boundary * se, pt - pc + boundary * se, "sequential_rci")
        est[name] = e
        rows.append({"name": name, "control": pc, "treatment": pt, "diff": e.estimate, "ci_low": e.ci_low,
                     "ci_high": e.ci_high, "margin": g["margin"], "harm_direction": g.get("harm_direction", 1)})
    return est, rows


def analyse(spec: ScenarioSpec, df: pd.DataFrame, policy_cfg: dict, run_id: str) -> dict:
    alpha = policy_cfg["alpha"]
    recs, clean = check_integrity(df, spec, policy_cfg, run_id)
    audit = pd.DataFrame([r.to_dict() for r in recs])
    status = integrity_status(audit).iloc[0]
    critical = [r for r in status["critical_failures"].split(",") if r]

    control = spec.arms[0]
    K = spec.planned_looks
    crit = boundaries(np.linspace(1 / K, 1, K), alpha, "obf")
    policy = dec.Policy.from_dict(spec.policy)
    clean = clean.sort_values("assigned_at", kind="stable")
    looks, stop = [], K
    for k in range(1, K + 1):
        d = clean.iloc[: int(round(len(clean) * k / K))]
        p = _primary_at(d, spec.primary_metric, control, alpha)
        z = p["estimate"] / p["se"] if p["se"] > 0 else 0.0
        look = {"look": k, "information_fraction": k / K, "n": len(d), "estimate": p["estimate"], "se": p["se"],
                "z": z, "boundary": float(crit[k - 1]), "rci_low": p["estimate"] - crit[k - 1] * p["se"],
                "rci_high": p["estimate"] + crit[k - 1] * p["se"], "crossed": bool(abs(z) >= crit[k - 1]),
                "action": "continue"}
        looks.append(look)
        if look["crossed"]:
            # Efficacy stop only once every required guardrail is resolved (harm or non-inferior);
            # otherwise keep collecting. RCIs stay valid at any look, and continuing past a
            # crossing can only lower the type I error of the primary test.
            g_est, _ = _guardrails_at(d, spec, control, look["boundary"])
            states = [dec.classify_guardrail(g, g_est.get(g.name))[0] for g in policy.guardrails if g.required]
            if k == K or all(s in ("harm", "non_inferior") for s in states):
                look["action"] = "stop"
                stop = k
                break
            look["action"] = "continue: guardrails unresolved"
    looks[-1]["action"] = "stop" if looks[-1]["crossed"] else "final look"
    at_stop = clean.iloc[: int(round(len(clean) * stop / K))]
    primary_detail = _primary_at(at_stop, spec.primary_metric, control, alpha)
    lk = looks[stop - 1]
    primary = dec.Estimate(lk["estimate"], lk["rci_low"], lk["rci_high"], "sequential_rci")
    guard_est, guard_rows = _guardrails_at(at_stop, spec, control, lk["boundary"])

    d = dec.decide(critical, primary, policy, guard_est)
    for row in guard_rows:
        step = next((s for s in d.trace if s["rule"] == f"guardrail:{row['name']}"), None)
        row["status"] = step["outcome"] if step else row.get("status")

    arm_summary = (clean.groupby("arm_id").agg(units=("unit_id", "size"), conversion=("conversion", "mean"),
                                                revenue_per_user=("revenue", "mean"))
                   .reindex(spec.arms).reset_index().to_dict("records"))
    return {
        "run_id": run_id, "experiment_id": spec.experiment_id, "source": "synthetic", "title": spec.title,
        "hypothesis": spec.hypothesis, "primary_metric": spec.primary_metric, "arms": spec.arms,
        "allocation": spec.allocation, "planned_units": spec.n_units, "planned_looks": K,
        "stop_look": stop, "stopped_early": stop < K, "looks": looks,
        "primary": {**asdict(primary), **{k: v for k, v in primary_detail.items() if k not in ("estimate", "se")},
                    "stop_estimate_se": lk["se"], "boundary": lk["boundary"]},
        "guardrails": guard_rows, "integrity_status": status["integrity_status"],
        "audit": audit.to_dict("records"), "arm_summary": arm_summary,
        "decision": d.to_dict(), "policy": spec.policy, "expected_recommendation": spec.expected_recommendation,
        "config_hash": stable_hash({"policy": policy_cfg, "scenario": asdict(spec)}), "engine_version": ENGINE_VERSION,
    }


def run(scenarios: str = "scenarios.yaml", policy: str = "policy.yaml", runs_dir: Path = DATA / "runs",
        only: list[str] | None = None) -> dict:
    sc_cfg, pol = load_yaml(scenarios), load_yaml(policy)
    specs = [ScenarioSpec.from_dict(s) for s in sc_cfg["scenarios"] if not only or s["experiment_id"] in only]
    run_id = "synthetic-" + stable_hash([sc_cfg["scenarios"], pol, ENGINE_VERSION], 10)
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=True)
    (DATA / "synthetic").mkdir(parents=True, exist_ok=True)
    results = []
    for spec in specs:
        df = generate(spec)
        df.to_parquet(DATA / "synthetic" / f"{spec.experiment_id}.parquet", index=False)
        res = analyse(spec, df, pol, run_id)
        results.append(res)
        print(f"{spec.experiment_id:<22} {res['decision']['recommendation']:<13} "
              f"(expected {spec.expected_recommendation:<12}) {','.join(res['decision']['reason_codes'])}")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str))
    rows = [{"run_id": r["run_id"], "experiment_id": r["experiment_id"], "metric": r["primary_metric"],
             "effect": r["primary"]["estimate"], "ci_low": r["primary"]["ci_low"], "ci_high": r["primary"]["ci_high"],
             "interval": r["primary"]["interval"], "integrity_status": r["integrity_status"],
             "recommendation": r["decision"]["recommendation"], "reason_codes": ",".join(r["decision"]["reason_codes"]),
             "trace": json.dumps(r["decision"]["trace"]), "config_hash": r["config_hash"],
             "engine_version": ENGINE_VERSION} for r in results]
    pd.DataFrame(rows).to_parquet(out / "decisions.parquet", index=False)
    pd.DataFrame([a for r in results for a in r["audit"]]).to_parquet(out / "audit_results.parquet", index=False)
    summary = {"run_id": run_id, "kind": "synthetic_product", "config_hash": stable_hash([sc_cfg, pol]),
               "engine_version": ENGINE_VERSION, "experiments": len(results),
               "recommendations": pd.Series([r["decision"]["recommendation"] for r in results]).value_counts().to_dict(),
               "all_match_expected": all(r["decision"]["recommendation"] == r["expected_recommendation"] for r in results)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _append_run(runs_dir, summary)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    s = run(only=a.only)
    print(json.dumps(s, indent=2))
    return 0 if s["all_match_expected"] else 1


if __name__ == "__main__":
    sys.exit(main())
