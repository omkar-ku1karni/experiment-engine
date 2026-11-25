"""Archive audit run: integrity rules, fixed-horizon CTR inference and historical readouts.

    data/processed/{experiments,arms}.parquet
      -> data/runs/<run_id>/audit_results.parquet  one row per (experiment, rule)
      -> data/runs/<run_id>/srm.parquet            per-experiment SRM statistics
      -> data/runs/<run_id>/comparisons.parquet    each arm vs the reference arm (Holm within test)
      -> data/runs/<run_id>/decisions.parquet      one historical readout per experiment
      -> data/runs/<run_id>/summary.json
      -> data/runs/runs.parquet                    run history (registry screen)

run_id is content-addressed (config + input checksums + engine version), so re-running on
the same inputs reproduces the same id and the same outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from expengine import decision as dec
from expengine.config import DATA, ENGINE_VERSION, load_yaml, stable_hash
from expengine.inference import holm, min_expected_cell, newcombe_diff_interval, pooled_z_test, relative_lift
from expengine.integrity import audit_archive, integrity_status


def relative_mde_vectorized(p0: np.ndarray, n0: np.ndarray, n1: np.ndarray, alpha=0.05, power=0.8, iters=30) -> np.ndarray:
    """Relative MDE (upward) at fixed arm sizes, by fixed-point iteration on the power equation."""
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    d = (za + zb) * np.sqrt(2 * p0 * (1 - p0) / np.minimum(n0, n1))
    for _ in range(iters):
        p1 = np.clip(p0 + d, 1e-12, 1 - 1e-12)
        pbar = (n0 * p0 + n1 * p1) / (n0 + n1)
        d = za * np.sqrt(pbar * (1 - pbar) * (1 / n0 + 1 / n1)) + zb * np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p0 > 0, d / p0, np.nan)


def compare_to_reference(arms: pd.DataFrame, experiments: pd.DataFrame, alpha: float) -> pd.DataFrame:
    a = arms[arms["arm_included"]][["experiment_id", "arm_id", "impressions", "clicks", "created_at", "headline"]].copy()
    a["impressions"] = a["impressions"].astype(float)
    a["clicks"] = a["clicks"].astype(float)
    ref_ids = experiments.set_index("experiment_id")["reference_arm_id"]
    a["is_reference"] = a["arm_id"].eq(a["experiment_id"].map(ref_ids))
    ref = a[a["is_reference"]][["experiment_id", "arm_id", "impressions", "clicks"]].rename(
        columns={"arm_id": "reference_arm_id", "impressions": "n0", "clicks": "x0"})
    c = a[~a["is_reference"]].merge(ref, on="experiment_id", how="inner", validate="many_to_one")
    c = c.rename(columns={"impressions": "n1", "clicks": "x1"})
    x0, n0, x1, n1 = (c[k].to_numpy() for k in ("x0", "n0", "x1", "n1"))
    c["ctr_reference"], c["ctr_arm"] = x0 / n0, x1 / n1
    c["diff"] = c["ctr_arm"] - c["ctr_reference"]
    c["ci_low"], c["ci_high"] = newcombe_diff_interval(x0, n0, x1, n1, alpha)
    _, p = pooled_z_test(x0, n0, x1, n1)
    sparse = min_expected_cell(x0, n0, x1, n1) < 5
    for i in np.flatnonzero(sparse):
        p[i] = stats.fisher_exact([[x1[i], n1[i] - x1[i]], [x0[i], n0[i] - x0[i]]]).pvalue
    c["p_value"], c["test"] = p, np.where(sparse, "fisher_exact", "pooled_z")
    c["rel_lift"], c["rel_lift_ci_low"], c["rel_lift_ci_high"] = relative_lift(x0, n0, x1, n1, alpha)
    c["lift_undefined"] = x0 == 0
    c["n_comparisons"] = c.groupby("experiment_id")["arm_id"].transform("size")
    c["p_holm"] = c.groupby("experiment_id")["p_value"].transform(lambda s: holm(s.to_numpy()))
    m = c["n_comparisons"].to_numpy()
    lo_b, hi_b = newcombe_diff_interval(x0, n0, x1, n1, alpha / m)
    c["ci_low_bonferroni"], c["ci_high_bonferroni"] = lo_b, hi_b
    c["interval_note"] = "ci_low/ci_high unadjusted; *_bonferroni simultaneous over the test's comparisons"
    return c.reset_index(drop=True)


def omnibus(arms: pd.DataFrame) -> pd.DataFrame:
    a = arms[arms["arm_included"]][["experiment_id", "impressions", "clicks"]].astype({"impressions": float, "clicks": float})
    g = a.groupby("experiment_id")
    P = g["clicks"].transform("sum") / g["impressions"].transform("sum")
    e1, e0 = a["impressions"] * P, a["impressions"] * (1 - P)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = np.where(e1 > 0, (a["clicks"] - e1) ** 2 / e1, 0) + np.where(e0 > 0, ((a["impressions"] - a["clicks"]) - e0) ** 2 / e0, 0)
    out = pd.DataFrame({"homogeneity_chi2": pd.Series(contrib, index=a.index).groupby(a["experiment_id"]).sum(),
                        "k": g.size()})
    out["homogeneity_p"] = stats.chi2.sf(out["homogeneity_chi2"], out["k"] - 1)
    return out.drop(columns="k")


def run(processed: Path = DATA / "processed", runs_dir: Path = DATA / "runs", config: str = "archive.yaml") -> dict:
    cfg = load_yaml(config)
    manifest = json.loads((processed / "manifest.json").read_text())
    experiments = pd.read_parquet(processed / "experiments.parquet")
    arms = pd.read_parquet(processed / "arms.parquet")
    alpha = cfg["analysis"]["alpha"]
    config_hash = stable_hash(cfg)
    run_id = "archive-" + stable_hash([config_hash, manifest["input_hash"], ENGINE_VERSION], 10)
    out = runs_dir / run_id
    out.mkdir(parents=True, exist_ok=True)

    audit, srm = audit_archive(experiments, arms, cfg, run_id)
    status = integrity_status(audit)
    comps = compare_to_reference(arms, experiments, alpha)
    omni = omnibus(arms)

    e = experiments.set_index("experiment_id")
    ref_arm = arms[arms["arm_included"]].set_index("arm_id").loc[e.loc[e["included"], "reference_arm_id"]]
    sens = pd.DataFrame({"experiment_id": ref_arm["experiment_id"].to_numpy(),
                         "p0": (ref_arm["clicks"].astype(float) / ref_arm["impressions"].astype(float)).to_numpy()})
    mean_n = arms[arms["arm_included"]].groupby("experiment_id")["impressions"].mean().astype(float)
    sens["mean_arm_impressions"] = sens["experiment_id"].map(mean_n).to_numpy()
    sens["relative_mde_80"] = relative_mde_vectorized(sens["p0"].to_numpy(), sens["mean_arm_impressions"].to_numpy(),
                                                      sens["mean_arm_impressions"].to_numpy(), alpha,
                                                      cfg["analysis"]["sensitivity_power"])
    sens = sens.set_index("experiment_id")

    best = comps.sort_values(["experiment_id", "ctr_arm", "arm_id"], ascending=[True, False, True]).groupby("experiment_id").first()
    any_sig = comps.groupby("experiment_id")["p_holm"].min() < alpha

    rows = []
    st = status.set_index("experiment_id")
    for eid in st.index:
        crit = [r for r in st.at[eid, "critical_failures"].split(",") if r]
        b = best.loc[eid] if eid in best.index else None
        primary = None if b is None else dec.Estimate(float(b["diff"]), float(b["ci_low_bonferroni"]), float(b["ci_high_bonferroni"]))
        evidence = None if b is None else ("EVID_DIFFERENCE_DETECTED" if any_sig.get(eid, False) else "EVID_NO_DIFFERENCE_DETECTED")
        d = dec.decide(crit, primary, None, historical=True, historical_evidence=evidence)
        rows.append({
            "run_id": run_id, "experiment_id": eid, "metric": "ctr",
            "comparison": None if b is None else f"{b['arm_id']} vs {b['reference_arm_id']} (highest-CTR arm vs reference)",
            "effect": None if b is None else float(b["diff"]),
            "ci_low": None if b is None else float(b["ci_low_bonferroni"]),
            "ci_high": None if b is None else float(b["ci_high_bonferroni"]),
            "interval": "newcombe, bonferroni-simultaneous",
            "p_value": None if b is None else float(b["p_holm"]),
            "p_value_note": "Holm-adjusted within experiment",
            "rel_lift": None if b is None else (None if np.isnan(b["rel_lift"]) else float(b["rel_lift"])),
            "integrity_status": st.at[eid, "integrity_status"],
            "recommendation": d.recommendation, "actionable": d.actionable,
            "reason_codes": ",".join(d.reason_codes), "trace": json.dumps(d.trace),
            "config_hash": config_hash, "engine_version": ENGINE_VERSION,
        })
    decisions = pd.DataFrame(rows)
    decisions = decisions.join(omni, on="experiment_id").join(sens[["p0", "mean_arm_impressions", "relative_mde_80"]], on="experiment_id")

    audit.to_parquet(out / "audit_results.parquet", index=False)
    srm.to_parquet(out / "srm.parquet", index=False)
    comps.to_parquet(out / "comparisons.parquet", index=False)
    decisions.to_parquet(out / "decisions.parquet", index=False)
    status.to_parquet(out / "integrity.parquet", index=False)

    summary = {
        "run_id": run_id, "kind": "archive_audit", "config_hash": config_hash,
        "input_hash": manifest["input_hash"], "engine_version": ENGINE_VERSION,
        "experiments_audited": int(len(status)),
        "integrity_status": status["integrity_status"].value_counts().to_dict(),
        "recommendations": decisions["recommendation"].value_counts().to_dict(),
        "audit_records": int(len(audit)), "comparisons": int(len(comps)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _append_run(runs_dir, summary)
    return summary


def _append_run(runs_dir: Path, summary: dict) -> None:
    idx = runs_dir / "runs.parquet"
    row = pd.DataFrame([{
        "run_id": summary["run_id"], "kind": summary["kind"], "config_hash": summary["config_hash"],
        "input_hash": summary.get("input_hash"), "engine_version": summary["engine_version"],
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiments": summary.get("experiments_audited", summary.get("experiments")),
        "summary": json.dumps(summary),
    }])
    if idx.exists():
        row = pd.concat([pd.read_parquet(idx), row], ignore_index=True)
    row.to_parquet(idx, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="archive.yaml")
    args = ap.parse_args()
    print(json.dumps(run(config=args.config), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
