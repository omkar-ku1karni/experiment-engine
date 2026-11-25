"""Seeded Monte Carlo benchmarks (all results are simulations, not archive replays).

  fixed       A/A and effect calibration of the fixed-horizon proportion test + Newcombe interval
  srm         false-flag rate under valid 50/50, 70/30 and 4-arm designs; detection of 60/40 under 50/50
  sequential  fixed horizon vs naive peeking (10 looks) vs Lan-DeMets OBF / Pocock group-sequential,
              evaluated on IDENTICAL simulated streams (paired comparison)
  cuped       variance ratio, bias, coverage, power and equivalent sample-size reduction across
              covariate correlations, with and without missing covariates

Every rate is reported with its Monte Carlo standard error sqrt(p(1-p)/R).
Seeds derive from (base seed, scenario hash), so any scenario can be re-run in isolation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from expengine.config import ROOT, load_yaml, stable_hash
from expengine.inference import min_expected_cell, newcombe_diff_interval, pooled_z_test
from expengine.power import sample_size_per_arm
from expengine.sequential import boundaries

OUT = ROOT / "simulations" / "results"


def _rng(base: int, scenario: dict) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([base, int(stable_hash(scenario, 8), 16)]))


def _rate(hits: np.ndarray) -> tuple[float, float]:
    p = float(np.mean(hits))
    return p, float(np.sqrt(p * (1 - p) / len(hits)))


# --------------------------------------------------------------------------- fixed horizon
def fixed_calibration(cfg: dict, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for p0, n_arm, share, rel in product(cfg["baselines"], cfg["n_per_arm"], cfg["treatment_shares"], cfg["relative_effects"]):
        sc = dict(bench="fixed", p0=p0, n_arm=n_arm, share=share, rel=rel)
        rng = _rng(seed, sc)
        n_tot = 2 * n_arm
        n1 = int(round(n_tot * share))
        n0 = n_tot - n1
        p1 = p0 * (1 + rel)
        x0, x1 = rng.binomial(n0, p0, reps), rng.binomial(n1, p1, reps)
        _, p = pooled_z_test(x0, n0, x1, n1)
        sparse = min_expected_cell(x0, n0, x1, n1) < 5
        for i in np.flatnonzero(sparse):
            p[i] = stats.fisher_exact([[x1[i], n1 - x1[i]], [x0[i], n0 - x0[i]]]).pvalue
        lo, hi = newcombe_diff_interval(x0, n0, x1, n1, 0.05)
        rej, rej_se = _rate(p < 0.05)
        cov, cov_se = _rate((lo <= p1 - p0) & (p1 - p0 <= hi))
        rows.append({**sc, "n0": n0, "n1": n1, "reps": reps, "rejection_rate": rej, "rejection_se": rej_se,
                     "coverage": cov, "coverage_se": cov_se, "share_sparse_fisher": float(sparse.mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- SRM
def srm_calibration(cfg: dict, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for case, n in product(cfg["cases"], cfg["n_total"]):
        sc = dict(bench="srm", case=case["name"], n=n)
        rng = _rng(seed, sc)
        exp_sh = np.asarray(case["expected"], float)
        act_sh = np.asarray(case["actual"], float)
        counts = rng.multinomial(n, act_sh / act_sh.sum(), size=reps)
        e = n * exp_sh / exp_sh.sum()
        chi = ((counts - e) ** 2 / e).sum(axis=1)
        pv = stats.chi2.sf(chi, len(e) - 1)
        rate, se = _rate(pv < cfg["alpha"])
        rows.append({**sc, "expected": case["expected"], "actual": case["actual"], "valid_design": case["valid"],
                     "reps": reps, "srm_flag_rate": rate, "srm_flag_se": se, "gate": cfg["alpha"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- sequential
def sequential(cfg: dict, reps: int, seed: int) -> pd.DataFrame:
    K, alpha = cfg["looks"], cfg["alpha"]
    t = np.linspace(1 / K, 1, K)
    crit = {"fixed": None, "naive_peeking": np.full(K, stats.norm.ppf(1 - alpha / 2))}
    for kind in cfg["spending"]:
        crit[f"group_sequential_{kind}"] = boundaries(t, alpha, kind)
    z_fixed = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for p0, share, mult in product(cfg["baselines"], cfg["treatment_shares"], cfg["effect_multiples"]):
        sc = dict(bench="sequential", p0=p0, share=share, effect_multiple=mult)
        rng = _rng(seed, sc)
        n0, n1 = sample_size_per_arm(p0, cfg["relative_mde"], alpha, cfg["power"], True, share)
        mde_abs = p0 * cfg["relative_mde"]
        p1 = p0 + mult * mde_abs
        d_true = p1 - p0
        # cumulative counts at each look from independent binomial increments (exact for counts)
        b0 = np.diff(np.round(np.concatenate([[0], t * n0])).astype(int))
        b1 = np.diff(np.round(np.concatenate([[0], t * n1])).astype(int))
        X0 = np.cumsum(rng.binomial(b0, p0, size=(reps, K)), axis=1)
        X1 = np.cumsum(rng.binomial(b1, p1, size=(reps, K)), axis=1)
        N0, N1 = np.cumsum(b0), np.cumsum(b1)
        Z, _ = pooled_z_test(X0, N0, X1, N1)
        est = X1 / N1 - X0 / N0
        se = np.sqrt((X0 / N0) * (1 - X0 / N0) / N0 + (X1 / N1) * (1 - X1 / N1) / N1)
        for policy, c in crit.items():
            if c is None:
                stop = np.full(reps, K - 1)
                rej = np.abs(Z[:, -1]) >= z_fixed
                q = np.full(reps, z_fixed)
            else:
                hit = np.abs(Z) >= c[None, :]
                any_hit = hit.any(axis=1)
                stop = np.where(any_hit, hit.argmax(axis=1), K - 1)
                rej = any_hit
                q = c[stop] if policy.startswith("group") else np.full(reps, z_fixed)
            idx = np.arange(reps)
            e_s, se_s = est[idx, stop], se[idx, stop]
            covered = (e_s - q * se_s <= d_true) & (d_true <= e_s + q * se_s)
            r, r_se = _rate(rej)
            cv, cv_se = _rate(covered)
            n_stop = (N0 + N1)[stop]
            rows.append({**sc, "policy": policy, "n_control_planned": n0, "n_treatment_planned": n1,
                         "true_diff": d_true, "reps": reps, "rejection_rate": r, "rejection_se": r_se,
                         "coverage": cv, "coverage_se": cv_se,
                         "mean_stop_n": float(n_stop.mean()), "mean_stop_fraction": float(n_stop.mean() / (n0 + n1)),
                         "bias": float(e_s.mean() - d_true),
                         "bias_se": float(e_s.std(ddof=1) / np.sqrt(reps)),
                         "bias_if_rejected": float(e_s[rej].mean() - d_true) if rej.any() else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- CUPED
def _welch_vec(y0, y1):
    m = y1.mean(1) - y0.mean(1)
    se = np.sqrt(y0.var(1, ddof=1) / y0.shape[1] + y1.var(1, ddof=1) / y1.shape[1])
    return m, se


def cuped(cfg: dict, reps: int, seed: int) -> pd.DataFrame:
    n = cfg["n_per_arm"]
    z = stats.norm.ppf(0.975)
    mde = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * np.sqrt(2 / n)   # raw-design MDE (sd units)
    rows = []
    for outcome, rho, miss, eff in product(cfg["outcomes"], cfg["correlations"], cfg["missing_rates"], cfg["effects"]):
        sc = dict(bench="cuped", outcome=outcome, rho=rho, missing=miss, effect=eff)
        rng = _rng(seed, sc)
        raw_d, raw_se, cu_d, cu_se, rhos = [], [], [], [], []
        tau = 0.0
        for start in range(0, reps, cfg["chunk"]):
            m = min(cfg["chunk"], reps - start)
            u = rng.standard_normal((m, 2 * n))
            e = rng.standard_normal((m, 2 * n))
            treat = np.zeros(2 * n, bool)
            treat[n:] = True
            if outcome == "continuous":
                tau = mde if eff == "mde" else 0.0
                y = tau * treat + rho * u + np.sqrt(1 - rho**2) * e
            else:  # binary conversion with latent correlation rho; effect = +20% relative
                p0 = cfg["binary_baseline"]
                p = np.where(treat, p0 * (1.2 if eff == "mde" else 1.0), p0)
                # higher engagement -> more likely to convert (positive correlation)
                y = ((rho * u + np.sqrt(1 - rho**2) * e) > stats.norm.isf(p)).astype(float)
                tau = p0 * (0.2 if eff == "mde" else 0.0)
            x = u.copy()
            if miss > 0:
                x[rng.random(x.shape) < miss] = np.nan
            xm = np.nanmean(x, axis=1, keepdims=True)
            xf = np.where(np.isnan(x), xm, x)
            ok = ~np.isnan(x)
            # treatment-blind pooled theta on observed covariates
            xc = np.where(ok, x - xm, 0.0)
            yc = np.where(ok, y - (np.where(ok, y, 0).sum(1, keepdims=True) / ok.sum(1, keepdims=True)), 0.0)
            theta = (xc * yc).sum(1) / (xc**2).sum(1)
            ystar = y - theta[:, None] * (xf - xm)
            d, s = _welch_vec(y[:, :n], y[:, n:])
            dc, sc_ = _welch_vec(ystar[:, :n], ystar[:, n:])
            raw_d.append(d); raw_se.append(s); cu_d.append(dc); cu_se.append(sc_)
            rhos.append(np.array([np.corrcoef(y[i, ok[i]], x[i, ok[i]])[0, 1] for i in range(min(m, 50))]))
        raw_d, raw_se, cu_d, cu_se = map(np.concatenate, (raw_d, raw_se, cu_d, cu_se))
        vr = cu_d.var(ddof=1) / raw_d.var(ddof=1)
        # delta-method MC SE of the variance ratio (log scale), from 200 bootstrap resamples of replicates
        bs = rng.integers(0, reps, size=(200, reps))
        vr_bs = cu_d[bs].var(1, ddof=1) / raw_d[bs].var(1, ddof=1)
        row = {**sc, "reps": reps, "n_per_arm": n, "true_effect": tau,
               "realized_correlation": float(np.nanmean(np.concatenate(rhos))),
               "variance_ratio": float(vr), "variance_ratio_se": float(vr_bs.std(ddof=1)),
               "theory_ratio_linear": float(1 - (1 - miss) * rho**2) if outcome == "continuous" else np.nan,
               "equivalent_sample_reduction": float(1 - vr)}
        for lab, dd, ss in (("raw", raw_d, raw_se), ("cuped", cu_d, cu_se)):
            rej, rej_se = _rate(np.abs(dd / ss) >= z)
            cov, cov_se = _rate(np.abs(dd - tau) <= z * ss)
            row.update({f"{lab}_bias": float(dd.mean() - tau), f"{lab}_bias_se": float(dd.std(ddof=1) / np.sqrt(reps)),
                        f"{lab}_rejection": rej, f"{lab}_rejection_se": rej_se,
                        f"{lab}_coverage": cov, f"{lab}_coverage_se": cov_se})
        rows.append(row)
    return pd.DataFrame(rows)


BENCHES = {"fixed": fixed_calibration, "srm": srm_calibration, "sequential": sequential, "cuped": cuped}


def run(config: str = "benchmark.yaml", reps: int | None = None, only: list[str] | None = None, out: Path = OUT) -> dict:
    cfg = load_yaml(config)
    reps = reps or cfg["replications"]
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config_hash": stable_hash(cfg), "replications": reps, "seed": cfg["seed"], "benches": {}}
    for name, fn in BENCHES.items():
        if only and name not in only:
            continue
        t0 = time.time()
        df = fn(cfg[name], reps, cfg["seed"])
        df.to_csv(out / f"{name}.csv", index=False)
        meta["benches"][name] = {"scenarios": len(df), "seconds": round(time.time() - t0, 1)}
        print(f"{name:<11} {len(df):>3} scenarios x {reps:,} reps  ({time.time() - t0:.1f}s)")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="benchmark.yaml")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    run(a.config, a.reps, a.only, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
