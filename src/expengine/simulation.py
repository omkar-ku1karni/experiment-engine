"""Seeded synthetic product experiments (user-level outcomes).

Everything here is SIMULATED. Outputs are labelled `source = "synthetic"` and are never
presented as historical replays of the archive.

Model per unit i (one latent engagement score u_i ~ N(0, 1) drives correlation):
  pre_period_metric  X = exp(mu_x + s_x * u)                 (pre-experiment spend; may be missing)
  conversion         C = 1{ r * u + sqrt(1 - r^2) * e1 > Phi^-1(1 - p_arm) }  (latent correlation r)
  order value        V = exp(mu_v + s_v * (r * u + sqrt(1 - r^2) * e2)) * (1 + value_lift_arm)
  revenue            Y = C * V                                 (zero-inflated, heavy-tailed)
  guardrails         Bernoulli(base + effect_arm)              (e.g. refund, support contact)

Configurable: baseline rate, allocation (designed vs actual, for SRM injection), effects,
enrollment period, pre-period correlation, covariate missingness, duplicate assignment,
guardrail deterioration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from expengine.config import stable_hash


@dataclass
class ScenarioSpec:
    experiment_id: str
    title: str
    hypothesis: str
    n_units: int
    days: int = 14
    arms: list[str] = field(default_factory=lambda: ["control", "treatment"])
    allocation: list[float] = field(default_factory=lambda: [0.5, 0.5])        # designed
    actual_allocation: list[float] | None = None                               # SRM injection
    baseline_conversion: float = 0.05
    conversion_lift: float = 0.0                                               # relative, treatment
    value_mu: float = 3.6
    value_sigma: float = 0.6
    value_lift: float = 0.0
    latent_correlation: float = 0.6
    pre_mu: float = 3.0
    pre_sigma: float = 0.8
    covariate_missing_rate: float = 0.0
    duplicate_assignment_rate: float = 0.0
    guardrails: dict[str, dict] = field(default_factory=dict)   # name -> {base, effect}
    drop_guardrails: list[str] = field(default_factory=list)    # simulate missing guardrail data
    seed: int = 0
    primary_metric: str = "revenue"                              # revenue | conversion
    planned_looks: int = 5
    policy: dict = field(default_factory=dict)
    expected_recommendation: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        return cls(**d)


def generate(spec: ScenarioSpec) -> pd.DataFrame:
    rng = np.random.default_rng(np.random.SeedSequence([spec.seed, int(stable_hash(spec.experiment_id, 8), 16)]))
    n = spec.n_units
    shares = np.asarray(spec.actual_allocation or spec.allocation, float)
    arm_idx = rng.choice(len(spec.arms), size=n, p=shares / shares.sum())
    is_t = arm_idx > 0

    u = rng.standard_normal(n)
    r = spec.latent_correlation
    p_arm = np.where(is_t, spec.baseline_conversion * (1 + spec.conversion_lift), spec.baseline_conversion)
    conv = (r * u + np.sqrt(1 - r**2) * rng.standard_normal(n)) > stats.norm.isf(p_arm)
    value = np.exp(spec.value_mu + spec.value_sigma * (r * u + np.sqrt(1 - r**2) * rng.standard_normal(n)))
    value *= np.where(is_t, 1 + spec.value_lift, 1.0)
    revenue = np.round(conv * value, 2)

    pre = np.round(np.exp(spec.pre_mu + spec.pre_sigma * u), 2)
    pre = np.where(rng.random(n) < spec.covariate_missing_rate, np.nan, pre)

    start = pd.Timestamp("2026-01-05")
    assigned_at = start + pd.to_timedelta(np.sort(rng.random(n)) * spec.days * 86400, unit="s")
    df = pd.DataFrame({
        "experiment_id": spec.experiment_id,
        "unit_id": [f"u{i:07d}" for i in range(n)],
        "arm_id": np.asarray(spec.arms)[arm_idx],
        "assigned_at": assigned_at,
        "pre_period_metric": pre,
        "conversion": conv.astype(int),
        "revenue": revenue,
    })
    for name, g in spec.guardrails.items():
        if name in spec.drop_guardrails:
            continue
        rate = np.clip(np.where(is_t, g["base"] + g.get("effect", 0.0), g["base"]), 0, 1)
        df[name] = (rng.random(n) < rate).astype(int)

    if spec.duplicate_assignment_rate > 0:   # same unit later logged in another arm
        dup = df.sample(frac=spec.duplicate_assignment_rate, random_state=spec.seed).copy()
        dup["arm_id"] = np.where(dup["arm_id"] == spec.arms[0], spec.arms[1], spec.arms[0])
        dup["assigned_at"] = dup["assigned_at"] + pd.Timedelta(hours=1)
        df = pd.concat([df, dup], ignore_index=True).sort_values("assigned_at", kind="stable").reset_index(drop=True)
    df["source"] = "synthetic"
    return df
