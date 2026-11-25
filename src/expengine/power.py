"""Experiment planning for binary metrics: sample size, power, MDE and duration.

Sample size per control arm for allocation ratio k = n_treatment / n_control
(k = 1 reproduces the textbook equal-arm formula):

    n0 = [ z(1-a/2) * sqrt(pbar(1-pbar)(1 + 1/k)) + z(power) * sqrt(p0(1-p0) + p1(1-p1)/k) ]^2 / (p1-p0)^2
    pbar = (p0 + k p1) / (1 + k)

Power is always assessed against a predeclared MDE, never the observed effect.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy import optimize, stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def _treatment_rate(p0: float, mde: float, relative: bool) -> float:
    p1 = p0 * (1 + mde) if relative else p0 + mde
    if not 0 < p0 < 1 or not 0 < p1 < 1:
        raise ValueError(f"rates must lie in (0, 1); got control {p0}, treatment {p1}")
    if p1 == p0:
        raise ValueError("MDE must be non-zero")
    return p1


def sample_size_per_arm(
    p0: float, mde: float, alpha: float = 0.05, power: float = 0.8, relative: bool = False,
    treatment_share: float = 0.5,
) -> tuple[int, int]:
    """(n_control, n_treatment), rounded up."""
    p1 = _treatment_rate(p0, mde, relative)
    k = treatment_share / (1 - treatment_share)
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    pbar = (p0 + k * p1) / (1 + k)
    num = za * math.sqrt(pbar * (1 - pbar) * (1 + 1 / k)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1) / k)
    n0 = num**2 / (p1 - p0) ** 2
    return math.ceil(n0), math.ceil(k * n0)


def power_two_proportions(p0: float, p1: float, n0: float, n1: float, alpha: float = 0.05) -> float:
    """Two-sided power of the pooled z-test (both rejection tails counted)."""
    za = stats.norm.ppf(1 - alpha / 2)
    pbar = (n0 * p0 + n1 * p1) / (n0 + n1)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n0 + 1 / n1))
    se1 = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    d = p1 - p0
    return float(stats.norm.cdf((d - za * se0) / se1) + stats.norm.cdf((-d - za * se0) / se1))


def mde_at_sample_size(
    p0: float, n0: float, n1: float, alpha: float = 0.05, power: float = 0.8, direction: int = 1
) -> float:
    """Smallest absolute effect detectable with `power` at the given arm sizes."""
    hi = (1 - p0) * 0.999999 if direction > 0 else p0 * 0.999999
    f = lambda d: power_two_proportions(p0, p0 + direction * d, n0, n1, alpha) - power  # noqa: E731
    if f(hi) < 0:
        return float("nan")
    return float(optimize.brentq(f, 1e-12, hi, xtol=1e-12))


def statsmodels_sample_size(p0: float, p1: float, alpha=0.05, power=0.8, treatment_share=0.5) -> float:
    """Independent cross-check via Cohen's h (arcsine) in statsmodels; close, not identical."""
    ratio = treatment_share / (1 - treatment_share)
    h = proportion_effectsize(p1, p0)
    return float(NormalIndPower().solve_power(effect_size=h, alpha=alpha, power=power, ratio=ratio))


@dataclass
class Plan:
    baseline_rate: float
    mde: float
    mde_type: str
    treatment_rate: float
    alpha: float
    power: float
    treatment_share: float
    n_control: int
    n_treatment: int
    n_total: int
    daily_eligible_traffic: float | None
    expected_duration_days: float | None
    statsmodels_n_control: float
    formula: str = "pooled-null / unpooled-alternative normal approximation"

    def to_dict(self) -> dict:
        return asdict(self)


def plan(
    baseline_rate: float, mde: float, relative: bool = True, alpha: float = 0.05, power: float = 0.8,
    treatment_share: float = 0.5, daily_traffic: float | None = None,
) -> Plan:
    p1 = _treatment_rate(baseline_rate, mde, relative)
    n0, n1 = sample_size_per_arm(baseline_rate, mde, alpha, power, relative, treatment_share)
    sm = statsmodels_sample_size(baseline_rate, p1, alpha, power, treatment_share)
    days = (n0 + n1) / daily_traffic if daily_traffic else None
    return Plan(
        baseline_rate, mde, "relative" if relative else "absolute", p1, alpha, power, treatment_share,
        n0, n1, n0 + n1, daily_traffic, None if days is None else math.ceil(days * 10) / 10, sm,
    )


def sensitivity_curve(p0: float, n0: float, n1: float, alpha: float = 0.05, effects=None) -> list[dict]:
    """Power at a fixed sample size across a range of relative effects."""
    effects = effects if effects is not None else np.linspace(0.01, 0.5, 50)
    out = []
    for e in effects:
        p1 = p0 * (1 + e)
        if 0 < p1 < 1:
            out.append({"relative_effect": float(e), "power": power_two_proportions(p0, p1, n0, n1, alpha)})
    return out
