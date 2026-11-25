"""Effect estimation for binary and continuous metrics.

All proportion functions are numpy-vectorised so the same code serves one experiment
in the UI and ~120k arm comparisons in the archive audit.

Binary metrics
  * difference in proportions, pooled two-sided z-test where large-sample assumptions hold;
  * Newcombe hybrid-score (Wilson-based) interval for the difference;
  * Fisher's exact test when any expected 2x2 cell count is below `sparse_threshold`;
  * relative lift with a Katz log-ratio interval; undefined (NaN + flag) when control rate is 0.
Continuous metrics
  * Welch's t-test / interval; percentile bootstrap as a heavy-tail sensitivity check.
Multiplicity
  * Holm step-down adjusted p-values; Bonferroni intervals as a compatible conservative set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats

ArrayLike = np.ndarray | float | int


def _z(alpha) -> np.ndarray | float:
    z = stats.norm.ppf(1 - np.asarray(alpha, float) / 2)
    return float(z) if np.ndim(z) == 0 else z


def wilson_interval(x: ArrayLike, n: ArrayLike, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    x, n = np.asarray(x, float), np.asarray(n, float)
    z = _z(alpha)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = x / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def newcombe_diff_interval(x0, n0, x1, n1, alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Newcombe (1998) method 10 interval for p1 - p0; well-behaved at 0 and 1."""
    x0, n0, x1, n1 = (np.asarray(v, float) for v in (x0, n0, x1, n1))
    p0, p1 = x0 / n0, x1 / n1
    l0, u0 = wilson_interval(x0, n0, alpha)
    l1, u1 = wilson_interval(x1, n1, alpha)
    d = p1 - p0
    lo = d - np.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return lo, hi


def pooled_z_test(x0, n0, x1, n1) -> tuple[np.ndarray, np.ndarray]:
    x0, n0, x1, n1 = (np.asarray(v, float) for v in (x0, n0, x1, n1))
    pp = (x0 + x1) / (n0 + n1)
    se = np.sqrt(pp * (1 - pp) * (1 / n0 + 1 / n1))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x1 / n1 - x0 / n0) / se
    p = 2 * stats.norm.sf(np.abs(z))
    # No events (or all events) in both arms: the data carry no evidence of a difference.
    degenerate = se == 0
    z = np.where(degenerate, 0.0, z)
    p = np.where(degenerate, 1.0, p)
    return z, p


def min_expected_cell(x0, n0, x1, n1) -> np.ndarray:
    x0, n0, x1, n1 = (np.asarray(v, float) for v in (x0, n0, x1, n1))
    pp = (x0 + x1) / (n0 + n1)
    return np.minimum.reduce([n0 * pp, n0 * (1 - pp), n1 * pp, n1 * (1 - pp)])


def relative_lift(x0, n0, x1, n1, alpha: float = 0.05):
    """(p1 - p0) / p0 with a Katz log risk-ratio interval. NaN when p0 == 0 (undefined)."""
    x0, n0, x1, n1 = (np.asarray(v, float) for v in (x0, n0, x1, n1))
    p0, p1 = x0 / n0, x1 / n1
    with np.errstate(invalid="ignore", divide="ignore"):
        lift = np.where(p0 > 0, p1 / p0 - 1, np.nan)
        se = np.sqrt(1 / x1 - 1 / n1 + 1 / x0 - 1 / n0)
        ok = (x0 > 0) & (x1 > 0)
        z = _z(alpha)
        log_rr = np.log(p1 / p0)
        lo = np.where(ok, np.exp(log_rr - z * se) - 1, np.nan)
        hi = np.where(ok, np.exp(log_rr + z * se) - 1, np.nan)
    return lift, lo, hi


@dataclass
class ProportionResult:
    n_control: int
    x_control: int
    n_treatment: int
    x_treatment: int
    rate_control: float
    rate_treatment: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    test: str
    ci_method: str
    rel_lift: float | None
    rel_lift_ci_low: float | None
    rel_lift_ci_high: float | None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def two_proportion(
    x0: int, n0: int, x1: int, n1: int, alpha: float = 0.05, sparse_threshold: float = 5.0
) -> ProportionResult:
    for x, n in ((x0, n0), (x1, n1)):
        if n <= 0:
            raise ValueError("each arm needs at least one unit")
        if not 0 <= x <= n:
            raise ValueError("successes must lie in [0, n]")
    flags: list[str] = []
    lo, hi = newcombe_diff_interval(x0, n0, x1, n1, alpha)
    if x0 + x1 == 0:
        flags.append("ZERO_EVENTS")
    if min_expected_cell(x0, n0, x1, n1) < sparse_threshold:
        flags.append("SPARSE_COUNTS")
        p = float(stats.fisher_exact([[x1, n1 - x1], [x0, n0 - x0]]).pvalue)
        test = "fisher_exact"
    else:
        p = float(pooled_z_test(x0, n0, x1, n1)[1])
        test = "pooled_z"
    lift, llo, lhi = relative_lift(x0, n0, x1, n1, alpha)
    if x0 == 0:
        flags.append("LIFT_UNDEFINED_ZERO_CONTROL")
    return ProportionResult(
        n_control=int(n0),
        x_control=int(x0),
        n_treatment=int(n1),
        x_treatment=int(x1),
        rate_control=x0 / n0,
        rate_treatment=x1 / n1,
        diff=x1 / n1 - x0 / n0,
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=p,
        test=test,
        ci_method="newcombe_hybrid_score",
        rel_lift=None if np.isnan(lift) else float(lift),
        rel_lift_ci_low=None if np.isnan(llo) else float(llo),
        rel_lift_ci_high=None if np.isnan(lhi) else float(lhi),
        flags=flags,
    )


@dataclass
class MeanResult:
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    diff: float
    se: float
    ci_low: float
    ci_high: float
    p_value: float
    df: float
    test: str = "welch_t"
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def welch(y0: np.ndarray, y1: np.ndarray, alpha: float = 0.05) -> MeanResult:
    """Welch's unequal-variance t-test. Zeros are real outcomes and must be included."""
    y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
    if len(y0) < 2 or len(y1) < 2:
        raise ValueError("each arm needs at least two units")
    m0, m1 = y0.mean(), y1.mean()
    v0, v1 = y0.var(ddof=1) / len(y0), y1.var(ddof=1) / len(y1)
    se = float(np.sqrt(v0 + v1))
    flags = []
    if se == 0:
        flags.append("ZERO_VARIANCE")
        return MeanResult(len(y0), len(y1), m0, m1, m1 - m0, 0.0, m1 - m0, m1 - m0, 1.0, np.inf, flags=flags)
    dof = (v0 + v1) ** 2 / (v0**2 / (len(y0) - 1) + v1**2 / (len(y1) - 1))
    t = (m1 - m0) / se
    q = stats.t.ppf(1 - alpha / 2, dof)
    return MeanResult(
        len(y0), len(y1), m0, m1, m1 - m0, se, m1 - m0 - q * se, m1 - m0 + q * se,
        float(2 * stats.t.sf(abs(t), dof)), float(dof), flags=flags,
    )


def bootstrap_mean_diff(y0, y1, alpha: float = 0.05, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for mean(y1) - mean(y0); sensitivity check for heavy tails."""
    rng = np.random.default_rng(seed)
    y0, y1 = np.asarray(y0, float), np.asarray(y1, float)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = rng.choice(y1, len(y1)).mean() - rng.choice(y0, len(y0)).mean()
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def holm(p_values: ArrayLike) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, float)
    m = len(p)
    if m == 0:
        return p
    order = np.argsort(p, kind="stable")
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def homogeneity_test(clicks: np.ndarray, impressions: np.ndarray) -> tuple[float, float, int]:
    """Omnibus chi-square test that all arms share one rate (2 x k table)."""
    clicks, impressions = np.asarray(clicks, float), np.asarray(impressions, float)
    table = np.vstack([clicks, impressions - clicks])
    if (table.sum(axis=1) == 0).any():
        return 0.0, 1.0, len(clicks) - 1
    res = stats.chi2_contingency(table, correction=False)
    return float(res.statistic), float(res.pvalue), int(res.dof)
