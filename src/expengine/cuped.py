"""CUPED variance reduction (Deng, Xu, Kohavi & Walker, 2013).

    Y* = Y - theta * (X - mean(X)),   theta = Cov(Y, X) / Var(X)

theta is fitted on the pooled sample, ignoring treatment labels (treatment-blind and
prespecified), so it cannot be tuned toward a preferred result. Under randomisation X is
independent of assignment, so the adjusted difference in means stays unbiased.

Edge cases handled explicitly:
  * missing covariate -> unit's X is set to the observed pooled mean (zero adjustment for it),
    with a flag; valid because missingness is a pre-treatment attribute;
  * zero-variance or all-missing covariate -> theta = 0 (CUPED reduces to the raw estimate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from expengine.inference import welch


@dataclass
class CupedResult:
    theta: float
    covariate_correlation: float
    raw_diff: float
    raw_se: float
    raw_ci: tuple[float, float]
    raw_p: float
    cuped_diff: float
    cuped_se: float
    cuped_ci: tuple[float, float]
    cuped_p: float
    variance_ratio: float            # Var(cuped estimator) / Var(raw estimator), plug-in
    equivalent_sample_reduction: float
    missing_covariate_share: float
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def fit_theta(y: np.ndarray, x: np.ndarray) -> tuple[float, list[str]]:
    flags = []
    ok = ~np.isnan(x)
    if ok.sum() < 2:
        return 0.0, ["COVARIATE_ALL_MISSING"]
    vx = np.var(x[ok], ddof=1)
    if vx == 0:
        return 0.0, ["COVARIATE_ZERO_VARIANCE"]
    theta = np.cov(y[ok], x[ok], ddof=1)[0, 1] / vx
    return float(theta), flags


def adjust(y: np.ndarray, x: np.ndarray, theta: float | None = None) -> tuple[np.ndarray, float, list[str]]:
    y, x = np.asarray(y, float), np.asarray(x, float)
    flags: list[str] = []
    if theta is None:
        theta, flags = fit_theta(y, x)
    miss = np.isnan(x)
    if miss.any():
        flags.append("COVARIATE_MISSING_IMPUTED_MEAN")
    xm = np.nanmean(x) if (~miss).any() else 0.0
    x_filled = np.where(miss, xm, x)
    return y - theta * (x_filled - xm), theta, flags


def cuped_effect(y0, x0, y1, x1, alpha: float = 0.05) -> CupedResult:
    y0, x0, y1, x1 = (np.asarray(v, float) for v in (y0, x0, y1, x1))
    y = np.concatenate([y0, y1])
    x = np.concatenate([x0, x1])
    ystar, theta, flags = adjust(y, x)
    raw = welch(y0, y1, alpha)
    adj = welch(ystar[: len(y0)], ystar[len(y0):], alpha)
    ok = ~np.isnan(x)
    rho = float(np.corrcoef(y[ok], x[ok])[0, 1]) if ok.sum() > 2 and np.var(x[ok]) > 0 else 0.0
    ratio = (adj.se / raw.se) ** 2 if raw.se > 0 else 1.0
    return CupedResult(
        theta, rho, raw.diff, raw.se, (raw.ci_low, raw.ci_high), raw.p_value,
        adj.diff, adj.se, (adj.ci_low, adj.ci_high), adj.p_value, ratio, 1 - ratio,
        float(np.isnan(x).mean()), flags,
    )
