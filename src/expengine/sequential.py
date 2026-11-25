"""Group-sequential monitoring with Lan-DeMets alpha spending.

Boundaries are computed by numerical integration of the canonical joint distribution
of the z-statistics (Brownian motion in information time), following Jennison & Turnbull
(2000, ch. 19): at each look the sub-density of the score on the continuation region is
propagated on a grid, and the critical value is solved so that the probability of first
crossing equals the alpha spent since the previous look.

Two-sided, symmetric boundaries. Spending functions:
  obf     alpha(t) = 2 - 2 * Phi(z_{1-alpha/2} / sqrt(t))      (O'Brien-Fleming-like)
  pocock  alpha(t) = alpha * ln(1 + (e - 1) t)                  (Pocock-like)
  obf_per_side   gsDesign sfLDOF convention (alpha/2 spent per tail); used to regression-test
                 against published gsDesign boundaries.

Repeated confidence intervals (estimate +/- c_k * se_k) are valid simultaneously over
all looks, so the interval reported at a data-dependent stopping time keeps its coverage.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import optimize, stats


def spent(t: np.ndarray, alpha: float, kind: str = "obf") -> np.ndarray:
    t = np.asarray(t, float)
    if kind == "obf":
        return 2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - alpha / 2) / np.sqrt(t))
    if kind == "obf_per_side":
        # gsDesign's sfLDOF convention for symmetric two-sided designs: the one-sided OBF
        # function with alpha/2 is spent independently in each tail.
        a1 = alpha / 2
        return 2 * (2 - 2 * stats.norm.cdf(stats.norm.ppf(1 - a1 / 2) / np.sqrt(t)))
    if kind == "pocock":
        return alpha * np.log(1 + (np.e - 1) * t)
    raise ValueError(f"unknown spending function {kind!r}")


@lru_cache(maxsize=256)
def _boundaries(fractions: tuple[float, ...], alpha: float, kind: str, grid: int) -> tuple[float, ...]:
    t = np.asarray(fractions, float)
    if np.any(np.diff(t) <= 0) or t[0] <= 0 or abs(t[-1] - 1) > 1e-9:
        raise ValueError("information fractions must increase strictly and end at 1")
    cum = spent(t, alpha, kind)
    inc = np.diff(np.concatenate([[0.0], cum]))

    c = [float(stats.norm.ppf(1 - inc[0] / 2))]
    # sub-density of B(t1) = Z1 * sqrt(t1) on the continuation region
    width = c[0] * np.sqrt(t[0])
    b = np.linspace(-width, width, grid)
    f = stats.norm.pdf(b, scale=np.sqrt(t[0]))

    for k in range(1, len(t)):
        sd = np.sqrt(t[k] - t[k - 1])
        w = np.full(grid, (b[1] - b[0]))  # trapezoid weights
        w[[0, -1]] /= 2
        mass = f * w

        def cross_prob(ck: float) -> float:
            edge = ck * np.sqrt(t[k])
            return float(np.sum(mass * (stats.norm.cdf((-edge - b) / sd) + stats.norm.sf((edge - b) / sd))))

        if inc[k] <= 1e-15:
            ck = np.inf
        else:
            ck = optimize.brentq(lambda x: cross_prob(x) - inc[k], 0.01, 40.0, xtol=1e-10)
        c.append(float(ck))
        if k < len(t) - 1:
            edge = ck * np.sqrt(t[k])
            b_new = np.linspace(-edge, edge, grid)
            kernel = stats.norm.pdf((b_new[:, None] - b[None, :]) / sd) / sd
            f = kernel @ mass
            b = b_new
    return tuple(c)


def boundaries(fractions, alpha: float = 0.05, kind: str = "obf", grid: int = 1201) -> np.ndarray:
    """Two-sided z critical values c_k for the given information fractions."""
    return np.asarray(_boundaries(tuple(float(x) for x in fractions), float(alpha), kind, int(grid)))


def first_crossing(z_paths: np.ndarray, crit: np.ndarray) -> np.ndarray:
    """Index of the first look where |z| >= crit (or -1 if never). z_paths: (reps, K)."""
    hit = np.abs(z_paths) >= np.asarray(crit)[None, :]
    first = hit.argmax(axis=1)
    return np.where(hit.any(axis=1), first, -1)


def repeated_ci(estimate: float, se: float, crit_k: float) -> tuple[float, float]:
    return estimate - crit_k * se, estimate + crit_k * se


def monitor(looks: list[dict], alpha: float = 0.05, kind: str = "obf", planned_looks: int | None = None) -> dict:
    """Apply a group-sequential design to observed looks.

    looks: [{"information_fraction": t, "estimate": d, "se": s}, ...] in order.
    Returns the stopping look (or the last look), its boundary, z and repeated CI.
    """
    if not looks:
        raise ValueError("no looks supplied")
    fr = [lk["information_fraction"] for lk in looks]
    if planned_looks and len(fr) < planned_looks and fr[-1] < 1:
        # design boundaries use the planned schedule; only observed looks are evaluated
        fr = list(np.linspace(1 / planned_looks, 1, planned_looks))
    crit = boundaries(fr, alpha, kind)
    trace = []
    for i, lk in enumerate(looks):
        z = lk["estimate"] / lk["se"] if lk["se"] > 0 else 0.0
        lo, hi = repeated_ci(lk["estimate"], lk["se"], crit[i])
        crossed = abs(z) >= crit[i]
        trace.append({"look": i + 1, "information_fraction": lk["information_fraction"], "z": z,
                      "boundary": float(crit[i]), "rci_low": lo, "rci_high": hi, "crossed": bool(crossed)})
        if crossed:
            break
    stop = trace[-1]
    return {"stopped_early": stop["crossed"] and stop["look"] < len(crit), "rejected": stop["crossed"],
            "stop_look": stop["look"], "boundary": stop["boundary"], "z": stop["z"],
            "rci_low": stop["rci_low"], "rci_high": stop["rci_high"], "trace": trace,
            "spending": kind, "planned_looks": len(crit)}
