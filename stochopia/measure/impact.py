"""Impact-law estimators for metaorder injection experiments.

Inputs are response paths ``[n_units, n_events]`` on the merged event clock of
an injection schedule (see :func:`stochopia.sim.world.injection_schedule`),
one row per independent unit (a trading day). Existence is decided before
magnitude: an exponent is reported only when the peak response is resolved
and the fitted exponent is interior to the search grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DELTA_GRID = np.round(np.arange(0.05, 1.5 + 1e-9, 0.01), 2)


def antisymmetrise(buy: np.ndarray, sell: np.ndarray) -> np.ndarray:
    """Combine buy and sell arms; direction-independent drift cancels."""
    return 0.5 * (np.asarray(buy, dtype=float) - np.asarray(sell, dtype=float))


def child_impact(paths: np.ndarray, schedule: np.ndarray) -> np.ndarray:
    """Response right after each injected child, ``[n_units, n_children]``."""
    return np.asarray(paths, dtype=float)[..., np.asarray(schedule, dtype=bool)]


def fit_power_law(q: np.ndarray, impact: np.ndarray) -> tuple[float, float]:
    """Least-squares fit of ``impact = a * q ** delta`` on raw signed values.

    For each delta on the grid the best amplitude is closed-form, so the fit
    needs no logarithms and no sign selection. Returns ``(delta, a)``.
    """
    q = np.asarray(q, dtype=float)
    impact = np.asarray(impact, dtype=float)
    basis = q[None, :] ** DELTA_GRID[:, None]
    amp = (basis * impact).sum(axis=1) / (basis**2).sum(axis=1)
    misfit = ((impact[None, :] - amp[:, None] * basis) ** 2).sum(axis=1)
    best = int(np.argmin(misfit))
    return float(DELTA_GRID[best]), float(amp[best])


@dataclass(frozen=True)
class ImpactLaw:
    identified: bool
    reason: str
    delta: float  # nan unless identified
    amplitude: float  # nan unless identified
    peak: float  # mean response after the last child
    peak_se: float


def identify_law(impact: np.ndarray, *, min_peak_se: float = 3.0) -> ImpactLaw:
    """Decide whether an impact law exists and, if so, fit it.

    ``impact`` is ``[n_units, n_children]``; participation is measured in
    children executed, ``q = 1..n_children``. The law is fitted on the unit
    mean, which is the pooled least-squares fit because every unit shares q.
    """
    impact = np.asarray(impact, dtype=float)
    if impact.ndim != 2 or impact.shape[0] < 2 or impact.shape[1] < 3:
        raise ValueError("impact must be [n_units >= 2, n_children >= 3]")
    n_units, n_children = impact.shape
    last = impact[:, -1]
    peak = float(last.mean())
    peak_se = float(last.std(ddof=1) / np.sqrt(n_units))
    nan = float("nan")
    if not peak > min_peak_se * peak_se:
        return ImpactLaw(False, "peak not resolved", nan, nan, peak, peak_se)
    delta, amp = fit_power_law(np.arange(1, n_children + 1), impact.mean(axis=0))
    if delta in (DELTA_GRID[0], DELTA_GRID[-1]):
        return ImpactLaw(False, "exponent on grid bound", nan, nan, peak, peak_se)
    return ImpactLaw(True, "ok", delta, amp, peak, peak_se)


def delta_bootstrap_ci(
    impact: np.ndarray, *, seed: int, n_resamples: int = 1000, confidence: float = 0.95
) -> tuple[float, float]:
    """Percentile CI for delta, resampling units (days) with replacement."""
    impact = np.asarray(impact, dtype=float)
    n_units, n_children = impact.shape
    q = np.arange(1, n_children + 1)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = impact[rng.integers(0, n_units, size=n_units)]
        deltas[i] = fit_power_law(q, sample.mean(axis=0))[0]
    tail = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(deltas, [tail, 1.0 - tail])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Relaxation:
    end_over_peak: float
    slope: float  # change per execution duration, as a fraction of the peak


def relaxation(mean_path: np.ndarray, schedule: np.ndarray, *, tail: int = 20) -> Relaxation:
    """Post-execution behaviour of a mean response path.

    The peak is the response right after the last child. The end level is the
    mean over the final ``tail`` events. The slope is a least-squares line
    through the post-execution path, per execution duration.
    """
    mean_path = np.asarray(mean_path, dtype=float)
    schedule = np.asarray(schedule, dtype=bool)
    children = np.flatnonzero(schedule)
    last = int(children[-1])
    post = mean_path[last:]
    if post.size < max(tail, 3):
        raise ValueError("schedule has no post-execution window to measure")
    peak = float(mean_path[last])
    duration = last + 1
    x = np.arange(post.size) / duration
    slope = float(np.polyfit(x, post, 1)[0] / peak)
    return Relaxation(end_over_peak=float(post[-tail:].mean() / peak), slope=slope)
