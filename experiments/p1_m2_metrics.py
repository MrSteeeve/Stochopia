"""Pure summaries for P1's development windows and paired injection paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from stochopia.measure.impact import (
    ImpactMetrics,
    ImpactReference,
    antisymmetrise,
    identify_law,
    relaxation,
    score_impact,
)


FLOW_LAGS = np.arange(1, 9)
PRICE_LAGS = np.array([1, 2, 4, 8])


@dataclass(frozen=True)
class TapeStats:
    """One tape's exogenous order-flow statistics and regular-grid book summaries."""

    events: int
    burst_rate: float
    action_share: np.ndarray
    flow_acf: np.ndarray
    price_msd: np.ndarray
    spread_ticks: float
    touch_shares: float
    drift_ticks: float
    two_sided_share: float


def _finite_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Average over finite observations. An empty support remains NaN."""
    valid = np.isfinite(values)
    counts = valid.sum(axis=axis)
    totals = np.where(valid, values, 0).sum(axis=axis)
    return np.divide(totals, counts, out=np.full_like(totals, np.nan, dtype=float), where=counts > 0)


def tape_statistics(events: np.ndarray, mids: np.ndarray, spreads: np.ndarray, depths: np.ndarray) -> TapeStats:
    """Measure [elapsed ms, action, ...] events and [grid, ...] book marks.

    ACF uses aggressive buy/sell signs only. Price scaling uses squared midpoint
    increments at fixed grid lags. We leave empty or constant samples missing.
    """
    action = events[:, 1].astype(int)
    signs = np.where(action[(action == 2) | (action == 3)] == 2, 1.0, -1.0)
    centered = signs - signs.mean() if len(signs) else signs
    variance = float(np.mean(centered**2)) if len(signs) else 0.0
    acf = np.full(len(FLOW_LAGS), np.nan)
    if variance > 0:
        for i, lag in enumerate(FLOW_LAGS):
            if len(signs) > lag + 1:
                acf[i] = np.mean(centered[lag:] * centered[:-lag]) / variance

    msd = np.array([_finite_mean((mids[lag:] - mids[:-lag]) ** 2) for lag in PRICE_LAGS])
    n_events = len(events)
    burst_rate = float((np.diff(events[:, 0]) == 0).mean()) if n_events > 1 else np.nan
    shares = np.bincount(action, minlength=6) / n_events if n_events else np.full(6, np.nan)
    active = np.isfinite(mids)
    return TapeStats(
        n_events, burst_rate, shares, acf, msd, float(_finite_mean(spreads)),
        float(_finite_mean(depths[active])), float(mids[-1] - mids[0]), float(active.mean()),
    )


def _log_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(y) & (y > 0)
    if valid.sum() < 3:
        return None
    return float(np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)[0])


def aggregate_tapes(rows: list[TapeStats]) -> dict[str, object]:
    means = {key: _finite_mean(np.array([asdict(row)[key] for row in rows]), axis=0) for key in asdict(rows[0])}
    gamma = _log_slope(FLOW_LAGS, means["flow_acf"])
    hurst = _log_slope(PRICE_LAGS, means["price_msd"])
    result = {key: value.tolist() for key, value in means.items()}
    result["gamma_flow"] = -gamma if gamma is not None else None
    result["hurst"] = hurst / 2 if hurst is not None else None
    result["mean_absolute_drift_ticks"] = float(_finite_mean(np.abs([row.drift_ticks for row in rows])))
    return result


def summarize_impact(
    *, visible_mid: np.ndarray, blind_mid: np.ndarray, child_visible: np.ndarray,
    child_blind: np.ndarray, child_filled: np.ndarray, sigma_ticks: float, daily_shares: float,
    schedule: np.ndarray, generated: dict[str, object], replay: dict[str, object],
) -> dict[str, object]:
    """Re-use the LOB-Impact scorer on day-clustered buy/sell paths.

    Mid paths are [day, side, grid], children and fills [day, side, child].
    Side order is buy then sell. Seeds and matched sessions are averaged within
    each day before calling; midday or seed replicates do not inflate n.
    """
    n_days = len(visible_mid)
    visible = antisymmetrise(visible_mid[:, 0], visible_mid[:, 1])
    blind = antisymmetrise(blind_mid[:, 0], blind_mid[:, 1])
    execution_visible = antisymmetrise(child_visible[:, 0], child_visible[:, 1])
    execution_blind = antisymmetrise(child_blind[:, 0], child_blind[:, 1])
    whole_days = np.isfinite(execution_visible).all(axis=1)
    valid_execution = execution_visible[whole_days]
    law = identify_law(valid_execution) if len(valid_execution) >= 2 else None

    peak_days = np.isfinite(execution_visible[:, -1])
    paired_days = peak_days & np.isfinite(execution_blind[:, -1])
    peak = float(_finite_mean(execution_visible[peak_days, -1]))
    visible_peak = float(_finite_mean(execution_visible[paired_days, -1]))
    blind_peak = float(_finite_mean(execution_blind[paired_days, -1]))
    traded = float(_finite_mean(child_filled[peak_days].sum(axis=-1)))
    denominator = sigma_ticks * np.sqrt(traded / daily_shares) if daily_shares > 0 else 0
    y = peak / denominator if denominator > 0 and np.isfinite(peak) else None

    visible_mean = _finite_mean(visible, axis=0)
    last_child = int(np.flatnonzero(schedule)[-1])
    recovery_days = peak_days & np.isfinite(visible[:, last_child:]).all(axis=1)
    recovery_mean = _finite_mean(visible[recovery_days], axis=0)
    recovery_mean[last_child] = _finite_mean(execution_visible[recovery_days, -1])
    rec = None
    if recovery_days.any() and len(recovery_mean) - last_child >= 3 and recovery_mean[last_child] > 0:
        rec = relaxation(recovery_mean, schedule, tail=1)

    ci = None
    if len(valid_execution) >= 2:
        peaks = valid_execution[:, -1]
        rng = np.random.default_rng(9127)
        draws = peaks[rng.integers(len(peaks), size=(2000, len(peaks)))].mean(axis=1)
        ci = tuple(float(val) for val in np.quantile(draws, [0.025, 0.975]))

    metrics = ImpactMetrics(
        y=y, delta=law.delta if law and law.identified else None,
        end_over_peak=rec.end_over_peak if rec else None,
        relaxation_slope=rec.slope if rec else None,
        law_identified=law.identified if law is not None else None, peak_interval_ticks=ci,
        visible_peak_ticks=visible_peak, blind_peak_ticks=blind_peak,
        gamma_flow=generated["gamma_flow"], hurst=generated["hurst"], idle_drift_ticks=generated["drift_ticks"],
    )
    reference = ImpactReference(
        gamma_flow=replay["gamma_flow"], hurst=replay["hurst"], idle_drift_ticks=replay["drift_ticks"],
        near_zero_peak_band_ticks=(-0.5, 0.5),
    )
    score = score_impact(metrics, reference)
    return {
        "paired_days": n_days, "days_with_complete_execution_marks": int(whole_days.sum()),
        "days_with_visible_last_child": int(peak_days.sum()), "days_with_paired_last_child": int(paired_days.sum()),
        "days_with_complete_recovery": int(recovery_days.sum()), "visible_peak_for_amplitude_ticks": peak,
        "visible_curve_ticks": visible_mean.tolist(), "blind_curve_ticks": _finite_mean(blind, axis=0).tolist(),
        "visible_child_ticks": _finite_mean(execution_visible, axis=0).tolist(),
        "blind_child_ticks": _finite_mean(execution_blind, axis=0).tolist(),
        "average_filled_shares": traded, "law": asdict(law) if law is not None else None,
        "measurements": asdict(metrics), "reference": asdict(reference), "score": asdict(score),
    }
