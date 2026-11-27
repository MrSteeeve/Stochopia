"""Pure summaries for market tapes and paired injection paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace
from stochopia.measure.impact import (
    ImpactLaw,
    ImpactMetrics,
    ImpactReference,
    antisymmetrise,
    fit_power_law,
    identify_law,
    continuous_impact_readouts,
    LEGACY_METRIC_SCHEMA,
    CONTINUOUS_METRIC_SCHEMA,
    relaxation,
    score_impact,
)


FLOW_LAGS = np.arange(1, 9)
PRICE_LAGS = np.array([1, 2, 4, 8])


# --- Tick and tape measurements ---


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


def finite_mean(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
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

    msd = np.array([finite_mean((mids[lag:] - mids[:-lag]) ** 2) for lag in PRICE_LAGS])
    n_events = len(events)
    burst_rate = float((np.diff(events[:, 0]) == 0).mean()) if n_events > 1 else np.nan
    shares = np.bincount(action, minlength=6) / n_events if n_events else np.full(6, np.nan)
    active = np.isfinite(mids)
    return TapeStats(
        n_events, burst_rate, shares, acf, msd, float(finite_mean(spreads)),
        float(finite_mean(depths[active])), float(mids[-1] - mids[0]), float(active.mean()),
    )


def log_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    valid = np.isfinite(y) & (y > 0)
    if valid.sum() < 3:
        return None
    return float(np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)[0])


def aggregate_tapes(
    rows: list[TapeStats], *, absolute_drifts: Sequence[float] | None = None,
) -> dict[str, object]:
    means = {key: finite_mean(np.array([asdict(row)[key] for row in rows]), axis=0) for key in asdict(rows[0])}
    gamma = log_slope(FLOW_LAGS, means["flow_acf"])
    hurst = log_slope(PRICE_LAGS, means["price_msd"])
    result = {key: value.tolist() for key, value in means.items()}
    result["gamma_flow"] = -gamma if gamma is not None else None
    result["hurst"] = hurst / 2 if hurst is not None else None
    # With K > 1, average each replicate's absolute drift before pooling.
    # Taking abs after averaging signed replicate drift understates the guard.
    if absolute_drifts is None:
        absolute_drifts = np.abs([row.drift_ticks for row in rows])
    if len(absolute_drifts) != len(rows):
        raise ValueError("absolute drifts must align with tape rows")
    result["mean_absolute_drift_ticks"] = float(finite_mean(np.asarray(absolute_drifts)))
    return result


# --- Pooled injection metrics ---


def _impact_law(paths: np.ndarray, law_admitted: bool | None) -> ImpactLaw | None:
    """Fit complete [day, child] responses after determining eligibility.

    The original panel applies the existing C2 test. Bootstrap resamples
    condition on that model's original result. Once admitted, the exponent
    refits on the same grid for each resampled mean, including grid boundaries.
    A draw still needs at least one complete day to fit a response.
    """
    if law_admitted is None:
        return identify_law(paths) if len(paths) >= 2 else None
    if not law_admitted or not len(paths):
        return None
    mean_path = paths.mean(axis=0)
    delta, amplitude = fit_power_law(np.arange(1, len(mean_path) + 1), mean_path)
    peak_se = float(paths[:, -1].std(ddof=1) / np.sqrt(len(paths))) if len(paths) >= 2 else float("nan")
    return ImpactLaw(True, "C2 admitted on original panel", delta, amplitude, float(mean_path[-1]), peak_se)


def summarize_impact(
    *, visible_mid: np.ndarray, blind_mid: np.ndarray, child_visible: np.ndarray,
    child_blind: np.ndarray, child_filled: np.ndarray, sigma_ticks: float, daily_shares: float,
    schedule: np.ndarray, generated: dict[str, object], replay: dict[str, object], paths_per_day: int = 1,
    include_peak_interval: bool = True, law_admitted: bool | None = None,
    metric_schema_version: str = LEGACY_METRIC_SCHEMA, grid_ms: np.ndarray | None = None,
    planned_child_shares: float | None = None,
    continuous_panel: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Re-use the LOB-Impact scorer on day-clustered buy/sell paths.

    Mid paths are [day, side, grid], children and fills [day, side, child].
    Side order is buy then sell. Seeds and matched sessions are averaged within
    each day before calling; midday or seed replicates do not inflate n.
    During outer bootstrap, C2 admission is fixed at the original-panel
    result and the peak diagnostic interval is omitted. Each draw still
    refits the admissible exponent and re-estimates the raw measurements.
    The continuous schema instead rejudges C2 on each draw (law_admitted=None); continuous
    readouts do not use admission. The fixed square-root reference
    uses historical daily volatility in ticks and intended total child
    shares / historical daily shares, never realized simulated fills.
    """
    if metric_schema_version not in (LEGACY_METRIC_SCHEMA, CONTINUOUS_METRIC_SCHEMA):
        raise ValueError("unknown market metric schema")
    n_days = len(visible_mid)
    visible = antisymmetrise(visible_mid[:, 0], visible_mid[:, 1])
    blind = antisymmetrise(blind_mid[:, 0], blind_mid[:, 1])
    execution_visible = antisymmetrise(child_visible[:, 0], child_visible[:, 1])
    execution_blind = antisymmetrise(child_blind[:, 0], child_blind[:, 1])
    whole_days = np.isfinite(execution_visible).all(axis=1)
    valid_execution = execution_visible[whole_days]
    law = _impact_law(valid_execution, law_admitted)

    peak_days = np.isfinite(execution_visible[:, -1])
    paired_days = peak_days & np.isfinite(execution_blind[:, -1])
    peak = float(finite_mean(execution_visible[peak_days, -1]))
    visible_peak = float(finite_mean(execution_visible[paired_days, -1]))
    blind_peak = float(finite_mean(execution_blind[paired_days, -1]))

    # Convert means back to sums in quarter ticks before forming the ratio.
    # Each signed buy/sell difference uses half-tick mids and a factor of 1/2.
    unit = 4 * paths_per_day
    visible_sum = int(np.rint(execution_visible[paired_days, -1] * unit).sum())
    blind_sum = int(np.rint(execution_blind[paired_days, -1] * unit).sum())
    peak_ratio = visible_sum / blind_sum if blind_sum > 0 else None
    traded = float(finite_mean(child_filled[peak_days].sum(axis=-1)))
    denominator = sigma_ticks * np.sqrt(traded / daily_shares) if daily_shares > 0 else 0
    y = float(peak / denominator) if denominator > 0 and np.isfinite(peak) else None

    visible_mean = finite_mean(visible, axis=0)
    last_child = int(np.flatnonzero(schedule)[-1])
    recovery_days = peak_days & np.isfinite(visible[:, last_child:]).all(axis=1)
    recovery_mean = finite_mean(visible[recovery_days], axis=0)
    recovery_mean[last_child] = finite_mean(execution_visible[recovery_days, -1])
    rec = None
    if recovery_days.any() and len(recovery_mean) - last_child >= 3 and recovery_mean[last_child] > 0:
        rec = relaxation(recovery_mean, schedule, tail=1)

    ci = None
    if include_peak_interval and len(valid_execution) >= 2:
        peaks = valid_execution[:, -1]
        rng = np.random.default_rng(9127)
        draws = peaks[rng.integers(len(peaks), size=(2000, len(peaks)))].mean(axis=1)
        ci = tuple(float(val) for val in np.quantile(draws, [0.025, 0.975]))

    metrics = ImpactMetrics(
        y=y, delta=law.delta if law and law.identified else None,
        end_over_peak=rec.end_over_peak if rec else None,
        relaxation_slope=rec.slope if rec else None,
        law_identified=law.identified if law is not None else None, peak_interval_ticks=ci,
        visible_to_blind_ratio=peak_ratio,
        gamma_flow=generated["gamma_flow"], hurst=generated["hurst"], idle_drift_ticks=generated["drift_ticks"],
    )
    reference = ImpactReference(
        gamma_flow=replay["gamma_flow"], hurst=replay["hurst"], idle_drift_ticks=replay["drift_ticks"],
        near_zero_peak_band_ticks=(-0.5, 0.5),
    )
    score = score_impact(metrics, reference)
    result = {
        "paired_days": n_days, "days_with_complete_execution_marks": int(whole_days.sum()),
        "days_with_visible_last_child": int(peak_days.sum()), "days_with_paired_last_child": int(paired_days.sum()),
        "days_with_complete_recovery": int(recovery_days.sum()), "visible_peak_for_amplitude_ticks": peak,
        "paired_visible_peak_ticks": visible_peak, "paired_blind_peak_ticks": blind_peak,
        "visible_curve_ticks": visible_mean.tolist(), "blind_curve_ticks": finite_mean(blind, axis=0).tolist(),
        "visible_child_ticks": finite_mean(execution_visible, axis=0).tolist(),
        "blind_child_ticks": finite_mean(execution_blind, axis=0).tolist(),
        "average_filled_shares": traded, "law": asdict(law) if law is not None else None,
        "measurements": asdict(metrics), "reference": asdict(reference), "score": asdict(score),
    }
    if metric_schema_version == CONTINUOUS_METRIC_SCHEMA:
        if grid_ms is None or planned_child_shares is None or not daily_shares > 0:
            raise ValueError("Continuous market metrics need a physical grid and fixed injection calibration")
        participation = planned_child_shares * int(np.count_nonzero(schedule)) / daily_shares
        continuous_visible = visible_mean
        continuous_blind = finite_mean(blind, axis=0)
        continuous_peak = peak
        joint_difference = None
        if continuous_panel is not None:
            # Equal-day means of each date's available physical window means.
            # The strict complete-replica panel above remains the legacy scorer's
            # input; the two estimands must never be silently conflated.
            continuous_visible = finite_mean(continuous_panel["visible"], axis=0)
            continuous_blind = None
            continuous_peak = float(finite_mean(continuous_panel["execution_peak"]))
            joint_difference = finite_mean(continuous_panel["joint_difference"], axis=0)
        continuous = dict(continuous_impact_readouts(
            visible_curve_ticks=continuous_visible, blind_curve_ticks=continuous_blind,
            grid_ms=grid_ms, schedule=schedule, scale_ticks=sigma_ticks, participation=participation,
            c2_pass=score.claims["C2"].status == "pass", execution_peak_ticks=continuous_peak,
            visible_minus_blind_curve_ticks=joint_difference,
        ))
        if continuous_panel is not None:
            from stochopia.market.rollout_eval import quote_support
            continuous["available_quote_support"] = quote_support(continuous_panel, paths_per_day)
        result["continuous"] = continuous
    return result


# --- Compact day-cluster observations ---


def day_paired_paths(
    paths: Sequence[Sequence[Mapping[RolloutArm, RolloutTrace]]],
    arms: tuple[RolloutArm, RolloutArm], field: str, *, relative: bool = False,
) -> np.ndarray:
    """Pool [seed, session, side, time] into [day, side, time].

    Runs contain the same two sessions per trading day and seed. Relative
    prices subtract each path's initial mid. If a paired mark is missing
    from any seed/session it remains missing at the day level.
    """
    by_seed = []
    for trial in paths:
        by_window = []
        for window in trial:
            sides = []
            for arm in arms:
                trace = window[arm]
                marks = getattr(trace, field)
                if relative:
                    marks = marks - trace.mid_ticks[0]
                sides.append(marks)
            by_window.append(sides)
        by_seed.append(by_window)
    data = np.asarray(by_seed)
    return data.reshape(len(paths), -1, 2, *data.shape[2:]).mean(axis=(0, 2))


def _cluster_tapes(rows: Sequence[TapeStats], replicates: int) -> dict[str, list]:
    """Store only the per-window terms entering these pooled statistics."""
    values = {}
    for field in ("events", "flow_acf", "price_msd", "drift_ticks"):
        ordered = np.array([getattr(row, field) for row in rows])
        values[field] = ordered.reshape(-1, replicates, *ordered.shape[1:]).tolist()
    return values


def market_bootstrap_payload(
    paths: Sequence[Sequence[Mapping[RolloutArm, RolloutTrace]]],
    generated: Sequence[TapeStats], replay: Sequence[TapeStats], *,
    cfg: Mapping[str, int | float], grid: np.ndarray, children: np.ndarray,
    replay_touch_shares: np.ndarray | None = None,
    num_replicates: int = 1, metric_schema_version: str = LEGACY_METRIC_SCHEMA,
    window_depth_errors: np.ndarray | None = None,
    generated_absolute_drifts: Sequence[float] | None = None,
    continuous_panel: Mapping[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Serialize chronological day clusters for exact repooling on each draw.

    Generated tapes arrive in seed-major, session-minor order. Historical
    touch shares is [window, grid, side]. The intra-day absolute touch error
    averages the absolute difference over all same-clock bid/ask cells and
    seeds, including zero usable depth. A missing historical panel leaves
    this optional readout unavailable. No event tape leaves the evaluator.
    Continuous-schema paths and tapes supplied here are already averaged *within* each
    seed/window over K replicates; the independent cluster count is still
    windows/2. Absolute depth/drift readouts are averaged before collapsing
    replicate signs. paths_per_day accounts for the mean's tick lattice.
    """
    if metric_schema_version not in (LEGACY_METRIC_SCHEMA, CONTINUOUS_METRIC_SCHEMA):
        raise ValueError("unknown market metric schema")
    if type(num_replicates) is not int or num_replicates < 1:
        raise ValueError("num_replicates must be a positive integer")
    if metric_schema_version == LEGACY_METRIC_SCHEMA and num_replicates != 1:
        raise ValueError("multiple replicas require the continuous metric schema")
    windows = len(replay)
    if not paths or not windows or windows % 2:
        raise ValueError("market clusters need seeds and two sessions per day")
    if any(len(trial) != windows for trial in paths) or len(generated) != windows * len(paths):
        raise ValueError("generated and historical tapes must align by seed and window")

    ordered = []
    for window in range(0, windows, 2):
        for seed in range(len(paths)):
            first = seed * windows + window
            ordered.extend(generated[first:first + 2])

    depth_errors = None
    if window_depth_errors is not None:
        errors = np.asarray(window_depth_errors, dtype=float)
        if errors.shape != (len(paths), windows):
            raise ValueError("window depth errors must align by seed and window")
        depth_errors = errors.reshape(len(paths), -1, 2).mean(axis=(0, 2)).tolist()
    elif replay_touch_shares is not None:
        observed = np.asarray(replay_touch_shares)
        if observed.shape != (windows, len(grid), 2):
            raise ValueError("historical touch shares must align by window, grid mark and side")
        free_depth = np.asarray([
            [window[RolloutArm.NO_INJECTION].touch_shares for window in trial]
            for trial in paths
        ], dtype=float)
        errors = np.abs(free_depth - observed[None]).reshape(len(paths), -1, 2, len(grid), 2)
        depth_errors = errors.mean(axis=(0, 2, 3, 4)).tolist()

    visible = (RolloutArm.BUY_VISIBLE, RolloutArm.SELL_VISIBLE)
    blind = (RolloutArm.BUY_BLIND, RolloutArm.SELL_BLIND)
    payload = {
        "visible_mid": day_paired_paths(paths, visible, "mid_ticks", relative=True).tolist(),
        "blind_mid": day_paired_paths(paths, blind, "mid_ticks", relative=True).tolist(),
        "child_visible": day_paired_paths(paths, visible, "after_child_mid_ticks", relative=True).tolist(),
        "child_blind": day_paired_paths(paths, blind, "after_child_mid_ticks", relative=True).tolist(),
        "child_filled": day_paired_paths(paths, visible, "child_fill_shares").tolist(),
        "paths_per_day": len(paths) * 2 * num_replicates,
        "sigma_ticks": float(cfg["sigma_ticks"]),
        "daily_shares": float(cfg["daily_shares"]),
        "schedule": np.isin(grid, children).tolist(),
        "generated_tapes": _cluster_tapes(ordered, len(paths) * 2),
        "historical_tapes": _cluster_tapes(replay, 2),
        "l1_depth_errors": depth_errors,
    }
    if metric_schema_version == CONTINUOUS_METRIC_SCHEMA:
        payload["metric_schema_version"] = metric_schema_version
        payload["grid_ms"] = grid.tolist()
        payload["planned_child_shares"] = float(cfg["child_shares"])
        payload["num_replicates"] = num_replicates
        if continuous_panel is not None:
            from stochopia.market.rollout_eval import decode_quote_panel
            panel = decode_quote_panel(continuous_panel, windows // 2, len(grid), payload["paths_per_day"])
            payload["continuous_panel"] = {key: values.tolist() for key, values in panel.items()}
        if generated_absolute_drifts is not None:
            values = np.asarray(generated_absolute_drifts, dtype=float)
            if values.shape != (len(generated),):
                raise ValueError("absolute drifts must align with generated tape rows")
            reordered = values.reshape(len(paths), -1, 2).transpose(1, 0, 2)
            payload["generated_tapes"]["absolute_drift_ticks"] = reordered.reshape(-1, len(paths) * 2).tolist()
    return payload
