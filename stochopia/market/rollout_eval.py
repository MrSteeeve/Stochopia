"""Paired window rollout, replica reduction and quote support diagnostics."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict

import numpy as np

from stochopia.measure.impact import antisymmetrise

from stochopia.market.tape_metrics import TapeStats, finite_mean, tape_statistics
from stochopia.market.windows import Window
from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace, RolloutWindow, rollout_paired_windows
from stochopia.neural.lob_interface import LOBWorldModel


# --- Physical window rollout and replica reduction ---

def window_replicates(
    model: LOBWorldModel, windows: Sequence[Window], *, child_times: np.ndarray, child_shares: int,
    grid: np.ndarray, rollout_seed: int, context_len: int, num_replicates: int, batch_windows: int,
) -> Iterator[tuple[dict[RolloutArm, RolloutTrace], ...]]:
    """Yield each physical window's K replicas, not K new statistical clusters."""
    for first in range(0, len(windows), batch_windows):
        chunk = windows[first:first + batch_windows]
        requests = [
            RolloutWindow(window.data.day, window.start, window.session, window.max_events)
            for window in chunk
        ]
        yield from rollout_paired_windows(
            model, requests, child_times_ms=child_times, child_shares=child_shares,
            grid_ms=grid, seed=rollout_seed, context_len=context_len, num_replicates=num_replicates,
        )


def _continuous_window(replicas: Sequence[dict[RolloutArm, RolloutTrace]]) -> dict[str, np.ndarray]:
    """Available paired marks; each price difference uses the same replica.

    MAE/relaxation use buy/sell visible marks only. AUC uses the within-replica
    visible-minus-blind difference and requires all four corresponding marks.
    Missing prices never become a zero response; report exact per-time counts.
    """
    def signed_response(buy: RolloutTrace, sell: RolloutTrace, field: str) -> np.ndarray:
        return antisymmetrise(
            getattr(buy, field) - buy.mid_ticks[0],
            getattr(sell, field) - sell.mid_ticks[0],
        )

    visible = []
    difference = []
    execution_peak = []
    for replica in replicas:
        buy = replica[RolloutArm.BUY_VISIBLE]
        sell = replica[RolloutArm.SELL_VISIBLE]
        response = signed_response(buy, sell, "mid_ticks")
        blind = signed_response(replica[RolloutArm.BUY_BLIND], replica[RolloutArm.SELL_BLIND], "mid_ticks")
        child = signed_response(buy, sell, "after_child_mid_ticks")
        visible.append(response)
        difference.append(response - blind)
        execution_peak.append(child[-1] if len(child) else np.nan)
    curves = np.stack(visible)
    contrast = np.stack(difference)
    peak = np.asarray(execution_peak)
    return {
        "visible": finite_mean(curves, axis=0),
        "joint_difference": finite_mean(contrast, axis=0),
        "execution_peak": finite_mean(peak),
        "visible_counts": np.isfinite(curves).sum(axis=0),
        "joint_counts": np.isfinite(contrast).sum(axis=0),
        "execution_peak_counts": np.asarray(np.isfinite(peak).sum()),
    }


def pool_continuous_days(rows: Sequence[dict[str, np.ndarray]], window_count: int) -> dict[str, np.ndarray]:
    """Equal available-window means within each day, with per-time K support.

    Input order is seed-major, window-minor. Each window has already reduced
    its K replicas. Count all finite replicas separately for transparency;
    the row count for bootstrap remains the number of physical trading days.
    """
    if window_count <= 0 or window_count % 2 or not rows or len(rows) % window_count:
        raise ValueError("Continuous panels require complete two-session day rows")
    seeds = len(rows) // window_count
    days = window_count // 2
    panel = {}
    for field in rows[0]:
        values = np.asarray([row[field] for row in rows])
        values = values.reshape(seeds, days, 2, *values.shape[1:])
        if field.endswith("_counts"):
            panel[field] = values.sum(axis=(0, 2))
        else:
            panel[field] = finite_mean(values, axis=(0, 2))
    return panel


def exhausted_replicate_rate(replicas: Sequence[dict[RolloutArm, RolloutTrace]]) -> float:
    """Fraction of window-replicate trajectories with any failed paired arm.

    Exhausted books, exhausted event budget and missing midpoint marks count
    once per replica even if multiple arms or marks fail. K identical copies
    of each trajectory leave this fraction unchanged. Missing arms are an
    invalid paired input; unavailable prices are never treated as zero.
    """
    if not replicas:
        raise ValueError("At least one replica is required")
    bad = 0
    for replica in replicas:
        if set(replica) != set(RolloutArm):
            raise ValueError("Incomplete paired trajectory")
        for trace in replica.values():
            if (
                trace.exhausted or trace.budget_exhausted or not np.isfinite(trace.mid_ticks).all()
                or not np.isfinite(trace.spread_ticks).all()
                or not np.isfinite(trace.after_child_mid_ticks).all()
            ):
                bad += 1
                break
    return bad / len(replicas)


def average_replicates(
    replicas: Sequence[dict[RolloutArm, RolloutTrace]], replay_depth: np.ndarray,
) -> tuple[dict[RolloutArm, RolloutTrace], TapeStats, float, float, float, dict[str, np.ndarray]]:
    """Reduce K trajectories to one paired physical-window observation.

    Curve and child marks are arithmetic averages, preserving missing prices
    when any required replicate lacks a two-sided midpoint. Absolute drift,
    depth error and exhaustion are measured per replica before averaging.
    The aggregated path contains no fictitious average event tape: generated
    event statistics must use the separately aggregated TapeStats record.
    """
    if not replicas:
        raise ValueError("At least one replica is required")
    paths = {}
    for arm in RolloutArm:
        traces = [replica[arm] for replica in replicas]
        if len(traces) == 1:
            paths[arm] = traces[0]
            continue
        marks = {}
        for field in ("mid_ticks", "spread_ticks", "touch_shares", "after_child_mid_ticks", "child_fill_shares"):
            marks[field] = np.mean([getattr(trace, field) for trace in traces], axis=0)
        paths[arm] = RolloutTrace(
            traces[0].grid_ms, marks["mid_ticks"], marks["spread_ticks"], marks["touch_shares"],
            np.empty((0, 4), dtype=np.int64), marks["after_child_mid_ticks"], marks["child_fill_shares"],
            any(trace.exhausted for trace in traces), any(trace.budget_exhausted for trace in traces),
        )

    tape_rows = []
    depth_errors = []
    exhausted = []
    for replica in replicas:
        free = replica[RolloutArm.NO_INJECTION]
        tape_rows.append(tape_statistics(free.events, free.mid_ticks, free.spread_ticks, free.touch_shares))
        depth_errors.append(float(np.abs(free.touch_shares.astype(float) - replay_depth).mean()))
        exhausted.append(free.exhausted or any(trace.budget_exhausted for trace in replica.values()))
    absolute_drift = float(finite_mean(np.abs([row.drift_ticks for row in tape_rows])))
    if len(tape_rows) == 1:
        summary = tape_rows[0]
    else:
        means = {}
        for key in asdict(tape_rows[0]):
            means[key] = finite_mean(np.asarray([getattr(row, key) for row in tape_rows]), axis=0)
        summary = TapeStats(**means)
    return (paths, summary, float(np.mean(depth_errors)), absolute_drift, float(np.mean(exhausted)),
            _continuous_window(replicas))


def stateless_projection(
    stateful: dict[str, float], stateless: dict[str, float], heads: Sequence[str],
) -> dict[str, object]:
    """Same event denominator for normal and reset-at-each-event NLL."""
    return {
        "stateless_nll_total": stateless["total"],
        "stateless_nll_gap": stateless["total"] - stateful["total"],
        "stateless_nll_by_stage": {head: stateless[head] for head in heads},
        "stateless_nll_gap_by_stage": {head: stateless[head] - stateful[head] for head in heads},
    }


# --- Matched available-quote panels ---

QUOTE_PANEL_FIELDS = (
    "visible", "joint_difference", "visible_counts", "joint_counts",
    "execution_peak", "execution_peak_counts",
)


def decode_quote_panel(
    payload: Mapping[str, object], days: int, marks: int, paths_per_day: int,
) -> dict[str, np.ndarray]:
    """Decode [day, grid] means/counts and [day] exact execution marks.

    Each physical mark's count is the number of available Monte Carlo
    replica/seed/session pairs; a finite mean is allowed iff at least one
    valid pair was observed. Joint differences require all four arms.
    """
    panel = {name: np.asarray(payload[name], dtype=float) for name in QUOTE_PANEL_FIELDS}
    for field, count_field in (
        ("visible", "visible_counts"), ("joint_difference", "joint_counts"),
        ("execution_peak", "execution_peak_counts"),
    ):
        values = panel[field]
        counts = panel[count_field]
        shape = (days,) if field == "execution_peak" else (days, marks)
        if values.shape != shape or counts.shape != shape:
            raise ValueError("available quote panel must align day and clock dimensions")
        if (not np.isfinite(counts).all() or (counts < 0).any() or (counts > paths_per_day).any()
                or (counts != np.floor(counts)).any()):
            raise ValueError("invalid available-quote replica counts")
        if not np.array_equal(np.isfinite(values), counts > 0):
            raise ValueError("finite available quote means must correspond to positive support counts")
    if (panel["joint_counts"] > panel["visible_counts"]).any():
        raise ValueError("joint visible/blind support cannot exceed visible support")
    return panel


def quote_support(panel: Mapping[str, np.ndarray], paths_per_day: int) -> dict[str, float | int]:
    """Describe observed support across [day,grid]; no count is a bootstrap unit."""
    n_days = len(panel["execution_peak"])
    expected = n_days * paths_per_day
    visible = panel["visible_counts"].sum(axis=0) / expected
    joint = panel["joint_counts"].sum(axis=0) / expected
    return {
        "days": n_days,
        "expected_pairs_per_grid_mark": expected,
        "visible_min_fraction": float(visible.min()),
        "visible_mean_fraction": float(visible.mean()),
        "joint_min_fraction": float(joint.min()),
        "joint_mean_fraction": float(joint.mean()),
        "execution_peak_fraction": float(panel["execution_peak_counts"].sum() / expected),
        "visible_complete_days": int(np.isfinite(panel["visible"]).all(axis=1).sum()),
        "joint_complete_days": int(np.isfinite(panel["joint_difference"]).all(axis=1).sum()),
    }


def common_quote_panels(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Retain common day/time supports across models for paired curve inference.

    The inputs are already within-window replica means; counts describe
    each model's observed replica support, not a claim of cross-model
    replicate-level completeness. A day/time with no mark in either model
    contributes to neither curve mean on that mark.
    """
    first = {}
    second = {}
    for field, count_field in (
        ("visible", "visible_counts"), ("joint_difference", "joint_counts"),
        ("execution_peak", "execution_peak_counts"),
    ):
        shared = np.isfinite(left[field]) & np.isfinite(right[field])
        first[field] = np.where(shared, left[field], np.nan)
        second[field] = np.where(shared, right[field], np.nan)
        first[count_field] = np.where(shared, left[count_field], 0)
        second[count_field] = np.where(shared, right[count_field], 0)
    return first, second
