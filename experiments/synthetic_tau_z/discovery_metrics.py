"""Synthetic discovery measurements; identical likelihood/CF panel to the reference study."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from experiments.synthetic_tau_z.design import BATCH_WINDOWS, CONTEXT, K, ROLLOUT_SEED
from experiments.synthetic_tau_z.evaluate import _day_likelihood, _mean_finite, _mean_marks, _window_metrics
from stochopia.data import SZSEDaySlice
from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace, RolloutWindow, rollout_paired_windows
from stochopia.measure.continuous_impact import continuous_impact_readouts
from stochopia.neural.lob_interface import LOBWorldModel
from stochopia.neural.lob_training import encode_day
from stochopia.sim.queue_reactive_world import CHILD_SHARES, CHILD_TIMES_MS, GRID_MS


@dataclass(frozen=True)
class MeasurementPanel:
    """Observable evaluation data only: no generator name, latent state, or training histories."""

    evaluation: tuple[SZSEDaySlice, ...]
    windows: tuple[RolloutWindow, ...]
    truth: np.ndarray
    scale_ticks: float
    participation: float


def reference_calibration(train: Sequence[SZSEDaySlice]) -> tuple[float, float]:
    """Train-only standard deviation of daily midpoint changes and planned/observed trade shares."""
    if len(train) < 2:
        raise ValueError("reference scale requires at least two independent training days")
    changes = []
    volumes = []
    for day in train:
        mids = day.mid_ticks_x2.astype(float) / 2
        changes.append(float(mids[-1] - mids[0]))
        aggressive = (day.action_type == 2) | (day.action_type == 3)
        volumes.append(float(day.volume_shares[aggressive].sum()))
    daily_volume = float(np.mean(volumes))
    if daily_volume <= 0:
        raise ValueError("historical aggressive shares must be positive")
    return float(np.std(changes, ddof=1)), CHILD_SHARES * len(CHILD_TIMES_MS) / daily_volume


def _signed(replica: dict[RolloutArm, RolloutTrace], buy: RolloutArm, sell: RolloutArm) -> np.ndarray:
    """Antisymmetric matched response, requiring also the same finite free mark (as M2)."""
    free = replica[RolloutArm.NO_INJECTION].mid_ticks
    return ((replica[buy].mid_ticks - free) - (replica[sell].mid_ticks - free)) / 2


def _impact_readouts(replicas: Sequence[dict[RolloutArm, RolloutTrace]], panel: MeasurementPanel) -> dict:
    visible = []
    differences = []
    children = []
    drift = []
    for replica in replicas:
        active = _signed(replica, RolloutArm.BUY_VISIBLE, RolloutArm.SELL_VISIBLE)
        blind = _signed(replica, RolloutArm.BUY_BLIND, RolloutArm.SELL_BLIND)
        visible.append(active)
        differences.append(active - blind)
        buy = replica[RolloutArm.BUY_VISIBLE]
        sell = replica[RolloutArm.SELL_VISIBLE]
        children.append(((buy.after_child_mid_ticks - buy.mid_ticks[0]) -
                         (sell.after_child_mid_ticks - sell.mid_ticks[0])) / 2)
        free = replica[RolloutArm.NO_INJECTION].mid_ticks
        drift.append(abs(float(free[-1] - free[0])))
    curve = _mean_marks(np.stack(visible))
    joint = _mean_marks(np.stack(differences))
    # C2 retains the strict, matched complete-replica child marks. It is evaluated
    # nonlinearly across days by the parent bootstrap, not as a per-window pass.
    child = np.mean(children, axis=0)
    readouts = continuous_impact_readouts(
        visible_curve_ticks=curve, blind_curve_ticks=None, grid_ms=GRID_MS,
        schedule=np.isin(GRID_MS, CHILD_TIMES_MS), scale_ticks=panel.scale_ticks,
        participation=panel.participation, c2_pass=False, execution_peak_ticks=float(child[-1]),
        visible_minus_blind_curve_ticks=joint,
    )
    result = {key: readouts[key] for key in (
        "stylized_impact_mae", "post_stop_relaxation_delta", "visible_minus_blind_auc",
    )}
    result["free_abs_drift_ticks"] = _mean_finite(drift)
    result["c2_child_ticks"] = child.tolist() if np.isfinite(child).all() else None
    return result


def _mean_complete(values: Sequence[float | None]) -> float | None:
    return float(np.mean(values)) if all(x is not None and isfinite(x) for x in values) else None


def _relative_clock_error(generated_cv2: float | None, historical_cv2: float | None) -> float | None:
    """Window-local CV² discrepancy, normalized by its matched historical flow."""
    if generated_cv2 is None or historical_cv2 is None:
        return None
    if not isfinite(generated_cv2) or not isfinite(historical_cv2):
        return None
    return abs(generated_cv2 - historical_cv2) / max(historical_cv2, 1.0)


def evaluate_discovery_model(model: LOBWorldModel, panel: MeasurementPanel) -> list[dict[str, Any]]:
    """Two sessions/physical day, K aligned replicas; no fabrication of unavailable quotations."""
    if panel.truth.shape != (len(panel.windows), len(GRID_MS)) or len(panel.windows) != 2 * len(panel.evaluation):
        raise ValueError("evaluation, truth, and physical window axes do not align")
    daily = []
    for day in panel.evaluation:
        row = _day_likelihood(model, encode_day(day))
        for old, new in (("action_nll", "event_nll"), ("rel_ticks_nll", "level_nll"), ("volume_nll", "size_nll")):
            row[new] = row.pop(old)
        row["trading_date"] = int(day.trading_date)
        daily.append(row)
    by_day: list[list[dict]] = [[] for _ in daily]
    for first in range(0, len(panel.windows), BATCH_WINDOWS):
        chunk = panel.windows[first:first+BATCH_WINDOWS]
        results = rollout_paired_windows(
            model, chunk, child_times_ms=CHILD_TIMES_MS, child_shares=CHILD_SHARES,
            grid_ms=GRID_MS, seed=ROLLOUT_SEED, context_len=CONTEXT, num_replicates=K,
        )
        for index, (window, replicas) in enumerate(zip(chunk, results, strict=True), start=first):
            if window.day.trading_date != daily[index//2]["trading_date"]:
                raise ValueError("window order changed relative to the registered day calendar")
            values = _window_metrics(window, replicas, panel.truth[index], GRID_MS)
            values["sign_acf_abs_error"] = values.pop("direction_acf_mae")
            values["interarrival_cv2_rel_error"] = _relative_clock_error(
                values["interarrival_cv2_generated"], values["interarrival_cv2_historical"],
            )
            values.update(_impact_readouts(replicas, panel))
            by_day[index//2].append(values)
    strict = ("counterfactual_impact_mae", "stylized_impact_mae", "post_stop_relaxation_delta",
              "visible_minus_blind_auc")
    for row, windows in zip(daily, by_day, strict=True):
        if len(windows) != 2:
            raise ValueError("every independent date requires two physical windows")
        for key in windows[0]:
            values = [w[key] for w in windows]
            if key == "c2_child_ticks":
                row[key] = None if any(x is None for x in values) else np.mean(values, axis=0).tolist()
            elif key in strict:
                row[key] = _mean_complete(values)
            else:
                row[key] = _mean_finite(values)
    return daily
