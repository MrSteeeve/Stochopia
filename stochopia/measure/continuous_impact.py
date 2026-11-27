"""Continuous price-response measurements independent of C2 admission.

The reference, response and relaxation difference are in ticks. Time for the
AUC is measured in execution durations (dimensionless), so its units are ticks
times execution durations. Missing quotes are not price zero.
"""

from __future__ import annotations

from math import isfinite, sqrt
from typing import TypedDict

import numpy as np


P2_METRIC_SCHEMA = "p2-m1"
P1_METRIC_SCHEMA = "p1"
RESPONSE_EPSILON_TICKS = 0.0
RELAXATION_DENOMINATOR_FLOOR_TICKS = 1e-3
IDENTIFIED_POWERLAW = "IDENTIFIED_POWERLAW"
NO_RESPONSE = "NO_RESPONSE"
NOISY_UNIDENTIFIED = "NOISY_UNIDENTIFIED"

CONTINUOUS_IMPACT_METRICS = (
    "impact_curve_mae", "post_stop_relaxation_delta",
    "bounded_relaxation_ratio", "visible_minus_blind_auc",
)
C2_PASS_METRIC = "c2_pass_prob"


class ContinuousImpactReadouts(TypedDict):
    impact_identifiability_state: str | None
    c2_pass_prob: float
    impact_curve_mae: float | None
    post_stop_relaxation_delta: float | None
    bounded_relaxation_ratio: float | None
    visible_minus_blind_auc: float | None
    reference_curve_ticks: list[float]
    reference_scale_ticks: float
    reference_participation: float


def impact_identifiability_state(*, c2_pass: bool, execution_peak_ticks: float) -> str | None:
    """Classify C2 using its original positive peak threshold (zero ticks).

    A positive but statistically unresolved or boundary-grid power-law fit
    is distinct from no positive response. No finite peak is unobservable,
    not evidence of no response.
    """
    if c2_pass:
        return IDENTIFIED_POWERLAW
    if not isfinite(execution_peak_ticks):
        return None
    if execution_peak_ticks <= RESPONSE_EPSILON_TICKS:
        return NO_RESPONSE
    return NOISY_UNIDENTIFIED


def reference_impact_curve(
    grid_ms: np.ndarray, execution_end_ms: float, *, scale_ticks: float, participation: float,
) -> np.ndarray:
    """Evaluate the fixed square-root propagator on a physical-clock grid.

    Input [grid] is elapsed milliseconds; T_exec must be strictly positive.
    With normalized time u=t/T_exec, the model-independent reference is
    S_mkt sqrt(eta) sqrt(u) during execution. After the last child it is
    S_mkt sqrt(eta) [sqrt(u)-sqrt(u-1)]. The amplitude is in ticks; neither
    model's realized fills nor model's estimated impact calibrate it.
    """
    grid = np.asarray(grid_ms, dtype=float)
    if (
        grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all()
        or grid[0] != 0 or (np.diff(grid) <= 0).any()
        or not isfinite(execution_end_ms) or not 0 < execution_end_ms < grid[-1]
    ):
        raise ValueError("reference grid must start at zero and cover execution and recovery")
    if not isfinite(scale_ticks) or scale_ticks < 0:
        raise ValueError("reference scale must be a finite nonnegative volatility in ticks")
    if not isfinite(participation) or participation <= 0:
        raise ValueError("reference participation must be a positive finite fraction")
    relative_time = grid / execution_end_ms
    shape = np.sqrt(relative_time) - np.sqrt(np.maximum(relative_time - 1.0, 0.0))
    return scale_ticks * sqrt(participation) * shape


def _finite_curve(values: np.ndarray, shape: tuple[int, ...]) -> bool:
    return values.shape == shape and bool(np.isfinite(values).all())


def continuous_impact_readouts(
    *, visible_curve_ticks: np.ndarray, blind_curve_ticks: np.ndarray | None,
    grid_ms: np.ndarray, schedule: np.ndarray, scale_ticks: float,
    participation: float, c2_pass: bool, execution_peak_ticks: float,
    visible_minus_blind_curve_ticks: np.ndarray | None = None,
) -> ContinuousImpactReadouts:
    """Measure [grid] pooled, antisymmetric signed responses on the same clock.

    MAE is mean absolute tick error over every grid mark against the fixed
    propagator. Relaxation is I(T_exec)-I(T_end); its bounded-denominator
    ratio divides by max(abs(I(T_exec)), 1e-3 ticks), without clipping the
    numerator. AUC is trapezoidal integral of visible-blind over t/T_exec.
    Complete finite response curves always yield finite readouts regardless
    of C2; if an entire pooled quote is missing, that measurement is invalid
    and remains unavailable rather than filling the missing price with zero.
    A disabled blind arm makes only the AUC unavailable. When supplied, the
    joint visible-minus-blind curve averages only replica marks with all
    four arms observed; this avoids subtracting different support sets.
    """
    grid = np.asarray(grid_ms, dtype=float)
    schedule = np.asarray(schedule, dtype=bool)
    if schedule.shape != grid.shape or not schedule.any():
        raise ValueError("injection schedule must align with the reference clock")
    last = int(np.flatnonzero(schedule)[-1])
    reference = reference_impact_curve(
        grid, float(grid[last]), scale_ticks=scale_ticks, participation=participation,
    )
    visible = np.asarray(visible_curve_ticks, dtype=float)
    blind = None if blind_curve_ticks is None else np.asarray(blind_curve_ticks, dtype=float)
    joint = None
    if visible_minus_blind_curve_ticks is not None:
        joint = np.asarray(visible_minus_blind_curve_ticks, dtype=float)
    if (
        visible.shape != grid.shape or (blind is not None and blind.shape != grid.shape)
        or (joint is not None and joint.shape != grid.shape)
    ):
        raise ValueError("response curves must align with the reference clock")

    mae = None
    relaxation_delta = None
    relaxation_ratio = None
    auc = None
    if _finite_curve(visible, grid.shape):
        mae = float(np.abs(visible - reference).mean())
    if joint is None and blind is not None and _finite_curve(visible, grid.shape) and _finite_curve(blind, grid.shape):
        joint = visible - blind
    if joint is not None and _finite_curve(joint, grid.shape):
        time_steps = np.diff(grid / grid[last])
        auc = float(np.sum((joint[1:] + joint[:-1]) * time_steps / 2.0))
    if np.isfinite(visible[[last, -1]]).all():
        relaxation_delta = float(visible[last] - visible[-1])
        denominator = max(abs(float(visible[last])), RELAXATION_DENOMINATOR_FLOOR_TICKS)
        relaxation_ratio = relaxation_delta / denominator

    return {
        "impact_identifiability_state": impact_identifiability_state(
            c2_pass=c2_pass, execution_peak_ticks=execution_peak_ticks,
        ),
        "c2_pass_prob": float(c2_pass),
        "impact_curve_mae": mae,
        "post_stop_relaxation_delta": relaxation_delta,
        "bounded_relaxation_ratio": relaxation_ratio,
        "visible_minus_blind_auc": auc,
        "reference_curve_ticks": reference.tolist(),
        "reference_scale_ticks": float(scale_ticks),
        "reference_participation": float(participation),
    }
