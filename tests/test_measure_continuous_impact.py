"""P2 impact measures: preserve the legacy score and recover continuous readouts."""

from __future__ import annotations

import numpy as np
import pytest

from stochopia.market.tape_metrics import TapeStats, market_bootstrap_payload, summarize_impact
from stochopia.market.szse_eval import MarketClusters
from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace
from stochopia.measure.impact import (
    IDENTIFIED_POWERLAW, NO_RESPONSE, NOISY_UNIDENTIFIED, CONTINUOUS_METRIC_SCHEMA,
    continuous_impact_readouts, reference_impact_curve,
)


GRID = np.arange(13)
CHILDREN = np.array([2, 4, 6, 8])
SCHEDULE = np.isin(GRID, CHILDREN)
SHARES = 100
DAILY = 1600
SIGMA = 10.0
TAPE = {"gamma_flow": None, "hurst": None, "drift_ticks": 0.0}


def _reference() -> np.ndarray:
    return reference_impact_curve(GRID, 8, scale_ticks=SIGMA, participation=0.25, permanent_fraction=0.0)


def _summarize(amplitudes: np.ndarray) -> dict[str, object]:
    paths = amplitudes[:, None] * _reference()[None, :]
    visible = np.stack((paths, -paths), axis=1)
    blind = visible / 3
    return summarize_impact(
        visible_mid=visible, blind_mid=blind,
        child_visible=visible[:, :, SCHEDULE], child_blind=blind[:, :, SCHEDULE],
        child_filled=np.full((len(amplitudes), 2, len(CHILDREN)), SHARES),
        sigma_ticks=SIGMA, daily_shares=DAILY, schedule=SCHEDULE,
        generated=TAPE, replay=TAPE, metric_schema_version=CONTINUOUS_METRIC_SCHEMA,
        grid_ms=GRID, planned_child_shares=SHARES,
    )


def test_square_root_reference_mae_zero_and_legacy_score_near_full() -> None:
    profile = _summarize(np.ones(6))
    readings = profile["continuous"]
    assert readings["reference_scale_ticks"] == SIGMA
    assert readings["reference_participation"] == 0.25
    assert readings["impact_curve_mae"] == pytest.approx(0.0, abs=1e-15)
    exact = continuous_impact_readouts(
        visible_curve_ticks=_reference(), blind_curve_ticks=_reference() / 3,
        grid_ms=GRID, schedule=SCHEDULE, scale_ticks=SIGMA, participation=0.25,
        c2_pass=True, execution_peak_ticks=5.0,
    )
    assert exact["impact_curve_mae"] == 0.0
    assert readings["impact_identifiability_state"] == IDENTIFIED_POWERLAW
    # T_end/T_exec=1.5 puts the propagator's terminal ratio in the original
    # legacy scorer band [1/2, 2/3]; the task's real 2.0 ratio need not.
    assert profile["score"]["score"] == pytest.approx(100.0)
    assert readings["post_stop_relaxation_delta"] == pytest.approx(5.0 - _reference()[-1])
    assert readings["visible_minus_blind_auc"] > 0


@pytest.mark.parametrize(
    ("amplitudes", "state"),
    [
        (np.ones(6), IDENTIFIED_POWERLAW),
        (np.zeros(6), NO_RESPONSE),
        (-np.ones(6), NO_RESPONSE),
        (np.array([-4, -4, 0, 2, 5, 7]), NOISY_UNIDENTIFIED),
    ],
)
def test_three_states_and_finite_continuous_fields(amplitudes: np.ndarray, state: str) -> None:
    profile = _summarize(amplitudes)
    values = profile["continuous"]
    assert values["impact_identifiability_state"] == state
    for key in ("impact_curve_mae", "post_stop_relaxation_delta",
                "bounded_relaxation_ratio", "visible_minus_blind_auc"):
        assert np.isfinite(values[key])
    if state != IDENTIFIED_POWERLAW:
        assert profile["score"]["score"] is None


def test_zero_and_missing_blind_use_distinct_readouts() -> None:
    readings = continuous_impact_readouts(
        visible_curve_ticks=np.zeros(len(GRID)), blind_curve_ticks=None,
        grid_ms=GRID, schedule=SCHEDULE, scale_ticks=SIGMA, participation=0.25,
        c2_pass=False, execution_peak_ticks=0.0,
    )
    assert readings["impact_identifiability_state"] == NO_RESPONSE
    assert readings["bounded_relaxation_ratio"] == 0.0
    assert readings["post_stop_relaxation_delta"] == 0.0
    assert np.isfinite(readings["impact_curve_mae"])
    assert readings["visible_minus_blind_auc"] is None


def _trace(arm: RolloutArm) -> RolloutTrace:
    sign = -1 if arm in (RolloutArm.SELL_VISIBLE, RolloutArm.SELL_BLIND) else 1
    curve = sign * _reference()
    if arm in (RolloutArm.BUY_BLIND, RolloutArm.SELL_BLIND):
        curve /= 3
    return RolloutTrace(
        GRID, 100 + curve, np.ones(len(GRID)), np.ones((len(GRID), 2)),
        np.empty((0, 4)), 100 + curve[SCHEDULE], np.full(len(CHILDREN), SHARES),
    )


def _tape() -> TapeStats:
    return TapeStats(100, 0.0, np.ones(6) / 6, 1 / np.arange(1, 9),
                     np.ones(4), 1.0, 1.0, 0.0, 1.0)


@pytest.mark.parametrize("replicas", (1, 4, 8))
def test_k_averages_remain_in_original_day_clusters(replicas: int) -> None:
    # Each input window is already the K-replica average, not a new day.
    windows = [{arm: _trace(arm) for arm in RolloutArm} for _ in range(12)]
    payload = market_bootstrap_payload(
        [windows, windows], [_tape()] * 24, [_tape()] * 12,
        cfg={"sigma_ticks": SIGMA, "daily_shares": DAILY, "child_shares": SHARES},
        grid=GRID, children=CHILDREN, metric_schema_version=CONTINUOUS_METRIC_SCHEMA,
        num_replicates=replicas, generated_absolute_drifts=np.full(24, 2.0),
        window_depth_errors=np.full((2, 12), 3.0),
    )
    clusters = MarketClusters.from_payload(payload)
    assert clusters.days == 6
    assert clusters.num_replicates == replicas
    assert clusters.paths_per_day == 4 * replicas
    point = clusters.estimate(np.arange(6))
    assert point["free_abs_drift_ticks"] == 2.0
    assert point["l1_depth_mae"] == 3.0
    assert point["impact_curve_mae"] == pytest.approx(0.0, abs=1e-12)


def test_available_quote_panel_does_not_force_strict_c2_admission() -> None:
    paths = np.tile(_reference(), (6, 1))
    strict = np.stack((paths, -paths), axis=1)
    strict[:, :, 4] = np.nan  # one failed replica poisons strict day paths
    joint = paths * 0.75
    available = {
        "visible": paths, "joint_difference": joint, "execution_peak": paths[:, 8],
        "visible_counts": np.full_like(paths, 7.0),
        "joint_counts": np.full_like(paths, 6.0),
        "execution_peak_counts": np.full(6, 7.0),
    }
    profile = summarize_impact(
        visible_mid=strict, blind_mid=strict / 4,
        child_visible=strict[:, :, SCHEDULE], child_blind=strict[:, :, SCHEDULE] / 4,
        child_filled=np.full((6, 2, len(CHILDREN)), SHARES), sigma_ticks=SIGMA,
        daily_shares=DAILY, schedule=SCHEDULE, generated=TAPE, replay=TAPE,
        paths_per_day=8, metric_schema_version=CONTINUOUS_METRIC_SCHEMA, grid_ms=GRID,
        planned_child_shares=SHARES, continuous_panel=available,
    )
    continuous = profile["continuous"]
    assert profile["score"]["score"] is None
    assert continuous["c2_pass_prob"] == 0.0
    assert continuous["impact_identifiability_state"] == NOISY_UNIDENTIFIED
    assert continuous["impact_curve_mae"] == pytest.approx(0.0, abs=1e-15)
    assert np.isfinite(continuous["visible_minus_blind_auc"])
    support = continuous["available_quote_support"]
    assert support["visible_min_fraction"] == 7 / 8
    assert support["joint_min_fraction"] == 6 / 8
