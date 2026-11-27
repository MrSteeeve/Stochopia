"""Check the day panel assembled from rollout arms and aligned history."""

from __future__ import annotations

import numpy as np
import pytest

from stochopia.market.tape_metrics import TapeStats, market_bootstrap_payload
from stochopia.market.szse_eval import market_metric_estimates
from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace


def _trace(grid: np.ndarray, arm: RolloutArm, bid_diff: int, *, missing: bool) -> RolloutTrace:
    sign = -1 if arm in (RolloutArm.SELL_VISIBLE, RolloutArm.SELL_BLIND) else 1
    tick_movements = np.r_[0.0, np.sqrt(np.arange(1, 4) / 3), 0.85, 0.7, 0.6]
    if arm in (RolloutArm.BUY_BLIND, RolloutArm.SELL_BLIND):
        tick_movements /= 2
    shifts = 100 + sign * tick_movements
    if missing:
        shifts[1] = np.nan
    return RolloutTrace(
        grid_ms=grid,
        mid_ticks=shifts,
        spread_ticks=np.full_like(grid, 2),
        touch_shares=np.tile([100 + bid_diff, 200], (len(grid), 1)),
        events=np.zeros((0, 4), dtype=int),
        after_child_mid_ticks=shifts[1:4],
        child_fill_shares=np.full(3, 100),
    )


def _tape(events: int) -> TapeStats:
    return TapeStats(events, 0.0, np.ones(6) / 6, 1 / np.arange(1, 9),
                     np.array([1.0, 2.0, 4.0, 8.0]), 2.0, 150.0, 0.0, 1.0)


def test_day_panel_preserves_session_seed_blocks_and_true_depth_mae() -> None:
    grid = np.arange(7)
    paths = []
    for seed in range(2):
        windows = []
        for window in range(4):
            missing = seed == 1 and window == 2
            windows.append({arm: _trace(grid, arm, seed + window, missing=missing) for arm in RolloutArm})
        paths.append(windows)
    observed = np.tile([100, 200], (4, len(grid), 1))
    generated = [_tape(count) for count in [10, 11, 12, 13, 20, 21, 22, 23]]
    replay = [_tape(count) for count in [5, 6, 7, 8]]
    payload = market_bootstrap_payload(
        paths, generated, replay, cfg={"sigma_ticks": 1.0, "daily_shares": 300},
        grid=grid, children=np.arange(1, 4), replay_touch_shares=observed,
    )
    assert payload["generated_tapes"]["events"] == [[10, 11, 20, 21], [12, 13, 22, 23]]
    assert payload["historical_tapes"]["events"] == [[5, 6], [7, 8]]
    assert payload["l1_depth_errors"] == pytest.approx([0.5, 1.5])
    assert np.isnan(payload["child_visible"][1][0][0])
    assert payload["child_visible"][0][0] == pytest.approx(np.sqrt(np.arange(1, 4) / 3))
    estimates = market_metric_estimates(payload)
    assert estimates["event_count_rel_gap"] == pytest.approx((16.5 - 6.5) / 6.5)
    assert estimates["l1_depth_mae"] == pytest.approx(1.0)
    assert estimates["lob_impact_score"] is None
