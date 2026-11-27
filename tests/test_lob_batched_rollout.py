"""V2 true batching, common-random tapes and physical-day replica reduction."""

from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec
from sys import version_info

import numpy as np
import pytest
import torch

from stochopia.market.tape_metrics import market_bootstrap_payload
from stochopia.market.szse_eval import MarketClusters
from stochopia.market.rollout_eval import average_replicates, pool_continuous_days
from stochopia.harness.lob_batch_matching import BatchedMatcher
from stochopia.harness.lob_rollout import (
    RolloutArm, RolloutTrace, RolloutWindow, _Order, _Path, rollout_paired_windows,
)
from stochopia.harness.lob_rollout_batch import background_uniforms
from stochopia.neural.lob_interface import State, StepObservation
from test_lob_rollout import _day, _fixture_rollout, _ScriptedModel


class BatchedScript(_ScriptedModel):
    def __init__(self, *, stochastic: bool = False) -> None:
        super().__init__()
        self.stochastic = stochastic
        self.batch_sizes = []

    def step_context(self, step_obs: StepObservation, state: State) -> tuple[torch.Tensor, State]:
        batch = len(step_obs.is_own)
        self.batch_sizes.append(batch)
        # A genuine recursive tensor tree with a leading batch axis.
        if state is None:
            state = {
                "counter": torch.zeros((batch, 1)),
                "matrix": torch.zeros((batch, 2, 2)),
            }
        assert isinstance(state, dict)
        next_state = {key: value + 1 for key, value in state.items()}
        return step_obs.is_own[:, None], next_state

    def head_logits(self, context: torch.Tensor, stage: int, prior: tuple[torch.Tensor, ...]) -> torch.Tensor:
        logits = super().head_logits(context, stage, prior)
        if self.stochastic and stage == 0:
            logits[:, 2] = 0
        return logits


def _run(model: BatchedScript, k: int) -> list[tuple[dict[RolloutArm, RolloutTrace], ...]]:
    day = _day()
    windows = [RolloutWindow(day, 2, 0, 8), RolloutWindow(replace(day, trading_date=20170104), 2, 0, 8)]
    return rollout_paired_windows(
        model, windows, child_times_ms=(10,), child_shares=90, grid_ms=(0, 10, 20),
        seed=1, context_len=2, num_replicates=k,
    )


def _assert_trace_equal(left: RolloutTrace, right: RolloutTrace) -> None:
    for field in ("grid_ms", "mid_ticks", "spread_ticks", "touch_shares", "events",
                  "after_child_mid_ticks", "child_fill_shares"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
    assert left.exhausted == right.exhausted
    assert left.budget_exhausted == right.budget_exhausted


def _no_jax_on_new_python() -> bool:
    return version_info >= (3, 13) and find_spec("jaxlob") is None


def test_batched_lifecycle_matches_scalar_and_packs_all_active_arms() -> None:
    model = BatchedScript()
    if _no_jax_on_new_python():
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            _run(model, 4)
        return
    expected = _fixture_rollout(_ScriptedModel(), _day())
    actual = _run(model, 4)
    assert len(actual) == 2 and all(len(window) == 4 for window in actual)
    assert max(model.batch_sizes) == 2 * 4 * 3
    assert any(size < max(model.batch_sizes) for size in model.batch_sizes)
    for window in actual:
        for replica in window:
            for arm in RolloutArm:
                _assert_trace_equal(replica[arm], expected[arm])


def test_crn_replica_prefix_and_blind_tapes_are_independent_of_batch_width() -> None:
    np.testing.assert_array_equal(background_uniforms(1, 20170103, 0, 0, 4),
                                  background_uniforms(1, 20170103, 0, 0, 9)[:4])
    assert not np.array_equal(background_uniforms(1, 20170103, 0, 0, 4),
                              background_uniforms(1, 20170103, 0, 1, 4))
    assert not np.array_equal(background_uniforms(1, 20170103, 0, 0, 4),
                              background_uniforms(1, 20170104, 0, 0, 4))
    if _no_jax_on_new_python():
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            _run(BatchedScript(stochastic=True), 8)
        return
    one = _run(BatchedScript(stochastic=True), 1)
    four = _run(BatchedScript(stochastic=True), 4)
    eight = _run(BatchedScript(stochastic=True), 8)
    for narrow, middle, wide in zip(one, four, eight, strict=True):
        for arm in RolloutArm:
            _assert_trace_equal(narrow[0][arm], wide[0][arm])
            for i in range(4):
                _assert_trace_equal(middle[i][arm], wide[i][arm])
        for replica in wide:
            free = replica[RolloutArm.NO_INJECTION].events
            np.testing.assert_array_equal(replica[RolloutArm.BUY_BLIND].events, free)
            np.testing.assert_array_equal(replica[RolloutArm.SELL_BLIND].events, free)


def test_vmapped_matching_retains_fifo_and_grows_without_changing_results() -> None:
    day = _day()
    ticks = day.pre_book_ticks[0]
    shares = day.pre_book_shares[0]
    if _no_jax_on_new_python():
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            BatchedMatcher([ticks], [shares])
        return
    batch = BatchedMatcher([ticks, ticks], [shares, shares], capacity=10)
    scalar = [_Path(ticks, shares, 128), _Path(ticks, shares, 128)]
    orders = [(_Order(i, 1, 0, i, 104), _Order(i, 0, 0, i, 100)) for i in range(1, 21)]
    orders.extend([
        (_Order(21, 5, 0, 90, 104), None),
        (_Order(22, 2, 0, 400, 999_999_999, True), _Order(22, 3, 0, 250, 0, True)),
    ])
    for pair in orders:
        fills = batch.apply(pair)
        for lane, order in enumerate(pair):
            if order is not None:
                assert fills[lane] == scalar[lane].apply(order)
    grid = np.arange(23)
    for left, right in zip(batch.paths, scalar, strict=True):
        _assert_trace_equal(left.trace(grid), right.trace(grid))


def _replica(i: int) -> dict[RolloutArm, RolloutTrace]:
    grid = np.array([0, 10, 20])
    base = 102.0
    drift = 2.0 if i % 2 == 0 else -2.0
    free_depth = np.full((3, 2), 0 if i % 2 == 0 else 200, dtype=np.int64)
    events = np.array([[10, 2, 0, 100], [20, 3, 0, 100]])
    paths = {}
    for arm in RolloutArm:
        midpoint = np.full(3, base)
        depth = np.full((3, 2), 100, dtype=np.int64)
        child = np.array([base])
        if arm == RolloutArm.NO_INJECTION:
            midpoint[-1] += drift
            depth = free_depth
            child = np.empty(0)
        elif arm in (RolloutArm.BUY_VISIBLE, RolloutArm.SELL_VISIBLE):
            sign = 1 if arm == RolloutArm.BUY_VISIBLE else -1
            midpoint += sign * np.array([0, 2, 1])
            child += sign * 2
        else:
            sign = 1 if arm == RolloutArm.BUY_BLIND else -1
            midpoint += sign * np.array([0, 1, 0.5])
            child += sign
        if i == 1 and arm == RolloutArm.BUY_VISIBLE:
            midpoint[1] = np.nan
        if i == 2 and arm == RolloutArm.SELL_BLIND:
            midpoint[-1] = np.nan
        paths[arm] = RolloutTrace(
            grid, midpoint, np.full(3, 1.0), depth, events,
            child, np.full(len(child), 100, dtype=np.int64),
        )
    return paths


@pytest.mark.parametrize("k", [1, 4, 8])
def test_window_replica_average_preserves_days_and_absolute_readouts(k: int) -> None:
    day_count = 3
    windows = day_count * 2
    seed_count = 3
    replicas = [_replica(i) for i in range(k)]
    paths, tape, depth, drift, exhausted, marks = average_replicates(replicas, np.full((3, 2), 100))
    panel = pool_continuous_days([marks] * windows * seed_count, windows)
    payload = market_bootstrap_payload(
        [[paths] * windows for _ in range(seed_count)], [tape] * windows * seed_count, [tape] * windows,
        cfg={"sigma_ticks": 2., "daily_shares": 1000., "child_shares": 100.},
        grid=np.array([0, 10, 20]), children=np.array([10]),
        num_replicates=k, metric_schema_version="p2-m1", continuous_panel=panel,
        window_depth_errors=np.full((seed_count, windows), depth),
        generated_absolute_drifts=[drift] * windows * seed_count,
    )
    assert MarketClusters.from_payload(payload).days == day_count
    assert len(payload["visible_mid"]) == day_count
    assert payload["paths_per_day"] == seed_count * 2 * k
    assert depth == 100.0
    assert drift == 2.0
    assert exhausted == 0
    if k > 1:
        assert np.isnan(paths[RolloutArm.BUY_VISIBLE].mid_ticks[1])
        assert tape.drift_ticks == 0
    np.testing.assert_array_equal(marks["visible"], [0., 2., 1.])
    np.testing.assert_array_equal(marks["joint_difference"], [0., 1., 0.5])
    expected = [k, k - int(k > 1), k]
    assert marks["visible_counts"].tolist() == expected
    assert marks["joint_counts"].tolist() == [expected[0], expected[1], k - int(k > 2)]
