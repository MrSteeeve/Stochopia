"""Small deterministic LOB paths exercise the paired matching protocol."""

from __future__ import annotations

from importlib.util import find_spec
from sys import version_info

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from experiments.p1_m2_metrics import summarize_impact
from stochopia.data import SZSEDaySlice
from stochopia.harness.lob_rollout import RolloutArm, RolloutTrace, _Order, _Path, _children, rollout_paired_window
from stochopia.neural.lob_interface import BUCKET_COUNTS, FiveTupleLogits, LOBBatch, LOBWorldModel, State, StepObservation, StepSampler
from stochopia.neural.lob_training import FitProtocol, encode_day, fit_lob_world_model
from stochopia.neural.primitives import SeedGRULOBModel


def _day() -> SZSEDaySlice:
    ticks = np.zeros((3, 2, 10), dtype=np.int32)
    shares = np.zeros((3, 2, 10), dtype=np.int64)
    ticks[:, 0, :3] = (100, 99, 98)
    ticks[:, 1, :3] = (104, 105, 106)
    shares[:, 0, :3] = (50, 70, 90)
    shares[:, 1, :3] = (60, 75, 90)
    return SZSEDaySlice(
        wind_code="000001.SZ", trading_date=20170103, prev_close_ticks=100,
        session_bounds=((0, 3), (3, 3)), time_ms=np.array([34_200_000, 34_200_010, 34_200_020]),
        seq_id=np.arange(3, dtype=np.uint64), dt_ms=np.array([0, 10, 10]),
        action_type=np.array([1, 1, 1], dtype=np.int8), rel_ticks=np.zeros(3, dtype=np.int32),
        volume_shares=np.full(3, 100), cancel_age_ms=np.zeros(3, dtype=np.int64),
        pre_book_ticks=ticks, pre_book_shares=shares, mid_ticks_x2=np.full(3, 204),
        spread_ticks=np.full(3, 4, dtype=np.int32),
    )


class _ScriptedModel(LOBWorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.inputs: list[StepObservation] = []

    def step_context(self, step_obs: StepObservation, state: State) -> tuple[Tensor, State]:
        self.inputs.append(step_obs)
        return step_obs.is_own[:, None], self.anchor

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        # Wait 10ms and quote 100 shares at the touch. An own child switches the side.
        fixed = (1, 1, 5, 1)
        index = (~context).long() if stage == 1 else torch.full_like(context, fixed[stage], dtype=torch.long)
        logits = torch.full((len(context), BUCKET_COUNTS[stage]), -torch.inf)
        return logits.scatter(-1, index, 0.0)


def _fixture_rollout(model: LOBWorldModel, day: SZSEDaySlice) -> dict[RolloutArm, RolloutTrace]:
    return rollout_paired_window(model, day, 2, child_times_ms=(10,), child_shares=90,
                                 grid_ms=(0, 10, 20), seed=1, context_len=2, max_events=8)


def test_five_arms_match_sweeps_and_reuse_the_absolute_blind_tape(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ScriptedModel()
    day = _day()
    if version_info >= (3, 13) and find_spec("jaxlob") is None:
        # The existing optional extra pins JAX-LOB to Python 3.10–3.12.
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            _fixture_rollout(model, day)
        return

    states: list[Tensor] = []
    sample = StepSampler.sample

    def record_draws(sampler: StepSampler, generator: torch.Generator) -> Tensor:
        states.append(generator.get_state())
        return sample(sampler, generator)

    monkeypatch.setattr(StepSampler, "sample", record_draws)
    arms = _fixture_rollout(model, day)
    assert set(arms) == set(RolloutArm)
    baseline = arms[RolloutArm.NO_INJECTION]
    np.testing.assert_array_equal(baseline.events, [[10, 1, 0, 100], [20, 1, 0, 100]])
    np.testing.assert_array_equal(baseline.mid_ticks, [102.0, 102.0, 102.0])
    assert baseline.child_fill_shares.size == 0

    for arm in (RolloutArm.BUY_VISIBLE, RolloutArm.BUY_BLIND):
        np.testing.assert_array_equal(arms[arm].child_fill_shares, [90])
        np.testing.assert_array_equal(arms[arm].after_child_mid_ticks, [102.5])
    for arm in (RolloutArm.SELL_VISIBLE, RolloutArm.SELL_BLIND):
        np.testing.assert_array_equal(arms[arm].child_fill_shares, [90])
        np.testing.assert_array_equal(arms[arm].after_child_mid_ticks, [101.5])

    # Both blind generators submit the same 104-tick ask after the tied child.
    buy_blind = arms[RolloutArm.BUY_BLIND]
    sell_blind = arms[RolloutArm.SELL_BLIND]
    np.testing.assert_array_equal(buy_blind.events, baseline.events)
    np.testing.assert_array_equal(sell_blind.events, baseline.events)
    np.testing.assert_array_equal(buy_blind.mid_ticks, [102.0, 102.0, 102.0])
    np.testing.assert_array_equal(buy_blind.touch_shares[:, 1], [60, 100, 200])
    np.testing.assert_array_equal(sell_blind.touch_shares[:, 0], [50, 30, 30])

    buy_visible = arms[RolloutArm.BUY_VISIBLE]
    np.testing.assert_array_equal(buy_visible.events, [[20, 0, 0, 100]])
    np.testing.assert_array_equal(buy_visible.mid_ticks, [102.0, 102.5, 102.5])
    assert sum(bool(obs.is_own[0]) for obs in model.inputs) == 2
    # A preempted proposal leaves the same first background draw available.
    assert len(states) == 9
    assert all(torch.equal(states[0], states[i]) for i in (3, 4, 6, 7))
    assert all(torch.equal(states[1], states[i]) for i in (5, 8))


def test_cancellation_at_price_can_span_fifo_orders_and_clips_to_depth() -> None:
    day = _day()
    if version_info >= (3, 13) and find_spec("jaxlob") is None:
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            _Path(day.pre_book_ticks[0], day.pre_book_shares[0], 12)
        return
    path = _Path(day.pre_book_ticks[0], day.pre_book_shares[0], 12)
    path.apply(_Order(0, 1, 0, 70, 104))
    path.apply(_Order(20, 5, 0, 100, 104))
    path.apply(_Order(30, 5, 0, 100, 104))
    trace = path.trace(np.array([0, 10, 20, 30]))
    np.testing.assert_array_equal(trace.touch_shares[:, 1], [60, 130, 30, 75])
    np.testing.assert_array_equal(trace.mid_ticks, [102, 102, 102, 102.5])


@pytest.mark.parametrize(("side", "filled"), [(0, 225), (1, 210)])
def test_ioc_child_discards_all_residue_when_opposite_book_is_empty(side: int, filled: int) -> None:
    day = _day()
    if version_info >= (3, 13) and find_spec("jaxlob") is None:
        with pytest.raises(ImportError, match="requires the harness-lob extra"):
            _Path(day.pre_book_ticks[0], day.pre_book_shares[0], 12)
        return
    path = _Path(day.pre_book_ticks[0], day.pre_book_shares[0], 12)
    path.apply(_children((10,), 500, side)[0])
    path.apply(_children((20,), 500, side)[0])
    np.testing.assert_array_equal(path.trace(np.array([0, 10, 20])).child_fill_shares, [filled, 0])
    np.testing.assert_array_equal(path.prices[side, :10], day.pre_book_ticks[0, side])
    np.testing.assert_array_equal(path.shares[side, :10], day.pre_book_shares[0, side])
    assert not path.shares[1 - side].any()


class _WarmupLogits(SeedGRULOBModel):
    def forward_sequence(self, batch: LOBBatch) -> FiveTupleLogits:
        logits = super().forward_sequence(batch)
        dt = logits.dt.clone()
        dt[:, 0] = torch.nan
        return logits._replace(dt=dt)


def test_fit_uses_burn_in_for_context_without_scoring_its_targets() -> None:
    day = encode_day(_day())
    protocol = FitProtocol(subseq_len=3, burn_in=1, steps=2, batch_size=1)
    fitted = fit_lob_world_model(_WarmupLogits, [day], 103, protocol=protocol)
    assert np.isfinite(fitted.last_nll)
    assert not fitted.model.training
    immediate_timeout = FitProtocol(subseq_len=3, burn_in=1, steps=2, batch_size=1, max_train_seconds=0)
    with pytest.raises(TimeoutError, match="training budget"):
        fit_lob_world_model(_WarmupLogits, [day], 103, protocol=immediate_timeout)


def test_warmup_cannot_cross_a_session_boundary() -> None:
    day = _day()
    with pytest.raises(ValueError, match="same-session history"):
        rollout_paired_window(_ScriptedModel(), day, 1, child_times_ms=(10,), child_shares=90,
                              grid_ms=(0, 10, 20), seed=1, context_len=2)


def test_rollout_rejects_a_crossed_initial_book() -> None:
    day = _day()
    day.pre_book_ticks[2, 0, 0] = 105
    with pytest.raises(ValueError, match="two uncrossed sides"):
        rollout_paired_window(_ScriptedModel(), day, 2, child_times_ms=(10,), child_shares=90,
                              grid_ms=(0, 10, 20), seed=1, context_len=2)


def test_impact_uses_common_day_support_for_paired_ratios() -> None:
    visible = np.array([[0, 1, 3, 4, 3, 3, 2], [0, 2, 6, 8, 7, 7, np.nan], [0, 1, 2, np.nan, 3, 3, 2]])
    blind = np.array([[0, 1, 1, 1, 1, 1, 1], [0, 1, 1, np.nan, 1, 1, 1], [0, 4, 6, 10, 10, 9, 9]])
    visible_mid = np.stack((visible, -visible), axis=1)
    blind_mid = np.stack((blind, -blind), axis=1)
    fills = np.full((3, 2, 3), 100)
    fills[2] = 10_000
    tape = {"gamma_flow": None, "hurst": None, "drift_ticks": 0.0}
    result = summarize_impact(
        visible_mid=visible_mid, blind_mid=blind_mid, child_visible=visible_mid[:, :, 1:4],
        child_blind=blind_mid[:, :, 1:4], child_filled=fills, sigma_ticks=10, daily_shares=1200,
        schedule=np.array([False, True, True, True, False, False, False]), generated=tape, replay=tape,
    )
    assert result["days_with_paired_last_child"] == 1
    assert result["days_with_complete_recovery"] == 1
    assert result["score"]["visible_to_blind_ratio"] == 4.0
    assert result["measurements"]["end_over_peak"] == 0.5
    assert type(result["measurements"]["y"]) is float
    assert result["measurements"]["y"] == 1.2


def test_exact_threefold_peak_ratio_passes_on_the_tick_grid() -> None:
    visible = np.zeros((12, 2, 7))
    blind = np.zeros_like(visible)
    visible[0, 0, 3] = 0.25
    blind[0, 0, 3] = 1 / 12
    tape = {"gamma_flow": None, "hurst": None, "drift_ticks": 0.0}
    result = summarize_impact(
        visible_mid=visible, blind_mid=blind, child_visible=visible[:, :, 1:4],
        child_blind=blind[:, :, 1:4], child_filled=np.ones((12, 2, 3)), sigma_ticks=1, daily_shares=10,
        schedule=np.array([False, True, True, True, False, False, False]), generated=tape, replay=tape,
        paths_per_day=6,
    )
    assert result["score"]["visible_to_blind_ratio"] == 3.0
    assert result["score"]["claims"]["C3"]["status"] == "pass"
