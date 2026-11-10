from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from stochopia.research import forecast_protocol as fp
from stochopia.research.market_panel import MarketPanel
from stochopia.research.world_model_generator import generate_scenario, snowball_events
from stochopia.research.world_models import TIERS, WORLD_MODELS, get_world_model


def _panel(sessions: int = 900, *, with_iv: bool = True) -> MarketPanel:
    rng = np.random.default_rng(11)
    days, current = [], date(2020, 1, 1)
    while len(days) < sessions:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    close = 5000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, size=(sessions, 2)), axis=0))
    iv = 0.2 + 0.01 * rng.standard_normal(sessions) if with_iv else np.full(sessions, np.nan)
    carry = 0.1 + 0.01 * rng.standard_normal((sessions, 2))
    return MarketPanel(tuple(days), close, np.full(sessions, 0.02), carry, iv, {"schema": "test"})


@pytest.mark.parametrize("model", sorted(WORLD_MODELS))
def test_registered_models_produce_valid_one_step_forecasts(model):
    panel = _panel()
    t = fp.MIN_HISTORY + 10
    returns, q = get_world_model(model).forecast()(fp.public_history(panel, t), fp.query_for("dev", t))
    fp.validate_forecast(returns, q)


def test_r02s01_is_marked_development_only():
    entry = get_world_model("r02s01")
    assert entry.tier == "draft"
    assert entry.development_only
    with pytest.raises(KeyError, match="unknown world model"):
        get_world_model("missing")


def test_tier_is_the_highest_consecutively_passed_check():
    entry = get_world_model("r02s01")
    checks = {"coherent": "passed", "predictive": "not_run", "decision_grade": "passed"}
    skipped = replace(entry, tier_checks=checks)
    assert skipped.tier == "coherent"
    assert skipped.development_only
    top = replace(entry, tier_checks={name: "passed" for name in TIERS[1:]})
    assert top.tier == "decision_grade"
    assert not top.development_only


@pytest.mark.parametrize("model", ["r02s01", "ewma_gbm_reference"])
def test_chained_scenario_shapes_and_determinism(model):
    panel = _panel()
    t = fp.MIN_HISTORY + 40
    first = generate_scenario(panel, t, model=model, n_paths=24, horizon_days=63, continuation_samples=16)
    second = generate_scenario(panel, t, model=model, n_paths=24, horizon_days=63, continuation_samples=16)
    assert first.returns.shape == (24, 63, 2)
    assert first.q.shape == (24, 3, 3)
    np.testing.assert_array_equal(first.returns, second.returns)
    other = generate_scenario(panel, t, model=model, n_paths=24, horizon_days=63, seed=1, continuation_samples=16)
    assert not np.array_equal(first.returns, other.returns)
    assert first.relative_spot.shape == (24, 64, 2)
    assert np.all(first.relative_spot[:, 0] == 1.0)
    assert first.manifest["measure"].startswith("P")
    assert first.manifest["tier"] == "draft"
    assert first.manifest["development_only"] is True
    assert first.as_of == panel.dates[t]


def test_scenario_never_uses_data_after_as_of():
    panel = _panel()
    t = fp.MIN_HISTORY + 40
    baseline = generate_scenario(panel, t, n_paths=16, horizon_days=42, continuation_samples=8)
    truncated = replace(
        panel,
        dates=panel.dates[: t + 1],
        close=panel.close[: t + 1],
        rate_3m=panel.rate_3m[: t + 1],
        carry=panel.carry[: t + 1],
        iv_1m=panel.iv_1m[: t + 1],
    )
    altered = generate_scenario(truncated, t, n_paths=16, horizon_days=42, continuation_samples=8)
    np.testing.assert_array_equal(baseline.returns, altered.returns)
    np.testing.assert_array_equal(baseline.q, altered.q)


def test_small_scenarios_subsample_the_scored_sample_size():
    panel = _panel()
    t = fp.MIN_HISTORY + 40
    full = generate_scenario(panel, t, n_paths=fp.N_PATHS, horizon_days=21)
    small = generate_scenario(panel, t, n_paths=10, horizon_days=21)
    np.testing.assert_array_equal(full.returns[:10], small.returns)


def test_seeds_follow_the_date_not_the_panel_index():
    panel = _panel()
    t = fp.MIN_HISTORY + 40
    shifted = replace(panel, dates=panel.dates[5:], close=panel.close[5:], rate_3m=panel.rate_3m[5:],
                      carry=panel.carry[5:], iv_1m=panel.iv_1m[5:])
    first = generate_scenario(panel, panel.dates[t], n_paths=8, horizon_days=21)
    second = generate_scenario(shifted, panel.dates[t], n_paths=8, horizon_days=21)
    np.testing.assert_array_equal(first.returns, second.returns)


def test_missing_quotes_are_flagged_and_never_written_into_history(monkeypatch):
    from stochopia.research import world_model_generator as gen

    seen = []
    original = gen.get_world_model

    def spying(name):
        entry = original(name)
        inner = entry.forecast()

        def forecast(history, query):
            seen.append(np.isfinite(history["iv_1m"]).sum())
            return inner(history, query)

        return replace(entry, loader=lambda: forecast)

    monkeypatch.setattr(gen, "get_world_model", spying)
    scenario = gen.generate_scenario(_panel(with_iv=False), fp.MIN_HISTORY + 30, n_paths=4, horizon_days=63,
                                     continuation_samples=4)
    assert scenario.manifest["q_supported"]["iv_1m_CSI1000"] is False
    assert scenario.manifest["q_supported"]["carry_CSI500"] is True
    assert seen and max(seen) == 0  # the default IV draw was never appended as a quote


def test_horizon_must_be_whole_steps():
    with pytest.raises(ValueError, match="multiple"):
        generate_scenario(_panel(), fp.MIN_HISTORY, horizon_days=30)


def test_snowball_events_on_deterministic_paths():
    up = np.linspace(1.0, 1.10, 64)[None, :]
    down = np.linspace(1.0, 0.80, 64)[None, :]
    events = snowball_events(np.vstack([up, down]), 63)
    assert events["knock_out"].tolist() == [1.0, 0.0]
    assert events["knock_in"].tolist() == [0.0, 1.0]
    assert events["payoff"][1] < 1.0 < events["payoff"][0]


PANEL_DIR = Path(__file__).resolve().parents[1] / "data" / "derived" / "world_model"


@pytest.mark.skipif(not (PANEL_DIR / "panel.npz").is_file(), reason="local real-data panel not built")
def test_vendored_r02s01_reproduces_the_loop_scores():
    from stochopia.research.market_panel import load_panel

    panel = load_panel(PANEL_DIR)
    forecast = get_world_model("r02s01").forecast()
    assert fp.evaluate_forecaster(forecast, panel, "dev")["score"] == pytest.approx(0.0478, abs=1e-4)
    assert fp.evaluate_forecaster(forecast, panel, "test")["score"] == pytest.approx(0.0783, abs=1e-4)
