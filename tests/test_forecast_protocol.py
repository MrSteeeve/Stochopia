from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from stochopia.research import forecast_protocol as fp
from stochopia.research.market_panel import MarketPanel


def _panel(sessions: int = 1000, seed: int = 3) -> MarketPanel:
    rng = np.random.default_rng(seed)
    days, current = [], date(2020, 1, 1)
    while len(days) < sessions:
        if current.weekday() < 5:
            days.append(current.isoformat())
        current += timedelta(days=1)
    returns = rng.normal(0.0, 0.012, size=(sessions, 2))
    close = 5000.0 * np.exp(np.cumsum(returns, axis=0))
    iv = np.where(np.arange(sessions) > 700, 0.2 + 0.01 * rng.standard_normal(sessions), np.nan)
    carry = np.column_stack([0.1 + 0.01 * rng.standard_normal(sessions), np.full(sessions, np.nan)])
    return MarketPanel(tuple(days), close, np.full(sessions, 0.02), carry, iv, {"schema": "test"})


def test_history_is_a_fixed_trailing_window_without_dates():
    panel = _panel()
    origins = fp.origin_indices(panel, "all")
    shapes = {tuple(v.shape for v in fp.public_history(panel, t).values()) for t in origins}
    assert len(shapes) == 1  # length never reveals the origin's position
    t = origins[0]
    history = fp.public_history(panel, t)
    assert set(history) == {"returns", "rate_3m", "carry", "iv_1m"}
    assert history["returns"].shape == (fp.RETURN_WINDOW, 2)
    assert history["carry"].shape == (fp.Q_WINDOW, 2)
    np.testing.assert_allclose(history["returns"], panel.log_returns[t + 1 - fp.RETURN_WINDOW : t + 1])
    assert np.all(np.isfinite(history["returns"]))
    with pytest.raises(ValueError):
        fp.public_history(panel, fp.MIN_HISTORY - 1)
    query = fp.query_for("dev", t)
    assert not any("date" in key for key in query)


def test_splits_are_disjoint_and_ordered_around_dev_end():
    panel = _panel()
    dev, test = fp.origin_indices(panel, "dev"), fp.origin_indices(panel, "test")
    assert dev and test
    assert max(panel.dates[t] for t in dev) <= fp.DEV_END < min(panel.dates[t] for t in test)
    assert all(t + fp.HORIZON < len(panel.dates) for t in dev + test)
    assert all(t >= fp.MIN_HISTORY for t in dev)


def test_crps_matches_brute_force_and_reduces_to_brier():
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(4, 50))
    observed = rng.normal(size=4)
    brute = [
        np.mean(np.abs(row - y)) - 0.5 * np.mean(np.abs(row[:, None] - row[None, :]))
        for row, y in zip(samples, observed)
    ]
    np.testing.assert_allclose(fp.crps(samples, observed), brute)
    binary = (rng.random((3, 40)) < 0.3).astype(float)
    outcome = np.array([0.0, 1.0, 1.0])
    np.testing.assert_allclose(fp.crps(binary, outcome), (binary.mean(axis=1) - outcome) ** 2)


def test_touch_probes_follow_the_path():
    up = np.full((1, fp.HORIZON), np.log(1.04) / 3)
    up[0, 3:] = 0.0
    payoff = fp._probe_payoffs(np.vstack([up, -up]))
    assert payoff["ko_touch_103"].tolist() == [1.0, 0.0]
    assert payoff["ki_touch_95"].tolist() == [0.0, 0.0]
    assert payoff["digital_up"].tolist() == [1.0, 0.0]


def test_reference_has_zero_skill_against_itself_and_overdispersion_loses():
    panel = _panel()
    reference = fp.reference_forecasts(panel, "dev")
    assert fp.evaluate_forecasts(reference, panel, reference=reference)["score"] == pytest.approx(0.0)

    def wide(history, query):
        returns, q = fp.reference_forecast(history, query)
        return returns * 2.0, q

    report = fp.evaluate_forecaster(wide, panel, "dev")
    assert report["score"] < 0
    assert report["family_skill"]["volatility"] < 0
    assert "q_carry_CSI1000" not in report["target_skill"]  # never observed in this panel
    assert set(report["slices"]) == {"low_trailing_vol", "mid_trailing_vol", "high_trailing_vol"}


def test_reference_is_deterministic_given_query_seed():
    panel = _panel()
    t = fp.origin_indices(panel, "dev")[0]
    first = fp.reference_forecast(fp.public_history(panel, t), fp.query_for("dev", t))
    second = fp.reference_forecast(fp.public_history(panel, t), fp.query_for("dev", t))
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


@pytest.mark.parametrize(
    "returns_shape, q_shape, message",
    [
        ((fp.N_PATHS, fp.HORIZON), (fp.N_PATHS, 3), "returns must have shape"),
        ((fp.N_PATHS, fp.HORIZON, 2), (fp.N_PATHS, 2), "q must have shape"),
    ],
)
def test_validate_rejects_wrong_shapes(returns_shape, q_shape, message):
    with pytest.raises(fp.ForecastError, match=message):
        fp.validate_forecast(np.zeros(returns_shape), np.ones(q_shape))


def test_extreme_q_cannot_remove_a_family_from_the_score():
    panel = _panel()

    def abstain(history, query):
        returns, q = fp.reference_forecast(history, query)
        return returns, np.full_like(q, 1e308)

    with pytest.raises(fp.ForecastError):
        fp.evaluate_forecaster(abstain, panel, "dev")
    report = fp.evaluate_forecaster(fp.reference_forecast, panel, "dev")
    assert "q_quote" in report["family_skill"]
    assert report["bootstrap_draws"] == fp.BOOTSTRAP_RESAMPLES
    assert set(report["scored_targets"]) == set(report["target_skill"])


def test_validate_rejects_non_finite_and_nonpositive_iv():
    returns = np.zeros((fp.N_PATHS, fp.HORIZON, 2))
    with pytest.raises(fp.ForecastError, match="finite"):
        fp.validate_forecast(returns + np.nan, np.ones((fp.N_PATHS, 3)))
    with pytest.raises(fp.ForecastError, match="implied volatility"):
        fp.validate_forecast(returns, np.zeros((fp.N_PATHS, 3)))
    with pytest.raises(fp.ForecastError, match="carry"):
        fp.validate_forecast(returns, np.column_stack([np.full(fp.N_PATHS, 0.2), np.full((fp.N_PATHS, 2), 5.0)]))
