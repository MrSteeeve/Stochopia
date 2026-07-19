"""Strict synthetic tests for market-data mathematics."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from mirage.market_data_math import (
    ExpiryPair,
    ExpiryPoint,
    ShiborCurve,
    black76_price,
    build_expiry_point,
    futures_implied_q,
    implied_vol_bisection,
    interpolate_total_variance,
    realized_volatility,
    trailing_drawdown,
    trailing_return,
)


def curve(asof=date(2025, 1, 31), percent=2.0):
    return ShiborCurve(asof, asof - timedelta(days=1), {d: percent for d in (1, 7, 14, 30, 91, 182, 274, 365)})


def point(days, vol):
    return ExpiryPoint(date(2025, 1, 1) + timedelta(days=days), days, 100.0, vol, (95, 100, 105), 6)


def test_shibor_simple_discount_log_interpolation_and_lag_gate():
    c = curve()
    expected = 1 / (1 + 0.02 * 91 / 365)
    assert c.discount(91) == pytest.approx(expected)
    assert c.log_discount(45) == pytest.approx(
        c.log_discount(30) + 15 / 61 * (c.log_discount(91) - c.log_discount(30))
    )
    assert math.exp(-c.continuous_rate(45) * 45 / 365) == pytest.approx(c.discount(45))
    with pytest.raises(ValueError):
        ShiborCurve(c.asof, c.asof - timedelta(days=6), c.percent_rates)
    with pytest.raises(ValueError):
        ShiborCurve(c.asof, c.asof, {d: 21 for d in c.percent_rates})


@pytest.mark.parametrize("call_put", ["C", "P"])
def test_black76_iv_round_trip_and_bounds(call_put):
    price = black76_price(102, 100, 0.5, 0.02, 0.27, call_put)
    assert implied_vol_bisection(price, 102, 100, 0.5, 0.02, call_put) == pytest.approx(0.27, abs=1e-7)
    with pytest.raises(ValueError):
        implied_vol_bisection(200, 102, 100, 0.5, 0.02, call_put)


def test_expiry_point_parity_liquidity_and_iv_quality():
    asof = date(2025, 1, 31)
    expiry = asof + timedelta(days=91)
    c = curve(asof)
    rate, maturity, forward, sigma = c.continuous_rate(91), 91 / 365, 101.0, 0.25
    pairs = [
        ExpiryPair(
            expiry, strike, "M", 10000,
            black76_price(forward, strike, maturity, rate, sigma, "C"),
            black76_price(forward, strike, maturity, rate, sigma, "P"), 10, 12,
        )
        for strike in (95.0, 100.0, 105.0, 108.0)
    ]
    result = build_expiry_point(pairs, spot=100, asof=asof, rcurve=c)
    assert result.forward == pytest.approx(forward)
    assert result.implied_vol == pytest.approx(sigma, abs=1e-7)
    assert len(result.strikes) == 3
    assert result.iv_observations == 6
    assert result.forward_mad_ratio <= 0.01
    assert result.forward_range_ratio <= 0.03
    with pytest.raises(ValueError):
        build_expiry_point(pairs[:2], spot=100, asof=asof, rcurve=c)
    with pytest.raises(ValueError):
        build_expiry_point(
            [ExpiryPair(**{**pair.__dict__, "expiry": asof + timedelta(days=6)}) for pair in pairs],
            spot=100, asof=asof, rcurve=c,
        )
    with pytest.raises(ValueError):
        build_expiry_point(
            [
                ExpiryPair(**{**pairs[0].__dict__, "call_settlement": 0}),
                ExpiryPair(**{**pairs[1].__dict__, "put_settlement": 0}),
                *pairs[2:],
            ],
            spot=100, asof=asof, rcurve=c,
        )


def test_total_variance_bracket_nearest_and_no_extrapolation():
    result = interpolate_total_variance([point(60, 0.2), point(120, 0.3)], 91)
    expected_w = 0.2**2 * 60 / 365 + 31 / 60 * (0.3**2 * 120 / 365 - 0.2**2 * 60 / 365)
    assert result == pytest.approx(math.sqrt(expected_w / (91 / 365)))
    assert interpolate_total_variance([point(100, 0.24)], 91) == 0.24
    with pytest.raises(ValueError):
        interpolate_total_variance([point(10, 0.2)], 91)


def test_futures_q_sign_and_forward_reconstruction():
    c = curve(percent=3.0)
    target, q = 91, 0.012
    rate = c.continuous_rate(target)
    forward = 100 * math.exp((rate - q) * target / 365)
    inferred = futures_implied_q([(60, 100 * math.exp((rate - q) * 60 / 365)), (120, 100 * math.exp((rate - q) * 120 / 365))], spot=100, rcurve=c)
    assert inferred == pytest.approx(q)
    assert 100 * math.exp((rate - inferred) * target / 365) == pytest.approx(forward)


def test_strict_windows_and_prefix_invariance():
    prefix = [100, 101, 99, 102, 104]
    assert trailing_return(prefix[:4], 4) is None
    assert realized_volatility(prefix[:4], 4) is None
    assert trailing_drawdown(prefix[:3], 4) is None
    assert trailing_return(prefix, 4) == pytest.approx(0.04)
    assert trailing_drawdown([100, 110, 105, 99], 4) == pytest.approx(99 / 110 - 1)
    future = prefix + [1, 999]
    assert realized_volatility(future[: len(prefix)], 4) == realized_volatility(prefix, 4)
    assert trailing_return(future[: len(prefix)], 4) == trailing_return(prefix, 4)
    assert trailing_drawdown(future[: len(prefix)], 4) == trailing_drawdown(prefix, 4)
