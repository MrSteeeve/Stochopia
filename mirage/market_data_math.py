"""Leakage-safe, dependency-free market-data mathematics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import median, pstdev
from typing import Mapping, Sequence


SHIBOR_NODES = (1, 7, 14, 30, 91, 182, 274, 365)


@dataclass(frozen=True)
class ShiborCurve:
    asof: date
    source_date: date
    percent_rates: Mapping[int, float]

    def __post_init__(self) -> None:
        lag = (self.asof - self.source_date).days
        if lag < 0 or lag > 5:
            raise ValueError("SHIBOR source must be no later than asof with lag <= 5 days")
        if set(self.percent_rates) != set(SHIBOR_NODES):
            raise ValueError("SHIBOR curve requires the frozen eight nodes")
        if any(not math.isfinite(float(rate)) or not 0 <= float(rate) <= 20 for rate in self.percent_rates.values()):
            raise ValueError("SHIBOR percentage rates must be in [0, 20]")

    def _node_log_discounts(self) -> dict[int, float]:
        return {
            days: -math.log1p((float(self.percent_rates[days]) / 100.0) * days / 365.0)
            for days in SHIBOR_NODES
        }

    def log_discount(self, days: int) -> float:
        if days == 0:
            return 0.0
        if days < SHIBOR_NODES[0] or days > SHIBOR_NODES[-1]:
            raise ValueError("SHIBOR interpolation target outside curve")
        values = self._node_log_discounts()
        if days in values:
            return values[days]
        upper = next(node for node in SHIBOR_NODES if node > days)
        lower = SHIBOR_NODES[SHIBOR_NODES.index(upper) - 1]
        weight = (days - lower) / (upper - lower)
        return values[lower] + weight * (values[upper] - values[lower])

    def discount(self, days: int) -> float:
        return math.exp(self.log_discount(days))

    def continuous_rate(self, days: int) -> float:
        if days <= 0:
            raise ValueError("rate horizon must be positive")
        return -self.log_discount(days) / (days / 365.0)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _call_put_sign(call_put: str) -> int:
    normalized = call_put.upper()
    if normalized not in {"C", "P"}:
        raise ValueError("call_put must be C or P")
    return 1 if normalized == "C" else -1


def black76_price(
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    sigma: float,
    call_put: str,
) -> float:
    sign = _call_put_sign(call_put)
    if forward <= 0 or strike <= 0 or maturity <= 0 or sigma <= 0:
        raise ValueError("positive forward, strike, maturity and sigma required")
    root_t = math.sqrt(maturity)
    d1 = (math.log(forward / strike) + 0.5 * sigma * sigma * maturity) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discount = math.exp(-rate * maturity)
    return discount * sign * (
        forward * _normal_cdf(sign * d1) - strike * _normal_cdf(sign * d2)
    )


def implied_vol_bisection(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    rate: float,
    call_put: str,
    *,
    sigma_min: float = 0.01,
    sigma_max: float = 3.0,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
) -> float | None:
    sign = _call_put_sign(call_put)
    if forward <= 0 or strike <= 0 or maturity <= 0 or price < 0:
        raise ValueError("invalid option inputs")
    discount = math.exp(-rate * maturity)
    lower_bound = discount * max(sign * (forward - strike), 0.0)
    upper_bound = discount * (forward if sign == 1 else strike)
    if price < lower_bound - tolerance or price > upper_bound + tolerance:
        raise ValueError("option price violates no-arbitrage bounds")
    low_price = black76_price(forward, strike, maturity, rate, sigma_min, call_put)
    high_price = black76_price(forward, strike, maturity, rate, sigma_max, call_put)
    if price < low_price - tolerance or price > high_price + tolerance:
        return None
    low, high = sigma_min, sigma_max
    for _ in range(max_iterations):
        middle = (low + high) / 2.0
        model = black76_price(forward, strike, maturity, rate, middle, call_put)
        if abs(model - price) <= tolerance:
            return middle
        if model < price:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


@dataclass(frozen=True)
class ExpiryPair:
    expiry: date
    strike: float
    version: str
    per_unit: float
    call_settlement: float
    put_settlement: float
    call_oi: float
    put_oi: float


@dataclass(frozen=True)
class ExpiryPoint:
    expiry: date
    dte: int
    forward: float
    implied_vol: float
    strikes: tuple[float, ...]
    iv_observations: int
    forward_mad_ratio: float = 0.0
    forward_range_ratio: float = 0.0


def build_expiry_point(
    pairs: Sequence[ExpiryPair],
    *,
    spot: float,
    asof: date,
    rcurve: ShiborCurve,
) -> ExpiryPoint:
    if not pairs or spot <= 0:
        raise ValueError("positive spot and option pairs required")
    identities = {(pair.expiry, pair.version, pair.per_unit) for pair in pairs}
    if len(identities) != 1 or next(iter(identities))[2] <= 0:
        raise ValueError("pairs must share expiry, version and positive per_unit")
    expiry = pairs[0].expiry
    dte = (expiry - asof).days
    if not 7 <= dte <= 365:
        raise ValueError("expiry DTE must be in [7, 365]")
    eligible = [
        pair for pair in pairs
        if pair.strike > 0
        and pair.call_settlement > 0
        and pair.put_settlement > 0
        and pair.call_oi > 0
        and pair.put_oi > 0
        and abs(math.log(pair.strike / spot)) <= 0.10
    ]
    if len({pair.strike for pair in eligible}) < 3 or len(eligible) != len({pair.strike for pair in eligible}):
        raise ValueError("at least three unique liquid near-money strikes required")
    maturity = dte / 365.0
    rate = rcurve.continuous_rate(dte)
    forwards = [
        pair.strike + math.exp(rate * maturity) * (pair.call_settlement - pair.put_settlement)
        for pair in eligible
    ]
    forward = median(forwards)
    mad = median(abs(value - forward) for value in forwards)
    if (
        forward <= 0
        or not 0.75 <= forward / spot <= 1.25
        or mad / forward > 0.01
        or (max(forwards) - min(forwards)) / forward > 0.03
    ):
        raise ValueError("put-call parity quality gate failed")
    selected = sorted(eligible, key=lambda pair: (abs(pair.strike - forward), pair.strike))[:3]
    ivs: list[float] = []
    for pair in selected:
        for settlement, call_put in (
            (pair.call_settlement, "C"), (pair.put_settlement, "P")
        ):
            try:
                iv = implied_vol_bisection(
                    settlement, forward, pair.strike, maturity, rate, call_put
                )
            except ValueError:
                iv = None
            if iv is not None:
                ivs.append(iv)
    if len(ivs) < 2:
        raise ValueError("fewer than two valid implied-vol observations")
    implied_vol = median(ivs)
    if not 0.01 <= implied_vol <= 1.5:
        raise ValueError("median implied volatility outside frozen range")
    return ExpiryPoint(
        expiry=expiry,
        dte=dte,
        forward=forward,
        implied_vol=implied_vol,
        strikes=tuple(pair.strike for pair in selected),
        iv_observations=len(ivs),
        forward_mad_ratio=mad / forward,
        forward_range_ratio=(max(forwards) - min(forwards)) / forward,
    )


def interpolate_total_variance(points: Sequence[ExpiryPoint], target_days: int) -> float:
    if target_days not in {30, 91, 182} or not points:
        raise ValueError("unsupported target or empty volatility term structure")
    ordered = sorted(points, key=lambda point: point.dte)
    if len({point.dte for point in ordered}) != len(ordered):
        raise ValueError("duplicate expiry horizons")
    exact = next((point for point in ordered if point.dte == target_days), None)
    if exact:
        return exact.implied_vol
    lower = [point for point in ordered if point.dte < target_days]
    upper = [point for point in ordered if point.dte > target_days]
    if lower and upper:
        left, right = lower[-1], upper[0]
        if right.dte - left.dte <= 120:
            weight = (target_days - left.dte) / (right.dte - left.dte)
            left_variance = left.implied_vol**2 * left.dte / 365.0
            right_variance = right.implied_vol**2 * right.dte / 365.0
            target_variance = left_variance + weight * (right_variance - left_variance)
            if target_variance < 0:
                raise ValueError("negative interpolated total variance")
            return math.sqrt(target_variance / (target_days / 365.0))
    nearest = min(ordered, key=lambda point: abs(point.dte - target_days))
    if abs(nearest.dte - target_days) / target_days <= 1 / 3:
        return nearest.implied_vol
    raise ValueError("volatility target cannot be bracketed or matched without extrapolation")


def futures_implied_q(
    futures: Sequence[tuple[int, float]],
    *,
    spot: float,
    rcurve: ShiborCurve,
    target_days: int = 91,
) -> float:
    if spot <= 0 or target_days != 91 or not futures:
        raise ValueError("invalid futures carry inputs")
    points = sorted((int(days), math.log(float(settlement) / spot)) for days, settlement in futures)
    if any(days <= 0 for days, _ in points) or len({days for days, _ in points}) != len(points):
        raise ValueError("positive unique futures horizons required")
    exact = next((basis for days, basis in points if days == target_days), None)
    if exact is not None:
        cumulative_basis = exact
    else:
        lower = [(days, basis) for days, basis in points if days < target_days]
        upper = [(days, basis) for days, basis in points if days > target_days]
        if lower and upper and upper[0][0] - lower[-1][0] <= 120:
            left, right = lower[-1], upper[0]
            weight = (target_days - left[0]) / (right[0] - left[0])
            cumulative_basis = left[1] + weight * (right[1] - left[1])
        else:
            nearest = min(points, key=lambda point: abs(point[0] - target_days))
            if abs(nearest[0] - target_days) / target_days > 1 / 3:
                raise ValueError("futures target cannot be bracketed or matched")
            cumulative_basis = nearest[1] * target_days / nearest[0]
    maturity = target_days / 365.0
    return rcurve.continuous_rate(target_days) - cumulative_basis / maturity


def trailing_return(closes: Sequence[float], lookback: int) -> float | None:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if len(closes) < lookback + 1:
        return None
    window = closes[-lookback - 1 :]
    if any(value <= 0 for value in window):
        raise ValueError("closes must be positive")
    return window[-1] / window[0] - 1.0


def realized_volatility(closes: Sequence[float], lookback: int) -> float | None:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if len(closes) < lookback + 1:
        return None
    window = closes[-lookback - 1 :]
    if any(value <= 0 for value in window):
        raise ValueError("closes must be positive")
    returns = [math.log(window[index] / window[index - 1]) for index in range(1, len(window))]
    return pstdev(returns) * math.sqrt(252.0)


def trailing_drawdown(closes: Sequence[float], lookback: int) -> float | None:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    if len(closes) < lookback:
        return None
    window = closes[-lookback:]
    if any(value <= 0 for value in window):
        raise ValueError("closes must be positive")
    return window[-1] / max(window) - 1.0
