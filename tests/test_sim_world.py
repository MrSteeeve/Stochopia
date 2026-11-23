"""Tier-0 simulator: flow statistics, propagator, exact counterfactuals."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

from stochopia.sim.flow import FlowSpec, simulate_flow  # noqa: E402
from stochopia.sim.impact import KernelSpec, kernel_values, propagate  # noqa: E402
from stochopia.sim.world import (  # noqa: E402
    WorldSpec,
    injection_schedule,
    simulate_day,
    true_response,
)


def _autocorr(signs, lag):
    x = signs.astype(float)
    return float(np.mean(x[:-lag] * x[lag:]))


def test_flow_signs_are_balanced_and_persistent():
    flow = simulate_flow(FlowSpec(), 400_000, np.random.default_rng(0))
    assert set(np.unique(flow.signs)) == {-1, 1}
    assert abs(flow.signs.mean()) < 0.05
    c10, c100 = _autocorr(flow.signs, 10), _autocorr(flow.signs, 100)
    assert c10 > c100 > 0.0
    # alpha = 1.5 gives a decay exponent near 0.5 between lags 10 and 100
    gamma = np.log(c10 / c100) / np.log(10.0)
    assert 0.3 < gamma < 0.8


def test_flow_metaorders_keep_one_sign_and_one_label():
    flow = simulate_flow(FlowSpec(informed_frac=0.3), 50_000, np.random.default_rng(1))
    order = np.argsort(flow.meta_id, kind="stable")
    ids, signs, informed = flow.meta_id[order], flow.signs[order], flow.informed[order]
    same = ids[1:] == ids[:-1]
    assert np.all(signs[1:][same] == signs[:-1][same])
    assert np.all(informed[1:][same] == informed[:-1][same])
    assert 0.2 < flow.informed.mean() < 0.4
    # P(length >= 2) = 2 ** -1.5, so single-trade metaorders are the majority
    lengths = np.bincount(flow.meta_id)
    assert np.mean(lengths >= 2) == pytest.approx(2.0**-1.5, abs=0.02)


def test_flow_without_informed_traders_has_no_informed_trades():
    flow = simulate_flow(FlowSpec(), 10_000, np.random.default_rng(2))
    assert not flow.informed.any()


def test_propagate_matches_direct_convolution():
    spec = KernelSpec()
    x = np.random.default_rng(3).choice([-1.0, 1.0], size=300)
    kernel = kernel_values(spec, np.arange(300))
    direct = np.array([np.dot(kernel[: t + 1][::-1], x[: t + 1]) for t in range(300)])
    assert np.allclose(propagate(spec, x), direct)
    assert kernel[0] == pytest.approx(spec.g0)


def _log_slope(spec):
    g10, g100 = kernel_values(spec, np.array([10.0, 100.0]))
    return float(np.log(g10 / g100) / np.log(10.0))


def test_kernel_decays_and_approaches_power_law_as_range_grows():
    lags = np.arange(0, 3000)
    assert np.all(np.diff(kernel_values(KernelSpec(), lags)) < 0)
    narrow = _log_slope(KernelSpec(beta=0.25, tau_max=1e3, n_scales=7))
    wide = _log_slope(KernelSpec(beta=0.25, tau_max=1e9, n_scales=60))
    assert narrow > wide > 0.25
    assert wide == pytest.approx(0.25, abs=0.03)


def test_simulate_day_is_reproducible_and_seeded_per_day():
    spec = WorldSpec(n_trades=2000, burn_in=200)
    a, b = simulate_day(spec, "t", 0), simulate_day(spec, "t", 0)
    c = simulate_day(spec, "t", 1)
    assert np.array_equal(a.returns, b.returns) and np.array_equal(a.signs, b.signs)
    assert not np.array_equal(a.signs, c.signs)
    assert a.signs.size == a.returns.size == 2000 and a.pre_signs.size == 200


def test_true_response_equals_kernel_sum_for_linear_world():
    spec = WorldSpec(n_trades=3000, burn_in=300)
    day = simulate_day(spec, "t", 0)
    schedule = injection_schedule(n_children=20, spacing=9, n_relax=200)
    buy = true_response(spec, day, 500, schedule, +1)
    sell = true_response(spec, day, 500, schedule, -1)
    child_times = np.flatnonzero(schedule)
    t = np.arange(schedule.size)
    lags = t[:, None] - child_times[None, :]
    expected = np.where(lags >= 0, kernel_values(spec.kernel, np.abs(lags)), 0.0).sum(axis=1)
    assert np.allclose(0.5 * (buy - sell), expected)
    assert not np.allclose(buy, expected)


def test_true_response_matches_event_by_event_recomputation():
    """Each side separately, including the clock shift of earlier trades."""
    spec = WorldSpec(n_trades=400, burn_in=60)
    day = simulate_day(spec, "t", 3)
    schedule = injection_schedule(n_children=6, spacing=3, n_relax=25)
    start = 120
    history = list(day.pre_signs) + list(day.signs[:start])
    background = list(day.signs[start : start + int((~schedule).sum())])

    def price(seq):
        seq = np.asarray(seq, dtype=float)
        return float(np.dot(kernel_values(spec.kernel, np.arange(seq.size))[::-1], seq))

    for side in (+1, -1):
        got = true_response(spec, day, start, schedule, side)
        with_inj, without, used = list(history), list(history), 0
        for t, is_child in enumerate(schedule):
            if is_child:
                with_inj.append(side)
            else:
                with_inj.append(background[used])
                without.append(background[used])
                used += 1
            assert got[t] == pytest.approx(price(with_inj) - price(without), abs=1e-9)


def test_information_moves_price_with_informed_flow_only():
    spec = WorldSpec(
        n_trades=200_000, burn_in=2000, flow=FlowSpec(informed_frac=0.3), info_drift=0.5
    )
    day = simulate_day(spec, "t", 0)
    signed = day.returns * day.signs
    gap = signed[day.informed].mean() - signed[~day.informed].mean()
    assert gap == pytest.approx(0.5, abs=0.05)


def test_true_response_rejects_window_outside_day():
    spec = WorldSpec(n_trades=500, burn_in=50)
    day = simulate_day(spec, "t", 0)
    schedule = injection_schedule(n_children=10, spacing=9, n_relax=500)
    with pytest.raises(ValueError):
        true_response(spec, day, 100, schedule, +1)
