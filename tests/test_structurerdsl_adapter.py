"""Cashflow valuation, unit conversion and supported contract semantics."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("structurerdsl")
from structurerdsl.interp import run
from structurerdsl.program import Program

from stochopia.adapters.structurer_dsl import (
    DSL_CONTRACT_VERSION, DSLContract, conditional_paths, future_value, observed_result, valuation,
)
from stochopia.benchmark import MarketSnapshot
from stochopia.pricing import bs_call, bs_greeks
from datetime import date


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def contract(body, n=6):
    program = Program.from_sexpr(
        f"(contract (meta (underlyings u) (maturity {n}) (schedules (obs {' '.join(map(str, range(1, n + 1)))}))) {body})"
    )
    return DSLContract.from_payload({"version": DSL_CONTRACT_VERSION, "program": program.to_json(), "notional": 1_000_000})


def market():
    return MarketSnapshot("dsl-test", 1, date(2023, 1, 31), "CSI500", 6000, 0.03, realized_vol_20d=0.25, carry_rate=0.01)


@pytest.mark.parametrize("cap", [None, 0.10])
def test_vanilla_and_spread_against_analytic_price(cap):
    expression = "(max (lin 1 (terminal (spot u)) -1.05) 0)" if cap is None else "(clip (lin 1 (terminal (spot u)) -1.05) 0 0.10)"
    c = contract(f"(premium (pay {expression} maturity))")
    m = market()
    paths = conditional_paths(c, m, (m.spot,), n_paths=80_000, seed=42)
    value, se = future_value(c, paths, 0)
    reference = bs_call(1, 1.05, 0.5, 0.03, 0.25, 0.01)
    if cap is not None:
        reference -= bs_call(1, 1.15, 0.5, 0.03, 0.25, 0.01)
    assert abs(value - reference * c.notional) <= 4 * se


def test_greek_units_against_analytic_call():
    c = contract("(premium (pay (max (lin 1 (terminal (spot u)) -1.05) 0) maturity))")
    m = market()
    result = valuation(c, m, (m.spot,), n_paths=80_000, seed=42)
    reference = bs_greeks(6000, 6300, 0.5, 0.03, 0.25, 0.01, "call")
    scale = c.notional / m.spot
    assert result["delta_units"] == pytest.approx(reference["delta"] * scale, rel=0.025)
    assert result["gamma_units"] == pytest.approx(reference["gamma"] * scale, rel=0.10)
    assert result["vega_dollars"] == pytest.approx(reference["vega"] * scale, rel=0.03)
    assert result["vega_absolute"] == pytest.approx(result["vega_dollars"] * 100)
    assert result["delta_dollars"] == pytest.approx(result["delta_units"] * m.spot)
    assert result["gamma_relative"] == pytest.approx(result["gamma_units"] * m.spot ** 2)


def test_conditional_paths_preserve_history_and_issue_fixing():
    c = contract("(premium (pay (avg_arith (spot u) obs) maturity))")
    m = replace(market(), spot=6300, round_num=3)
    paths = conditional_paths(c, m, (6000, 6100, 6300), n_paths=16, seed=42, spot_factor=1.1)
    assert torch.all(paths.spot["u"][:, 0] == 6000)
    assert torch.all(paths.spot["u"][:, 1] == 6100)
    assert torch.all(paths.spot["u"][:, 2] == 6300)  # today's observed spot is never bumped
    assert torch.all(paths.fixing["u"] == 6000)
    base = conditional_paths(c, m, (6000, 6100, 6300), n_paths=16, seed=42)
    assert torch.allclose(paths.spot["u"][:, 3:] / base.spot["u"][:, 3:], torch.full((16, 4), 1.1, dtype=torch.float64))


def test_spot_bump_does_not_move_todays_observed_fixing_across_a_knockout():
    # spot/fixing = 1.025 today, just below the 1.03 autocall level; a +1%
    # bump of the already-observed date used to wipe every path.
    c = contract("(funded 1 $issue_price (terminate_if (gt (spot u) 1.03) obs 0.05))")
    m = replace(market(), spot=6150, round_num=2, as_of=date(2023, 2, 28))
    history = (6000, 6150)
    base = future_value(c, conditional_paths(c, m, history, n_paths=4096, seed=42), 1)[0]
    up = future_value(c, conditional_paths(c, m, history, n_paths=4096, seed=42, spot_factor=1.01), 1)[0]
    assert up > 0.9 * base
    assert abs(valuation(c, m, history, n_paths=4096, seed=42)["delta_dollars"]) < 10 * c.notional


def test_initial_payment_is_not_an_asset_value():
    c = contract("(funded 1 $issue_price (pay 0 maturity))")
    m = market()
    paths = conditional_paths(c, m, (6000,), n_paths=16, seed=42)
    fair, _ = future_value(c, paths, 0)
    interpreted = run(c.program, paths, c.params(issue_price=fair / c.notional))
    assert interpreted.cashflows[0, 0] * c.notional == pytest.approx(-fair)
    assert fair == pytest.approx(c.notional * torch.exp(torch.tensor(-0.03 * 0.5)).item(), rel=1e-7)
    whole_trade_npv = (interpreted.cashflows * torch.exp(-paths.rate * paths.t_years)).sum(dim=1) * c.notional
    assert torch.max(torch.abs(whole_trade_npv)) < 1e-8
    assert future_value(c, paths, c.program.maturity)[0] == 0


def test_automatic_termination_and_conservative_floor():
    c = contract("(funded 1 $issue_price (both (pay 1 maturity) (terminate_if (gt (spot u) 1.01) obs 0)))", n=3)
    assert c.guaranteed_ratio == 1.0  # suppressed maturity leg is not a guarantee
    result = observed_result(c, (6000, 6100))
    assert int(result.termination[0]) == 1
    assert result.cashflows[0, 1] == 1
    assert result.cashflows[0, 2:].sum() == 0


def test_q_value_does_not_use_physical_trend():
    c = contract("(premium (pay (max (lin 1 (terminal (spot u)) -1) 0) maturity))")
    m = market()
    q0 = conditional_paths(c, m, (6000,), n_paths=32, seed=42)
    q1 = conditional_paths(c, replace(m, trend_alpha=0.5), (6000,), n_paths=32, seed=42)
    physical = conditional_paths(c, replace(m, trend_alpha=0.5), (6000,), n_paths=32, seed=42, physical=True)
    assert torch.equal(q0.spot["u"], q1.spot["u"])
    assert not torch.equal(q0.spot["u"], physical.spot["u"])


@pytest.mark.parametrize("body", [
    "(premium (anytime (pay 1 maturity)))",
    "(funded 1 1 (pay 0 maturity))",
    "(premium (coupon_on obs 0.1))",
    "(premium (pay rate maturity))",
    "(premium (pay $issue_price maturity))",
    "(premium (pay -1 maturity))",
])
def test_unsupported_semantics_are_explicit(body):
    with pytest.raises(ValueError):
        contract(body)


def test_payload_detaches_program_and_rejects_nonfinite_values():
    c = contract("(premium (pay 1 maturity))")
    payload = c.to_payload()
    detached = DSLContract.from_payload(payload)
    payload["program"]["body"]["args"][0]["args"][0] = 9
    assert detached.to_payload()["program"]["body"]["args"][0]["args"][0] == 1
    payload["notional"] = float("inf")
    with pytest.raises(ValueError):
        DSLContract.from_payload(payload)


@pytest.mark.parametrize("expression", [
    "(clip (lin 10 (terminal (spot u)) -7.9) 0.85 1)",
    "(lin 1 (mul_const 0.15 (clip (lin 100 (terminal (spot u)) -88) 0 1)) 0.85)",
])
def test_unconditional_maturity_payment_keeps_its_expression_floor(expression):
    c = contract(f"(premium (pay {expression} maturity))", n=3)
    assert c.guaranteed_ratio == pytest.approx(0.85)
    result = observed_result(c, (6000, 3000, 1200, 600))
    assert result.cashflows[0, 1:].sum().item() == pytest.approx(0.85)


def test_conditional_maturity_payment_does_not_guarantee_its_expression_floor():
    c = contract("(premium (when (gt (terminal (spot u)) 1.2) maturity (pay 0.85 maturity)))", n=3)
    assert c.guaranteed_ratio == 0
    assert observed_result(c, (6000, 6000, 6000, 6000)).cashflows[0, 1:].sum().item() == 0


@pytest.mark.parametrize("base,coupon", [(1., .24), (.85, .19)])
def test_unconditional_floor_survives_nonnegative_conditional_sibling(base, coupon):
    c = contract(f"(premium (both (pay {base} maturity) "
                 f"(when (gt (terminal (spot u)) 1.2) maturity (pay {coupon} maturity))))", n=3)
    assert c.guaranteed_ratio == pytest.approx(base)
    assert observed_result(c, (6000, 6000, 6000, 6000)).cashflows[0, 1:].sum().item() == pytest.approx(base)
    assert observed_result(c, (6000, 6000, 6000, 7800)).cashflows[0, 1:].sum().item() == pytest.approx(base + coupon)


def test_premium_unconditional_floor_can_be_suppressed_by_sibling_termination():
    c = contract("(premium (both (pay 0.85 maturity) (terminate_if (gt (spot u) 1.01) obs 0)))", n=3)
    assert c.guaranteed_ratio == 0
    assert observed_result(c, (6000, 6100)).cashflows[0, 1:].sum().item() == 0


def test_equivalent_maturity_payment_encodings_have_same_floor():
    direct = contract("(premium (pay (clip (lin 10 (terminal (spot u)) -7.9) 0.85 1) maturity))", n=3)
    funded = contract("(funded 1 $issue_price (give (pay (clip (lin -10 (terminal (spot u)) 8.9) 0 0.15) maturity)))", n=3)
    assert direct.guaranteed_ratio == pytest.approx(funded.guaranteed_ratio)
    for terminal in (600, 5250, 5280, 5340, 6000, 6600):
        history = (6000, 6000, 6000, terminal)
        torch.testing.assert_close(observed_result(direct, history).cashflows[:, 1:],
                                   observed_result(funded, history).cashflows[:, 1:])


def test_positive_payment_floor_does_not_protect_a_higher_purchase_price():
    c = contract("(premium (pay (clip (lin 10 (terminal (spot u)) -7.9) 0.85 1) maturity))", n=3)
    m = market()
    purchase_price = valuation(c, m, (m.spot,), n_paths=4096, seed=42)["fair_value"]
    payment_floor = c.guaranteed_ratio * c.notional
    realized = observed_result(c, (m.spot, 3000, 1200, 600))
    worst_payment = float(realized.cashflows[0, 1:].sum()) * c.notional
    assert worst_payment == pytest.approx(payment_floor)
    assert purchase_price > payment_floor
    assert (purchase_price - payment_floor) / purchase_price > 0
