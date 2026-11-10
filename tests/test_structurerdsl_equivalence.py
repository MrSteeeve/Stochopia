"""Equivalence of StructurerDSL programs with Stochopia's legacy pricers.

Stochopia is the consumer of the DSL library: these tests pin the DSL
interpreter to the deterministic pricing functions the v3 protocol relies on.

* snowball / autocallable: the DSL is run on exactly the random paths that
  ``pricing._mc_stats`` draws (same ``random.Random(seed)`` stream, same
  early-stop behaviour), so fair value, knock-in and knock-out probabilities
  must agree to floating-point precision, not just within MC error.
* vanilla: DSL Monte Carlo vs ``bs_call``/``bs_put`` within 3 standard errors.
* barrier: DSL daily-monitored barrier vs the continuous Reiner–Rubinstein
  ``barrier_option`` after the Broadie–Glasserman–Kou barrier shift.
"""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")
structurerdsl = pytest.importorskip("structurerdsl")

from structurerdsl import catalog, price  # noqa: E402
from structurerdsl.gbm import gbm_paths  # noqa: E402
from structurerdsl.path import Paths  # noqa: E402
from structurerdsl.program import Program  # noqa: E402

from stochopia import pricing as legacy  # noqa: E402


def _legacy_paths(n_paths: int, n_steps: int, r: float, sigma: float, q: float, seed: int, ko: float | None) -> Paths:
    """Reproduce the monthly GBM ratios drawn by ``_mc_stats`` (drift r, dt=1/12),
    including its early stop at knock-out (later steps held constant)."""
    dt = 1.0 / 12.0
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    rng = random.Random(seed)
    rows = []
    for _ in range(n_paths):
        ratio = 1.0
        row = [1.0]
        stopped = False
        for _t in range(1, n_steps + 1):
            if not stopped:
                z = rng.gauss(0.0, 1.0)
                ratio = ratio * math.exp(drift + vol * z)
                if ko is not None and ratio >= ko:
                    stopped = True
            row.append(ratio)
        rows.append(row)
    spot = torch.tensor(rows, dtype=torch.float64)
    t = torch.arange(n_steps + 1, dtype=torch.float64) / 12.0
    return Paths(spot={"u": spot}, t_years=t, rate=r, div={"u": q})


def _legacy_snowball_program(n_steps: int, ko: float, ki: float, coupon: float) -> Program:
    """Snowball with Stochopia's exact maturity rule: loss only if knocked in
    AND below fixing at maturity; otherwise full coupon (even after knock-in)."""
    obs = " ".join(str(i) for i in range(1, n_steps + 1))
    loss_cond = f"(and (touched (spot u) {ki} down obs) (lt (terminal (spot u)) 1.0))"
    body = (
        "(funded 1.0 1.0 (both "
        f"(terminate_if (gt (spot u) {ko}) obs (mul_const {coupon} time)) "
        f"(when (not (gt (spot u) {ko})) maturity (both "
        f"(when {loss_cond} maturity (pay (lin 1 (terminal (spot u)) -1.0) maturity)) "
        f"(when (not {loss_cond}) maturity (pay (mul_const {coupon} time) maturity))))))"
    )
    return Program.from_sexpr(f"(contract (meta (underlyings u) (maturity {n_steps}) (schedules (obs {obs}))) {body})")


@pytest.mark.parametrize("ko,ki,coupon,sigma", [(1.03, 0.75, 0.12, 0.25), (1.00, 0.80, 0.20, 0.40), (1.05, 0.70, 0.08, 0.15)])
def test_snowball_matches_legacy_mc_exactly(ko, ki, coupon, sigma):
    T_months, r, q, n_paths, seed = 24, 0.03, 0.01, 4000, 42
    legacy_res = legacy.snowball_price(
        1.0, T_months, r, sigma, q, coupon, knock_in_pct=ki, knock_out_pct=ko, n_paths=n_paths, seed=seed
    )
    paths = _legacy_paths(n_paths, T_months, r, sigma, q, seed, ko)
    prog = _legacy_snowball_program(T_months, ko, ki, coupon)
    pr = price(prog, paths)
    # legacy fair_value includes the principal; DSL funded note nets the issue price at t=0
    assert abs((pr.fair_value + 1.0) - legacy_res["fair_value"]) < 1e-9, (pr.fair_value + 1.0, legacy_res["fair_value"])
    term = pr.result.termination
    ko_prob = float((term <= T_months).to(torch.float64).mean())
    assert abs(ko_prob - legacy_res["knock_out_prob"]) < 1e-12
    st = paths.spot["u"][:, -1]
    min_ratio = paths.spot["u"].min(dim=1).values
    ki_loss = (~(term <= T_months)) & (min_ratio <= ki) & (st < 1.0)
    assert abs(float(ki_loss.to(torch.float64).mean()) - legacy_res["knock_in_prob"]) < 1e-12


@pytest.mark.parametrize("ki", [None, 0.75])
def test_autocallable_matches_legacy_mc_exactly(ki):
    T_months, r, q, sigma, n_paths, seed = 12, 0.02, 0.0, 0.3, 3000, 42
    ko, coupon = 1.02, 0.10
    legacy_res = legacy.autocallable_price(1.0, T_months, r, sigma, q, coupon, ki, autocall_pct=ko, n_paths=n_paths, seed=seed)
    paths = _legacy_paths(n_paths, T_months, r, sigma, q, seed, ko)
    obs = " ".join(str(i) for i in range(1, T_months + 1))
    if ki is None:
        # never called -> full coupon at maturity
        tail = f"(pay (mul_const {coupon} time) maturity)"
    else:
        # legacy: knocked in (monthly min <= ki) -> min(S_T, 1) with no coupon; else full coupon
        touched = f"(touched (spot u) {ki} down obs)"
        tail = (
            f"(both (when {touched} maturity (pay (lin 1 (min (terminal (spot u)) 1.0) -1.0) maturity)) "
            f"(when (not {touched}) maturity (pay (mul_const {coupon} time) maturity)))"
        )
    body = (
        "(funded 1.0 1.0 (both "
        f"(terminate_if (gt (spot u) {ko}) obs (mul_const {coupon} time)) "
        f"(when (not (gt (spot u) {ko})) maturity {tail})))"
    )
    prog = Program.from_sexpr(f"(contract (meta (underlyings u) (maturity {T_months}) (schedules (obs {obs}))) {body})")
    pr = price(prog, paths)
    assert abs((pr.fair_value + 1.0) - legacy_res["fair_value"]) < 1e-9, (pr.fair_value + 1.0, legacy_res["fair_value"])


def test_vanilla_matches_legacy_bs():
    N, T, r, q, sigma, k = 252, 1.0, 0.03, 0.01, 0.25, 1.05
    paths = gbm_paths(200_000, N, T, sigma=sigma, rate=r, div=q, seed=3)
    c = price(catalog.vanilla_call(N, k), paths)
    p = price(catalog.vanilla_put(N, k), paths)
    assert abs(c.fair_value - legacy.bs_call(1.0, k, T, r, sigma, q)) < 3 * c.mc_se
    assert abs(p.fair_value - legacy.bs_put(1.0, k, T, r, sigma, q)) < 3 * p.mc_se


def test_barrier_matches_legacy_after_bgk_shift():
    N, T, r, q, sigma, k, barrier = 252, 1.0, 0.03, 0.01, 0.25, 1.0, 0.80
    paths = gbm_paths(200_000, N, T, sigma=sigma, rate=r, div=q, seed=5)
    dsl = price(catalog.down_and_out_put(N, k, barrier), paths)
    shifted = barrier * math.exp(-0.5826 * sigma * math.sqrt(T / N))
    ref = legacy.barrier_option(1.0, k, T, r, sigma, q, shifted, "knock_out", "put", "down")
    assert abs(dsl.fair_value - ref) < 3 * dsl.mc_se + 2e-4, (dsl.fair_value, ref, dsl.mc_se)
