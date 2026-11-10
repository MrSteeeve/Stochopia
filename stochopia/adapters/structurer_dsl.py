"""Single-underlying, monthly-grid cashflow adapter for StructurerDSL.

Contracts pay at maturity or on automatic termination. Funding is an initial
account movement; asset values always contain only remaining future payments.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace

import torch
from structurerdsl.ast import Node, Param, walk
from structurerdsl.check import check
from structurerdsl.gbm import gbm_paths
from structurerdsl.interp import run
from structurerdsl.path import Paths
from structurerdsl.program import Program

from ..benchmark import MarketSnapshot
from ..products import ProductSpec


DSL_CONTRACT_VERSION = "stochopia.dsl-contract.monthly.v1"
DSL_PRICING_VERSION = "dsl-monthly-gbm-future-cashflows.v2"
ALLOWED_OPS = frozenset({
    "spot", "time", "terminal", "running_max", "running_min", "touched",
    "avg_arith", "avg_geo", "count_above", "count_within", "occupation", "ret",
    "lin", "pl", "clip", "step", "smooth_step", "add", "mul_const", "mul", "max", "min",
    "gt", "lt", "and", "or", "not", "at", "pay", "both", "scale", "give", "when",
    "terminate_if", "until", "activate_if", "premium", "funded",
})


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _contains(value, atom) -> bool:
    if isinstance(value, Node):
        return any(_contains(arg, atom) for arg in value.args)
    if isinstance(value, tuple):
        return any(_contains(arg, atom) for arg in value)
    return value == atom


def _payment_bounds(program: Program, node: Node, params: dict, *, maturity_always_paid=False) -> tuple[float, float]:
    """Conservative bounds even when termination suppresses positive legs.

    Unguarded maturity payments keep their floor when no early trigger exists.
    Conditional or suppressible payments can be absent. Bounding legs separately
    avoids inferring a guarantee from a maturity leg an earlier trigger erases.
    This is limited to the supported cashflow grammar, not a general verifier.
    """
    if node.op in {"pay", "terminate_if"}:
        expr = node.args[0] if node.op == "pay" else node.args[2]
        standalone = replace(program, body=Node("premium", (Node("pay", (expr, "maturity")),)))
        report = check(standalone, params=params, max_years=program.maturity / 12)
        if node.op == "pay" and maturity_always_paid:
            return report.client_lo, report.client_hi
        return min(0.0, report.client_lo), max(0.0, report.client_hi)
    if node.op == "both":
        lo1, hi1 = _payment_bounds(program, node.args[0], params, maturity_always_paid=maturity_always_paid)
        lo2, hi2 = _payment_bounds(program, node.args[1], params, maturity_always_paid=maturity_always_paid)
        return lo1 + lo2, hi1 + hi2
    if node.op in {"scale", "give"}:
        scale = -1.0 if node.op == "give" else node.args[0]
        scale = params[scale.name] if isinstance(scale, Param) else float(scale)
        if scale == 0:
            return 0.0, 0.0
        lo, hi = _payment_bounds(program, node.args[-1], params, maturity_always_paid=maturity_always_paid)
        return min(scale * lo, scale * hi), max(scale * lo, scale * hi)
    if node.op in {"when", "until", "activate_if"}:
        return _payment_bounds(program, node.args[-1], params)
    if node.op == "premium":
        return _payment_bounds(program, node.args[0], params, maturity_always_paid=maturity_always_paid)
    if node.op == "funded":
        lo, hi = _payment_bounds(program, node.args[2], params, maturity_always_paid=maturity_always_paid)
        return 1.0 + lo, 1.0 + hi
    raise ValueError(f"unsupported contract node: {node.op}")


@dataclass(frozen=True)
class DSLContract:
    program_json: str
    params_json: str
    notional: float
    guaranteed_ratio: float
    target_client: str = "current_client"
    pitch: str = ""
    hedging_plan: str = "monthly static delta hedge"
    version: str = DSL_CONTRACT_VERSION

    @property
    def program(self) -> Program:
        return Program.from_json(json.loads(self.program_json))

    def params(self, issue_price: float = 0.0) -> dict:
        return {**json.loads(self.params_json), "issue_price": issue_price}

    @classmethod
    def from_payload(cls, payload: dict) -> "DSLContract":
        if payload.get("version") != DSL_CONTRACT_VERSION:
            raise ValueError("unsupported DSL contract version")
        raw = payload["program"]
        if type(raw["maturity"]) is not int or raw["maturity"] < 1:
            raise ValueError("maturity must be a positive integer month count")
        for dates in raw.get("schedules", {}).values():
            if any(type(i) is not int for i in dates):
                raise ValueError("schedule dates must be integer grid indices")
        program = Program.from_json(raw)
        if len(program.underlyings) != 1:
            raise ValueError("DSL experiment supports one underlying")
        notional = float(payload["notional"])
        if not math.isfinite(notional) or notional <= 0:
            raise ValueError("notional must be positive and finite")
        params = dict(payload.get("params", {}))
        if "issue_price" in params:
            raise ValueError("issue_price is set by the environment")
        params = {key: float(value) for key, value in params.items()}
        _canonical(params)
        root = program.body
        if root.op not in {"premium", "funded"}:
            raise ValueError("program must have a premium or funded root")
        if root.op == "funded" and root.args[:2] != (1.0, Param("issue_price")):
            raise ValueError("funded root must be (funded 1 $issue_price ...)")
        if _contains(root, "rate"):
            raise ValueError("rate-dependent payments are outside the spot-only experiment")
        if _contains(root.args[-1], Param("issue_price")):
            raise ValueError("issue_price is only the initial funding movement")
        bound_params = {**params, "issue_price": 0.0}
        report = check(program, params=bound_params, max_years=program.maturity / 12)
        if not report.valid:
            raise ValueError("invalid DSL program: " + "; ".join(report.errors))
        for node in walk(root):
            if node.op not in ALLOWED_OPS:
                raise ValueError(f"unsupported DSL primitive: {node.op}")
            if node.op == "spot" and node.args != program.underlyings:
                raise ValueError("spot must refer to the declared underlying")
            if node.op == "pay" and node.args[1] != "maturity":
                raise ValueError("scheduled payments currently occur at maturity")
            if node.op == "when" and node.args[1] != "maturity":
                raise ValueError("when currently gates maturity payments")
            if node.op == "terminate_if" and 0 in program.schedule(str(node.args[1])):
                raise ValueError("automatic termination must occur after issuance")
        # A sibling trigger suppresses all remaining maturity payments.
        floor, _ = _payment_bounds(program, root, bound_params,
            maturity_always_paid=not any(node.op == "terminate_if" for node in walk(root)))
        if not math.isfinite(floor) or floor < -1e-12:
            raise ValueError("this experiment requires a nonnegative client payment floor")
        return cls(
            _canonical(program.to_json()), _canonical(params), notional, max(floor, 0.0),
            target_client=payload.get("target_client", "current_client"),
            pitch=payload.get("pitch", ""), hedging_plan=payload.get("hedging_plan", "monthly static delta hedge"),
        )

    def to_payload(self) -> dict:
        return {
            "version": self.version, "program": json.loads(self.program_json),
            "params": json.loads(self.params_json), "notional": self.notional,
            "target_client": self.target_client, "pitch": self.pitch, "hedging_plan": self.hedging_plan,
        }


@dataclass
class DSLProductSpec(ProductSpec):
    contract: DSLContract | None = None

    @classmethod
    def from_contract(cls, contract: DSLContract) -> "DSLProductSpec":
        return cls(
            product_type="custom", notional=contract.notional,
            maturity_months=contract.program.maturity, strike_pct=1.0,
            barrier_pct=None, barrier_type=None, coupon_rate=None, participation_rate=1.0,
            principal_protected=False, target_client=contract.target_client,
            pitch=contract.pitch, hedging_plan=contract.hedging_plan,
            funding_style="funded_note" if contract.program.body.op == "funded" else "premium_paid",
            face_value=contract.notional, protected_amount=contract.guaranteed_ratio * contract.notional,
            contract=contract,
        )


def conditional_paths(
    contract: DSLContract, snapshot: MarketSnapshot, history: tuple[float, ...],
    *, n_paths: int, seed: int, sigma: float | None = None, spot_factor: float = 1.0,
    physical: bool = False,
) -> Paths:
    """Keep observed history (today included) and issue fixing; sample only the remaining grid.

    ``spot_factor`` bumps the generated future columns only; today's observed
    spot in column ``elapsed`` is never bumped.
    """
    program = contract.program
    name = program.underlyings[0]
    elapsed = len(history) - 1
    remaining = program.maturity - elapsed
    sigma = snapshot.pricing_volatility(max(remaining, 1))[0] if sigma is None else sigma
    spot = torch.tensor(history, dtype=torch.float64).expand(n_paths, -1)
    if remaining > 0:
        generated = gbm_paths(
            n_paths, remaining, remaining / 12, spot=snapshot.spot * spot_factor,
            sigma=sigma, rate=snapshot.risk_free_rate + (snapshot.trend_alpha if physical else 0),
            div=snapshot.carry_rate, seed=seed, antithetic=False, name=name, device="cpu",
        ).spot[name]
        spot = torch.cat((spot, generated[:, 1:]), dim=1)
    return Paths(
        spot={name: spot},
        t_years=torch.arange(program.maturity + 1, dtype=torch.float64) / 12,
        rate=snapshot.risk_free_rate, div={name: snapshot.carry_rate},
        fixing={name: torch.full((n_paths,), history[0], dtype=torch.float64)},
    )


def future_value(contract: DSLContract, paths: Paths, elapsed: int) -> tuple[float, float]:
    cf = run(contract.program, paths, contract.params()).cashflows * contract.notional
    discount = torch.exp(-paths.rate * (paths.t_years - elapsed / 12))
    pv = (cf[:, elapsed + 1:] * discount[None, elapsed + 1:]).sum(dim=1)
    if not torch.isfinite(pv).all():
        raise ValueError("DSL produced nonfinite future cashflows")
    se = float(pv.std(unbiased=True) / math.sqrt(paths.n_paths)) if paths.n_paths > 1 else 0.0
    return float(pv.mean()), se


def valuation(
    contract: DSLContract, snapshot: MarketSnapshot, history: tuple[float, ...],
    *, n_paths: int = 4096, seed: int = 42,
) -> dict:
    elapsed = len(history) - 1
    sigma = snapshot.pricing_volatility(max(contract.program.maturity - elapsed, 1))[0]

    def value(factor=1.0, vol=sigma):
        paths = conditional_paths(contract, snapshot, history, n_paths=n_paths, seed=seed, sigma=vol, spot_factor=factor)
        return future_value(contract, paths, elapsed)

    fair, se = value()
    h = 0.01
    up, _ = value(1 + h)
    down, _ = value(1 - h)
    delta_relative = (up - down) / (2 * h)
    gamma_relative = (up - 2 * fair + down) / h ** 2
    vol_h = min(0.01, sigma / 2)
    vega_absolute = (value(vol=sigma + vol_h)[0] - value(vol=sigma - vol_h)[0]) / (2 * vol_h)
    delta_units = delta_relative / snapshot.spot
    client_loss = dealer_loss = hedged_loss = 0.0
    for factor, vol in ((0.8, sigma), (1.2, sigma), (1.0, max(sigma - 0.1, 0.001)), (1.0, sigma + 0.1)):
        stressed = value(factor, vol)[0]
        client_loss = max(client_loss, fair - stressed)
        dealer_loss = max(dealer_loss, stressed - fair)
        hedged_loss = max(hedged_loss, stressed - fair - delta_units * snapshot.spot * (factor - 1))
    return {
        "fair_value": fair, "mc_se": se,
        "delta_dollars": delta_relative, "gamma_relative": gamma_relative,
        "delta_units": delta_units, "gamma_units": gamma_relative / snapshot.spot ** 2,
        "vega_dollars": vega_absolute / 100, "vega_absolute": vega_absolute,
        "client_value_loss": client_loss, "dealer_unhedged_liability_loss": dealer_loss,
        "dealer_hedged_stress_loss": hedged_loss,
    }


def observed_result(contract: DSLContract, history: tuple[float, ...]):
    """Interpret the observed prefix; future padding is never paid or observed."""
    program = contract.program
    padded = (*history, *((history[-1],) * (program.maturity + 1 - len(history))))
    name = program.underlyings[0]
    paths = Paths(
        spot={name: torch.tensor([padded], dtype=torch.float64)},
        t_years=torch.arange(program.maturity + 1, dtype=torch.float64) / 12, rate=0.0,
        fixing={name: torch.tensor([history[0]], dtype=torch.float64)},
    )
    return run(program, paths, contract.params())
