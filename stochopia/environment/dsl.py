"""Opt-in DSL contract-design environment; trades at model fair value."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Literal

import torch

from ..adapters.structurer_dsl import (
    ALLOWED_OPS, DSL_CONTRACT_VERSION, DSL_PRICING_VERSION, DSLContract, DSLProductSpec,
    conditional_paths, observed_result, valuation,
)
from ..benchmark import (
    BenchmarkError, HardConstraintEngine, LongHorizonEnvironment,
    PortfolioState, Position, Quote, TradingDesk, _accrue_static_delta_hedge,
)
from ..pricing import QuoteUnavailableError
from ..products import ProductError
from .core import StochopiaStructurerEnv
from .types import InvalidAction, RequestQuote, SubmitProduct


DSL_ENVIRONMENT_VERSION = "v3-spine-dsl-experimental.2"
DSL_ACTION_SCHEMA = "stochopia.environment.dsl-action-schema.v1"
DSL_AGENT_PROMPT = (
    "Design a contract using the DSL grammar in action_schema. Return exactly one JSON action. "
    "Use ask_client, request_dsl_quote with a contract payload, submit_design with a quote_id, or skip. "
    "The environment values the contract and sets its purchase price; you design only its payments."
)


@dataclass(frozen=True)
class RequestDSLQuote(RequestQuote):
    action: Literal["request_dsl_quote"] = field(default="request_dsl_quote", init=False)


def parse_dsl_action(raw):
    from ..benchmark_runner import extract_action_json
    from .agent_runner import AgentActionError, parse_environment_action

    try:
        payload = extract_action_json(raw) if isinstance(raw, str) else dict(raw)
    except (TypeError, ValueError) as exc:
        raise AgentActionError(f"invalid action JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise AgentActionError("action JSON must be an object")
    if payload.get("action") == "request_dsl_quote":
        if not isinstance(payload.get("contract"), dict):
            raise AgentActionError("request_dsl_quote.contract must be an object")
        try:
            contract = DSLContract.from_payload(payload["contract"])
            return RequestDSLQuote(DSLProductSpec.from_contract(contract))
        # Malformed program trees surface from the StructurerDSL parser as
        # AttributeError as well; all of these are the policy's formatting fault.
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AgentActionError(f"invalid DSL contract: {exc}") from exc
    if payload.get("action") not in {"ask_client", "submit_design", "skip"}:
        raise AgentActionError("unsupported DSL experiment action")
    return parse_environment_action(payload)


@dataclass
class _DSLPosition(Position):
    observed_spots: tuple[float, ...] = ()


class _DSLPortfolio(PortfolioState):
    def __init__(self, *, n_paths, seed, **kwargs):
        super().__init__(**kwargs)
        self.n_paths, self.seed = n_paths, seed

    def _advance_position(self, position, snapshot):
        _accrue_static_delta_hedge(position, snapshot)
        position.observed_spots += (snapshot.spot,)
        position.elapsed_months += 1
        position.remaining_months -= 1
        observed = observed_result(position.product.contract, position.observed_spots)
        if int(observed.termination[0]) <= position.elapsed_months:
            position.status = "autocalled"
        elif position.remaining_months == 0:
            position.status = "matured"
        if position.status == "active":
            values = valuation(
                position.product.contract, snapshot, position.observed_spots,
                n_paths=self.n_paths, seed=self.seed,
            )
        else:
            values = {key: 0.0 for key in (
                "fair_value", "delta_dollars", "vega_dollars", "client_value_loss",
                "dealer_unhedged_liability_loss", "dealer_hedged_stress_loss",
            )}
        for field_name, value_name in (
            ("current_fair_value", "fair_value"), ("delta_dollars", "delta_dollars"),
            ("vega_dollars", "vega_dollars"), ("client_value_loss", "client_value_loss"),
            ("dealer_unhedged_liability_loss", "dealer_unhedged_liability_loss"),
            ("dealer_hedged_stress_loss", "dealer_hedged_stress_loss"),
        ):
            setattr(position, field_name, values[value_name])
        position.last_valuation_round = snapshot.round_num

    def _settlement_amount(self, position, snapshot):
        observed = observed_result(position.product.contract, position.observed_spots)
        return float(observed.cashflows[0, position.elapsed_months]) * position.product.notional


class _DSLConstraints(HardConstraintEngine):
    def _payoff_floor(self, product):
        return product.contract.guaranteed_ratio


class _DSLDesk(TradingDesk):
    def __init__(self, *args, n_paths, seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_paths, self.seed = n_paths, seed

    def quote(self, product, snapshot, client, portfolio, state_version, quote_index):
        contract = product.contract
        values = valuation(contract, snapshot, (snapshot.spot,), n_paths=self.n_paths, seed=self.seed)
        fair = values["fair_value"]
        if fair <= 0:
            # Legal but unpriceable: quote_unavailable, not an invalid action.
            raise QuoteUnavailableError("DSL contract must have positive future cashflow value")
        protected = product.effective_protected_amount
        product.principal_protected = protected + 1e-9 >= fair
        product.issue_price_pct = fair / product.notional if product.effective_funding_style == "funded_note" else None
        p_paths = conditional_paths(contract, snapshot, (snapshot.spot,), n_paths=self.n_paths, seed=self.seed, physical=True)
        from structurerdsl.interp import run

        result = run(contract.program, p_paths, contract.params())
        total = result.cashflows[:, 1:].sum(dim=1) * product.notional
        life = result.termination.clamp(max=contract.program.maturity).to(torch.float64) / 12
        hit_prob = float(((total / fair - 1) / life >= client.min_return_pct).to(torch.float64).mean())
        pricing = {
            **values, "client_price": fair, "cash_outlay": fair,
            "premium": fair if product.effective_funding_style == "premium_paid" else None,
            "face_value": product.notional, "issue_price_pct": product.issue_price_pct,
            "protected_amount": protected, "dealer_fee": 0.0, "dealer_margin": 0.0,
            "loss_frac": max(fair - protected, 0.0) / fair,
            "hurdle_hit_prob": hit_prob,
            "greeks": {
                "delta": values["delta_dollars"] / product.notional,
                "gamma": values["gamma_relative"] / product.notional,
                "vega_pct": values["vega_dollars"] / product.notional,
            },
            "pricing_details": {"expected_loss_frac": float((fair - total).clamp(min=0).mean()) / (
                product.notional if product.effective_funding_style == "funded_note" else fair
            )},
        }
        checks = self.hard_checks.evaluate(
            product, pricing, client, portfolio,
            **{name: values[name] for name in (
                "delta_dollars", "vega_dollars", "client_value_loss",
                "dealer_unhedged_liability_loss", "dealer_hedged_stress_loss",
            )},
        )
        return Quote(
            quote_id=f"DSL-r{snapshot.round_num}-n{quote_index}", state_version=state_version,
            fair_value=fair, client_price=fair, dealer_margin=0.0,
            **{name: values[name] for name in (
                "delta_dollars", "vega_dollars", "client_value_loss",
                "dealer_unhedged_liability_loss", "dealer_hedged_stress_loss",
            )},
            volatility_source=snapshot.pricing_volatility(product.maturity_months)[1],
            checks=tuple(checks), warnings=(), hurdle_hit_prob=hit_prob,
            hedge_model="dsl-monthly-static-delta-before-costs.v1",
            funding_style=product.effective_funding_style, face_value=product.notional,
            issue_price_pct=product.issue_price_pct, cash_outlay=fair,
            premium=pricing["premium"], protected_amount=protected,
            quote_solver_version="dsl-fair-value-trade.v1", pricing=pricing,
        )


class _DSLBackend(LongHorizonEnvironment):
    def __init__(self, *args, n_paths, seed, **kwargs):
        super().__init__(*args, **kwargs)  # EpisodeTask enforces the monthly grid.
        self.portfolio = _DSLPortfolio(
            n_paths=n_paths, seed=seed,
            client_account=self.portfolio.client_account, dealer_account=self.portfolio.dealer_account,
        )
        self.desk = _DSLDesk(
            _DSLConstraints(self.desk.hard_checks.risk_budget), policy=self.quote_policy,
            domain=self.domain, n_paths=n_paths, seed=seed,
        )

    def _make_position(self, **kwargs):
        return _DSLPosition(**kwargs, observed_spots=(self.snapshot.spot,))

    def request_quote(self, product):
        if self.submitted:
            raise BenchmarkError("the round is already submitted")
        if self.quote_count >= self.max_quotes_per_round:
            raise BenchmarkError("quote budget exhausted")
        if product.target_client != self.client.id:
            raise BenchmarkError("TARGET_CLIENT_MATCH: wrong settlement client")
        quoted = copy.deepcopy(product)
        quote = self.desk.quote(quoted, self.snapshot, self.client, self.portfolio, self.state_version, self.quote_count + 1)
        self.quote_count += 1  # only a priced quote consumes budget
        self.quotes[quote.quote_id] = (quote, quoted)
        payload = self.public_quote_payload(quote)
        payload["dsl_valuation"] = {
            "mc_se": quote.pricing["mc_se"], "gamma_units": quote.pricing["gamma_units"],
            "delta_units": quote.pricing["delta_units"], "vega_per_unit_sigma": quote.pricing["vega_absolute"],
            "trade_price_rule": "future_cashflow_fair_value", "pricing_version": DSL_PRICING_VERSION,
        }
        return payload


class StochopiaDSLEnv(StochopiaStructurerEnv):
    environment_version = DSL_ENVIRONMENT_VERSION
    pricing_version = DSL_PRICING_VERSION
    level = "DSLExperimental"

    def __init__(self, task, *, dsl_paths=4096, dsl_seed=42, **kwargs):
        super().__init__(task, **kwargs)
        if type(dsl_paths) is not int or dsl_paths < 2:
            raise ValueError("dsl_paths must be an integer >= 2")
        self.dsl_paths, self.dsl_seed = dsl_paths, dsl_seed

    @property
    def configuration(self):
        return {**super().configuration, "dsl_paths": self.dsl_paths, "dsl_seed": self.dsl_seed}

    def _make_backend(self, *args, **kwargs):
        return _DSLBackend(*args, n_paths=self.dsl_paths, seed=self.dsl_seed, **kwargs)

    def _require_product(self, product):
        if not isinstance(product, DSLProductSpec):
            raise ProductError("DSL environment expects a DSL contract")
        try:
            contract = DSLContract.from_payload(product.contract.to_payload())
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ProductError(f"invalid DSL contract: {exc}") from exc
        return DSLProductSpec.from_contract(contract)

    def step(self, action):
        if isinstance(action, (RequestQuote, SubmitProduct)) and not isinstance(action, RequestDSLQuote):
            action = InvalidAction(reason="use request_dsl_quote in the DSL experiment")
        return super().step(action)

    @property
    def action_schema(self):
        return {
            "schema_version": DSL_ACTION_SCHEMA, "contract_version": DSL_CONTRACT_VERSION,
            "client_topic_enum": sorted(self._require_state().backend.CLIENT_TOPICS),
            "actions": ["ask_client", "request_dsl_quote", "submit_design", "skip"],
            "target_client": "current_client", "underlyings": "one declared spot series",
            "notional_rule": "positive continuous CNY amount; portfolio and client limits apply",
            "time_grid": "integer monthly indices, t_years=i/12; fixed issue spot",
            "allowed_primitives": sorted(ALLOWED_OPS),
            "payment_rule": "maturity payments plus automatic termination payments; nonnegative aggregate floor",
            "funded_root": "(funded 1 $issue_price body); issue_price is environment-bound",
            "trade_price_rule": "future_cashflow_fair_value",
        }

    def _observation(self, state):
        observation = super()._observation(state)
        actions = tuple(
            "request_dsl_quote" if name == "request_quote" else name
            for name in observation.available_actions if name != "submit_product"
        )
        return replace(observation, available_actions=actions)
