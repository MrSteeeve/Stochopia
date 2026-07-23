"""Canonical synchronous MIRAGE v3-spine environment (Level 0).

This is deliberately a thin, deterministic state-transition layer.  Level 0
reuses the existing finite product lattice and settlement engine, fixed to the
partial-observation/dynamic-world semantics.  Prompting, LLM calls, strategies,
forced submissions, oracle search, replay, and scalar reward choices live
outside this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from ..benchmark import BenchmarkCondition, BenchmarkError, LongHorizonEnvironment
from ..pricing import PricingError
from ..products import ProductError, ProductSpec, parse_product_spec
from .types import (
    AskClient,
    ConstraintSignals,
    EnvironmentAction,
    EpisodeTask,
    InvalidAction,
    Observation,
    RequestQuote,
    RewardComponents,
    RewardTerm,
    REWARD_SCHEMA_VERSION,
    Skip,
    StepTransition,
    SubmitDesign,
    SubmitProduct,
)


ENVIRONMENT_VERSION = "v3-spine-level0.1-economic-ledger"
PRICING_VERSION = "benchmark-pricing-v3-funding-stress-equilibrium"


def _jsonable(value: Any) -> Any:
    """Return a stable JSON-compatible view without leaking object identity."""

    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        converted = [_jsonable(item) for item in value]
        return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def _stable_hash(value: Any) -> str:
    raw = json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class _EpisodeState:
    """The sole mutable state entry for the Level-0 environment."""

    backend: LongHorizonEnvironment
    seed: int
    options: dict[str, Any]
    step_index: int = 0
    round_step_index: int = 0
    disclosed_client: dict[str, Any] = field(default_factory=dict)
    quote_payloads: list[dict[str, Any]] = field(default_factory=list)
    last_event: dict[str, Any] | None = None
    terminated: bool = False
    truncated: bool = False


class MirageStructurerEnv:
    """Synchronous, no-LLM, single-agent MIRAGE environment.

    ``reset`` returns ``(Observation, info)``.  ``step`` returns a typed
    :class:`StepTransition`, which can also be unpacked as the usual five-value
    step result.  A submission or skip ends the current market round and
    advances automatically; the same action on the last round terminates the
    episode.
    """

    environment_version = ENVIRONMENT_VERSION
    pricing_version = PRICING_VERSION
    level = "Level0"

    def __init__(
        self,
        task: EpisodeTask,
        *,
        query_cost: float = -0.01,
        quote_cost: float = -0.02,
        invalid_action_cost: float = -1.0,
        skip_cost: float = 0.0,
        max_steps_per_round: int = 12,
        expose_privileged_info: bool = False,
    ) -> None:
        if not isinstance(task, EpisodeTask):
            raise TypeError("task must be an EpisodeTask")
        self.task = task
        costs = {
            "query_cost": query_cost,
            "quote_cost": quote_cost,
            "invalid_action_cost": invalid_action_cost,
            "skip_cost": skip_cost,
        }
        converted_costs: dict[str, float] = {}
        for name, value in costs.items():
            if isinstance(value, bool):
                raise TypeError(f"{name} must be a finite number")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} must be finite")
            converted_costs[name] = converted
        self.query_cost = converted_costs["query_cost"]
        self.quote_cost = converted_costs["quote_cost"]
        self.invalid_action_cost = converted_costs["invalid_action_cost"]
        self.skip_cost = converted_costs["skip_cost"]
        if (
            isinstance(max_steps_per_round, bool)
            or not isinstance(max_steps_per_round, int)
            or max_steps_per_round < 1
        ):
            raise ValueError("max_steps_per_round must be a positive integer")
        if not isinstance(expose_privileged_info, bool):
            raise TypeError("expose_privileged_info must be boolean")
        self.max_steps_per_round = max_steps_per_round
        self.expose_privileged_info = expose_privileged_info
        self._state: _EpisodeState | None = None

    @property
    def configuration(self) -> dict[str, Any]:
        """Knobs that change rewards, termination, or returned information."""
        return {
            "query_cost": self.query_cost,
            "quote_cost": self.quote_cost,
            "invalid_action_cost": self.invalid_action_cost,
            "skip_cost": self.skip_cost,
            "max_steps_per_round": self.max_steps_per_round,
            "expose_privileged_info": self.expose_privileged_info,
        }

    @property
    def is_reset(self) -> bool:
        return self._state is not None

    @property
    def terminated(self) -> bool:
        return bool(self._require_state().terminated)

    @property
    def truncated(self) -> bool:
        return bool(self._require_state().truncated)

    @property
    def done(self) -> bool:
        state = self._require_state()
        return bool(state.terminated or state.truncated)

    @property
    def state_hash(self) -> str:
        return self._state_hash(self._require_state())

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        """Rebuild the complete episode state and return its public projection."""

        effective_seed = self.task.task_seed if seed is None else seed
        if isinstance(effective_seed, bool) or not isinstance(effective_seed, int):
            raise TypeError("seed must be an int or None")
        copied_options = copy.deepcopy(dict(options or {}))

        # The v2 client is mutable, and the backend mutates portfolio/client
        # memory.  Fresh deep copies make reset genuinely deterministic and
        # prevent one rollout from contaminating another.
        snapshots, client, risk_budget, domain, quote_policy = self.task.materialize()
        backend = LongHorizonEnvironment(
            snapshots,
            client,
            risk_budget,
            BenchmarkCondition(full_information=False, dynamic=True),
            quote_policy=quote_policy,
            domain=domain,
            env_agents=None,
        )
        self._state = _EpisodeState(
            backend=backend,
            seed=effective_seed,
            options=copied_options,
        )
        observation = self._observation(self._state)
        info = self._info(self._state)
        return observation, info

    def step(self, action: EnvironmentAction) -> StepTransition:
        """Apply one typed action through the sole state-transition entrypoint."""

        state = self._require_state()
        if state.terminated or state.truncated:
            raise RuntimeError("step() called after the episode ended; call reset()")

        previous_observation = self._observation(state)
        state_hash_before = self._state_hash(state)
        state.step_index += 1
        state.round_step_index += 1

        effective_action: EnvironmentAction
        if isinstance(
            action,
            (AskClient, RequestQuote, SubmitDesign, SubmitProduct, Skip, InvalidAction),
        ):
            # Preserve product values at the call boundary even when a caller
            # later mutates a v2 ProductSpec instance. InvalidAction.raw may be
            # deliberately non-copyable, so only product actions need copying.
            effective_action = (
                copy.deepcopy(action)
                if isinstance(action, (RequestQuote, SubmitProduct))
                else action
            )
        else:
            effective_action = InvalidAction(
                reason=f"unsupported action object: {type(action).__name__}",
                raw=action,
            )

        try:
            rewards, constraints, ends_round = self._apply(state, effective_action)
        except (
            BenchmarkError,
            ProductError,
            PricingError,
            ArithmeticError,
            TypeError,
            ValueError,
        ) as exc:
            message = str(exc)
            exhausted: tuple[str, ...] = ()
            if "query budget exhausted" in message:
                exhausted = ("client_queries",)
            elif "quote budget exhausted" in message:
                exhausted = ("quotes",)
            constraints = ConstraintSignals(
                action_valid=False,
                exhausted_budgets=exhausted,
                error=message,
            )
            rewards = RewardComponents(
                operational_cost=self._cost_term(
                    self.invalid_action_cost,
                    "invalid-action-cost-v1",
                )
            )
            state.last_event = {
                "type": "action_error",
                "action": effective_action.action,
                "error": message,
            }
            ends_round = False

        if ends_round:
            self._finish_round(state)
        elif state.round_step_index >= self.max_steps_per_round:
            state.truncated = True
            event = copy.deepcopy(state.last_event) if state.last_event is not None else {}
            event["time_limit"] = {
                "max_steps_per_round": self.max_steps_per_round,
                "round": state.backend.snapshot.round_num,
            }
            state.last_event = event

        observation = self._observation(state)
        state_hash_after = self._state_hash(state)
        info = self._info(state)
        return StepTransition(
            previous_observation=previous_observation,
            action=effective_action,
            observation=observation,
            tool_result=copy.deepcopy(state.last_event),
            reward_components=rewards,
            constraint_signals=constraints,
            terminated=state.terminated,
            truncated=state.truncated,
            info=info,
            state_hash_before=state_hash_before,
            state_hash_after=state_hash_after,
        )

    def _apply(
        self,
        state: _EpisodeState,
        action: EnvironmentAction,
    ) -> tuple[RewardComponents, ConstraintSignals, bool]:
        backend = state.backend

        if isinstance(action, InvalidAction):
            state.last_event = {
                "type": "invalid_action",
                "reason": action.reason,
                "raw": _jsonable(action.raw),
            }
            return (
                RewardComponents(
                    operational_cost=self._cost_term(
                        self.invalid_action_cost,
                        "invalid-action-cost-v1",
                    )
                ),
                ConstraintSignals(action_valid=False, error=action.reason),
                False,
            )

        if isinstance(action, AskClient):
            if not isinstance(action.topic, str) or not action.topic:
                raise ValueError("AskClient.topic must be a non-empty string")
            result = backend.query_client(action.topic)
            state.disclosed_client[action.topic] = copy.deepcopy(result.get("answer"))
            state.last_event = {"type": "client_answer", "payload": copy.deepcopy(result)}
            return (
                RewardComponents(
                    query_cost=self._cost_term(
                        self.query_cost,
                        "client-query-cost-v1",
                    )
                ),
                ConstraintSignals(),
                False,
            )

        if isinstance(action, RequestQuote):
            product = self._require_product(action.product)
            quote = backend.request_quote(product)
            state.quote_payloads.append(copy.deepcopy(quote))
            state.last_event = {"type": "quote", "payload": copy.deepcopy(quote)}
            hard_failures = self._failed_check_ids(quote.get("checks", ()), severity="HARD")
            return (
                RewardComponents(
                    quote_cost=self._cost_term(
                        self.quote_cost,
                        "desk-quote-cost-v1",
                    )
                ),
                ConstraintSignals(
                    hard_pass=bool(quote.get("hard_pass")),
                    hard_failures=hard_failures,
                ),
                False,
            )

        if isinstance(action, SubmitDesign):
            if not isinstance(action.quote_id, str) or not action.quote_id:
                raise ValueError("SubmitDesign.quote_id must be a non-empty string")
            result = backend.submit_design(action.quote_id, action.explanation)
            state.last_event = {"type": "submission", "payload": copy.deepcopy(result)}
            rewards, constraints = self._submission_vectors(result, quote_cost=0.0)
            return rewards, constraints, True

        if isinstance(action, SubmitProduct):
            product = self._require_product(action.product)
            quote = backend.request_quote(product)
            state.quote_payloads.append(copy.deepcopy(quote))
            result = backend.submit_design(quote["quote_id"], action.explanation)
            state.last_event = {
                "type": "quote_and_submission",
                "quote": copy.deepcopy(quote),
                "submission": copy.deepcopy(result),
            }
            rewards, constraints = self._submission_vectors(
                result, quote_cost=self.quote_cost
            )
            return rewards, constraints, True

        if isinstance(action, Skip):
            # v2 has no skip method.  Marking the round submitted preserves its
            # invariant while this wrapper owns the automatic round advance.
            backend.submitted = True
            state.last_event = {
                "type": "skip",
                "reason": action.reason,
                "round": backend.snapshot.round_num,
            }
            return (
                RewardComponents(
                    operational_cost=self._cost_term(
                        self.skip_cost,
                        "round-skip-cost-v1",
                    )
                ),
                ConstraintSignals(),
                True,
            )

        # Kept defensive even though step() normalises unknown objects above.
        raise TypeError(f"unsupported action: {type(action).__name__}")

    def _submission_vectors(
        self,
        result: Mapping[str, Any],
        *,
        quote_cost: float,
    ) -> tuple[RewardComponents, ConstraintSignals]:
        accepted = bool(result.get("accepted"))
        hard_pass = bool(result.get("hard_executable"))
        contract_pass = bool(result.get("client_contract_pass"))
        hard_failures = self._failed_check_ids(result.get("hard_failures", ()))
        contract_failures = self._failed_check_ids(result.get("contract_failures", ()))
        dealer_value = float(result.get("dealer_margin", 0.0))
        face_value = float(result.get("face_value", 0.0))
        rewards = RewardComponents(
            # Acceptance belongs to ConstraintSignals.  Level 0 has no
            # calibrated client utility model.
            dealer_economics=RewardTerm.measured(
                dealer_value,
                provenance="quote-equilibrium-inception-margin-v1",
                units="CNY",
                normalization="per_face_value",
                normalized_value=(
                    dealer_value / face_value if face_value > 0 else 0.0
                ),
            ),
            quote_cost=self._cost_term(
                quote_cost,
                "desk-quote-cost-v1",
            ),
        )
        constraints = ConstraintSignals(
            hard_pass=hard_pass,
            client_contract_pass=contract_pass,
            accepted=accepted,
            hard_failures=hard_failures,
            contract_failures=contract_failures,
        )
        return rewards, constraints

    @staticmethod
    def _cost_term(value: float, provenance: str) -> RewardTerm:
        return RewardTerm.measured(
            value,
            provenance=provenance,
            units="reward_units",
            normalization="identity",
            normalized_value=value,
        )

    @staticmethod
    def _failed_check_ids(
        checks: Any,
        *,
        severity: str | None = None,
    ) -> tuple[str, ...]:
        failures: list[str] = []
        for check in checks or ():
            if not isinstance(check, Mapping):
                continue
            if severity is not None and check.get("severity") != severity:
                continue
            status = check.get("status")
            passed = check.get("passed")
            if status == "FAIL" or passed is False:
                failures.append(str(check.get("check_id", "UNKNOWN")))
        return tuple(failures)

    @staticmethod
    def _require_product(product: ProductSpec | None) -> ProductSpec:
        if not isinstance(product, ProductSpec):
            raise TypeError("action.product must be a ProductSpec")
        forged: list[str] = []
        if product.reference_spot is not None:
            forged.append("reference_spot")
        if product.barrier_touched:
            forged.append("barrier_touched")
        if product.knock_in_active:
            forged.append("knock_in_active")
        if product.elapsed_months != 0:
            forged.append("elapsed_months")
        if forged:
            raise ProductError(
                "policy actions cannot set environment-maintained product fields: "
                + ", ".join(forged)
            )
        # Typed callers bypass the JSON runner. Re-enter its canonical parser
        # and return a fresh snapshot before the quote boundary.
        return parse_product_spec({
            "product_type": product.product_type,
            "notional": product.notional,
            "maturity_months": product.maturity_months,
            "strike_pct": product.strike_pct,
            "barrier_pct": product.barrier_pct,
            "barrier_type": product.barrier_type,
            "barrier_direction": product.barrier_direction,
            "coupon_rate": product.coupon_rate,
            "participation_rate": product.participation_rate,
            "principal_protected": product.principal_protected,
            "target_client": product.target_client,
            "pitch": product.pitch,
            "hedging_plan": product.hedging_plan,
            "funding_style": product.funding_style,
            "face_value": product.face_value,
            "issue_price_pct": product.issue_price_pct,
            "protected_amount": product.protected_amount,
        })

    def _finish_round(self, state: _EpisodeState) -> None:
        backend = state.backend
        event = copy.deepcopy(state.last_event) if state.last_event is not None else {}
        if backend.round_index >= len(backend.snapshots) - 1:
            state.terminated = True
            event["episode_terminated"] = True
            event["open_at_horizon"] = [
                {
                    "position_id": position.position_id,
                    "status": position.status,
                    "remaining_months": position.remaining_months,
                    "current_fair_value": position.current_fair_value,
                    "issue_cash_outlay": position.issue_cash_outlay,
                }
                for position in backend.portfolio.positions
            ]
            state.last_event = event
            return

        closed_ids = backend.advance_round()
        lifecycle_events = list(backend.portfolio.last_lifecycle_events)
        state.round_step_index = 0
        state.disclosed_client.clear()
        state.quote_payloads.clear()
        event["advanced_to_round"] = backend.snapshot.round_num
        event["closed_position_ids"] = closed_ids
        event["closed_positions"] = [
            {
                "position_id": item.position_id,
                "close_reason": item.close_reason,
                "settlement_amount": item.settlement_amount,
                "client_realized_pnl": item.client_realized_pnl,
                "dealer_liability_realized_pnl": (
                    item.dealer_liability_realized_pnl
                ),
                "dealer_total_pnl": item.dealer_total_pnl,
            }
            for item in lifecycle_events
        ]
        event["lifecycle_events"] = _jsonable(lifecycle_events)
        # Deprecated compatibility field, now correctly restricted to actual
        # maturity rather than every closure reason.
        event["matured_positions"] = [
            item.position_id
            for item in lifecycle_events
            if item.close_reason == "matured"
        ]
        state.last_event = event

    def _observation(self, state: _EpisodeState) -> Observation:
        backend = state.backend
        brief = backend.get_round_brief()
        excluded = {
            "condition",
            "quote_budget",
            "portfolio_summary",
            "client_history",
            # Defensive: the backend is fixed to partial, but this ensures a
            # future v2 change cannot accidentally expose the full profile.
            "client_constraints",
        }
        market = {key: copy.deepcopy(value) for key, value in brief.items() if key not in excluded}
        portfolio = copy.deepcopy(brief.get("portfolio_summary", {}))
        client_history = copy.deepcopy(brief.get("client_history", {}))
        budgets = {
            "client_queries_left": backend.max_client_queries_per_round - backend.query_count,
            "quotes_left": backend.max_quotes_per_round - backend.quote_count,
            "steps_left": max(self.max_steps_per_round - state.round_step_index, 0),
        }
        actions: tuple[str, ...]
        if state.terminated or state.truncated:
            actions = ()
        else:
            base: list[str] = []
            if budgets["client_queries_left"] > 0:
                base.append("ask_client")
            if budgets["quotes_left"] > 0:
                base.append("request_quote")
                base.append("submit_product")
            if state.quote_payloads:
                base.append("submit_design")
            base.append("skip")
            actions = tuple(base)

        return Observation(
            schema=self.task.schema,
            task_version=self.task.version,
            episode_id=backend.snapshot.episode_id,
            round_num=backend.snapshot.round_num,
            step_index=state.step_index,
            state_version=backend.state_version,
            market=market,
            portfolio=portfolio,
            client_history=client_history,
            disclosed_client=copy.deepcopy(state.disclosed_client),
            action_budget=budgets,
            quotes=tuple(copy.deepcopy(state.quote_payloads)),
            last_event=copy.deepcopy(state.last_event),
            available_actions=actions,
            terminated=state.terminated,
            truncated=state.truncated,
        )

    def _privileged_state(self, state: _EpisodeState) -> dict[str, Any]:
        backend = state.backend
        quotes: dict[str, Any] = {}
        for quote_id, (quote, product) in sorted(backend.quotes.items()):
            quotes[quote_id] = {
                "quote": _jsonable(quote),
                "product": _jsonable(product),
            }
        return {
            "seed": state.seed,
            "round_index": backend.round_index,
            "snapshot": _jsonable(backend.snapshot),
            "client": _jsonable(backend.client),
            "risk_budget": _jsonable(backend.desk.hard_checks.risk_budget),
            "domain": _jsonable(backend.domain),
            "quote_policy": _jsonable(backend.quote_policy),
            "portfolio": _jsonable(backend.portfolio),
            "client_memory": _jsonable(backend.client_memory),
            "quotes": quotes,
            "query_count": backend.query_count,
            "quote_count": backend.quote_count,
            "submitted": backend.submitted,
            "terminated": state.terminated,
            "truncated": state.truncated,
        }

    def _info(self, state: _EpisodeState) -> dict[str, Any]:
        info = {
            "state_hash": self._state_hash(state),
            "schema_version": self.task.schema,
            "environment_version": self.environment_version,
            "pricing_version": self.pricing_version,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "task_version": self.task.version,
            "level": self.level,
            # reset(seed=...) is run identity only; economic common-random
            # numbers remain frozen by the pricing protocol.
            "run_id_seed": state.seed,
            "seed": state.seed,
            "seed_role": "run_id_only",
            "options_hash": _stable_hash(state.options),
        }
        if self.expose_privileged_info:
            info["privileged_state"] = self._privileged_state(state)
            info["reset_options"] = copy.deepcopy(state.options)
        return info

    def _state_hash(self, state: _EpisodeState) -> str:
        payload = {
            "task_schema": self.task.schema,
            "task_version": self.task.version,
            "environment_version": self.environment_version,
            "pricing_version": self.pricing_version,
            "reward_schema_version": REWARD_SCHEMA_VERSION,
            "environment_configuration": self.configuration,
            "seed": state.seed,
            "options": state.options,
            "step_index": state.step_index,
            "round_step_index": state.round_step_index,
            "disclosed_client": state.disclosed_client,
            "quote_payloads": state.quote_payloads,
            "last_event": state.last_event,
            "terminated": state.terminated,
            "truncated": state.truncated,
            "privileged_state": self._privileged_state(state),
        }
        return _stable_hash(payload)

    def _require_state(self) -> _EpisodeState:
        if self._state is None:
            raise RuntimeError("reset() must be called before accessing or stepping the environment")
        return self._state


__all__ = [
    "ENVIRONMENT_VERSION",
    "PRICING_VERSION",
    "MirageStructurerEnv",
]
