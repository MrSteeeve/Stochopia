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
from dataclasses import dataclass, field, fields, is_dataclass, replace
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
    V3_PUBLIC_CLIENT_ALIAS,
)


ENVIRONMENT_VERSION = "v3-spine-level0.3-terminal-snapshot"
PRICING_VERSION = "benchmark-pricing-v3.1-funding-stress-equilibrium"


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
    quote_aliases: dict[str, str] = field(default_factory=dict)
    last_event: dict[str, Any] | None = None
    reward_raw_totals: dict[str, float] = field(default_factory=dict)
    reward_normalized_totals: dict[str, float] = field(default_factory=dict)
    reward_available_counts: dict[str, int] = field(default_factory=dict)
    constraint_summary: dict[str, Any] = field(
        default_factory=lambda: {
            "steps": 0,
            "valid_actions": 0,
            "invalid_actions": 0,
            "accepted_submissions": 0,
            "hard_failure_counts": {},
            "contract_failure_counts": {},
        }
    )
    terminated: bool = False
    truncated: bool = False


class MirageStructurerEnv:
    """Synchronous, no-LLM, single-agent MIRAGE environment.

    ``reset`` returns ``(Observation, info)``.  ``step`` returns a typed
    :class:`StepTransition`, which can also be unpacked as the usual five-value
    step result.  A submission or skip ends the current market round and
    advances automatically.  The final market snapshot is terminal valuation
    truth rather than a decision round, so an issued contract always carries
    at least one market interval before it can be marked at the horizon.
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

    @property
    def action_schema(self) -> dict[str, Any]:
        """Current machine-readable policy contract."""

        return copy.deepcopy(self._require_state().backend.policy_action_schema())

    @property
    def reset_options(self) -> dict[str, Any]:
        """Canonical reset options committed to a replayable trajectory."""

        return copy.deepcopy(self._require_state().options)

    @property
    def run_id_seed(self) -> int:
        """Run-identity seed required to reconstruct the initial state hash."""

        return self._require_state().seed

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
        if "curriculum" in copied_options:
            raise ValueError(
                "reset(options['curriculum']) is not implemented and must not "
                "be used as a no-op label; construct an EpisodeTask from a "
                "versioned TaskGenerator before reset"
            )

        # The v2 client is mutable, and the backend mutates portfolio/client
        # memory.  Fresh deep copies make reset genuinely deterministic and
        # prevent one rollout from contaminating another.
        snapshots, client, risk_budget, domain, quote_policy = self.task.materialize()
        # The hidden client identifier is routing metadata, not a policy input.
        # v3 exposes a fixed alias so published trajectories cannot turn a
        # stable client id into a task-specific lookup key.
        client = replace(client, id=V3_PUBLIC_CLIENT_ALIAS)
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
        economic_before = state.backend.economic_state()
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

        economic_after_action = state.backend.economic_state()
        time_advanced_years = 0.0
        if ends_round:
            time_advanced_years = self._finish_round(state)
        elif state.round_step_index >= self.max_steps_per_round:
            state.truncated = True
            event = copy.deepcopy(state.last_event) if state.last_event is not None else {}
            event["time_limit"] = {
                "max_steps_per_round": self.max_steps_per_round,
                "round": state.backend.snapshot.round_num,
            }
            state.last_event = event

        if state.last_event is not None:
            state.last_event = self._public_event_payload(state.last_event)

        rewards = self._economic_reward_delta(
            rewards,
            economic_before,
            state.backend.economic_state(),
            occupancy_state=economic_after_action,
            time_advanced_years=time_advanced_years,
        )
        self._accumulate_episode_summary(state, rewards, constraints)
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
            quote = self._register_public_quote(
                state,
                backend.request_quote(product),
            )
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
            internal_quote_id = state.quote_aliases.get(action.quote_id)
            if internal_quote_id is None:
                raise BenchmarkError("unknown public quote_id")
            result = self._public_submission_payload(
                state,
                backend.submit_design(
                    internal_quote_id,
                    action.explanation,
                ),
            )
            state.last_event = {"type": "submission", "payload": copy.deepcopy(result)}
            rewards, constraints = self._submission_vectors(result, quote_cost=0.0)
            return rewards, constraints, True

        if isinstance(action, SubmitProduct):
            product = self._require_product(action.product)
            internal_quote = backend.request_quote(product)
            quote = self._register_public_quote(state, internal_quote)
            state.quote_payloads.append(copy.deepcopy(quote))
            result = self._public_submission_payload(
                state,
                backend.submit_design(
                    internal_quote["quote_id"],
                    action.explanation,
                ),
            )
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
        quoted_dealer_value = float(result.get("dealer_margin", 0.0))
        dealer_value = quoted_dealer_value if accepted else 0.0
        face_value = float(result.get("face_value", 0.0))
        rewards = RewardComponents(
            # Acceptance belongs to ConstraintSignals.  Level 0 has no
            # calibrated client utility model. A rejected quote is only a
            # counterfactual and cannot create inception dealer economics.
            dealer_economics=RewardTerm.measured(
                dealer_value,
                provenance="accepted-inception-margin-v2",
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
    def _economic_reward_delta(
        rewards: RewardComponents,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        occupancy_state: Mapping[str, Any],
        time_advanced_years: float,
    ) -> RewardComponents:
        """Attach flow-adjusted wealth and time-weighted occupancy rewards."""

        before_client = before.get("client", {})
        after_client = after.get("client", {})
        occupancy_client = occupancy_state.get("client", {})
        before_dealer = before.get("dealer", {})
        after_dealer = after.get("dealer", {})
        occupancy_risk = occupancy_state.get("active_risk", {})
        initial_cash = float(after_client.get("initial_cash", 0.0))
        risk_limit = float(after_dealer.get("risk_capital_limit", 0.0))

        external_inflow_delta = (
            float(after_client.get("external_inflows", 0.0))
            - float(before_client.get("external_inflows", 0.0))
        )
        external_outflow_delta = (
            float(after_client.get("external_outflows", 0.0))
            - float(before_client.get("external_outflows", 0.0))
        )
        client_wealth_delta = (
            float(after_client.get("net_liquidation_wealth", 0.0))
            - float(before_client.get("net_liquidation_wealth", 0.0))
            - external_inflow_delta
            + external_outflow_delta
        )
        capital_time = (
            float(occupancy_client.get("locked_cash", 0.0))
            * time_advanced_years
        )
        stress_time = (
            float(occupancy_risk.get("dealer_hedged_stress_loss", 0.0))
            * time_advanced_years
        )
        lifecycle_count_delta = int(after.get("lifecycle_event_count", 0)) - int(
            before.get("lifecycle_event_count", 0)
        )
        dealer_lifecycle_delta = (
            float(after_dealer.get("realised_hedged_pnl", 0.0))
            - float(before_dealer.get("realised_hedged_pnl", 0.0))
        )

        lifecycle_term = rewards.terminal_lifecycle_pnl
        if lifecycle_count_delta > 0:
            lifecycle_term = RewardTerm.measured(
                dealer_lifecycle_delta,
                provenance="monthly-static-delta-before-transaction-costs-v1",
                units="CNY",
                normalization="per_dealer_risk_capital",
                normalized_value=(
                    dealer_lifecycle_delta / risk_limit
                    if risk_limit > 0
                    else 0.0
                ),
            )

        return replace(
            rewards,
            client_utility=RewardTerm.measured(
                client_wealth_delta,
                provenance="client-flow-adjusted-net-liquidation-wealth-v2",
                units="CNY",
                normalization="per_initial_client_cash",
                normalized_value=(
                    client_wealth_delta / initial_cash
                    if initial_cash > 0
                    else 0.0
                ),
            ),
            capital_efficiency=RewardTerm.measured(
                -capital_time,
                provenance="negative-client-locked-capital-time-v2",
                units="CNY-year",
                normalization="per_initial_client_cash-year",
                normalized_value=(
                    -capital_time / initial_cash if initial_cash > 0 else 0.0
                ),
            ),
            risk_change=RewardTerm.measured(
                -stress_time,
                provenance="negative-dealer-hedged-stress-time-v2",
                units="CNY-year",
                normalization="per_dealer_risk-capital-year",
                normalized_value=(
                    -stress_time / risk_limit if risk_limit > 0 else 0.0
                ),
            ),
            terminal_lifecycle_pnl=lifecycle_term,
        )

    @staticmethod
    def _accumulate_episode_summary(
        state: _EpisodeState,
        rewards: RewardComponents,
        constraints: ConstraintSignals,
    ) -> None:
        for name in RewardComponents.component_names():
            term = getattr(rewards, name)
            if not term.available:
                continue
            state.reward_available_counts[name] = (
                state.reward_available_counts.get(name, 0) + 1
            )
            state.reward_raw_totals[name] = (
                state.reward_raw_totals.get(name, 0.0) + float(term.value or 0.0)
            )
            if term.normalized_value is not None:
                state.reward_normalized_totals[name] = (
                    state.reward_normalized_totals.get(name, 0.0)
                    + float(term.normalized_value)
                )

        summary = state.constraint_summary
        summary["steps"] += 1
        if constraints.action_valid:
            summary["valid_actions"] += 1
        else:
            summary["invalid_actions"] += 1
        if constraints.accepted is True:
            summary["accepted_submissions"] += 1
        for field_name, destination in (
            ("hard_failures", "hard_failure_counts"),
            ("contract_failures", "contract_failure_counts"),
        ):
            counts = summary[destination]
            for check_id in getattr(constraints, field_name):
                counts[check_id] = counts.get(check_id, 0) + 1

    @staticmethod
    def _episode_summary(state: _EpisodeState) -> dict[str, Any]:
        return {
            "reward": {
                "raw_totals": dict(state.reward_raw_totals),
                "normalized_totals": dict(state.reward_normalized_totals),
                "available_step_counts": dict(state.reward_available_counts),
                "total_steps": state.step_index,
            },
            "constraints": copy.deepcopy(state.constraint_summary),
        }

    @staticmethod
    def _cost_term(value: float, provenance: str) -> RewardTerm:
        return RewardTerm.measured(
            value,
            provenance=provenance,
            units="reward_units",
            normalization="identity",
            normalized_value=value,
        )

    def _public_state_version(self, state: _EpisodeState) -> str:
        """Policy-visible state label with no hidden episode/client preimage."""

        return _stable_hash(
            {
                "public_task_id": self.task.public_task_id,
                "round_num": state.backend.snapshot.round_num,
                "portfolio_revision": state.backend.portfolio.revision,
            }
        )[:16]

    def _register_public_quote(
        self,
        state: _EpisodeState,
        quote: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Replace the v2 hidden-state quote key with a public round alias."""

        internal_quote_id = quote.get("quote_id")
        if not isinstance(internal_quote_id, str) or not internal_quote_id:
            raise BenchmarkError("desk quote is missing its internal quote_id")
        alias = next(
            (
                public_id
                for public_id, internal_id in state.quote_aliases.items()
                if internal_id == internal_quote_id
            ),
            None,
        )
        if alias is None:
            alias = (
                f"quote-r{state.backend.snapshot.round_num}-"
                f"n{len(state.quote_aliases) + 1}"
            )
            state.quote_aliases[alias] = internal_quote_id
        public_quote = copy.deepcopy(dict(quote))
        public_quote["quote_id"] = alias
        public_quote["valid_for_state"] = self._public_state_version(state)
        return public_quote

    @staticmethod
    def _public_submission_payload(
        state: _EpisodeState,
        result: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a settlement payload keyed by the public quote alias."""

        public_result = copy.deepcopy(dict(result))
        internal_quote_id = public_result.get("quote_id")
        if isinstance(internal_quote_id, str):
            alias = next(
                (
                    public_id
                    for public_id, stored_id in state.quote_aliases.items()
                    if stored_id == internal_quote_id
                ),
                None,
            )
            if alias is None:
                raise BenchmarkError(
                    "submission refers to an unregistered internal quote"
                )
            public_result["quote_id"] = alias
        return public_result

    @classmethod
    def _public_event_payload(cls, value: Any) -> Any:
        """Recursively remove evaluator-only identities and account truth."""

        private_keys = {
            "episode_id",
            "task_hash",
            "position_id",
            "event_id",
            "cashflow_id",
            "closed_position_ids",
            "matured_positions",
            "horizon_liquidated_position_ids",
            "client_account",
            "dealer_account",
        }
        if isinstance(value, Mapping):
            return {
                str(key): cls._public_event_payload(item)
                for key, item in value.items()
                if key not in private_keys
            }
        if isinstance(value, (list, tuple)):
            return [cls._public_event_payload(item) for item in value]
        return copy.deepcopy(value)

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

    def _finish_round(self, state: _EpisodeState) -> float:
        """Advance one decision interval and terminate on the final snapshot."""

        backend = state.backend
        event = copy.deepcopy(state.last_event) if state.last_event is not None else {}
        if backend.round_index >= len(backend.snapshots) - 1:
            # Defensive only: EpisodeTask exposes the last snapshot as terminal
            # truth, so a policy should never receive an action at this point.
            closed_ids = backend.liquidate_horizon()
            lifecycle_events = list(backend.portfolio.last_lifecycle_events)
            state.terminated = True
            event["episode_terminated"] = True
            event["horizon_liquidated_position_ids"] = closed_ids
            event["horizon_liquidations"] = [
                {
                    "position_id": item.position_id,
                    "settlement_amount": item.settlement_amount,
                    "client_realized_pnl": item.client_realized_pnl,
                    "dealer_liability_realized_pnl": (
                        item.dealer_liability_realized_pnl
                    ),
                    "dealer_hedge_pnl": item.dealer_hedge_pnl,
                    "dealer_hedged_realized_pnl": (
                        item.dealer_hedged_realized_pnl
                    ),
                    "pnl_provenance": item.pnl_provenance,
                }
                for item in lifecycle_events
            ]
            event["lifecycle_events"] = _jsonable(lifecycle_events)
            event["open_at_horizon"] = []
            state.last_event = event
            return 0.0

        previous_date = backend.snapshot.as_of
        closed_ids = backend.advance_round()
        next_date = backend.snapshot.as_of
        time_advanced_years = (next_date - previous_date).days / 365.25
        lifecycle_events = list(backend.portfolio.last_lifecycle_events)
        state.round_step_index = 0
        state.disclosed_client.clear()
        state.quote_payloads.clear()
        state.quote_aliases.clear()
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
                "dealer_hedge_pnl": item.dealer_hedge_pnl,
                "dealer_hedged_realized_pnl": item.dealer_hedged_realized_pnl,
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

        if backend.round_index >= len(backend.snapshots) - 1:
            horizon_ids = backend.liquidate_horizon()
            horizon_events = list(backend.portfolio.last_lifecycle_events)
            lifecycle_events.extend(horizon_events)
            state.terminated = True
            event["episode_terminated"] = True
            event["terminal_snapshot"] = _jsonable(backend.snapshot)
            event["horizon_liquidated_position_ids"] = horizon_ids
            event["horizon_liquidations"] = [
                {
                    "position_id": item.position_id,
                    "settlement_amount": item.settlement_amount,
                    "client_realized_pnl": item.client_realized_pnl,
                    "dealer_liability_realized_pnl": (
                        item.dealer_liability_realized_pnl
                    ),
                    "dealer_hedge_pnl": item.dealer_hedge_pnl,
                    "dealer_hedged_realized_pnl": (
                        item.dealer_hedged_realized_pnl
                    ),
                    "pnl_provenance": item.pnl_provenance,
                }
                for item in horizon_events
            ]
            event["lifecycle_events"] = _jsonable(lifecycle_events)
            event["open_at_horizon"] = []
        state.last_event = event
        return time_advanced_years

    def _observation(self, state: _EpisodeState) -> Observation:
        backend = state.backend
        brief = backend.get_round_brief()
        excluded = {
            "episode_id",
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
        client_history = {
            key: copy.deepcopy(value)
            for key, value in brief.get("client_history", {}).items()
            if key != "trust"
        }
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
            public_task_id=self.task.public_task_id,
            round_num=backend.snapshot.round_num,
            step_index=state.step_index,
            state_version=self._public_state_version(state),
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
        client_truth = _jsonable(backend.client)
        client_truth["id"] = self.task.manifest["client"]["id"]
        return {
            "seed": state.seed,
            "round_index": backend.round_index,
            "snapshot": _jsonable(backend.snapshot),
            "client": client_truth,
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
            "episode_summary": self._episode_summary(state),
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
            "quote_aliases": state.quote_aliases,
            "last_event": state.last_event,
            "reward_raw_totals": state.reward_raw_totals,
            "reward_normalized_totals": state.reward_normalized_totals,
            "reward_available_counts": state.reward_available_counts,
            "constraint_summary": state.constraint_summary,
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
