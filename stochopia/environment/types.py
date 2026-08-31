"""Typed Level-0 contracts for the Stochopia v3 environment spine.

The classes in this module intentionally contain no runner, prompting, oracle,
or model logic.  They are the small, serialisable boundary shared by training
loops and the synchronous environment in :mod:`stochopia.environment.core`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from typing import Any, Literal, Mapping, TypeAlias

from ..benchmark import (
    MarketSnapshot,
    ProductDomainSpec,
    RiskBudget,
    resolve_client_profile,
)
from ..pricing import QuotePolicy
from ..products import ClientProfile, ProductSpec


TASK_SCHEMA = "stochopia.environment.task.v1"
TASK_VERSION = "v3-spine-level0.3-terminal-evaluation"
REWARD_SCHEMA_VERSION = "stochopia.reward-vector.v1"
SCALARIZATION_VERSION = "stochopia.scalarization.explicit.v1"
V3_PUBLIC_NOTIONAL_BASE = 10_000_000.0
V3_PUBLIC_CLIENT_ALIAS = "current_client"


def _manifest_default(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported task manifest value: {type(value).__name__}")


@dataclass(frozen=True)
class EpisodeTask:
    """All deterministic inputs needed to construct one training episode.

    ``ClientProfile`` predates the v3 spine and is mutable.  The environment
    therefore deep-copies it on every reset; callers can safely reuse one
    frozen ``EpisodeTask`` for multiple episodes.
    """

    snapshots: tuple[MarketSnapshot, ...]
    client: ClientProfile
    risk_budget: RiskBudget
    domain: ProductDomainSpec = field(default_factory=ProductDomainSpec)
    quote_policy: QuotePolicy = field(default_factory=QuotePolicy)
    task_seed: int = 0
    schema: str = TASK_SCHEMA
    version: str = TASK_VERSION
    _manifest_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # A tuple prevents callers from changing episode length/order through a
        # list after construction.  Nested v2 values are copied by reset().
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        if len(self.snapshots) < 2:
            raise ValueError(
                "EpisodeTask requires at least one decision snapshot and one "
                "terminal valuation snapshot"
            )
        if self.domain.public_notional_base is None:
            object.__setattr__(
                self,
                "domain",
                replace(
                    self.domain,
                    public_notional_base=V3_PUBLIC_NOTIONAL_BASE,
                ),
            )
        if isinstance(self.task_seed, bool) or not isinstance(self.task_seed, int):
            raise TypeError("EpisodeTask.task_seed must be an int")
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("EpisodeTask.schema must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("EpisodeTask.version must be a non-empty string")
        episode_ids = {snapshot.episode_id for snapshot in self.snapshots}
        if len(episode_ids) != 1:
            raise ValueError("EpisodeTask snapshots must belong to one episode")
        rounds = [snapshot.round_num for snapshot in self.snapshots]
        if rounds != list(range(1, len(rounds) + 1)):
            raise ValueError("EpisodeTask snapshot rounds must be contiguous from one")
        dates = [snapshot.as_of for snapshot in self.snapshots]
        if any(right <= left for left, right in zip(dates, dates[1:])):
            raise ValueError(
                "EpisodeTask snapshot dates must be strictly increasing"
            )
        for snapshot in self.snapshots:
            required = (
                snapshot.spot,
                snapshot.risk_free_rate,
                snapshot.carry_rate,
                snapshot.trend_alpha,
            )
            optional = (
                snapshot.return_20d,
                snapshot.realized_vol_20d,
                snapshot.realized_vol_60d,
                snapshot.drawdown_6m,
                snapshot.atm_iv_1m,
                snapshot.atm_iv_3m,
                snapshot.atm_iv_6m,
            )
            if not all(math.isfinite(value) for value in required):
                raise ValueError("EpisodeTask market required numerics must be finite")
            if snapshot.spot <= 0:
                raise ValueError("EpisodeTask market spot must be positive")
            if not all(
                value is None or math.isfinite(value) for value in optional
            ):
                raise ValueError("EpisodeTask market optional numerics must be finite")
            snapshot.pricing_volatility()
            resolve_client_profile(self.client, snapshot.round_num)

        payload = {
            "snapshots": [asdict(snapshot) for snapshot in self.snapshots],
            "client": asdict(self.client),
            "risk_budget": asdict(self.risk_budget),
            "domain": asdict(self.domain),
            "quote_policy": asdict(self.quote_policy),
            "task_seed": self.task_seed,
            "schema": self.schema,
            "version": self.version,
        }
        object.__setattr__(
            self,
            "_manifest_json",
            json.dumps(
                payload,
                default=_manifest_default,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )

    @property
    def schema_version(self) -> str:
        """Explicit alias used by trajectory metadata."""

        return self.schema

    @property
    def manifest(self) -> dict[str, Any]:
        """Canonical immutable-at-construction task preimage."""

        return json.loads(self._manifest_json)

    @property
    def task_hash(self) -> str:
        return hashlib.sha256(self._manifest_json.encode("utf-8")).hexdigest()

    @property
    def public_task_id(self) -> str:
        """Non-privileged task-family identifier safe to expose to a policy."""

        payload = {
            "schema": self.schema,
            "version": self.version,
            "underlyings": sorted({item.underlying for item in self.snapshots}),
            "decision_intervals": len(self.snapshots) - 1,
            "domain_version": self.domain.version,
            "public_notional_base": self.domain.public_notional_base,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"public-{hashlib.sha256(encoded).hexdigest()[:16]}"

    def materialize(self) -> tuple[
        tuple[MarketSnapshot, ...],
        ClientProfile,
        RiskBudget,
        ProductDomainSpec,
        QuotePolicy,
    ]:
        """Reconstruct fresh runtime inputs from the captured manifest."""

        payload = self.manifest
        snapshots = tuple(
            MarketSnapshot(
                **{
                    **row,
                    "as_of": date.fromisoformat(row["as_of"]),
                }
            )
            for row in payload["snapshots"]
        )
        client = ClientProfile(**payload["client"])
        risk_budget = RiskBudget(**payload["risk_budget"])
        domain_payload = payload["domain"]
        for name in (
            "product_types",
            "notional_fractions",
            "maturities",
            "strikes",
            "barriers",
            "coupons",
            "participations",
            "principal_protected",
        ):
            domain_payload[name] = tuple(domain_payload[name])
        domain = ProductDomainSpec(**domain_payload)
        quote_policy = QuotePolicy(**payload["quote_policy"])
        return snapshots, client, risk_budget, domain, quote_policy

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "EpisodeTask":
        """Reconstruct and revalidate a task from a stored canonical manifest."""

        payload = json.loads(
            json.dumps(
                dict(manifest),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        try:
            snapshots = tuple(
                MarketSnapshot(
                    **{
                        **row,
                        "as_of": date.fromisoformat(row["as_of"]),
                    }
                )
                for row in payload["snapshots"]
            )
            client = ClientProfile(**payload["client"])
            risk_budget = RiskBudget(**payload["risk_budget"])
            domain_payload = payload["domain"]
            for name in (
                "product_types",
                "notional_fractions",
                "maturities",
                "strikes",
                "barriers",
                "coupons",
                "participations",
                "principal_protected",
            ):
                domain_payload[name] = tuple(domain_payload[name])
            domain = ProductDomainSpec(**domain_payload)
            quote_policy = QuotePolicy(**payload["quote_policy"])
            return cls(
                snapshots=snapshots,
                client=client,
                risk_budget=risk_budget,
                domain=domain,
                quote_policy=quote_policy,
                task_seed=payload["task_seed"],
                schema=payload["schema"],
                version=payload["version"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid EpisodeTask manifest: {exc}") from exc


@dataclass(frozen=True)
class Observation:
    """Leakage-safe state visible to a structurer policy.

    There is deliberately no ``client`` or ``privileged_state`` field.  Hidden
    client values appear only after an explicit :class:`AskClient` action, in
    ``disclosed_client``. Complete truth is returned in ``info`` only when the
    harness explicitly constructs the environment with evaluator-only
    ``expose_privileged_info=True``; the default policy API omits it.
    """

    schema: str
    task_version: str
    public_task_id: str
    round_num: int
    step_index: int
    state_version: str
    market: Mapping[str, Any]
    portfolio: Mapping[str, Any]
    client_history: Mapping[str, Any]
    disclosed_client: Mapping[str, Any]
    action_budget: Mapping[str, int]
    quotes: tuple[Mapping[str, Any], ...] = ()
    last_event: Mapping[str, Any] | None = None
    available_actions: tuple[str, ...] = ()
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class AskClient:
    """Reveal one permitted client topic, spending one query action."""

    action: Literal["ask_client"] = field(default="ask_client", init=False)
    topic: str

    @property
    def type(self) -> str:
        return self.action

    @property
    def kind(self) -> str:
        return self.action


@dataclass(frozen=True)
class RequestQuote:
    """Request one state-bound deterministic quote."""

    action: Literal["request_quote"] = field(default="request_quote", init=False)
    product: ProductSpec

    @property
    def type(self) -> str:
        return self.action

    @property
    def kind(self) -> str:
        return self.action


@dataclass(frozen=True)
class SubmitDesign:
    """Submit a quote previously issued in the current state."""

    action: Literal["submit_design"] = field(default="submit_design", init=False)
    quote_id: str
    explanation: str = ""

    @property
    def type(self) -> str:
        return self.action

    @property
    def kind(self) -> str:
        return self.action


@dataclass(frozen=True)
class SubmitProduct:
    """Atomic request-quote plus submit convenience action."""

    action: Literal["submit_product"] = field(default="submit_product", init=False)
    product: ProductSpec
    explanation: str = ""

    @property
    def type(self) -> str:
        return self.action

    @property
    def kind(self) -> str:
        return self.action


@dataclass(frozen=True)
class Skip:
    """End the current round without a design."""

    action: Literal["skip"] = field(default="skip", init=False)
    reason: str = ""

    @property
    def type(self) -> str:
        return self.action

    @property
    def kind(self) -> str:
        return self.action


@dataclass(frozen=True)
class InvalidAction:
    """A parser/adapter can preserve malformed input without raising in core."""

    action: Literal["invalid"] = field(default="invalid", init=False)
    reason: str = "invalid action"
    raw: Any = None

    @property
    def type(self) -> str:
        return self.action

    @property
    def kind(self) -> str:
        return self.action


EnvironmentAction: TypeAlias = (
    AskClient | RequestQuote | SubmitDesign | SubmitProduct | Skip | InvalidAction
)
# Short alias convenient for policy/trainer annotations.
Action: TypeAlias = EnvironmentAction


@dataclass(frozen=True)
class RewardTerm:
    """One measured or explicitly unavailable reward component."""

    value: float | None
    available: bool
    provenance: str
    units: str
    normalization: str | None = None
    normalized_value: float | None = None

    def __post_init__(self) -> None:
        if self.available != (self.value is not None):
            raise ValueError("RewardTerm.available must agree with value presence")
        for name, value in (
            ("value", self.value),
            ("normalized_value", self.normalized_value),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"RewardTerm.{name} must be finite or None")
        if not self.provenance:
            raise ValueError("RewardTerm.provenance must be non-empty")
        if not self.units:
            raise ValueError("RewardTerm.units must be non-empty")

    @classmethod
    def unavailable(
        cls,
        provenance: str = "not-implemented-level0",
        *,
        units: str = "unavailable",
    ) -> "RewardTerm":
        return cls(None, False, provenance, units, None, None)

    @classmethod
    def measured(
        cls,
        value: float,
        *,
        provenance: str,
        units: str,
        normalization: str | None = None,
        normalized_value: float | None = None,
    ) -> "RewardTerm":
        return cls(
            float(value),
            True,
            provenance,
            units,
            normalization,
            (
                float(normalized_value)
                if normalized_value is not None
                else None
            ),
        )


def _unavailable_reward() -> RewardTerm:
    return RewardTerm.unavailable()


@dataclass(frozen=True)
class RewardComponents:
    """Versioned raw/normalised reward vector without implicit scalarisation."""

    client_utility: RewardTerm = field(default_factory=_unavailable_reward)
    dealer_economics: RewardTerm = field(default_factory=_unavailable_reward)
    capital_efficiency: RewardTerm = field(default_factory=_unavailable_reward)
    risk_change: RewardTerm = field(default_factory=_unavailable_reward)
    relationship_delta: RewardTerm = field(default_factory=_unavailable_reward)
    query_cost: RewardTerm = field(default_factory=_unavailable_reward)
    quote_cost: RewardTerm = field(default_factory=_unavailable_reward)
    communication_faithfulness: RewardTerm = field(default_factory=_unavailable_reward)
    operational_cost: RewardTerm = field(default_factory=_unavailable_reward)
    terminal_lifecycle_pnl: RewardTerm = field(
        default_factory=lambda: RewardTerm.unavailable(
            "no-lifecycle-event-this-step",
            units="CNY",
        )
    )
    schema_version: str = REWARD_SCHEMA_VERSION

    @property
    def dealer_margin(self) -> float | None:
        return self.dealer_economics.value

    @property
    def raw_reward_components(self) -> dict[str, float | None]:
        return {
            name: getattr(self, name).value
            for name in self.component_names()
        }

    @property
    def normalised_reward_components(self) -> dict[str, float | None]:
        return {
            name: getattr(self, name).normalized_value
            for name in self.component_names()
        }

    @property
    def available_components(self) -> tuple[str, ...]:
        return tuple(
            name for name in self.component_names() if getattr(self, name).available
        )

    @staticmethod
    def component_names() -> tuple[str, ...]:
        return (
            "client_utility",
            "dealer_economics",
            "capital_efficiency",
            "risk_change",
            "relationship_delta",
            "query_cost",
            "quote_cost",
            "communication_faithfulness",
            "operational_cost",
            "terminal_lifecycle_pnl",
        )

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: asdict(getattr(self, name))
            for name in self.component_names()
        }


@dataclass(frozen=True)
class ScalarizationSpec:
    """Explicit, versioned downstream scalarisation contract."""

    weights: Mapping[str, float]
    version: str = SCALARIZATION_VERSION

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(RewardComponents.component_names())
        if unknown:
            raise ValueError(f"unknown scalarization components: {sorted(unknown)}")
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            for value in self.weights.values()
        ):
            raise ValueError("scalarization weights must be finite")
        if (
            self.weights.get("dealer_economics", 0.0) != 0.0
            and self.weights.get("terminal_lifecycle_pnl", 0.0) != 0.0
        ):
            raise ValueError(
                "one scalarization cannot combine ex-ante dealer_economics "
                "with ex-post terminal_lifecycle_pnl"
            )


def scalarize_reward(
    reward: RewardComponents,
    spec: ScalarizationSpec,
) -> float:
    """Scalarise only explicitly selected, available normalized components."""

    total = 0.0
    for name, weight in spec.weights.items():
        term = getattr(reward, name)
        if not term.available:
            if term.provenance == "no-lifecycle-event-this-step":
                continue
            raise ValueError(f"cannot scalarize unavailable component {name!r}")
        value = (
            term.normalized_value
            if term.normalized_value is not None
            else term.value
        )
        if value is None:
            raise ValueError(f"component {name!r} has no scalar value")
        total += float(weight) * value
    return total


@dataclass(frozen=True)
class ConstraintSignals:
    """Non-reward protocol, hard-risk, and contract signals."""

    action_valid: bool = True
    hard_pass: bool | None = None
    client_contract_pass: bool | None = None
    accepted: bool | None = None
    hard_failures: tuple[str, ...] = ()
    contract_failures: tuple[str, ...] = ()
    exhausted_budgets: tuple[str, ...] = ()
    error: str | None = None

    @property
    def hard_violations(self) -> tuple[str, ...]:
        return self.hard_failures

    @property
    def contract_violations(self) -> tuple[str, ...]:
        return self.contract_failures


@dataclass(frozen=True)
class StepTransition:
    """One complete state-action-state transition returned by ``step``.

    Iteration yields the familiar five values ``(observation, reward,
    terminated, truncated, info)`` without taking a dependency on Gymnasium.
    The explicit fields retain the previous observation, action, constraint
    vector, and both state hashes for lossless trajectory recording.
    """

    previous_observation: Observation
    action: EnvironmentAction
    observation: Observation
    tool_result: Mapping[str, Any] | None
    reward_components: RewardComponents
    constraint_signals: ConstraintSignals
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]
    state_hash_before: str
    state_hash_after: str

    @property
    def reward(self) -> RewardComponents:
        return self.reward_components

    @property
    def constraints(self) -> ConstraintSignals:
        return self.constraint_signals

    @property
    def next_observation(self) -> Observation:
        return self.observation

    def __iter__(self):
        yield self.observation
        yield self.reward_components
        yield self.terminated
        yield self.truncated
        yield self.info


__all__ = [
    "REWARD_SCHEMA_VERSION",
    "SCALARIZATION_VERSION",
    "TASK_SCHEMA",
    "TASK_VERSION",
    "Action",
    "AskClient",
    "ConstraintSignals",
    "EnvironmentAction",
    "EpisodeTask",
    "InvalidAction",
    "Observation",
    "RequestQuote",
    "RewardComponents",
    "RewardTerm",
    "ScalarizationSpec",
    "Skip",
    "StepTransition",
    "SubmitDesign",
    "SubmitProduct",
    "scalarize_reward",
]
