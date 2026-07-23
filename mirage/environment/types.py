"""Typed Level-0 contracts for the MIRAGE v3 environment spine.

The classes in this module intentionally contain no runner, prompting, oracle,
or model logic.  They are the small, serialisable boundary shared by training
loops and the synchronous environment in :mod:`mirage.environment.core`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, TypeAlias

from ..benchmark import MarketSnapshot, ProductDomainSpec, RiskBudget
from ..pricing import QuotePolicy
from ..products import ClientProfile, ProductSpec


TASK_SCHEMA = "mirage.environment.task.v3"
TASK_VERSION = "v3-spine-level0"


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

    def __post_init__(self) -> None:
        # A tuple prevents callers from changing episode length/order through a
        # list after construction.  Nested v2 values are copied by reset().
        object.__setattr__(self, "snapshots", tuple(self.snapshots))
        if not self.snapshots:
            raise ValueError("EpisodeTask.snapshots must not be empty")
        if isinstance(self.task_seed, bool) or not isinstance(self.task_seed, int):
            raise TypeError("EpisodeTask.task_seed must be an int")
        if not isinstance(self.schema, str) or not self.schema.strip():
            raise ValueError("EpisodeTask.schema must be a non-empty string")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ValueError("EpisodeTask.version must be a non-empty string")

    @property
    def schema_version(self) -> str:
        """Explicit alias used by trajectory metadata."""

        return self.schema


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
    episode_id: str
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
class RewardComponents:
    """Unscalarised outcome/cost vector for one environment action.

    The components intentionally use their native scales.  A downstream
    experiment may scalarise them, but the environment never silently chooses
    weights on the researcher's behalf.
    """

    client_utility: float = 0.0
    dealer_economics: float = 0.0
    capital_efficiency: float = 0.0
    risk_change: float = 0.0
    relationship_delta: float = 0.0
    query_cost: float = 0.0
    quote_cost: float = 0.0
    communication_faithfulness: float = 0.0
    operational_cost: float = 0.0
    terminal_lifecycle_pnl: float = 0.0

    @property
    def dealer_margin(self) -> float:
        """Level-0 compatibility name for ``dealer_economics``."""

        return self.dealer_economics

    def as_dict(self) -> dict[str, float]:
        return {
            "client_utility": self.client_utility,
            "dealer_economics": self.dealer_economics,
            "capital_efficiency": self.capital_efficiency,
            "risk_change": self.risk_change,
            "relationship_delta": self.relationship_delta,
            "query_cost": self.query_cost,
            "quote_cost": self.quote_cost,
            "communication_faithfulness": self.communication_faithfulness,
            "operational_cost": self.operational_cost,
            "terminal_lifecycle_pnl": self.terminal_lifecycle_pnl,
        }


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
    "Skip",
    "StepTransition",
    "SubmitDesign",
    "SubmitProduct",
]
