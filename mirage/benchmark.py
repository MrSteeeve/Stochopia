"""MIRAGE research benchmark protocol.

This module adds the reproducible long-horizon layer on top of the original
Structurer Playground pricing primitives.  It deliberately keeps market data,
pricing and hard constraints inside the environment: a tested model can only
observe a compact brief and request state-bound desk quotes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Sequence

if TYPE_CHECKING:  # runtime import avoided (env_agents imports CheckResult from here)
    from .env_agents import FormalFacts, FrozenEnvAgent, RoleResponse

from .pricing import (
    QuotePolicy,
    _fair_value_stress_loss,
    client_loss_measure,
    evaluate_product,
    hurdle_hit_prob,
    mc_diagnostics,
    quote_economics,
)
from .products import PRODUCT_TYPES, ClientProfile, MarketState, ProductSpec

# Fixed diagnostic MC seed for desk/oracle quotes. Constant across candidates so
# the structure-keyed mc_diagnostics cache is shared during oracle enumeration.
QUOTE_DIAGNOSTIC_SEED = 42


class BenchmarkError(Exception):
    """Benchmark configuration or state transition is invalid."""


@dataclass(frozen=True)
class BenchmarkCondition:
    """One cell of the Full/Partial x Static/Dynamic experiment."""

    full_information: bool
    dynamic: bool

    @property
    def id(self) -> str:
        info = "full" if self.full_information else "partial"
        horizon = "dynamic" if self.dynamic else "static"
        return f"{info}_{horizon}"


@dataclass(frozen=True)
class MarketSnapshot:
    """Leakage-safe monthly market observation used by an episode."""

    episode_id: str
    round_num: int
    as_of: date
    underlying: str
    spot: float
    risk_free_rate: float
    return_20d: float | None = None
    realized_vol_20d: float | None = None
    realized_vol_60d: float | None = None
    drawdown_6m: float | None = None
    atm_iv_1m: float | None = None
    atm_iv_3m: float | None = None
    atm_iv_6m: float | None = None
    carry_rate: float = 0.0
    regime: str = "unknown"
    source: str = ""

    def pricing_volatility(self, maturity_months: int | None = None) -> tuple[float, str]:
        """Return the documented ATM -> realized-vol fallback."""
        if maturity_months is None:
            atm_choices = (
                (self.atm_iv_3m, "atm_iv_3m"),
                (self.atm_iv_1m, "atm_iv_1m"),
                (self.atm_iv_6m, "atm_iv_6m"),
            )
        elif maturity_months <= 2:
            atm_choices = ((self.atm_iv_1m, "atm_iv_1m"), (self.atm_iv_3m, "atm_iv_3m"), (self.atm_iv_6m, "atm_iv_6m"))
        elif maturity_months <= 4:
            atm_choices = ((self.atm_iv_3m, "atm_iv_3m"), (self.atm_iv_1m, "atm_iv_1m"), (self.atm_iv_6m, "atm_iv_6m"))
        else:
            atm_choices = ((self.atm_iv_6m, "atm_iv_6m"), (self.atm_iv_3m, "atm_iv_3m"), (self.atm_iv_1m, "atm_iv_1m"))
        choices = atm_choices + (
            (self.realized_vol_20d, "realized_vol_20d"),
            (self.realized_vol_60d, "realized_vol_60d"),
        )
        for value, source in choices:
            if value is not None and math.isfinite(value) and value > 0:
                return float(value), source
        raise BenchmarkError(
            f"{self.episode_id} round {self.round_num} has no usable volatility"
        )

    def to_market_state(self, maturity_months: int | None = None) -> MarketState:
        vol, _ = self.pricing_volatility(maturity_months)
        return MarketState(
            round_num=self.round_num,
            index_name=self.underlying,
            spot=self.spot,
            volatility=vol,
            risk_free_rate=self.risk_free_rate,
            # Carry is inferred from the index-futures term structure when present.
            dividend_yield=self.carry_rate,
            recent_trend=self.regime,
            vix_level=self.regime,
        )

    def public_brief(self) -> dict:
        """Fields available to the tested model; no raw chain or future data."""
        vol, vol_source = self.pricing_volatility()
        return {
            "episode_id": self.episode_id,
            "round": self.round_num,
            "date": self.as_of.isoformat(),
            "underlying": self.underlying,
            "spot": self.spot,
            "return_20d": self.return_20d,
            "realized_vol_20d": self.realized_vol_20d,
            "realized_vol_60d": self.realized_vol_60d,
            "drawdown_6m": self.drawdown_6m,
            "atm_iv_1m": self.atm_iv_1m,
            "atm_iv_3m": self.atm_iv_3m,
            "atm_iv_6m": self.atm_iv_6m,
            "pricing_volatility": vol,
            "volatility_source": vol_source,
            "market_regime": self.regime,
        }


def _optional_float(row: dict[str, str], name: str) -> float | None:
    raw = row.get(name, "").strip()
    return None if not raw else float(raw)


def load_market_snapshots(path: str | Path) -> list[MarketSnapshot]:
    """Load and validate monthly snapshots from a provenance-carrying CSV."""
    p = Path(path)
    if not p.is_file():
        raise BenchmarkError(f"market snapshot file does not exist: {p}")
    required = {
        "episode_id", "round", "date", "underlying", "spot",
        "risk_free_rate", "source",
    }
    snapshots: list[MarketSnapshot] = []
    with p.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise BenchmarkError(f"market snapshot CSV missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            try:
                snapshot = MarketSnapshot(
                    episode_id=row["episode_id"].strip(),
                    round_num=int(row["round"]),
                    as_of=date.fromisoformat(row["date"].strip()),
                    underlying=row["underlying"].strip(),
                    spot=float(row["spot"]),
                    risk_free_rate=float(row["risk_free_rate"]),
                    return_20d=_optional_float(row, "return_20d"),
                    realized_vol_20d=_optional_float(row, "realized_vol_20d"),
                    realized_vol_60d=_optional_float(row, "realized_vol_60d"),
                    drawdown_6m=_optional_float(row, "drawdown_6m"),
                    atm_iv_1m=_optional_float(row, "atm_iv_1m"),
                    atm_iv_3m=_optional_float(row, "atm_iv_3m"),
                    atm_iv_6m=_optional_float(row, "atm_iv_6m"),
                    carry_rate=_optional_float(row, "carry_rate") or 0.0,
                    regime=row.get("regime", "unknown").strip() or "unknown",
                    source=row["source"].strip(),
                )
                if not snapshot.episode_id or not snapshot.underlying or not snapshot.source:
                    raise ValueError("episode_id, underlying and source must be non-empty")
                if snapshot.round_num < 1 or snapshot.spot <= 0:
                    raise ValueError("round must be >=1 and spot must be positive")
                snapshot.pricing_volatility()
            except (TypeError, ValueError, BenchmarkError) as exc:
                raise BenchmarkError(f"invalid market snapshot at {p}:{line_no}: {exc}") from exc
            snapshots.append(snapshot)

    keys = [(s.episode_id, s.round_num) for s in snapshots]
    if len(keys) != len(set(keys)):
        raise BenchmarkError("duplicate (episode_id, round) in market snapshot CSV")
    by_episode: dict[str, list[int]] = {}
    for snapshot in snapshots:
        by_episode.setdefault(snapshot.episode_id, []).append(snapshot.round_num)
    for episode_id, rounds in by_episode.items():
        ordered = sorted(rounds)
        if ordered != list(range(1, len(ordered) + 1)):
            raise BenchmarkError(f"episode {episode_id} rounds must be contiguous from 1")
    return sorted(snapshots, key=lambda s: (s.episode_id, s.round_num))


def option_implied_forward(call: float, put: float, strike: float, rate: float, years: float) -> float:
    """Compute F=K+exp(rT)(C-P), absorbing dividend/carry expectations."""
    values = (call, put, strike, rate, years)
    if not all(math.isfinite(v) for v in values) or call < 0 or put < 0:
        raise BenchmarkError("invalid put-call parity input")
    if strike <= 0 or years <= 0:
        raise BenchmarkError("strike and years must be positive")
    forward = strike + math.exp(rate * years) * (call - put)
    if forward <= 0:
        raise BenchmarkError("option-implied forward is non-positive")
    return forward


@dataclass(frozen=True)
class RiskBudget:
    notional: float
    net_delta: float
    gross_delta: float
    net_vega: float
    stress_loss: float

    def scaled(self, factor: float) -> "RiskBudget":
        if factor <= 0:
            raise BenchmarkError("risk budget scale must be positive")
        return replace(
            self,
            notional=self.notional * factor,
            net_delta=self.net_delta * factor,
            gross_delta=self.gross_delta * factor,
            net_vega=self.net_vega * factor,
            stress_loss=self.stress_loss * factor,
        )


@dataclass
class Position:
    """Lifecycle state for an issued contract.

    ``product`` is the issued contract, while the mutable fields below are the
    environment truth carried between market observations.  In particular,
    strike/barrier levels are stored in absolute units so a later revaluation
    cannot silently re-anchor the contract to the new spot.
    """

    position_id: str
    product: ProductSpec
    remaining_months: int
    delta_dollars: float
    vega_dollars: float
    stress_loss: float
    quote_margin: float
    trade_date: date | None = None
    initial_fixing: float | None = None
    absolute_strike: float | None = None
    absolute_barrier: float | None = None
    barrier_direction: str | None = None
    elapsed_months: int = 0
    running_min: float | None = None
    running_max: float | None = None
    barrier_touched: bool = False
    knock_in_state: bool = False
    knock_out_state: bool = False
    autocall_state: bool = False
    accrued_coupon: float = 0.0
    current_fair_value: float = 0.0
    status: str = "active"
    last_valuation_round: int | None = None


_CLIENT_OVERRIDE_FIELDS = frozenset({
    "capital", "max_loss_pct", "min_return_pct", "risk_appetite", "preferences",
    "max_maturity_months", "principal_protection_required", "allowed_product_types",
    "accepting_new_products", "min_hit_prob", "current_focus",
})


def resolve_client_profile(client: ClientProfile, round_num: int) -> ClientProfile:
    """Resolve a base client profile into deterministic round-specific truth.

    The benchmark CLI loads ``ClientProfile`` directly from JSON, bypassing the
    legacy scenario loader.  Keeping this resolver beside the environment makes
    ``round_overrides`` effective on every production path and rejects malformed
    schedules before they can change settlement semantics mid-episode.
    """
    combined: dict = {}
    for index, entry in enumerate(client.round_overrides):
        if not isinstance(entry, dict):
            raise BenchmarkError(f"client round_overrides[{index}] must be an object")
        rounds = entry.get("rounds")
        if not isinstance(rounds, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) and item >= 1
            for item in rounds
        ):
            raise BenchmarkError(
                f"client round_overrides[{index}].rounds must contain positive integers"
            )
        unknown = set(entry) - _CLIENT_OVERRIDE_FIELDS - {"rounds"}
        if unknown:
            raise BenchmarkError(
                f"client round_overrides[{index}] has unsupported fields: {sorted(unknown)}"
            )
        if round_num in rounds:
            combined.update({key: value for key, value in entry.items() if key != "rounds"})

    combined.setdefault("current_focus", "")
    resolved = replace(client, **combined)
    numeric = (
        resolved.capital, resolved.max_loss_pct, resolved.min_return_pct,
        resolved.min_hit_prob,
    )
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in numeric):
        raise BenchmarkError("resolved client numeric fields must be finite")
    if resolved.capital <= 0:
        raise BenchmarkError("resolved client capital must be positive")
    if not 0.0 <= resolved.max_loss_pct <= 1.0:
        raise BenchmarkError("resolved client max_loss_pct must be in [0, 1]")
    if not 0.0 <= resolved.min_hit_prob <= 1.0:
        raise BenchmarkError("resolved client min_hit_prob must be in [0, 1]")
    if (
        not isinstance(resolved.max_maturity_months, int)
        or isinstance(resolved.max_maturity_months, bool)
        or resolved.max_maturity_months < 1
    ):
        raise BenchmarkError("resolved client max_maturity_months must be positive")
    if resolved.risk_appetite not in {"conservative", "moderate", "aggressive"}:
        raise BenchmarkError("resolved client risk_appetite is invalid")
    if not isinstance(resolved.principal_protection_required, bool):
        raise BenchmarkError("resolved client principal_protection_required must be boolean")
    if not isinstance(resolved.accepting_new_products, bool):
        raise BenchmarkError("resolved client accepting_new_products must be boolean")
    if resolved.allowed_product_types is not None:
        if not isinstance(resolved.allowed_product_types, list):
            raise BenchmarkError("resolved client allowed_product_types must be a list or null")
        if not all(isinstance(item, str) for item in resolved.allowed_product_types):
            raise BenchmarkError("resolved client allowed_product_types must contain strings")
        invalid = set(resolved.allowed_product_types) - set(PRODUCT_TYPES)
        if invalid:
            raise BenchmarkError(f"resolved client product types are invalid: {sorted(invalid)}")
    return resolved


def _barrier_direction(product: ProductSpec) -> str | None:
    explicit = getattr(product, "barrier_direction", None)
    if explicit is not None:
        return explicit
    if product.barrier_pct is None:
        return None
    return "down" if product.barrier_pct <= 1.0 else "up"


def _process_position_observation(position: Position, snapshot: MarketSnapshot) -> None:
    """Apply one observed month to persistent path state before revaluation."""
    spot = snapshot.spot
    position.running_min = spot if position.running_min is None else min(position.running_min, spot)
    position.running_max = spot if position.running_max is None else max(position.running_max, spot)
    position.elapsed_months += 1
    position.remaining_months -= 1
    position.accrued_coupon = (
        (position.product.coupon_rate or 0.0) * position.elapsed_months / 12.0
    )

    barrier = position.absolute_barrier
    direction = position.barrier_direction
    touched = bool(
        barrier is not None
        and ((direction == "down" and spot <= barrier) or (direction == "up" and spot >= barrier))
    )
    position.barrier_touched = position.barrier_touched or touched

    if position.product.product_type in {"barrier_call", "barrier_put"} and touched:
        if position.product.barrier_type == "knock_out":
            position.knock_out_state = True
            # A knock-out terminates only the option component.  A protected
            # note still carries its bond floor and remains a live liability
            # until maturity; an unprotected option can be closed immediately.
            if not position.product.principal_protected:
                position.status = "knocked_out"
        else:
            position.knock_in_state = True

    if position.product.product_type in {"snowball", "autocallable"}:
        if barrier is not None and spot <= barrier:
            position.knock_in_state = True
        if position.initial_fixing is not None:
            autocall_ratio = 1.03 if position.product.product_type == "snowball" else 1.05
            if spot >= position.initial_fixing * autocall_ratio:
                position.autocall_state = True
                position.status = "autocalled"

    if position.remaining_months <= 0 and position.status == "active":
        position.status = "matured"


def _revalue_position(position: Position, snapshot: MarketSnapshot) -> None:
    """Recompute FV/Greeks/stress from current market and persistent path truth."""
    if position.status != "active":
        position.current_fair_value = 0.0
        position.delta_dollars = 0.0
        position.vega_dollars = 0.0
        position.stress_loss = 0.0
        position.last_valuation_round = snapshot.round_num
        return

    if position.initial_fixing is None or position.absolute_strike is None:
        raise BenchmarkError(f"position {position.position_id} is missing issue fixing state")

    product = replace(
        position.product,
        barrier_direction=position.barrier_direction,
        reference_spot=position.initial_fixing,
        barrier_touched=position.barrier_touched,
        knock_in_active=position.knock_in_state,
        elapsed_months=position.elapsed_months,
    )
    market = snapshot.to_market_state(max(position.remaining_months, 1))
    pricing = evaluate_product(product, market)
    if pricing is None:
        raise BenchmarkError(f"cannot revalue unsupported position {position.position_id}")
    stress_loss, _ = _fair_value_stress_loss(product, market)
    position.current_fair_value = float(pricing["fair_value"])
    position.delta_dollars = float(pricing["greeks"].get("delta", 0.0)) * product.notional
    position.vega_dollars = float(pricing["greeks"].get("vega_pct", 0.0)) * product.notional
    position.stress_loss = float(stress_loss)
    position.last_valuation_round = snapshot.round_num


@dataclass
class PortfolioState:
    """Outstanding products and risk carried across dynamic rounds."""

    positions: list[Position] = field(default_factory=list)
    revision: int = 0

    def totals(self) -> dict[str, float]:
        return {
            "notional": sum(p.product.notional for p in self.positions),
            "fair_value": sum(p.current_fair_value for p in self.positions),
            "net_delta": sum(p.delta_dollars for p in self.positions),
            "gross_delta": sum(abs(p.delta_dollars) for p in self.positions),
            "net_vega": sum(p.vega_dollars for p in self.positions),
            "stress_loss": sum(max(p.stress_loss, 0.0) for p in self.positions),
        }

    def advance_month(self, snapshot: MarketSnapshot | None = None) -> list[str]:
        """Advance positions and, when a snapshot is supplied, revalue them.

        ``snapshot=None`` remains a compatibility path for callers that only
        need mechanical maturity.  The benchmark environment always supplies
        the next observation, so portfolio limits never use stale issue-time
        risk after the first round.
        """
        closed: list[str] = []
        active: list[Position] = []
        for position in self.positions:
            if snapshot is None:
                position.remaining_months -= 1
                position.elapsed_months += 1
            else:
                _process_position_observation(position, snapshot)
                _revalue_position(position, snapshot)
            if position.remaining_months <= 0 or position.status != "active":
                closed.append(position.position_id)
            else:
                active.append(position)
        self.positions = active
        self.revision += 1
        return closed

    def add(self, position: Position) -> None:
        self.positions.append(position)
        self.revision += 1

    def reset(self) -> None:
        self.positions.clear()
        self.revision += 1


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    status: str
    observed: float | str | bool | None
    limit: float | str | bool | None
    severity: str
    reason: str

    @property
    def passed(self) -> bool:
        return self.status == "PASS"


# Exact client mandate values are hidden in Partial.  A quote may reveal which
# client gate passed/failed (that is useful, budgeted interaction), but it must
# not turn one quote into a bulk dump of every numeric threshold.
_PRIVATE_CLIENT_CHECK_IDS = frozenset({
    "CLIENT_ACCEPTING",
    "CLIENT_CAPITAL",
    "CLIENT_MATURITY",
    "CLIENT_PRODUCT_WHITELIST",
    "CLIENT_LOSS_BUDGET_V2",
    "CLIENT_PROTECTION",
    "CONTRACT_ACCEPTING",
    "CONTRACT_CAPITAL",
    "CONTRACT_MATURITY",
    "CONTRACT_WHITELIST",
    "CONTRACT_PROTECTION",
    "CONTRACT_LOSS",
    "CONTRACT_HURDLE",
})


@dataclass(frozen=True)
class Quote:
    quote_id: str
    state_version: str
    fair_value: float
    client_price: float
    dealer_margin: float
    delta_dollars: float
    vega_dollars: float
    stress_loss: float
    volatility_source: str
    checks: tuple[CheckResult, ...]
    warnings: tuple[str, ...]
    margin_rate: float = 0.0
    suitability: float = 1.0
    hurdle_hit_prob: float | None = None
    worst_stress_id: str = ""
    # Full engine-side pricing dict (enriched with hurdle_hit_prob / stress_loss).
    # Excluded from equality/repr so the frozen dataclass stays lightweight and the
    # dict never leaks through public_payload. Consumed by client_contract_pass.
    pricing: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def hard_pass(self) -> bool:
        return all(item.passed for item in self.checks if item.severity == "HARD")

    def public_payload(self) -> dict:
        # Deliberately exposes margin_rate and suitability, but never the margin
        # breakdown, so the tested model cannot reverse-engineer the quote formula.
        return {
            "quote_id": self.quote_id,
            "fair_value": self.fair_value,
            "client_price": self.client_price,
            "dealer_margin": self.dealer_margin,
            "margin_rate": self.margin_rate,
            "suitability": self.suitability,
            "delta_dollars": self.delta_dollars,
            "vega_dollars": self.vega_dollars,
            "stress_loss": self.stress_loss,
            "volatility_source": self.volatility_source,
            "hard_pass": self.hard_pass,
            "checks": [asdict(check) for check in self.checks],
            "warnings": list(self.warnings),
            "valid_for_state": self.state_version,
        }


def _check(check_id: str, passed: bool, observed, limit, reason: str, severity: str = "HARD") -> CheckResult:
    return CheckResult(
        check_id=check_id,
        status="PASS" if passed else "FAIL",
        observed=observed,
        limit=limit,
        severity=severity,
        reason=reason,
    )


def worst_case_payoff_ratio(product: ProductSpec) -> float:
    """Conservative template-level terminal payoff floor.

    The original product grammar represents options and structured notes rather
    than arbitrary payoff code.  For a declared protected template the bond
    floor is one; otherwise the conservative payoff floor is zero.
    """
    if product.principal_protected and product.product_type not in {"snowball"}:
        if product.product_type != "autocallable" or product.barrier_pct is None:
            return 1.0
    return 0.0


@dataclass(frozen=True)
class ProductDomainSpec:
    """Frozen finite action lattice shared by the tested agent and the oracle.

    Any change to these tuples is a protocol change and must be re-frozen before
    evaluation. ``notional_fractions`` are expressed relative to client capital so
    the same lattice transfers across clients; the absolute notional for a
    candidate is ``round(client.capital * fraction)``.
    """

    product_types: tuple[str, ...] = (
        "vanilla_call", "vanilla_put", "barrier_call", "barrier_put",
        "autocallable", "snowball",
    )
    notional_fractions: tuple[float, ...] = (
        .005, .01, .02, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.00,
    )
    maturities: tuple[int, ...] = (3, 6, 12)
    strikes: tuple[float, ...] = (.95, 1.00, 1.05)
    barriers: tuple[float, ...] = (.75, .85, 1.10, 1.20)
    coupons: tuple[float, ...] = (.04, .08)
    participations: tuple[float, ...] = (.5, 1.0)
    principal_protected: tuple[bool, ...] = (False, True)
    version: str = "csi-domain-v1"


def _domain_notionals(client: ClientProfile, domain: ProductDomainSpec) -> list[int]:
    return sorted({int(round(client.capital * f)) for f in domain.notional_fractions if f > 0})


def _domain_signature(product: ProductSpec) -> tuple:
    """Structural key used to test lattice membership (float-normalised)."""
    return (
        product.product_type,
        int(round(product.notional)),
        int(product.maturity_months),
        round(product.strike_pct, 9),
        round(product.barrier_pct, 9) if product.barrier_pct is not None else None,
        product.barrier_type,
        _barrier_direction(product),
        round(product.coupon_rate, 9) if product.coupon_rate is not None else None,
        round(product.participation_rate, 9),
        bool(product.principal_protected),
    )


def enumerate_domain(client: ClientProfile, domain: ProductDomainSpec | None = None):
    """Yield every ProductSpec in the frozen lattice for this client.

    Pure construction, no pricing: the oracle prices these, and validate_domain
    tests membership against exactly this set, so agent and oracle share one domain.
    """
    domain = domain or ProductDomainSpec()

    def _spec(product_type, notional, maturity, *, strike=1.0, barrier=None,
              barrier_type=None, coupon=None, participation=1.0, protected=False):
        return ProductSpec(
            product_type=product_type,
            notional=float(notional),
            maturity_months=maturity,
            strike_pct=strike,
            barrier_pct=barrier,
            barrier_type=barrier_type,
            coupon_rate=coupon,
            participation_rate=participation,
            principal_protected=protected,
            target_client=client.id,
            pitch="domain lattice candidate",
            hedging_plan="delta hedge with listed futures",
            barrier_direction=(
                "down" if barrier is not None and barrier <= 1.0
                else "up" if barrier is not None
                else None
            ),
        )

    lower_barriers = tuple(b for b in domain.barriers if b < 1.0)
    for notional in _domain_notionals(client, domain):
        for maturity in domain.maturities:
            for ptype in domain.product_types:
                if ptype in ("vanilla_call", "vanilla_put"):
                    for strike in domain.strikes:
                        for part in domain.participations:
                            for pp in domain.principal_protected:
                                yield _spec(ptype, notional, maturity, strike=strike,
                                            participation=part, protected=pp)
                elif ptype in ("barrier_call", "barrier_put"):
                    for strike in domain.strikes:
                        for barrier in domain.barriers:
                            btype = "knock_out" if barrier > 1.0 else "knock_in"
                            for part in domain.participations:
                                for pp in domain.principal_protected:
                                    yield _spec(ptype, notional, maturity, strike=strike,
                                                barrier=barrier, barrier_type=btype,
                                                participation=part, protected=pp)
                elif ptype == "autocallable":
                    for coupon in domain.coupons:
                        for pp in domain.principal_protected:
                            yield _spec(ptype, notional, maturity, coupon=coupon, protected=pp)
                            for barrier in lower_barriers:
                                yield _spec(ptype, notional, maturity, coupon=coupon,
                                            barrier=barrier, barrier_type="knock_in", protected=pp)
                elif ptype == "snowball":
                    for coupon in domain.coupons:
                        for barrier in lower_barriers:
                            yield _spec(ptype, notional, maturity, coupon=coupon,
                                        barrier=barrier, barrier_type="knock_in", protected=False)


_DOMAIN_SET_CACHE: dict = {}


def _domain_signature_set(client: ClientProfile, domain: ProductDomainSpec) -> frozenset:
    """Cached set of in-lattice structural signatures for this (capital, domain)."""
    key = (int(round(client.capital)), domain)
    cached = _DOMAIN_SET_CACHE.get(key)
    if cached is None:
        cached = frozenset(_domain_signature(candidate) for candidate in enumerate_domain(client, domain))
        _DOMAIN_SET_CACHE[key] = cached
    return cached


def validate_domain(
    product: ProductSpec, client: ClientProfile, domain: ProductDomainSpec | None = None
) -> tuple[CheckResult, ...]:
    """Return a single HARD DOMAIN check; PASS iff the product is in the lattice.

    Membership is defined as identity to some ``enumerate_domain`` candidate, so a
    quotable agent action is always an oracle candidate (attainment can never exceed 1).
    """
    domain = domain or ProductDomainSpec()
    ok = _domain_signature(product) in _domain_signature_set(client, domain)
    reason = (
        f"product is inside the frozen action lattice {domain.version}"
        if ok else
        f"product parameters are outside the frozen action lattice {domain.version}"
    )
    return (_check("DOMAIN", ok, product.product_type, domain.version, reason, "HARD"),)


def client_contract_pass(
    product: ProductSpec, quote_pricing: dict, client: ClientProfile
) -> tuple[bool, list[CheckResult]]:
    """Deterministic client contract gate (the settlement counterpart of would_buy).

    Checks capital, maturity, whitelist, protection preference, the continuous
    loss proxy and hurdle_hit_prob >= min_hit_prob. Uses the enriched engine
    pricing dict (must carry ``loss_frac``, ``stress_loss``, ``hurdle_hit_prob``).
    Mirrors mirage.products.ClientProfile.would_buy but on the continuous loss
    measure. The client LLM's attitude never overrides this gate.
    """
    stress_loss = float(quote_pricing.get("stress_loss", 0.0))
    worst_stress_id = str(quote_pricing.get("worst_stress_id", ""))
    measure = client_loss_measure(product, quote_pricing, stress_loss, worst_stress_id=worst_stress_id)
    allowed = client.allowed_product_types is None or product.product_type in client.allowed_product_types
    needs_protection = client.principal_protection_required or client.risk_appetite == "conservative"
    protection_ok = not needs_protection or product.principal_protected
    hh = quote_pricing.get("hurdle_hit_prob")
    hurdle_ok = hh is None or hh >= client.min_hit_prob - 1e-9

    checks = [
        _check("CONTRACT_ACCEPTING", client.accepting_new_products,
               client.accepting_new_products, True,
               "client must be accepting new products", "CONTRACT"),
        _check("CONTRACT_CAPITAL", product.notional <= client.capital + 1e-9,
               product.notional, client.capital,
               "notional must not exceed available capital", "CONTRACT"),
        _check("CONTRACT_MATURITY", product.maturity_months <= client.max_maturity_months,
               product.maturity_months, client.max_maturity_months,
               "maturity must fit the client mandate", "CONTRACT"),
        _check("CONTRACT_WHITELIST", allowed, product.product_type,
               "all" if client.allowed_product_types is None else ",".join(client.allowed_product_types),
               "product family must be allowed for the client", "CONTRACT"),
        _check("CONTRACT_PROTECTION", protection_ok, product.principal_protected, needs_protection,
               "a protection-seeking client only contracts on protected structures", "CONTRACT"),
        _check("CONTRACT_LOSS", measure.observed_loss_frac <= client.max_loss_pct + 1e-9,
               measure.observed_loss_frac, client.max_loss_pct,
               f"continuous loss proxy expected={measure.expected_loss_frac:.4f} "
               f"premium={measure.premium_at_risk_frac:.4f} stress={measure.stress_loss_frac:.4f}",
               "CONTRACT"),
        _check("CONTRACT_HURDLE", hurdle_ok,
               hh if hh is not None else "n/a", client.min_hit_prob,
               "realised-return hit probability must reach the client threshold", "CONTRACT"),
    ]
    return all(c.passed for c in checks), checks


class HardConstraintEngine:
    """Deterministic, auditable checks; an LLM judge cannot override them."""

    def __init__(self, risk_budget: RiskBudget, description_tolerance_bp: float = 1.0):
        self.risk_budget = risk_budget
        self.description_tolerance_bp = description_tolerance_bp

    def evaluate(
        self,
        product: ProductSpec,
        pricing: dict,
        client: ClientProfile,
        portfolio: PortfolioState,
        *,
        delta_dollars: float,
        vega_dollars: float,
        stress_loss: float,
        worst_stress_id: str = "",
    ) -> list[CheckResult]:
        current = portfolio.totals()
        protection_tolerance = self.description_tolerance_bp / 10_000.0
        floor = worst_case_payoff_ratio(product)
        client_loss = client_loss_measure(product, pricing, stress_loss, worst_stress_id=worst_stress_id)
        post_notional = current["notional"] + product.notional
        post_net_delta = current["net_delta"] + delta_dollars
        post_gross_delta = current["gross_delta"] + abs(delta_dollars)
        post_net_vega = current["net_vega"] + vega_dollars
        post_stress = current["stress_loss"] + max(stress_loss, 0.0)

        allowed = client.allowed_product_types is None or product.product_type in client.allowed_product_types
        protection_ok = not client.principal_protection_required or floor >= 1.0 - protection_tolerance
        claim_ok = not product.principal_protected or floor >= 1.0 - protection_tolerance
        normalized_pitch = product.pitch.lower().replace(" ", "")
        protected_words = ("保本", "本金保障", "本金保证", "principalprotected", "capitalguaranteed")
        pitch_claims_protection = any(word in normalized_pitch for word in protected_words)
        description_ok = not pitch_claims_protection or claim_ok
        return [
            _check("CLIENT_ACCEPTING", client.accepting_new_products, client.accepting_new_products, True,
                   "client must be accepting new products"),
            _check("CLIENT_CAPITAL", product.notional <= client.capital + 1e-9,
                   product.notional, client.capital, "notional must not exceed available capital"),
            _check("CLIENT_MATURITY", product.maturity_months <= client.max_maturity_months,
                   product.maturity_months, client.max_maturity_months, "maturity must fit the client mandate"),
            _check("CLIENT_PRODUCT_WHITELIST", allowed, product.product_type,
                   "all" if client.allowed_product_types is None else ",".join(client.allowed_product_types),
                   "product family must be allowed for the client"),
            _check("CLIENT_LOSS_BUDGET_V2",
                   client_loss.observed_loss_frac <= client.max_loss_pct + 1e-9,
                   client_loss.observed_loss_frac, client.max_loss_pct,
                   f"continuous loss proxy must fit the hard loss limit "
                   f"(expected={client_loss.expected_loss_frac:.4f} "
                   f"premium={client_loss.premium_at_risk_frac:.4f} "
                   f"stress={client_loss.stress_loss_frac:.4f} "
                   f"worst_stress={client_loss.worst_stress_id or 'n/a'})"),
            _check("CLIENT_PROTECTION", protection_ok, floor, 1.0,
                   "a protection requirement needs a deterministic bond floor"),
            _check("PROTECTION_CLAIM", claim_ok, floor, 1.0 - protection_tolerance,
                   "a principal-protected claim must match the payoff floor within 1 bp"),
            _check("DESCRIPTION_PROTECTION", description_ok, pitch_claims_protection,
                   product.principal_protected,
                   "a natural-language protection claim must match the deterministic payoff floor"),
            _check("PORTFOLIO_NOTIONAL", post_notional <= self.risk_budget.notional + 1e-9,
                   post_notional, self.risk_budget.notional, "post-trade notional budget"),
            _check("PORTFOLIO_NET_DELTA", abs(post_net_delta) <= self.risk_budget.net_delta + 1e-9,
                   abs(post_net_delta), self.risk_budget.net_delta, "post-trade absolute net delta budget"),
            _check("PORTFOLIO_GROSS_DELTA", post_gross_delta <= self.risk_budget.gross_delta + 1e-9,
                   post_gross_delta, self.risk_budget.gross_delta, "post-trade gross delta budget"),
            _check("PORTFOLIO_NET_VEGA", abs(post_net_vega) <= self.risk_budget.net_vega + 1e-9,
                   abs(post_net_vega), self.risk_budget.net_vega, "post-trade absolute vega budget"),
            _check("PORTFOLIO_STRESS_LOSS", post_stress <= self.risk_budget.stress_loss + 1e-9,
                   post_stress, self.risk_budget.stress_loss, "post-trade stress-loss budget"),
        ]


# ---------------------------------------------------------------------------
# Structure-level pricing caches (notional-independent). The heavy MC in
# evaluate_product / _fair_value_stress_loss / hurdle_hit_prob is invariant to
# notional (fair value and stress loss scale linearly; hurdle depends only on the
# per-notional client price), so oracle enumeration shares one MC per structure
# across the whole notional ladder instead of repeating it 14x.
# ---------------------------------------------------------------------------

_PRICE_REF_NOTIONAL = 1_000_000.0
_STRUCT_PRICE_CACHE: dict = {}
_HURDLE_CACHE: dict = {}


def _structure_key(product: ProductSpec, market: MarketState) -> tuple:
    return (
        product.product_type,
        int(product.maturity_months),
        product.strike_pct,
        product.barrier_pct,
        product.barrier_type,
        getattr(product, "barrier_direction", None),
        product.coupon_rate,
        product.participation_rate,
        product.principal_protected,
        getattr(product, "reference_spot", None),
        bool(getattr(product, "barrier_touched", False)),
        bool(getattr(product, "knock_in_active", False)),
        int(getattr(product, "elapsed_months", 0)),
        market.spot,
        market.volatility,
        market.risk_free_rate,
        market.dividend_yield,
    )


def _structural_pricing(product: ProductSpec, market: MarketState) -> dict:
    """Notional-independent pricing fractions for one structure, MC'd once."""
    key = _structure_key(product, market)
    cached = _STRUCT_PRICE_CACHE.get(key)
    if cached is not None:
        return cached
    ref = replace(product, notional=_PRICE_REF_NOTIONAL)
    pr = evaluate_product(ref, market)
    if pr is None:
        raise BenchmarkError("custom products are outside the finite benchmark grammar")
    stress_ref, worst_stress_id = _fair_value_stress_loss(ref, market)
    cached = {
        "frac_fair": pr["fair_value"] / _PRICE_REF_NOTIONAL,
        "greeks": pr["greeks"],
        "loss_frac": pr["loss_frac"],
        "pricing_details": pr["pricing_details"],
        "stress_frac": stress_ref / _PRICE_REF_NOTIONAL,
        "worst_stress_id": worst_stress_id,
    }
    _STRUCT_PRICE_CACHE[key] = cached
    return cached


def _cached_hurdle(
    product: ProductSpec, market: MarketState, min_return: float, client_price_per_notional: float
) -> float | None:
    """hurdle_hit_prob keyed by structure + per-notional client price (notional-free)."""
    key = (
        _structure_key(product, market),
        round(min_return, 9),
        round(client_price_per_notional, 12),
    )
    if key in _HURDLE_CACHE:
        return _HURDLE_CACHE[key]
    ref = replace(product, notional=_PRICE_REF_NOTIONAL)
    value = hurdle_hit_prob(ref, market, min_return, client_price_per_notional * _PRICE_REF_NOTIONAL)
    _HURDLE_CACHE[key] = value
    return value


class TradingDesk:
    """State-bound deterministic quote service.

    Uses the v2 quote economics: client_price / hedging_cost / dealer_margin come
    from ``quote_economics`` (dealer_margin = N·(r_c·suitability − r_h), not the old
    1% constant). Domain membership is enforced as a HARD DOMAIN check so out-of-
    lattice products are refused a feasible quote.
    """

    def __init__(
        self,
        hard_checks: HardConstraintEngine,
        *,
        policy: QuotePolicy | None = None,
        domain: ProductDomainSpec | None = None,
    ):
        self.hard_checks = hard_checks
        self.policy = policy or QuotePolicy()
        self.domain = domain or ProductDomainSpec()

    def quote(
        self,
        product: ProductSpec,
        snapshot: MarketSnapshot,
        client: ClientProfile,
        portfolio: PortfolioState,
        state_version: str,
        quote_index: int,
    ) -> Quote:
        market = snapshot.to_market_state(product.maturity_months)
        struct = _structural_pricing(product, market)
        pricing = {
            "fair_value": struct["frac_fair"] * product.notional,
            "greeks": struct["greeks"],
            "loss_frac": struct["loss_frac"],
            "pricing_details": struct["pricing_details"],
        }
        stress_loss = struct["stress_frac"] * product.notional
        worst_stress_id = struct["worst_stress_id"]
        pricing["stress_loss"] = stress_loss
        pricing["worst_stress_id"] = worst_stress_id

        diag = mc_diagnostics(product, market, n_paths=self.policy.diagnostic_paths, seed=QUOTE_DIAGNOSTIC_SEED)
        totals = portfolio.totals()
        capacity_notional = self.hard_checks.risk_budget.notional
        post_notional = totals["notional"] + product.notional

        # Two-pass so suitability reflects the client's hurdle probability: the
        # first pass (no hurdle in pricing -> neutral) yields a client_price to
        # size hurdle_hit_prob, which the second pass folds into suitability.
        econ0 = quote_economics(
            pricing, diag, stress_loss=stress_loss, post_notional=post_notional,
            capacity_notional=capacity_notional, policy=self.policy, product=product, client=client,
        )
        cpn = econ0.client_price / product.notional if product.notional else 0.0
        hh = _cached_hurdle(product, market, client.min_return_pct, cpn)
        pricing["hurdle_hit_prob"] = hh
        econ = quote_economics(
            pricing, diag, stress_loss=stress_loss, post_notional=post_notional,
            capacity_notional=capacity_notional, policy=self.policy, product=product, client=client,
        )
        pricing["client_price"] = econ.client_price
        pricing["hedging_cost"] = econ.hedging_cost
        pricing["dealer_margin"] = econ.dealer_margin

        delta_dollars = float(pricing["greeks"].get("delta", 0.0)) * product.notional
        vega_dollars = float(pricing["greeks"].get("vega_pct", 0.0)) * product.notional

        domain_checks = validate_domain(product, client, self.domain)
        hard = self.hard_checks.evaluate(
            product, pricing, client, portfolio,
            delta_dollars=delta_dollars,
            vega_dollars=vega_dollars,
            stress_loss=stress_loss,
            worst_stress_id=worst_stress_id,
        )
        checks = list(domain_checks) + hard

        warnings: list[str] = []
        usages = {
            "delta": abs(totals["net_delta"] + delta_dollars) / self.hard_checks.risk_budget.net_delta,
            "vega": abs(totals["net_vega"] + vega_dollars) / self.hard_checks.risk_budget.net_vega,
        }
        for name, usage in usages.items():
            if usage >= 0.9:
                warnings.append(f"post-trade {name} utilization is {usage:.1%}")
        digest = hashlib.sha256(
            f"{state_version}|{quote_index}|{product}".encode("utf-8")
        ).hexdigest()[:12]
        vol_source = snapshot.pricing_volatility(product.maturity_months)[1]
        return Quote(
            quote_id=f"Q-{digest}",
            state_version=state_version,
            fair_value=econ.fair_value,
            client_price=econ.client_price,
            dealer_margin=econ.dealer_margin,
            delta_dollars=delta_dollars,
            vega_dollars=vega_dollars,
            stress_loss=stress_loss,
            volatility_source=vol_source,
            checks=tuple(checks),
            warnings=tuple(warnings),
            margin_rate=econ.margin_rate,
            suitability=econ.suitability,
            hurdle_hit_prob=hh,
            worst_stress_id=worst_stress_id,
            pricing=pricing,
        )


@dataclass
class ClientMemory:
    trust: float = 0.5
    accepted_products: int = 0
    rejected_products: int = 0


# ---------------------------------------------------------------------------
# v2 workflow layer (second, LLM-driven outcome; never touches primary metrics)
# ---------------------------------------------------------------------------

_NUMERIC_TOKEN_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _collect_numeric_strings(obj) -> tuple[str, ...]:
    """Every number reachable in a JSON-able structure, as raw strings.

    Numeric leaves are stringified; strings are scanned for embedded numeric
    tokens. Booleans are skipped (they are not numbers for grounding). The
    result is the ``allowed_numeric_strings`` an env role may echo in narrative;
    ``validate_grounding`` normalises both sides so ``5000000.0`` and
    ``5,000,000`` both match.
    """
    out: list[str] = []

    def walk(value) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            out.append(repr(value) if isinstance(value, float) else str(value))
        elif isinstance(value, str):
            out.extend(match.group(0) for match in _NUMERIC_TOKEN_RE.finditer(value))
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    walk(obj)
    return tuple(out)


_CONSULT_FALLBACK = {
    "trading_desk": "（交易台暂时无法给出定性判断；请以正式 request_quote 报价事实为准。）",
    "risk_control": "（风控暂时无法给出定性预检；请以正式报价的硬检查结果为准。）",
    "client": "（客户暂时未表态；这不改变确定性合同门的裁决。）",
}


def _consult_fallback(role: str) -> str:
    return _CONSULT_FALLBACK.get(role, "（该环境角色暂时无法响应。）")


@dataclass(frozen=True)
class WorkflowOutcome:
    """Second-layer, LLM-driven result of a submission (draft_codex §1.1).

    ``workflow_deal`` requires the deterministic hard-executability AND all three
    env roles affirming (desk issues, risk approves, client accepts). It is a
    behavioural signal only: the primary settlement (hard ∧ contract) in
    :meth:`LongHorizonEnvironment.submit_design` never depends on it.
    """

    workflow_deal: bool
    desk_action: str
    risk_action: str
    client_action: str
    degraded: bool


def settle_submission(
    hard_executable: bool,
    desk: "RoleResponse | None",
    risk: "RoleResponse | None",
    client: "RoleResponse | None",
) -> WorkflowOutcome:
    """Pure (no-I/O) composition of the three env-role decisions.

    A missing role (``None``) is treated as a non-affirmative, degraded signal
    so ``workflow_deal`` can only be True when every role genuinely affirmed.
    """
    def action(resp, default: str) -> str:
        return resp.action if resp is not None else default

    def degraded(resp) -> bool:
        return resp is None or bool(resp.degraded)

    desk_action = action(desk, "decline")
    risk_action = action(risk, "escalate")
    client_action = action(client, "reject")
    workflow_deal = bool(
        hard_executable
        and desk_action == "issue"
        and risk_action == "approve"
        and client_action == "accept"
    )
    is_degraded = degraded(desk) or degraded(risk) or degraded(client)
    return WorkflowOutcome(workflow_deal, desk_action, risk_action, client_action, is_degraded)


class LongHorizonEnvironment:
    """Tool-facing benchmark environment with strict information boundaries."""

    CLIENT_TOPIC_FIELDS = {
        "capital": ("capital",),
        "loss_tolerance": ("max_loss_pct",),
        "maturity": ("max_maturity_months",),
        "product_types": ("allowed_product_types",),
        "protection": ("principal_protection_required",),
        "preferences": ("preferences", "current_focus"),
        "purchase_status": ("accepting_new_products",),
        "risk_appetite": ("risk_appetite",),
        "return_hurdle": ("min_return_pct", "min_hit_prob"),
    }
    CLIENT_TOPICS = frozenset(CLIENT_TOPIC_FIELDS)

    def __init__(
        self,
        snapshots: Sequence[MarketSnapshot],
        client: ClientProfile,
        risk_budget: RiskBudget,
        condition: BenchmarkCondition,
        *,
        max_quotes_per_round: int = 3,
        max_client_queries_per_round: int = 3,
        max_consults_per_round: int = 3,
        quote_policy: QuotePolicy | None = None,
        domain: ProductDomainSpec | None = None,
        env_agents: "dict[str, FrozenEnvAgent] | None" = None,
    ):
        if not snapshots:
            raise BenchmarkError("an episode needs at least one market snapshot")
        episode_ids = {item.episode_id for item in snapshots}
        if len(episode_ids) != 1:
            raise BenchmarkError("one environment instance must contain exactly one episode")
        self.snapshots = sorted(snapshots, key=lambda s: s.round_num)
        expected = list(range(1, len(self.snapshots) + 1))
        if [s.round_num for s in self.snapshots] != expected:
            raise BenchmarkError("episode rounds must be contiguous from one")
        # Resolve every configured round up front so a malformed schedule fails
        # before an episode starts rather than halfway through a benchmark job.
        for item in self.snapshots:
            resolve_client_profile(client, item.round_num)
        self.base_client = client
        self.client = resolve_client_profile(client, self.snapshots[0].round_num)
        self.condition = condition
        self.portfolio = PortfolioState()
        self.client_memory = ClientMemory()
        self.quote_policy = quote_policy or QuotePolicy()
        self.domain = domain or ProductDomainSpec()
        self.desk = TradingDesk(
            HardConstraintEngine(risk_budget), policy=self.quote_policy, domain=self.domain
        )
        self.max_quotes_per_round = max_quotes_per_round
        self.max_client_queries_per_round = max_client_queries_per_round
        self.max_consults_per_round = max_consults_per_round
        # None -> pure deterministic fallback: every existing byte-for-byte
        # behaviour is unchanged and no LLM env role is ever consulted.
        self.env_agents = env_agents
        self.round_index = 0
        self.quote_count = 0
        self.query_count = 0
        self.consult_count = 0
        self._turn_id = 0
        self.quotes: dict[str, tuple[Quote, ProductSpec]] = {}
        self.submitted = False

    @property
    def snapshot(self) -> MarketSnapshot:
        return self.snapshots[self.round_index]

    @property
    def state_version(self) -> str:
        raw = (
            f"{self.snapshot.episode_id}|{self.snapshot.round_num}|"
            f"{self.portfolio.revision}|{self.condition.id}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def get_round_brief(self) -> dict:
        brief = self.snapshot.public_brief()
        # Routing identity is public protocol metadata, not a hidden preference.
        # ProductSpec.target_client is mandatory, so withholding this value made
        # otherwise valid Partial actions impossible to construct from the brief.
        brief["client_id"] = self.client.id
        brief["condition"] = self.condition.id
        brief["quote_budget"] = self.max_quotes_per_round - self.quote_count
        if self.condition.full_information:
            brief["client_constraints"] = self._full_client_payload()
        if self.condition.dynamic:
            totals = self.portfolio.totals()
            brief["portfolio_summary"] = {
                "outstanding_products": len(self.portfolio.positions),
                **totals,
            }
            brief["client_history"] = asdict(self.client_memory)
        return brief

    def _full_client_payload(self) -> dict:
        return {
            "capital": self.client.capital,
            "max_loss_pct": self.client.max_loss_pct,
            "min_return_pct": self.client.min_return_pct,
            "min_hit_prob": self.client.min_hit_prob,
            "risk_appetite": self.client.risk_appetite,
            "max_maturity_months": self.client.max_maturity_months,
            "principal_protection_required": self.client.principal_protection_required,
            "allowed_product_types": self.client.allowed_product_types,
            "accepting_new_products": self.client.accepting_new_products,
            "preferences": self.client.preferences,
            "current_focus": self.client.current_focus,
        }

    def query_client(self, topic: str) -> dict:
        if self.condition.full_information:
            return {"topic": topic, "answer": "already disclosed in the full client profile"}
        if topic not in self.CLIENT_TOPICS:
            raise BenchmarkError(f"unknown client topic: {topic}")
        if self.query_count >= self.max_client_queries_per_round:
            raise BenchmarkError("client query budget exhausted")
        self.query_count += 1
        profile, _ = self._client_profile_facts(self.CLIENT_TOPIC_FIELDS[topic])
        answer = next(iter(profile.values())) if len(profile) == 1 else profile
        return {"topic": topic, "answer": answer,
                "queries_left": self.max_client_queries_per_round - self.query_count}

    def _visible_check(self, check: CheckResult) -> CheckResult:
        """Return a condition-safe view of one deterministic check."""
        if self.condition.full_information or check.check_id not in _PRIVATE_CLIENT_CHECK_IDS:
            return check
        return replace(
            check,
            observed=None,
            limit=None,
            reason="client mandate value hidden; query the relevant client topic",
        )

    def public_quote_payload(self, quote: Quote) -> dict:
        """Condition-aware quote payload used at every policy-facing boundary."""
        payload = quote.public_payload()
        payload["checks"] = [asdict(self._visible_check(check)) for check in quote.checks]
        return payload

    def request_quote(self, product: ProductSpec) -> dict:
        if self.submitted:
            raise BenchmarkError("the round is already submitted")
        if self.quote_count >= self.max_quotes_per_round:
            raise BenchmarkError("quote budget exhausted")
        forged_state = (
            product.reference_spot is not None
            or product.barrier_touched
            or product.knock_in_active
            or product.elapsed_months != 0
        )
        if forged_state:
            raise BenchmarkError(
                "new quote requests cannot set environment-maintained lifecycle state"
            )
        self.quote_count += 1
        # ProductSpec is mutable for v2 compatibility.  Bind the quote to an
        # immutable-by-convention value snapshot so a caller cannot mutate the
        # original object between request_quote and submit_design (TOCTOU).
        quoted_product = replace(product)
        quote = self.desk.quote(
            quoted_product, self.snapshot, self.client, self.portfolio,
            self.state_version, self.quote_count,
        )
        self.quotes[quote.quote_id] = (quote, quoted_product)
        return self.public_quote_payload(quote)

    def submit_design(self, quote_id: str, explanation: str = "") -> dict:
        if self.submitted:
            raise BenchmarkError("the round is already submitted")
        if quote_id not in self.quotes:
            raise BenchmarkError("unknown quote_id")
        quote, product = self.quotes[quote_id]
        if quote.state_version != self.state_version:
            raise BenchmarkError("quote expired because the environment state changed")
        self.submitted = True
        hard_executable = quote.hard_pass
        contract_ok, contract_checks = client_contract_pass(product, quote.pricing, self.client)
        accepted = hard_executable and contract_ok
        if accepted:
            if self.condition.dynamic:
                fixing = self.snapshot.spot
                direction = _barrier_direction(product)
                issued_product = replace(
                    product,
                    barrier_direction=direction,
                    reference_spot=fixing,
                    barrier_touched=False,
                    knock_in_active=False,
                    elapsed_months=0,
                )
                self.portfolio.add(Position(
                    position_id=f"P-{quote.quote_id[2:]}",
                    product=issued_product,
                    remaining_months=product.maturity_months,
                    delta_dollars=quote.delta_dollars,
                    vega_dollars=quote.vega_dollars,
                    stress_loss=quote.stress_loss,
                    quote_margin=quote.dealer_margin,
                    trade_date=self.snapshot.as_of,
                    initial_fixing=fixing,
                    absolute_strike=product.strike_pct * fixing,
                    absolute_barrier=(
                        product.barrier_pct * fixing if product.barrier_pct is not None else None
                    ),
                    barrier_direction=direction,
                    running_min=fixing,
                    running_max=fixing,
                    current_fair_value=quote.fair_value,
                    last_valuation_round=self.snapshot.round_num,
                ))
            self.client_memory.accepted_products += 1
            self.client_memory.trust = min(1.0, self.client_memory.trust + 0.05)
        else:
            self.client_memory.rejected_products += 1
            self.client_memory.trust = max(0.0, self.client_memory.trust - 0.05)
        return {
            "accepted": accepted,
            "hard_executable": hard_executable,
            "client_contract_pass": contract_ok,
            "quote_id": quote_id,
            "hard_failures": [
                asdict(self._visible_check(c)) for c in quote.checks if not c.passed
            ],
            "contract_failures": [
                asdict(self._visible_check(c)) for c in contract_checks if not c.passed
            ],
            "dealer_margin": quote.dealer_margin if accepted else 0.0,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    # v2 LLM env-role dialogue layer. All of these are no-ops / errors when
    # env_agents is None, and none of them ever mutate the primary settlement.
    # ------------------------------------------------------------------

    def has_client_llm(self) -> bool:
        """Partial-observability rounds route query_client through the client LLM."""
        return (
            self.env_agents is not None
            and "client" in self.env_agents
            and not self.condition.full_information
        )

    def _consult_deterministic(self, role: str, draft: dict):
        """Build the deterministic block + grounding facts for a drafted consult.

        The consultative quote is priced but never stored: it does not consume
        the request_quote budget, never enters the archive, and cannot be used
        for submit (state binding still requires a real request_quote).
        """
        from .env_agents import FormalFacts
        from .products import ProductError, parse_product_draft

        try:
            product = parse_product_draft(draft)
            quote = self.desk.quote(
                product, self.snapshot, self.client, self.portfolio,
                self.state_version, -(self.consult_count + 1),
            )
        except (ProductError, BenchmarkError, ValueError, TypeError):
            return None, FormalFacts(state_version=self.state_version)

        if role == "trading_desk":
            block = self.public_quote_payload(quote)
            visible_checks = tuple(self._visible_check(c) for c in quote.checks)
            facts = FormalFacts(
                state_version=self.state_version,
                quote={"quote_id": quote.quote_id},
                checks=visible_checks,
                allowed_numeric_strings=_collect_numeric_strings(block),
            )
        else:  # risk_control: PASS/FAIL subset only, never any price
            visible_checks = tuple(self._visible_check(c) for c in quote.checks)
            checks_block = [
                {"check_id": c.check_id, "status": c.status,
                 "observed": c.observed, "limit": c.limit, "severity": c.severity}
                for c in visible_checks
            ]
            block = {"checks": checks_block, "hard_pass": quote.hard_pass}
            facts = FormalFacts(
                state_version=self.state_version,
                checks=visible_checks,
                allowed_numeric_strings=_collect_numeric_strings(checks_block),
            )
        return block, facts

    async def consult(self, role: str, message: str, draft: dict | None = None) -> dict:
        """Qualitative consult with an env role; never raises.

        With a valid ``draft`` and role in {trading_desk, risk_control} a
        deterministic block (consultative quote payload / hard-check subset) is
        attached and its numbers become the only ones the role may echo. A
        degraded role returns a fixed fallback string; an exhausted budget or a
        missing role returns an error dict.
        """
        from .env_agents import FormalFacts, RoleRequest

        if self.env_agents is None or role not in self.env_agents:
            return {"role": role, "error": f"no LLM agent configured for role {role!r}",
                    "reply": _consult_fallback(role), "degraded": True}
        if self.consult_count >= self.max_consults_per_round:
            return {"role": role, "error": "consult budget exhausted",
                    "reply": _consult_fallback(role), "degraded": True}

        block = None
        facts: "FormalFacts" = FormalFacts(state_version=self.state_version)
        if draft is not None and role in ("trading_desk", "risk_control"):
            block, facts = self._consult_deterministic(role, draft)

        self.consult_count += 1
        self._turn_id += 1
        payload: dict = {"message": message}
        if block is not None:
            payload["deterministic_block"] = block
        request = RoleRequest(
            kind=f"{role}_review" if role != "client" else "client_query",
            episode_id=self.snapshot.episode_id,
            round_num=self.snapshot.round_num,
            turn_id=self._turn_id,
            payload=payload,
            state_version=self.state_version,
            recipient=role,
        )
        resp = await self.env_agents[role].respond(request, facts=facts)
        return {
            "role": role,
            "action": resp.action,
            "reply": _consult_fallback(role) if resp.degraded else resp.narrative,
            "deterministic_block": block,
            "cache_hit": resp.cache_hit,
            "degraded": resp.degraded,
        }

    def _client_profile_facts(self, fields: Iterable[str] | None = None):
        """Full hidden client profile injected to the client LLM as grounding facts."""
        from .env_agents import FormalFacts

        profile = {
            **self._full_client_payload(),
            "accepting_new_products": self.client.accepting_new_products,
            "min_hit_prob": self.client.min_hit_prob,
            "min_return_pct": self.client.min_return_pct,
            "risk_appetite": self.client.risk_appetite,
            "current_focus": self.client.current_focus,
        }
        if fields is not None:
            selected = tuple(fields)
            profile = {key: profile[key] for key in selected}
        facts = FormalFacts(
            state_version=self.state_version,
            fact_ids=tuple(profile.keys()),
            public_client_fields=profile,
            allowed_numeric_strings=_collect_numeric_strings(profile),
        )
        return profile, facts

    async def query_client_llm(self, topic: str) -> dict:
        """Partial-observability client query routed through the client LLM.

        The numeric truth comes from the hidden ClientProfile (injected as
        grounding facts); the LLM only decides disclosure and wording. Budget
        and topic gating mirror the deterministic ``query_client``.
        """
        from .env_agents import RoleRequest

        if topic not in self.CLIENT_TOPICS:
            raise BenchmarkError(f"unknown client topic: {topic}")
        if self.query_count >= self.max_client_queries_per_round:
            raise BenchmarkError("client query budget exhausted")
        self.query_count += 1
        profile, facts = self._client_profile_facts(self.CLIENT_TOPIC_FIELDS[topic])
        self._turn_id += 1
        queries_left = self.max_client_queries_per_round - self.query_count
        request = RoleRequest(
            kind="client_query",
            episode_id=self.snapshot.episode_id,
            round_num=self.snapshot.round_num,
            turn_id=self._turn_id,
            payload={"topic": topic, "client_profile": profile, "queries_left": queries_left},
            state_version=self.state_version,
            recipient="client",
        )
        resp = await self.env_agents["client"].respond(request, facts=facts)
        if resp.degraded:
            return {"topic": topic, "reply": _consult_fallback("client"),
                    "degraded": True, "queries_left": queries_left}
        return {
            "topic": topic,
            "action": resp.action,
            "reply": resp.narrative,
            "disclosed_fields": list(resp.cited_fact_ids),
            "queries_left": queries_left,
            "cache_hit": resp.cache_hit,
            "degraded": False,
        }

    async def _role_review(self, role: str, quote: "Quote", product: ProductSpec, explanation: str):
        """One review request to an env role for a final submission; None if absent."""
        from .env_agents import FormalFacts, RoleRequest

        if self.env_agents is None or role not in self.env_agents:
            return None
        payload_public = quote.public_payload()
        if role == "trading_desk":
            payload = {"product": asdict(product), "quote": payload_public}
            facts = FormalFacts(
                state_version=self.state_version,
                quote={"quote_id": quote.quote_id},
                checks=quote.checks,
                allowed_numeric_strings=_collect_numeric_strings(payload_public),
            )
            kind = "desk_review"
        elif role == "risk_control":
            checks_block = [
                {"check_id": c.check_id, "status": c.status,
                 "observed": c.observed, "limit": c.limit, "severity": c.severity}
                for c in quote.checks
            ]
            payload = {"product": asdict(product), "checks": checks_block,
                       "hard_pass": quote.hard_pass}
            facts = FormalFacts(
                state_version=self.state_version,
                checks=quote.checks,
                allowed_numeric_strings=_collect_numeric_strings(checks_block),
            )
            kind = "risk_review"
        else:  # client
            profile, cfacts = self._client_profile_facts()
            decision_view = {
                "product": asdict(product),
                "client_price": payload_public.get("client_price"),
                "client_profile": profile,
                "explanation": explanation,
            }
            facts = FormalFacts(
                state_version=self.state_version,
                fact_ids=cfacts.fact_ids,
                public_client_fields=profile,
                allowed_numeric_strings=cfacts.allowed_numeric_strings
                + _collect_numeric_strings(payload_public.get("client_price")),
            )
            payload = decision_view
            kind = "client_decision"
        self._turn_id += 1
        request = RoleRequest(
            kind=kind,
            episode_id=self.snapshot.episode_id,
            round_num=self.snapshot.round_num,
            turn_id=self._turn_id,
            payload=payload,
            state_version=self.state_version,
            recipient=role,
        )
        return await self.env_agents[role].respond(request, facts=facts)

    async def workflow_review(self, quote_id: str, explanation: str = "") -> "WorkflowOutcome | None":
        """Gather desk/risk/client decisions on a submission into a WorkflowOutcome.

        Second-layer only: it reads the already-settled quote and never mutates
        the portfolio or client memory. Returns None if no env roles are wired
        or the quote is unknown.
        """
        if self.env_agents is None:
            return None
        entry = self.quotes.get(quote_id)
        if entry is None:
            return None
        quote, product = entry
        desk = await self._role_review("trading_desk", quote, product, explanation)
        risk = await self._role_review("risk_control", quote, product, explanation)
        client = await self._role_review("client", quote, product, explanation)
        return settle_submission(quote.hard_pass, desk, risk, client)

    def advance_round(self) -> list[str]:
        if self.round_index >= len(self.snapshots) - 1:
            raise BenchmarkError("episode is complete")
        matured: list[str] = []
        next_snapshot = self.snapshots[self.round_index + 1]
        if self.condition.dynamic:
            matured = self.portfolio.advance_month(next_snapshot)
        else:
            self.portfolio.reset()
            self.client_memory = ClientMemory()
        self.round_index += 1
        self.client = resolve_client_profile(self.base_client, self.snapshot.round_num)
        self.quote_count = 0
        self.query_count = 0
        self.consult_count = 0
        self.quotes.clear()
        self.submitted = False
        return matured


@dataclass
class CandidateRecord:
    product: ProductSpec
    quote: Quote


class CandidateArchive:
    """Lightweight method component used by the benchmark intervention."""

    def __init__(self):
        self.records: list[CandidateRecord] = []

    def add(self, product: ProductSpec, quote: Quote) -> None:
        self.records.append(CandidateRecord(replace(product), quote))

    def best_feasible(self) -> CandidateRecord | None:
        feasible = [record for record in self.records if record.quote.hard_pass]
        return max(feasible, key=lambda record: record.quote.dealer_margin, default=None)

    def prompt_summary(self, limit: int = 5) -> str:
        rows = sorted(self.records, key=lambda r: r.quote.dealer_margin, reverse=True)[:limit]
        return json.dumps([
            {
                "product_type": r.product.product_type,
                "maturity_months": r.product.maturity_months,
                "hard_pass": r.quote.hard_pass,
                "dealer_margin": r.quote.dealer_margin,
                "failed_checks": [c.check_id for c in r.quote.checks if not c.passed],
            }
            for r in rows
        ], ensure_ascii=False)


def constraint_ledger(quote: Quote, portfolio: PortfolioState, budget: RiskBudget) -> dict:
    """Serializable explicit memory intervention for the tested agent."""
    totals = portfolio.totals()
    return {
        "remaining_notional": budget.notional - totals["notional"],
        "remaining_net_delta": budget.net_delta - abs(totals["net_delta"]),
        "remaining_gross_delta": budget.gross_delta - totals["gross_delta"],
        "remaining_net_vega": budget.net_vega - abs(totals["net_vega"]),
        "remaining_stress_loss": budget.stress_loss - totals["stress_loss"],
        "last_failed_checks": [item.check_id for item in quote.checks if not item.passed],
    }


def oracle_candidate_grid(
    client: ClientProfile, domain: ProductDomainSpec | None = None
) -> list[ProductSpec]:
    """Frozen finite oracle candidate list, materialised from the shared lattice.

    This is exactly ``enumerate_domain``: the oracle and the agent share one
    domain, so any quotable agent action is an oracle candidate.
    """
    return list(enumerate_domain(client, domain or ProductDomainSpec()))


def oracle_best_quote(
    domain: ProductDomainSpec | Iterable[ProductSpec],
    snapshot: MarketSnapshot,
    client: ClientProfile,
    portfolio: PortfolioState,
    risk_budget: RiskBudget,
    state_version: str = "oracle-frozen-state",
    *,
    policy: QuotePolicy | None = None,
) -> tuple[ProductSpec, Quote] | None:
    """Deterministic one-step margin frontier after hard filtering, over the lattice.

    ``domain`` may be a ProductDomainSpec (enumerated internally) or, for
    backwards compatibility, an explicit iterable of ProductSpec candidates.
    mc_diagnostics caches by structure key, so enumeration shares MC across the
    notional ladder. oracle and agent use the same TradingDesk / domain, so
    one_step_attainment can never exceed 1.
    """
    if isinstance(domain, ProductDomainSpec):
        spec_domain = domain
        products: Iterable[ProductSpec] = enumerate_domain(client, domain)
    else:
        spec_domain = ProductDomainSpec()
        products = domain
    desk = TradingDesk(HardConstraintEngine(risk_budget), policy=policy, domain=spec_domain)
    feasible: list[tuple[ProductSpec, Quote]] = []
    for index, product in enumerate(products, start=1):
        quote = desk.quote(product, snapshot, client, portfolio, state_version, index)
        if quote.hard_pass:
            feasible.append((product, quote))
    return max(feasible, key=lambda item: item[1].dealer_margin, default=None)


def calibrate_risk_budget(
    products: Sequence[ProductSpec],
    development_cases: Sequence[tuple[MarketSnapshot, ClientProfile, PortfolioState]],
    base_budget: RiskBudget,
    *,
    target: tuple[float, float] = (0.20, 0.40),
    factors: Sequence[float] = tuple(value / 10 for value in range(1, 31)),
    policy: QuotePolicy | None = None,
) -> dict:
    """Choose a budget scale on development cases, never on test episodes.

    ``policy`` lets the caller pin a (possibly non-default) ``QuotePolicy`` --
    e.g. one loaded from ``--quote-policy-json`` -- so budget calibration uses
    the same margin economics as the run it is calibrating for; ``None`` keeps
    the previous behaviour (``TradingDesk`` default policy).
    """
    if not products or not development_cases:
        raise BenchmarkError("budget calibration needs products and development cases")
    if not (0 <= target[0] <= target[1] <= 1):
        raise BenchmarkError("invalid target feasibility interval")
    midpoint = sum(target) / 2.0
    rows: list[dict] = []
    for factor in factors:
        candidate_budget = base_budget.scaled(float(factor))
        desk = TradingDesk(HardConstraintEngine(candidate_budget), policy=policy)
        passed = 0
        total = 0
        for case_index, (snapshot, client, portfolio) in enumerate(development_cases):
            for product_index, product in enumerate(products):
                quote = desk.quote(
                    product, snapshot, client, portfolio,
                    f"calibration-{case_index}", product_index + 1,
                )
                passed += int(quote.hard_pass)
                total += 1
        rate = passed / total
        rows.append({"factor": float(factor), "feasibility_rate": rate, "passed": passed, "total": total})
    in_band = [row for row in rows if target[0] <= row["feasibility_rate"] <= target[1]]
    pool = in_band or rows
    selected = min(pool, key=lambda row: (abs(row["feasibility_rate"] - midpoint), row["factor"]))
    return {
        "selected_factor": selected["factor"],
        "selected_budget": asdict(base_budget.scaled(selected["factor"])),
        "selected_feasibility_rate": selected["feasibility_rate"],
        "target": list(target),
        "within_target": selected in in_band,
        "grid": rows,
        "warning": "freeze this result before evaluating held-out episodes",
    }


def carry_sensitivity(
    product: ProductSpec,
    snapshot: MarketSnapshot,
    client: ClientProfile,
    portfolio: PortfolioState,
    risk_budget: RiskBudget,
    *,
    shock_bp: float = 25.0,
) -> dict:
    """Requote after +/- carry shocks and report FV changes as notional fractions."""
    shock = shock_bp / 10_000.0
    desk = TradingDesk(HardConstraintEngine(risk_budget))
    rows = []
    for label, amount in (("down", -shock), ("base", 0.0), ("up", shock)):
        shocked = replace(snapshot, carry_rate=snapshot.carry_rate + amount)
        quote = desk.quote(product, shocked, client, portfolio, f"carry-{label}", len(rows) + 1)
        rows.append({"scenario": label, "carry_rate": shocked.carry_rate, "fair_value": quote.fair_value})
    base = rows[1]["fair_value"]
    max_change = max(abs(row["fair_value"] - base) for row in rows)
    return {
        "shock_bp": shock_bp,
        "rows": rows,
        "max_fv_change": max_change,
        "max_fv_change_pct_notional": max_change / product.notional,
        "below_0_5pct_notional": max_change / product.notional < 0.005,
    }
