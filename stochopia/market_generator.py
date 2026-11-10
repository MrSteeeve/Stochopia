"""Strictly experimental offline synthetic market scenarios.

This module pre-generates an external, exogenous regime-switching GBM path and
freezes it as existing :class:`~stochopia.benchmark.MarketSnapshot` values plus a
separate privileged provenance artifact.  It is not action-conditioned, is not
a predictive world model, and is not benchmark evidence.  It deliberately has
no ``reset``/``step``, TaskSuite, CLI, evaluator, aggregate, customer, or product
wiring.

Every daily step is one verified exchange open session from an explicit
offline calendar artifact (``<calendar_id>.json``).  No calendar or market-data
network access occurs at generator runtime.

The latent regime label is never copied into a public snapshot, and public
``trend_alpha`` is the stationary (unconditional) P drift, constant over the
episode, so no public field is a function of the current regime.  Q inputs are
selected from a frozen point-in-time Q bundle at the scenario start and remain
constant across the synthetic horizon; they are not emitted from the latent P
regime.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Real
from statistics import pstdev
from typing import Any, Mapping, Sequence

from .benchmark import MarketSnapshot
from .common import canonical_json
from .q_side_bundle import QScenarioInput
from .trading_calendar import FrozenTradingCalendar, TRANSITION_INTERVAL


SCHEMA_VERSION = "stochopia.synthetic-regime-scenario.v2"
GENERATOR_VERSION = "regime-switching-gbm-v1"

_REGIME_SEED_DOMAIN = "stochopia.seed.market.regime-transition.v2"
_INNOVATION_SEED_DOMAIN = "stochopia.seed.market.physical-gbm-innovation.v2"
_RNG_ALGORITHM = "python.random.Random(MT19937)+box-muller-v0"
_TRANSITION_INTERVAL = TRANSITION_INTERVAL
_RETURN_LOOKBACK_DAYS = 20
_REALIZED_VOLATILITY_LOOKBACK_DAYS = (20, 60)


__all__ = [
    "SCHEMA_VERSION",
    "RegimeParameters",
    "RegimeSwitchingGBMSpecV1",
    "ScenarioProvenance",
    "GeneratedMarketPath",
    "generate_regime_switching_gbm",
]


def _non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field_name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_float(value: object, field_name: str) -> float:
    result = _finite_float(value, field_name)
    if result <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _strict_int(value: object, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int (not bool)")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def _stationary_distribution(matrix: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Solve pi P = pi, sum(pi) = 1 (least squares picks one if not unique)."""

    import numpy as np  # research extra; this generator is research-only

    transition = np.asarray(matrix, dtype=float)
    size = transition.shape[0]
    system = np.vstack((transition.T - np.eye(size), np.ones(size)))
    target = np.zeros(size + 1)
    target[-1] = 1.0
    solution = np.linalg.lstsq(system, target, rcond=None)[0]
    return tuple(float(value) for value in solution)


def _monitoring_grid(trading_days_per_year: int, drawdown_lookback_days: int) -> str:
    return (
        "exchange-open-session-daily;monthly-last-open-session;"
        f"regime-transition={_TRANSITION_INTERVAL};"
        f"return={_RETURN_LOOKBACK_DAYS}-intervals;"
        "realized-vol=20,60-log-return-intervals;"
        f"drawdown={drawdown_lookback_days}-levels;"
        "state-t-governs-return-into-t"
    )


def _normalize_optional_cutoff(value: object) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        raise TypeError("calibration_cutoff must be a date without a time")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("calibration_cutoff must be non-empty when provided")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("calibration_cutoff must be an ISO date") from exc
    raise TypeError("calibration_cutoff must be a date, ISO date string, or None")


@dataclass(frozen=True)
class RegimeParameters:
    """Explicit P-path parameters for one latent regime."""

    name: str
    physical_annual_return: float
    physical_annual_volatility: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _non_empty_string(self.name, "RegimeParameters.name"))
        object.__setattr__(
            self,
            "physical_annual_return",
            _finite_float(
                self.physical_annual_return,
                f"regime {self.name!r} physical_annual_return",
            ),
        )
        for field_name in ("physical_annual_volatility",):
            normalized_value = _positive_float(
                getattr(self, field_name),
                f"regime {self.name!r} {field_name}",
            )
            if not math.isfinite(normalized_value * normalized_value):
                raise ValueError(
                    f"regime {self.name!r} {field_name} is too large for "
                    "finite variance calculations"
                )
            object.__setattr__(
                self,
                field_name,
                normalized_value,
            )

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physical_annual_return": self.physical_annual_return,
            "physical_annual_volatility": self.physical_annual_volatility,
        }


@dataclass(frozen=True)
class RegimeSwitchingGBMSpecV1:
    """Fully explicit P parameters and exchange clock; Q is a separate input."""

    regimes: tuple[RegimeParameters, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    initial_distribution: tuple[float, ...]
    calendar: FrozenTradingCalendar
    p_parameter_source: str
    p_calibration_cutoff: date | str | None = None
    trading_days_per_year: int = 252

    def __post_init__(self) -> None:
        try:
            regimes = tuple(self.regimes)
        except TypeError as exc:
            raise TypeError("regimes must be an iterable of RegimeParameters") from exc
        object.__setattr__(self, "regimes", regimes)
        if len(regimes) < 2:
            raise ValueError("at least two regimes are required")
        if not all(isinstance(regime, RegimeParameters) for regime in regimes):
            raise TypeError("every regime must be a RegimeParameters instance")
        names = tuple(regime.name for regime in regimes)
        if len(names) != len(set(names)):
            raise ValueError("regime names must be unique")

        try:
            matrix_rows = tuple(tuple(row) for row in self.transition_matrix)
        except TypeError as exc:
            raise TypeError("transition_matrix must be an iterable of rows") from exc
        if len(matrix_rows) != len(regimes) or any(
            len(row) != len(regimes) for row in matrix_rows
        ):
            raise ValueError("transition_matrix must be square and match the regime count")
        matrix = tuple(
            tuple(
                _finite_float(value, f"transition_matrix[{row_index}][{column_index}]")
                for column_index, value in enumerate(row)
            )
            for row_index, row in enumerate(matrix_rows)
        )
        for row_index, row in enumerate(matrix):
            if any(value < 0.0 for value in row):
                raise ValueError(f"transition_matrix row {row_index} must be non-negative")
            if not math.isclose(math.fsum(row), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"transition_matrix row {row_index} must sum to 1")
        object.__setattr__(self, "transition_matrix", matrix)

        try:
            initial_values = tuple(self.initial_distribution)
        except TypeError as exc:
            raise TypeError("initial_distribution must be an iterable") from exc
        if len(initial_values) != len(regimes):
            raise ValueError("initial_distribution must match the regime count")
        initial_distribution = tuple(
            _finite_float(value, f"initial_distribution[{index}]")
            for index, value in enumerate(initial_values)
        )
        if any(value < 0.0 for value in initial_distribution):
            raise ValueError("initial_distribution must be non-negative")
        if not math.isclose(
            math.fsum(initial_distribution),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("initial_distribution must sum to 1")
        object.__setattr__(self, "initial_distribution", initial_distribution)

        if not isinstance(self.calendar, FrozenTradingCalendar):
            raise TypeError("calendar must be a FrozenTradingCalendar artifact")
        object.__setattr__(
            self,
            "p_parameter_source",
            _non_empty_string(self.p_parameter_source, "p_parameter_source"),
        )
        object.__setattr__(
            self,
            "p_calibration_cutoff",
            _normalize_optional_cutoff(self.p_calibration_cutoff),
        )
        object.__setattr__(
            self,
            "trading_days_per_year",
            _strict_int(
                self.trading_days_per_year,
                "trading_days_per_year",
                minimum=1,
            ),
        )
    @property
    def manifest(self) -> dict[str, Any]:
        cutoff = self.p_calibration_cutoff
        return {
            "schema": SCHEMA_VERSION,
            "transition_interval": _TRANSITION_INTERVAL,
            "transition_unit": "one transition per exchange open session",
            "regimes": [regime.manifest for regime in self.regimes],
            "transition_matrix": [list(row) for row in self.transition_matrix],
            "initial_distribution": list(self.initial_distribution),
            "calendar": self.calendar.manifest,
            "p_parameter_source": self.p_parameter_source,
            "p_calibration_cutoff": cutoff.isoformat() if cutoff is not None else None,
            "trading_days_per_year": self.trading_days_per_year,
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.manifest)


def _spec_from_manifest(payload: object) -> RegimeSwitchingGBMSpecV1:
    """Rebuild and fully validate a canonical spec embedded in provenance."""

    if not isinstance(payload, dict):
        raise ValueError("normalized spec must be a JSON object")
    if payload.get("schema") == "stochopia.synthetic-regime-scenario.v1" or payload.get(
        "calendar_id"
    ) == "weekday-only-v0":
        raise ValueError("weekday-only-v0 specs require explicit migration; automatic reinterpretation is forbidden")
    try:
        if payload.get("transition_interval") != _TRANSITION_INTERVAL:
            raise ValueError("unexpected transition interval")
        if payload.get("transition_unit") != "one transition per exchange open session":
            raise ValueError("unexpected transition unit")
        regimes_payload = payload["regimes"]
        if not isinstance(regimes_payload, list):
            raise TypeError("regimes must be a list")
        spec = RegimeSwitchingGBMSpecV1(
            regimes=tuple(RegimeParameters(**item) for item in regimes_payload),
            transition_matrix=tuple(
                tuple(row) for row in payload["transition_matrix"]
            ),
            initial_distribution=tuple(payload["initial_distribution"]),
            calendar=FrozenTradingCalendar.from_manifest(payload["calendar"]),
            p_parameter_source=payload["p_parameter_source"],
            p_calibration_cutoff=payload["p_calibration_cutoff"],
            trading_days_per_year=payload["trading_days_per_year"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("normalized spec does not match the V1 exchange-calendar schema") from exc
    return spec


@dataclass(frozen=True)
class ScenarioProvenance:
    """What is needed to regenerate one path: spec, seeds, Q input and horizon."""

    schema: str
    generator_version: str
    spec: Mapping[str, Any]
    root_seed: int
    regime_seed: int
    innovation_seed: int
    rng_algorithm: str
    calendar_id: str
    monitoring_grid: str
    trading_days_per_year: int
    return_lookback_days: int
    realized_volatility_lookback_days: tuple[int, ...]
    drawdown_lookback_days: int
    p_parameter_source: str
    p_calibration_cutoff: str | None
    q_selection: Mapping[str, Any]
    q_cutoff: str
    q_observation_date: str
    q_dynamics: str
    episode_id: str
    underlying: str
    start_date: str
    terminal_date: str
    start_spot: float
    decision_rounds: int

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"ScenarioProvenance.schema must be {SCHEMA_VERSION!r}")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError(
                f"ScenarioProvenance.generator_version must be {GENERATOR_VERSION!r}"
            )
        object.__setattr__(self, "spec", self.normalized_spec.manifest)
        object.__setattr__(self, "q_selection", self.q_input.manifest)
        object.__setattr__(
            self,
            "realized_volatility_lookback_days",
            tuple(self.realized_volatility_lookback_days),
        )
        object.__setattr__(self, "start_spot", _positive_float(self.start_spot, "start_spot"))
        _strict_int(self.decision_rounds, "decision_rounds", minimum=1)

    @property
    def normalized_spec(self) -> RegimeSwitchingGBMSpecV1:
        return _spec_from_manifest(dict(self.spec))

    @property
    def q_input(self) -> QScenarioInput:
        return QScenarioInput.from_manifest(dict(self.q_selection))

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generator_version": self.generator_version,
            "spec": dict(self.spec),
            "root_seed": self.root_seed,
            "regime_seed": self.regime_seed,
            "innovation_seed": self.innovation_seed,
            "rng_algorithm": self.rng_algorithm,
            "calendar_id": self.calendar_id,
            "monitoring_grid": self.monitoring_grid,
            "trading_days_per_year": self.trading_days_per_year,
            "return_lookback_days": self.return_lookback_days,
            "realized_volatility_lookback_days": list(self.realized_volatility_lookback_days),
            "drawdown_lookback_days": self.drawdown_lookback_days,
            "p_parameter_source": self.p_parameter_source,
            "p_calibration_cutoff": self.p_calibration_cutoff,
            "q_selection": dict(self.q_selection),
            "q_cutoff": self.q_cutoff,
            "q_observation_date": self.q_observation_date,
            "q_dynamics": self.q_dynamics,
            "episode_id": self.episode_id,
            "underlying": self.underlying,
            "start_date": self.start_date,
            "terminal_date": self.terminal_date,
            "start_spot": self.start_spot,
            "decision_rounds": self.decision_rounds,
        }


def _snapshot_manifest(snapshot: MarketSnapshot) -> dict[str, Any]:
    return {
        "episode_id": snapshot.episode_id,
        "round_num": snapshot.round_num,
        "as_of": snapshot.as_of.isoformat(),
        "underlying": snapshot.underlying,
        "spot": snapshot.spot,
        "risk_free_rate": snapshot.risk_free_rate,
        "return_20d": snapshot.return_20d,
        "realized_vol_20d": snapshot.realized_vol_20d,
        "realized_vol_60d": snapshot.realized_vol_60d,
        "drawdown_6m": snapshot.drawdown_6m,
        "atm_iv_1m": snapshot.atm_iv_1m,
        "atm_iv_3m": snapshot.atm_iv_3m,
        "atm_iv_6m": snapshot.atm_iv_6m,
        "carry_rate": snapshot.carry_rate,
        "regime": snapshot.regime,
        "source": snapshot.source,
        "trend_alpha": snapshot.trend_alpha,
    }


def _daily_path_payload(
    daily_dates: Sequence[date],
    daily_spots: Sequence[float],
    latent_regime_path: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "kind": "stochopia.market.privileged-daily-path.v2",
        "daily_dates": [item.isoformat() for item in daily_dates],
        "daily_spots": list(daily_spots),
        "latent_regime_path": list(latent_regime_path),
    }


@dataclass(frozen=True)
class GeneratedMarketPath:
    """Frozen public snapshots and privileged daily path.

    ``latent_regime_path`` is daily and privileged.  It must not be copied into
    ``MarketSnapshot.regime`` or exposed as ordinary benchmark observation.
    """

    snapshots: tuple[MarketSnapshot, ...]
    daily_dates: tuple[date, ...]
    daily_spots: tuple[float, ...]
    latent_regime_path: tuple[str, ...]
    provenance: ScenarioProvenance

    def __post_init__(self) -> None:
        try:
            snapshots = tuple(self.snapshots)
            daily_dates = tuple(self.daily_dates)
            daily_spots_raw = tuple(self.daily_spots)
            latent_regime_path = tuple(self.latent_regime_path)
        except TypeError as exc:
            raise TypeError("GeneratedMarketPath sequence fields must be iterable") from exc
        if not isinstance(self.provenance, ScenarioProvenance):
            raise TypeError("provenance must be ScenarioProvenance")
        if not all(isinstance(snapshot, MarketSnapshot) for snapshot in snapshots):
            raise TypeError("snapshots must contain MarketSnapshot values")
        if not daily_dates:
            raise ValueError("daily_dates must be non-empty")
        if not (
            len(daily_dates) == len(daily_spots_raw) == len(latent_regime_path)
        ):
            raise ValueError(
                "daily_dates, daily_spots, and latent_regime_path must have equal length"
            )
        for item in daily_dates:
            if isinstance(item, datetime) or not isinstance(item, date):
                raise TypeError("daily_dates must contain date values without times")
        if any(right <= left for left, right in zip(daily_dates, daily_dates[1:])):
            raise ValueError("daily_dates must be strictly increasing")
        daily_spots = tuple(
            _positive_float(value, f"daily_spots[{index}]")
            for index, value in enumerate(daily_spots_raw)
        )
        for index, regime_name in enumerate(latent_regime_path):
            _non_empty_string(regime_name, f"latent_regime_path[{index}]")

        object.__setattr__(self, "snapshots", snapshots)
        object.__setattr__(self, "daily_dates", daily_dates)
        object.__setattr__(self, "daily_spots", daily_spots)
        object.__setattr__(self, "latent_regime_path", latent_regime_path)

        provenance = self.provenance
        regime_names = {regime["name"] for regime in provenance.spec["regimes"]}
        if any(name not in regime_names for name in latent_regime_path):
            raise ValueError("latent_regime_path contains a regime absent from the spec")
        if len(snapshots) != provenance.decision_rounds + 1:
            raise ValueError("snapshot count must equal decision_rounds + 1")
        if daily_dates[0].isoformat() != provenance.start_date:
            raise ValueError("daily path start date disagrees with provenance")
        if daily_dates[-1].isoformat() != provenance.terminal_date:
            raise ValueError("daily path terminal date disagrees with provenance")

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a fresh, JSON-compatible copy of the complete frozen artifact."""

        return {
            "schema": SCHEMA_VERSION,
            "kind": "stochopia.market.generated-path.v2",
            "snapshots": [_snapshot_manifest(snapshot) for snapshot in self.snapshots],
            "daily_path": _daily_path_payload(
                self.daily_dates,
                self.daily_spots,
                self.latent_regime_path,
            ),
            "provenance": self.provenance.manifest,
        }

def _next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _monthly_snapshot_dates(
    trading_calendar: FrozenTradingCalendar,
    start_date: date,
    decision_rounds: int,
) -> tuple[date, ...]:
    result = [start_date]
    year, month = start_date.year, start_date.month
    try:
        for _ in range(decision_rounds):
            year, month = _next_month(year, month)
            result.append(trading_calendar.month_end(year, month))
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            "decision_rounds extends beyond the frozen calendar coverage"
        ) from exc
    return tuple(result)


def _derive_domain_seed(root_seed: int, domain: str) -> int:
    preimage = {
        "schema": SCHEMA_VERSION,
        "root_seed": root_seed,
        "domain": domain,
    }
    return int.from_bytes(
        hashlib.sha256(canonical_json(preimage).encode("utf-8")).digest(),
        "big",
    )


def _sample_categorical(weights: Sequence[float], rng: random.Random) -> int:
    total = math.fsum(weights)
    target = rng.random() * total
    cumulative = 0.0
    last_positive = 0
    for index, weight in enumerate(weights):
        if weight > 0.0:
            last_positive = index
        cumulative += weight
        if target < cumulative:
            return index
    # Only reachable through floating-point accumulation at the upper boundary.
    return last_positive


def _standard_normal(rng: random.Random) -> float:
    """Box-Muller using exactly two innovation-domain MT19937 draws."""

    first_uniform = 1.0 - rng.random()  # (0, 1], so log is always defined.
    second_uniform = rng.random()
    return math.sqrt(-2.0 * math.log(first_uniform)) * math.cos(
        2.0 * math.pi * second_uniform
    )


def _trailing_return(spots: Sequence[float], lookback: int) -> float | None:
    if len(spots) < lookback + 1:
        return None
    result = spots[-1] / spots[-lookback - 1] - 1.0
    if not math.isfinite(result):
        raise ValueError("generated return is non-finite")
    return result


def _realized_volatility(
    spots: Sequence[float],
    lookback: int,
    trading_days_per_year: int,
) -> float | None:
    if len(spots) < lookback + 1:
        return None
    window = spots[-lookback - 1 :]
    log_returns = [
        math.log(window[index]) - math.log(window[index - 1])
        for index in range(1, len(window))
    ]
    result = pstdev(log_returns) * math.sqrt(float(trading_days_per_year))
    if not math.isfinite(result):
        raise ValueError("generated realized volatility is non-finite")
    return result


def _trailing_drawdown(spots: Sequence[float], lookback: int) -> float | None:
    if len(spots) < lookback:
        return None
    window = spots[-lookback:]
    result = window[-1] / max(window) - 1.0
    if not math.isfinite(result):
        raise ValueError("generated drawdown is non-finite")
    return result


def generate_regime_switching_gbm(
    spec: RegimeSwitchingGBMSpecV1,
    *,
    q_input: QScenarioInput,
    episode_id: str,
    underlying: str,
    start_date: date,
    start_spot: float,
    decision_rounds: int,
    seed: int,
) -> GeneratedMarketPath:
    """Pre-generate one frozen exogenous scenario and its provenance.

    The physical path uses only each regime's P return/volatility, the Markov
    probabilities, the calendar, and the innovation stream.  Risk-free rate,
    carry, and pricing IVs are Q snapshot inputs and cannot affect that path.
    The result is experimental synthetic stress input, not a forecast, an
    action-conditioned world model, or evidence that a benchmark is realistic.
    The same root seed deliberately reuses common random numbers across calls;
    callers must allocate distinct root seeds for statistically independent
    episodes.  Raw snapshots and privileged provenance are evaluator-side
    artifacts; a policy boundary must expose only ``MarketSnapshot.public_brief``.
    """

    if not isinstance(spec, RegimeSwitchingGBMSpecV1):
        raise TypeError("spec must be RegimeSwitchingGBMSpecV1")
    if not isinstance(q_input, QScenarioInput):
        raise TypeError("q_input must be a complete QScenarioInput")
    episode_id = _non_empty_string(episode_id, "episode_id")
    underlying = _non_empty_string(underlying, "underlying")
    if isinstance(start_date, datetime) or not isinstance(start_date, date):
        raise TypeError("start_date must be a date without a time")
    if start_date != spec.calendar.month_end(start_date.year, start_date.month):
        raise ValueError(
            "start_date must be the month's final exchange open session in the "
            "frozen calendar"
        )
    if q_input.snapshot.underlying != underlying:
        raise ValueError("Q input underlying must match the generated scenario")
    if q_input.snapshot.as_of != start_date:
        raise ValueError("Q input must be observed at the scenario start_date")
    start_spot = _positive_float(start_spot, "start_spot")
    decision_rounds = _strict_int(decision_rounds, "decision_rounds", minimum=1)
    seed = _strict_int(seed, "seed")

    snapshot_dates = _monthly_snapshot_dates(
        spec.calendar,
        start_date,
        decision_rounds,
    )
    daily_dates = spec.calendar.open_dates_between(start_date, snapshot_dates[-1])
    if len(daily_dates) < 2:
        raise AssertionError("a positive decision horizon must contain daily steps")

    regime_seed = _derive_domain_seed(seed, _REGIME_SEED_DOMAIN)
    innovation_seed = _derive_domain_seed(seed, _INNOVATION_SEED_DOMAIN)
    regime_rng = random.Random(regime_seed)
    innovation_rng = random.Random(innovation_seed)

    current_regime_index = _sample_categorical(spec.initial_distribution, regime_rng)
    regime_indices = [current_regime_index]
    daily_spots = [start_spot]
    step_size = 1.0 / float(spec.trading_days_per_year)
    root_step_size = math.sqrt(step_size)

    for as_of in daily_dates[1:]:
        current_regime_index = _sample_categorical(
            spec.transition_matrix[current_regime_index],
            regime_rng,
        )
        regime_indices.append(current_regime_index)
        regime = spec.regimes[current_regime_index]
        sigma = regime.physical_annual_volatility
        variance = sigma * sigma
        log_increment = (
            (regime.physical_annual_return - 0.5 * variance) * step_size
            + sigma * root_step_size * _standard_normal(innovation_rng)
        )
        if not math.isfinite(variance) or not math.isfinite(log_increment):
            raise ValueError(
                f"physical parameters produce a non-finite GBM increment at {as_of}"
            )
        next_log_spot = math.log(daily_spots[-1]) + log_increment
        try:
            next_spot = math.exp(next_log_spot)
        except OverflowError as exc:
            raise ValueError(f"physical path overflows at {as_of}") from exc
        if not math.isfinite(next_spot) or next_spot <= 0.0:
            raise ValueError(f"physical path is non-positive or non-finite at {as_of}")
        daily_spots.append(next_spot)

    latent_regime_path = tuple(
        spec.regimes[index].name for index in regime_indices
    )
    date_to_index = {as_of: index for index, as_of in enumerate(daily_dates)}
    # Episode id and seed stay in the privileged provenance, not the public source.
    source = f"synthetic;schema={SCHEMA_VERSION}"
    q_fields = q_input.snapshot.fields
    risk_free_rate = float(q_fields["risk_free_rate"].value)
    carry_rate = float(q_fields["carry_rate"].value)
    atm_iv_1m = float(q_fields["atm_iv_1m"].value)
    atm_iv_3m = float(q_fields["atm_iv_3m"].value)
    atm_iv_6m = float(q_fields["atm_iv_6m"].value)
    drawdown_lookback_days = max(1, spec.trading_days_per_year // 2)
    # Public snapshots must not reveal the latent regime, so trend_alpha uses
    # the stationary (unconditional) P drift and is constant over the episode.
    # Existing core code represents physical drift as r + alpha - q, hence
    # alpha = mu_P - r + q.
    stationary_drift = math.fsum(
        weight * regime.physical_annual_return
        for weight, regime in zip(
            _stationary_distribution(spec.transition_matrix), spec.regimes
        )
    )
    trend_alpha = stationary_drift - risk_free_rate + carry_rate
    snapshots: list[MarketSnapshot] = []

    for round_num, as_of in enumerate(snapshot_dates, start=1):
        daily_index = date_to_index[as_of]
        history = daily_spots[: daily_index + 1]
        snapshot = MarketSnapshot(
            episode_id=episode_id,
            round_num=round_num,
            as_of=as_of,
            underlying=underlying,
            spot=history[-1],
            risk_free_rate=risk_free_rate,
            return_20d=_trailing_return(history, _RETURN_LOOKBACK_DAYS),
            realized_vol_20d=_realized_volatility(
                history,
                _REALIZED_VOLATILITY_LOOKBACK_DAYS[0],
                spec.trading_days_per_year,
            ),
            realized_vol_60d=_realized_volatility(
                history,
                _REALIZED_VOLATILITY_LOOKBACK_DAYS[1],
                spec.trading_days_per_year,
            ),
            drawdown_6m=_trailing_drawdown(history, drawdown_lookback_days),
            atm_iv_1m=atm_iv_1m,
            atm_iv_3m=atm_iv_3m,
            atm_iv_6m=atm_iv_6m,
            carry_rate=carry_rate,
            regime="synthetic",
            source=source,
            trend_alpha=trend_alpha,
        )
        snapshot.pricing_volatility()
        snapshots.append(snapshot)

    snapshots_tuple = tuple(snapshots)
    daily_spots_tuple = tuple(daily_spots)
    cutoff = spec.p_calibration_cutoff
    provenance = ScenarioProvenance(
        schema=SCHEMA_VERSION,
        generator_version=GENERATOR_VERSION,
        spec=spec.manifest,
        root_seed=seed,
        regime_seed=regime_seed,
        innovation_seed=innovation_seed,
        rng_algorithm=_RNG_ALGORITHM,
        calendar_id=spec.calendar.calendar_id,
        monitoring_grid=_monitoring_grid(spec.trading_days_per_year, drawdown_lookback_days),
        trading_days_per_year=spec.trading_days_per_year,
        return_lookback_days=_RETURN_LOOKBACK_DAYS,
        realized_volatility_lookback_days=_REALIZED_VOLATILITY_LOOKBACK_DAYS,
        drawdown_lookback_days=drawdown_lookback_days,
        p_parameter_source=spec.p_parameter_source,
        p_calibration_cutoff=cutoff.isoformat() if cutoff is not None else None,
        q_selection=q_input.manifest,
        q_cutoff=q_input.q_cutoff.isoformat(),
        q_observation_date=q_input.snapshot.as_of.isoformat(),
        q_dynamics="frozen_at_scenario_start",
        episode_id=episode_id,
        underlying=underlying,
        start_date=start_date.isoformat(),
        terminal_date=snapshot_dates[-1].isoformat(),
        start_spot=start_spot,
        decision_rounds=decision_rounds,
    )
    return GeneratedMarketPath(
        snapshots=snapshots_tuple,
        daily_dates=daily_dates,
        daily_spots=daily_spots_tuple,
        latent_regime_path=latent_regime_path,
        provenance=provenance,
    )
