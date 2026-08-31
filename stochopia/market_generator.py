"""Strictly experimental offline synthetic market scenarios.

This module pre-generates an external, exogenous regime-switching GBM path and
freezes it as existing :class:`~stochopia.benchmark.MarketSnapshot` values plus a
separate privileged provenance artifact.  It is not action-conditioned, is not
a predictive world model, and is not benchmark evidence.  It deliberately has
no ``reset``/``step``, TaskSuite, CLI, evaluator, aggregate, customer, or product
wiring.

The calendar is explicitly ``weekday-only-v0``: Monday through Friday with no
exchange holidays.  In particular, the generated dates must not be described
as a Chinese exchange trading calendar.

The latent regime label is never copied into a public snapshot.  Its
regime-conditioned IV emission can nevertheless make the state inferable, so
V0 must not be used as evidence about hidden-state inference quality.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from numbers import Real
from pathlib import Path
from statistics import pstdev
from typing import Any, Sequence

from .benchmark import MarketSnapshot


SCHEMA_VERSION = "stochopia.synthetic-regime-scenario.v1"
GENERATOR_VERSION = "regime-switching-gbm-v0"

_CALENDAR_ID = "weekday-only-v0"
_REGIME_SEED_DOMAIN = "stochopia.seed.market.regime-transition.v1"
_INNOVATION_SEED_DOMAIN = "stochopia.seed.market.physical-gbm-innovation.v1"
_RNG_ALGORITHM = "python.random.Random(MT19937)+box-muller-v0"
_HASH_ALGORITHM = "sha256"
_TRANSITION_INTERVAL = "each-weekday-v0"
_RETURN_LOOKBACK_DAYS = 20
_REALIZED_VOLATILITY_LOOKBACK_DAYS = (20, 60)


__all__ = [
    "SCHEMA_VERSION",
    "RegimeParameters",
    "RegimeSwitchingGBMSpecV0",
    "ScenarioProvenance",
    "GeneratedMarketPath",
    "generate_regime_switching_gbm",
]


def _canonical_json(payload: Any) -> str:
    """Return the one canonical JSON encoding used by every V0 hash."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


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


def _trend_alpha_for_physical_drift(
    physical_annual_return: float,
    risk_free_rate: float,
    carry_rate: float,
) -> float:
    """Map P drift into the existing ``r + alpha - q`` representation safely."""

    trend_alpha = physical_annual_return - risk_free_rate + carry_rate
    if not math.isfinite(trend_alpha):
        raise ValueError("P/Q inputs produce a non-finite trend_alpha")
    reconstructed_return = risk_free_rate + trend_alpha - carry_rate
    tolerance = 1e-12 * max(1.0, abs(physical_annual_return))
    if not math.isfinite(reconstructed_return) or not math.isclose(
        reconstructed_return,
        physical_annual_return,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        raise ValueError(
            "P/Q inputs cannot represent physical_annual_return without material "
            "floating-point cancellation"
        )
    return trend_alpha


def _monitoring_grid(trading_days_per_year: int, drawdown_lookback_days: int) -> str:
    return (
        "weekday-only-daily;monthly-last-weekday;"
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


def _require_sha256(value: object, field_name: str) -> str:
    text = _non_empty_string(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
    return text


@dataclass(frozen=True)
class RegimeParameters:
    """Explicit P path parameters and Q snapshot inputs for one latent regime."""

    name: str
    physical_annual_return: float
    physical_annual_volatility: float
    pricing_iv_1m: float
    pricing_iv_3m: float
    pricing_iv_6m: float

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
        for field_name in (
            "physical_annual_volatility",
            "pricing_iv_1m",
            "pricing_iv_3m",
            "pricing_iv_6m",
        ):
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

        # Necessary ATM calendar sanity only; this is not a full surface
        # no-arbitrage guarantee.  Total variance must not decrease with tenor.
        total_variances = (
            self.pricing_iv_1m * self.pricing_iv_1m,
            3.0 * self.pricing_iv_3m * self.pricing_iv_3m,
            6.0 * self.pricing_iv_6m * self.pricing_iv_6m,
        )
        tolerance = 1e-12 * max(1.0, *total_variances)
        if any(
            right + tolerance < left
            for left, right in zip(total_variances, total_variances[1:])
        ):
            raise ValueError(
                f"regime {self.name!r} pricing IVs imply decreasing ATM total "
                "variance across 1m/3m/6m"
            )

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "physical_annual_return": self.physical_annual_return,
            "physical_annual_volatility": self.physical_annual_volatility,
            "pricing_iv_1m": self.pricing_iv_1m,
            "pricing_iv_3m": self.pricing_iv_3m,
            "pricing_iv_6m": self.pricing_iv_6m,
        }


@dataclass(frozen=True)
class RegimeSwitchingGBMSpecV0:
    """Fully explicit V0 parameters; no defaults, fitting, or calibration logic."""

    regimes: tuple[RegimeParameters, ...]
    transition_matrix: tuple[tuple[float, ...], ...]
    initial_distribution: tuple[float, ...]
    risk_free_rate: float
    carry_rate: float
    parameter_source: str
    calibration_cutoff: date | str | None = None
    calibration_data_hash: str | None = None
    trading_days_per_year: int = 252
    calendar_id: str = _CALENDAR_ID

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

        object.__setattr__(
            self,
            "risk_free_rate",
            _finite_float(self.risk_free_rate, "risk_free_rate"),
        )
        object.__setattr__(
            self,
            "carry_rate",
            _finite_float(self.carry_rate, "carry_rate"),
        )
        object.__setattr__(
            self,
            "parameter_source",
            _non_empty_string(self.parameter_source, "parameter_source"),
        )
        object.__setattr__(
            self,
            "calibration_cutoff",
            _normalize_optional_cutoff(self.calibration_cutoff),
        )
        if self.calibration_data_hash is not None:
            object.__setattr__(
                self,
                "calibration_data_hash",
                _require_sha256(self.calibration_data_hash, "calibration_data_hash"),
            )
        if (self.calibration_cutoff is None) != (self.calibration_data_hash is None):
            raise ValueError(
                "calibration_cutoff and calibration_data_hash must be provided together"
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
        _non_empty_string(self.calendar_id, "calendar_id")
        if self.calendar_id != _CALENDAR_ID:
            raise ValueError(
                "V0 implements only calendar_id='weekday-only-v0'; it is not an "
                "exchange trading calendar"
            )

        for regime in regimes:
            _trend_alpha_for_physical_drift(
                regime.physical_annual_return,
                self.risk_free_rate,
                self.carry_rate,
            )

    @property
    def manifest(self) -> dict[str, Any]:
        cutoff = self.calibration_cutoff
        return {
            "schema": SCHEMA_VERSION,
            "transition_interval": _TRANSITION_INTERVAL,
            "regimes": [regime.manifest for regime in self.regimes],
            "transition_matrix": [list(row) for row in self.transition_matrix],
            "initial_distribution": list(self.initial_distribution),
            "risk_free_rate": self.risk_free_rate,
            "carry_rate": self.carry_rate,
            "parameter_source": self.parameter_source,
            "calibration_cutoff": cutoff.isoformat() if cutoff is not None else None,
            "calibration_data_hash": self.calibration_data_hash,
            "trading_days_per_year": self.trading_days_per_year,
            "calendar_id": self.calendar_id,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.manifest)

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _spec_from_manifest(payload: object) -> RegimeSwitchingGBMSpecV0:
    """Rebuild and fully validate a canonical spec embedded in provenance."""

    if not isinstance(payload, dict):
        raise ValueError("normalized spec must be a JSON object")
    try:
        if payload.get("transition_interval") != _TRANSITION_INTERVAL:
            raise ValueError("unexpected transition interval")
        regimes_payload = payload["regimes"]
        if not isinstance(regimes_payload, list):
            raise TypeError("regimes must be a list")
        spec = RegimeSwitchingGBMSpecV0(
            regimes=tuple(RegimeParameters(**item) for item in regimes_payload),
            transition_matrix=tuple(
                tuple(row) for row in payload["transition_matrix"]
            ),
            initial_distribution=tuple(payload["initial_distribution"]),
            risk_free_rate=payload["risk_free_rate"],
            carry_rate=payload["carry_rate"],
            parameter_source=payload["parameter_source"],
            calibration_cutoff=payload["calibration_cutoff"],
            calibration_data_hash=payload["calibration_data_hash"],
            trading_days_per_year=payload["trading_days_per_year"],
            calendar_id=payload["calendar_id"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("normalized spec does not match the V0 schema") from exc
    if spec.manifest != payload:
        raise ValueError("normalized spec contains missing, extra, or changed fields")
    return spec


@dataclass(frozen=True)
class ScenarioProvenance:
    """Independent, deterministic provenance for one generated path artifact."""

    schema: str
    generator_version: str
    generator_source_hash: str
    hash_algorithm: str
    spec_hash: str
    spec_canonical_json: str
    root_seed: int
    regime_seed_domain: str
    regime_seed: int
    innovation_seed_domain: str
    innovation_seed: int
    rng_algorithm: str
    calendar_id: str
    monitoring_grid: str
    trading_days_per_year: int
    return_lookback_days: int
    realized_volatility_lookback_days: tuple[int, ...]
    drawdown_lookback_days: int
    parameter_source: str
    calibration_cutoff: str | None
    calibration_data_hash: str | None
    episode_id: str
    underlying: str
    start_date: str
    terminal_date: str
    start_spot: float
    decision_rounds: int
    snapshot_count: int
    daily_observation_count: int
    daily_path_hash_scope: str
    daily_path_hash: str
    snapshot_hash: str

    def __post_init__(self) -> None:
        if self.schema != SCHEMA_VERSION:
            raise ValueError(f"ScenarioProvenance.schema must be {SCHEMA_VERSION!r}")
        if self.generator_version != GENERATOR_VERSION:
            raise ValueError(
                f"ScenarioProvenance.generator_version must be {GENERATOR_VERSION!r}"
            )
        object.__setattr__(
            self,
            "generator_source_hash",
            _require_sha256(self.generator_source_hash, "generator_source_hash"),
        )
        object.__setattr__(
            self,
            "spec_hash",
            _require_sha256(self.spec_hash, "spec_hash"),
        )
        spec_canonical_json = _non_empty_string(
            self.spec_canonical_json,
            "spec_canonical_json",
        )
        try:
            normalized_spec = json.loads(spec_canonical_json)
        except json.JSONDecodeError as exc:
            raise ValueError("spec_canonical_json must contain valid JSON") from exc
        if _canonical_json(normalized_spec) != spec_canonical_json:
            raise ValueError("spec_canonical_json must use the canonical JSON encoding")
        if hashlib.sha256(spec_canonical_json.encode("utf-8")).hexdigest() != self.spec_hash:
            raise ValueError("spec_canonical_json disagrees with spec_hash")
        validated_spec = _spec_from_manifest(normalized_spec)
        object.__setattr__(
            self,
            "daily_path_hash",
            _require_sha256(self.daily_path_hash, "daily_path_hash"),
        )
        object.__setattr__(
            self,
            "snapshot_hash",
            _require_sha256(self.snapshot_hash, "snapshot_hash"),
        )
        if self.hash_algorithm != _HASH_ALGORITHM:
            raise ValueError(f"hash_algorithm must be {_HASH_ALGORITHM!r}")

        _strict_int(self.root_seed, "root_seed")
        _strict_int(self.regime_seed, "regime_seed", minimum=0)
        _strict_int(self.innovation_seed, "innovation_seed", minimum=0)
        if self.regime_seed == self.innovation_seed:
            raise ValueError("regime_seed and innovation_seed must be independent")
        if self.regime_seed_domain != _REGIME_SEED_DOMAIN:
            raise ValueError("unexpected regime_seed_domain")
        if self.innovation_seed_domain != _INNOVATION_SEED_DOMAIN:
            raise ValueError("unexpected innovation_seed_domain")
        if self.regime_seed != _derive_domain_seed(
            self.root_seed,
            self.regime_seed_domain,
        ):
            raise ValueError("regime_seed disagrees with root_seed/domain derivation")
        if self.innovation_seed != _derive_domain_seed(
            self.root_seed,
            self.innovation_seed_domain,
        ):
            raise ValueError("innovation_seed disagrees with root_seed/domain derivation")
        if self.rng_algorithm != _RNG_ALGORITHM:
            raise ValueError("unexpected rng_algorithm")

        if self.calendar_id != _CALENDAR_ID:
            raise ValueError("ScenarioProvenance must disclose weekday-only-v0")
        trading_days_per_year = _strict_int(
            self.trading_days_per_year,
            "trading_days_per_year",
            minimum=1,
        )
        if validated_spec.trading_days_per_year != trading_days_per_year:
            raise ValueError("normalized spec disagrees with trading_days_per_year")
        if validated_spec.calendar_id != self.calendar_id:
            raise ValueError("normalized spec disagrees with calendar_id")
        return_lookback_days = _strict_int(
            self.return_lookback_days,
            "return_lookback_days",
            minimum=1,
        )
        if return_lookback_days != _RETURN_LOOKBACK_DAYS:
            raise ValueError("unexpected return lookback")
        try:
            volatility_lookbacks = tuple(self.realized_volatility_lookback_days)
        except TypeError as exc:
            raise TypeError(
                "realized_volatility_lookback_days must be an iterable of ints"
            ) from exc
        for index, lookback in enumerate(volatility_lookbacks):
            _strict_int(
                lookback,
                f"realized_volatility_lookback_days[{index}]",
                minimum=1,
            )
        if volatility_lookbacks != _REALIZED_VOLATILITY_LOOKBACK_DAYS:
            raise ValueError("unexpected realized-volatility lookbacks")
        object.__setattr__(
            self,
            "realized_volatility_lookback_days",
            volatility_lookbacks,
        )
        drawdown_lookback_days = _strict_int(
            self.drawdown_lookback_days,
            "drawdown_lookback_days",
            minimum=1,
        )
        if drawdown_lookback_days != max(1, trading_days_per_year // 2):
            raise ValueError("drawdown lookback disagrees with trading_days_per_year")
        if self.monitoring_grid != _monitoring_grid(
            trading_days_per_year,
            drawdown_lookback_days,
        ):
            raise ValueError("monitoring_grid disagrees with the implemented algorithm")

        _non_empty_string(self.parameter_source, "parameter_source")
        normalized_cutoff = _normalize_optional_cutoff(self.calibration_cutoff)
        if self.calibration_data_hash is not None:
            _require_sha256(self.calibration_data_hash, "calibration_data_hash")
        if (normalized_cutoff is None) != (self.calibration_data_hash is None):
            raise ValueError(
                "calibration_cutoff and calibration_data_hash must be provided together"
            )
        if normalized_spec.get("schema") != SCHEMA_VERSION:
            raise ValueError("normalized spec has an unexpected schema")
        if validated_spec.parameter_source != self.parameter_source:
            raise ValueError("normalized spec disagrees with parameter_source")
        expected_cutoff = (
            normalized_cutoff.isoformat() if normalized_cutoff is not None else None
        )
        object.__setattr__(self, "calibration_cutoff", expected_cutoff)
        validated_cutoff = validated_spec.calibration_cutoff
        if (
            validated_cutoff.isoformat() if validated_cutoff is not None else None
        ) != expected_cutoff:
            raise ValueError("normalized spec disagrees with calibration_cutoff")
        if validated_spec.calibration_data_hash != self.calibration_data_hash:
            raise ValueError("normalized spec disagrees with calibration_data_hash")
        _non_empty_string(self.episode_id, "episode_id")
        _non_empty_string(self.underlying, "underlying")
        try:
            parsed_start = date.fromisoformat(self.start_date)
            parsed_terminal = date.fromisoformat(self.terminal_date)
        except (TypeError, ValueError) as exc:
            raise ValueError("start_date and terminal_date must be ISO dates") from exc
        if parsed_terminal <= parsed_start:
            raise ValueError("terminal_date must be after start_date")
        object.__setattr__(
            self,
            "start_spot",
            _positive_float(self.start_spot, "start_spot"),
        )
        decision_rounds = _strict_int(
            self.decision_rounds,
            "decision_rounds",
            minimum=1,
        )
        snapshot_count = _strict_int(self.snapshot_count, "snapshot_count", minimum=2)
        _strict_int(
            self.daily_observation_count,
            "daily_observation_count",
            minimum=2,
        )
        if snapshot_count != decision_rounds + 1:
            raise ValueError("snapshot_count must equal decision_rounds + 1")
        if self.daily_path_hash_scope != "daily_dates+daily_spots+latent_regime_path":
            raise ValueError("unexpected daily_path_hash_scope")

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generator_version": self.generator_version,
            "generator_source_hash": self.generator_source_hash,
            "hash_algorithm": self.hash_algorithm,
            "spec_hash": self.spec_hash,
            "normalized_spec": json.loads(self.spec_canonical_json),
            "root_seed": self.root_seed,
            "regime_seed_domain": self.regime_seed_domain,
            "regime_seed": self.regime_seed,
            "innovation_seed_domain": self.innovation_seed_domain,
            "innovation_seed": self.innovation_seed,
            "rng_algorithm": self.rng_algorithm,
            "calendar_id": self.calendar_id,
            "monitoring_grid": self.monitoring_grid,
            "trading_days_per_year": self.trading_days_per_year,
            "return_lookback_days": self.return_lookback_days,
            "realized_volatility_lookback_days": list(
                self.realized_volatility_lookback_days
            ),
            "drawdown_lookback_days": self.drawdown_lookback_days,
            "parameter_source": self.parameter_source,
            "calibration_cutoff": self.calibration_cutoff,
            "calibration_data_hash": self.calibration_data_hash,
            "episode_id": self.episode_id,
            "underlying": self.underlying,
            "start_date": self.start_date,
            "terminal_date": self.terminal_date,
            "start_spot": self.start_spot,
            "decision_rounds": self.decision_rounds,
            "snapshot_count": self.snapshot_count,
            "daily_observation_count": self.daily_observation_count,
            "daily_path_hash_scope": self.daily_path_hash_scope,
            "daily_path_hash": self.daily_path_hash,
            "snapshot_hash": self.snapshot_hash,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.manifest)

    @property
    def provenance_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


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


def _snapshot_payload(snapshots: Sequence[MarketSnapshot]) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "kind": "stochopia.market.snapshots.v1",
        "snapshots": [_snapshot_manifest(snapshot) for snapshot in snapshots],
    }


def _daily_path_payload(
    daily_dates: Sequence[date],
    daily_spots: Sequence[float],
    latent_regime_path: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema": SCHEMA_VERSION,
        "kind": "stochopia.market.privileged-daily-path.v1",
        "daily_dates": [item.isoformat() for item in daily_dates],
        "daily_spots": list(daily_spots),
        "latent_regime_path": list(latent_regime_path),
    }


@dataclass(frozen=True)
class GeneratedMarketPath:
    """Frozen public snapshots and privileged daily path with stable hashes.

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
            if item.weekday() >= 5:
                raise ValueError("daily_dates may contain only Monday through Friday")
        if any(right <= left for left, right in zip(daily_dates, daily_dates[1:])):
            raise ValueError("daily_dates must be strictly increasing")
        expected_daily_dates = _weekday_dates(daily_dates[0], daily_dates[-1])
        if daily_dates != expected_daily_dates:
            raise ValueError("daily_dates must contain every weekday without gaps")
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
        if len(snapshots) != provenance.snapshot_count:
            raise ValueError("snapshot count disagrees with provenance")
        if len(daily_dates) != provenance.daily_observation_count:
            raise ValueError("daily observation count disagrees with provenance")
        if daily_dates[0].isoformat() != provenance.start_date:
            raise ValueError("daily path start date disagrees with provenance")
        if daily_dates[-1].isoformat() != provenance.terminal_date:
            raise ValueError("daily path terminal date disagrees with provenance")
        if daily_spots[0] != provenance.start_spot:
            raise ValueError("daily path start spot disagrees with provenance")

        expected_snapshot_dates = _monthly_snapshot_dates(
            daily_dates[0], provenance.decision_rounds
        )
        if tuple(snapshot.as_of for snapshot in snapshots) != expected_snapshot_dates:
            raise ValueError("snapshot dates must be consecutive weekday-only month ends")
        date_to_spot = dict(zip(daily_dates, daily_spots))
        date_to_index = {item: index for index, item in enumerate(daily_dates)}
        spec = _spec_from_manifest(json.loads(provenance.spec_canonical_json))
        regime_by_name = {regime.name: regime for regime in spec.regimes}
        if any(name not in regime_by_name for name in latent_regime_path):
            raise ValueError("latent_regime_path contains a regime absent from the spec")
        expected_source = _snapshot_source(provenance.spec_hash, provenance.root_seed)
        for expected_round, snapshot in enumerate(snapshots, start=1):
            if snapshot.round_num != expected_round:
                raise ValueError("snapshot rounds must be contiguous from one")
            if snapshot.episode_id != provenance.episode_id:
                raise ValueError("snapshot episode_id disagrees with provenance")
            if snapshot.underlying != provenance.underlying:
                raise ValueError("snapshot underlying disagrees with provenance")
            if snapshot.regime != "synthetic":
                raise ValueError("public MarketSnapshot.regime must remain 'synthetic'")
            if snapshot.source != expected_source:
                raise ValueError("snapshot source disagrees with frozen provenance")
            if snapshot.spot != date_to_spot[snapshot.as_of]:
                raise ValueError("snapshot spot disagrees with daily path")
            daily_index = date_to_index[snapshot.as_of]
            regime = regime_by_name[latent_regime_path[daily_index]]
            expected_q_fields = (
                spec.risk_free_rate,
                spec.carry_rate,
                regime.pricing_iv_1m,
                regime.pricing_iv_3m,
                regime.pricing_iv_6m,
            )
            actual_q_fields = (
                snapshot.risk_free_rate,
                snapshot.carry_rate,
                snapshot.atm_iv_1m,
                snapshot.atm_iv_3m,
                snapshot.atm_iv_6m,
            )
            if actual_q_fields != expected_q_fields:
                raise ValueError("snapshot Q inputs disagree with the normalized spec")
            expected_trend_alpha = _trend_alpha_for_physical_drift(
                regime.physical_annual_return,
                spec.risk_free_rate,
                spec.carry_rate,
            )
            if snapshot.trend_alpha != expected_trend_alpha:
                raise ValueError("snapshot trend_alpha disagrees with the P/Q mapping")
            history = daily_spots[: daily_index + 1]
            expected_history_fields = (
                _trailing_return(history, _RETURN_LOOKBACK_DAYS),
                _realized_volatility(
                    history,
                    _REALIZED_VOLATILITY_LOOKBACK_DAYS[0],
                    spec.trading_days_per_year,
                ),
                _realized_volatility(
                    history,
                    _REALIZED_VOLATILITY_LOOKBACK_DAYS[1],
                    spec.trading_days_per_year,
                ),
                _trailing_drawdown(history, provenance.drawdown_lookback_days),
            )
            actual_history_fields = (
                snapshot.return_20d,
                snapshot.realized_vol_20d,
                snapshot.realized_vol_60d,
                snapshot.drawdown_6m,
            )
            if actual_history_fields != expected_history_fields:
                raise ValueError("snapshot history metrics disagree with the daily path")
            snapshot.pricing_volatility()

        daily_path_hash = _canonical_hash(
            _daily_path_payload(daily_dates, daily_spots, latent_regime_path)
        )
        if daily_path_hash != provenance.daily_path_hash:
            raise ValueError("daily path hash disagrees with provenance")
        snapshot_hash = _canonical_hash(_snapshot_payload(snapshots))
        if snapshot_hash != provenance.snapshot_hash:
            raise ValueError("snapshot hash disagrees with provenance")

    @property
    def manifest(self) -> dict[str, Any]:
        """Return a fresh, JSON-compatible copy of the complete frozen artifact."""

        return {
            "schema": SCHEMA_VERSION,
            "kind": "stochopia.market.generated-path.v1",
            "snapshots": [_snapshot_manifest(snapshot) for snapshot in self.snapshots],
            "daily_path": _daily_path_payload(
                self.daily_dates,
                self.daily_spots,
                self.latent_regime_path,
            ),
            "provenance": self.provenance.manifest,
        }

    @property
    def canonical_json(self) -> str:
        return _canonical_json(self.manifest)

    @property
    def root_hash(self) -> str:
        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()


def _last_weekday_of_month(year: int, month: int) -> date:
    candidate = date(year, month, calendar.monthrange(year, month)[1])
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def _next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _monthly_snapshot_dates(start_date: date, decision_rounds: int) -> tuple[date, ...]:
    result = [start_date]
    year, month = start_date.year, start_date.month
    try:
        for _ in range(decision_rounds):
            year, month = _next_month(year, month)
            result.append(_last_weekday_of_month(year, month))
    except (OverflowError, ValueError) as exc:
        raise ValueError("decision_rounds extends beyond the supported date range") from exc
    return tuple(result)


def _weekday_dates(start: date, end: date) -> tuple[date, ...]:
    result: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            result.append(current)
        if current == end:
            break
        current += timedelta(days=1)
    return tuple(result)


def _derive_domain_seed(root_seed: int, domain: str) -> int:
    preimage = {
        "schema": SCHEMA_VERSION,
        "root_seed": root_seed,
        "domain": domain,
    }
    return int.from_bytes(
        hashlib.sha256(_canonical_json(preimage).encode("utf-8")).digest(),
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


def _generator_source_hash() -> str:
    """Hash this module's bytes and fail closed if they cannot be read."""

    source_path = Path(__file__)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise RuntimeError(
            f"cannot hash generator module bytes at {source_path}; generation aborted"
        ) from exc
    if not source_bytes:
        raise RuntimeError("generator module is empty; generation aborted")
    return hashlib.sha256(source_bytes).hexdigest()


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


def _snapshot_source(spec_hash: str, seed: int) -> str:
    return (
        f"synthetic;schema={SCHEMA_VERSION};spec_hash={spec_hash};seed={seed}"
    )


def generate_regime_switching_gbm(
    spec: RegimeSwitchingGBMSpecV0,
    *,
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

    if not isinstance(spec, RegimeSwitchingGBMSpecV0):
        raise TypeError("spec must be RegimeSwitchingGBMSpecV0")
    episode_id = _non_empty_string(episode_id, "episode_id")
    underlying = _non_empty_string(underlying, "underlying")
    if isinstance(start_date, datetime) or not isinstance(start_date, date):
        raise TypeError("start_date must be a date without a time")
    if start_date != _last_weekday_of_month(start_date.year, start_date.month):
        raise ValueError(
            "start_date must be the month's final Monday-through-Friday date "
            "under weekday-only-v0"
        )
    start_spot = _positive_float(start_spot, "start_spot")
    decision_rounds = _strict_int(decision_rounds, "decision_rounds", minimum=1)
    seed = _strict_int(seed, "seed")

    snapshot_dates = _monthly_snapshot_dates(start_date, decision_rounds)
    daily_dates = _weekday_dates(start_date, snapshot_dates[-1])
    if len(daily_dates) < 2:
        raise AssertionError("a positive decision horizon must contain daily steps")

    spec_hash = spec.spec_hash
    generator_source_hash = _generator_source_hash()
    regime_seed = _derive_domain_seed(seed, _REGIME_SEED_DOMAIN)
    innovation_seed = _derive_domain_seed(seed, _INNOVATION_SEED_DOMAIN)
    if regime_seed == innovation_seed:
        raise RuntimeError("SHA-256 domain separation failed")
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
    source = _snapshot_source(spec_hash, seed)
    drawdown_lookback_days = max(1, spec.trading_days_per_year // 2)
    snapshots: list[MarketSnapshot] = []

    for round_num, as_of in enumerate(snapshot_dates, start=1):
        daily_index = date_to_index[as_of]
        history = daily_spots[: daily_index + 1]
        regime = spec.regimes[regime_indices[daily_index]]

        # Existing core code represents physical drift as r + alpha - q, hence
        # alpha = mu_P - r + q.  The core still reuses Q volatility in some
        # suitability calculations, so this bridge is not a complete P/Q API.
        trend_alpha = _trend_alpha_for_physical_drift(
            regime.physical_annual_return,
            spec.risk_free_rate,
            spec.carry_rate,
        )
        snapshot = MarketSnapshot(
            episode_id=episode_id,
            round_num=round_num,
            as_of=as_of,
            underlying=underlying,
            spot=history[-1],
            risk_free_rate=spec.risk_free_rate,
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
            atm_iv_1m=regime.pricing_iv_1m,
            atm_iv_3m=regime.pricing_iv_3m,
            atm_iv_6m=regime.pricing_iv_6m,
            carry_rate=spec.carry_rate,
            regime="synthetic",
            source=source,
            trend_alpha=trend_alpha,
        )
        snapshot.pricing_volatility()
        snapshots.append(snapshot)

    snapshots_tuple = tuple(snapshots)
    daily_spots_tuple = tuple(daily_spots)
    daily_path_hash = _canonical_hash(
        _daily_path_payload(daily_dates, daily_spots_tuple, latent_regime_path)
    )
    snapshot_hash = _canonical_hash(_snapshot_payload(snapshots_tuple))
    cutoff = spec.calibration_cutoff
    monitoring_grid = _monitoring_grid(
        spec.trading_days_per_year,
        drawdown_lookback_days,
    )
    provenance = ScenarioProvenance(
        schema=SCHEMA_VERSION,
        generator_version=GENERATOR_VERSION,
        generator_source_hash=generator_source_hash,
        hash_algorithm=_HASH_ALGORITHM,
        spec_hash=spec_hash,
        spec_canonical_json=spec.canonical_json,
        root_seed=seed,
        regime_seed_domain=_REGIME_SEED_DOMAIN,
        regime_seed=regime_seed,
        innovation_seed_domain=_INNOVATION_SEED_DOMAIN,
        innovation_seed=innovation_seed,
        rng_algorithm=_RNG_ALGORITHM,
        calendar_id=spec.calendar_id,
        monitoring_grid=monitoring_grid,
        trading_days_per_year=spec.trading_days_per_year,
        return_lookback_days=_RETURN_LOOKBACK_DAYS,
        realized_volatility_lookback_days=_REALIZED_VOLATILITY_LOOKBACK_DAYS,
        drawdown_lookback_days=drawdown_lookback_days,
        parameter_source=spec.parameter_source,
        calibration_cutoff=cutoff.isoformat() if cutoff is not None else None,
        calibration_data_hash=spec.calibration_data_hash,
        episode_id=episode_id,
        underlying=underlying,
        start_date=start_date.isoformat(),
        terminal_date=snapshot_dates[-1].isoformat(),
        start_spot=start_spot,
        decision_rounds=decision_rounds,
        snapshot_count=len(snapshots_tuple),
        daily_observation_count=len(daily_dates),
        daily_path_hash_scope="daily_dates+daily_spots+latent_regime_path",
        daily_path_hash=daily_path_hash,
        snapshot_hash=snapshot_hash,
    )
    return GeneratedMarketPath(
        snapshots=snapshots_tuple,
        daily_dates=daily_dates,
        daily_spots=daily_spots_tuple,
        latent_regime_path=latent_regime_path,
        provenance=provenance,
    )
