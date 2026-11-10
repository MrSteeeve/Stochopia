"""Observed, point-in-time Q-side inputs for synthetic scenario generation.

This module is an adapter over the repository's formal SHIBOR/futures/option
builders.  It does not implement a second IV or carry algorithm and never
turns missing Q observations into zeroes, realized volatility, or synthetic
values.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from .common import as_date, atomic_write_text, canonical_json, parse_yyyymmdd, read_csv_rows


Q_BUNDLE_SCHEMA = "stochopia.q-side-bundle.v1"
Q_SELECTION_SCHEMA = "stochopia.q-scenario-input.v1"
Q_FIELD_NAMES = (
    "risk_free_rate",
    "carry_rate",
    "atm_iv_1m",
    "atm_iv_3m",
    "atm_iv_6m",
)
IV_TARGET_DAYS = {"atm_iv_1m": 30, "atm_iv_3m": 91, "atm_iv_6m": 182}
COMMON_Q_START = date(2022, 9, 19)
Q_CUTOFF = date(2025, 12, 31)


def _json_copy(value: object, name: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON-compatible data") from exc


@dataclass(frozen=True)
class QFieldObservation:
    """One observed Q field, or an explicit unavailable marker."""

    value: float | None
    unit: str
    source: str
    observation_date: date | str | None
    target_days: int | None
    method: str | None = None
    unavailable_reason: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("Q field unit must be non-empty")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Q field source must be non-empty")
        if self.target_days is not None and (
            isinstance(self.target_days, bool)
            or not isinstance(self.target_days, int)
            or self.target_days <= 0
        ):
            raise ValueError("Q field target_days must be a positive integer or null")
        if self.value is None:
            if self.observation_date is not None:
                raise ValueError("unavailable Q field cannot have an observation date")
            if not isinstance(self.unavailable_reason, str) or not self.unavailable_reason:
                raise ValueError("unavailable Q field requires an explicit reason")
        else:
            if isinstance(self.value, bool) or not math.isfinite(float(self.value)):
                raise ValueError("observed Q field value must be finite")
            object.__setattr__(self, "value", float(self.value))
            if self.observation_date is None:
                raise ValueError("observed Q field requires an observation date")
            object.__setattr__(
                self,
                "observation_date",
                as_date(self.observation_date, "Q field observation_date"),
            )
            if self.unavailable_reason is not None:
                raise ValueError("observed Q field cannot have an unavailable reason")
        if self.method is not None and (not isinstance(self.method, str) or not self.method):
            raise ValueError("Q field method must be non-empty when provided")
        object.__setattr__(self, "details", _json_copy(dict(self.details or {}), "details"))

    @property
    def available(self) -> bool:
        return self.value is not None

    @property
    def availability_status(self) -> str:
        if self.available:
            return "available"
        prefix = str(self.unavailable_reason).split(":", 1)[0]
        if prefix in {
            "structural_prelaunch",
            "not_collected",
            "permission_denied",
            "quality_rejected",
        }:
            return prefix
        return "not_collected"

    @property
    def manifest(self) -> dict[str, Any]:
        observation_date = self.observation_date
        return {
            "available": self.available,
            "availability_status": self.availability_status,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "observation_date": (
                observation_date.isoformat() if isinstance(observation_date, date) else None
            ),
            "target_days": self.target_days,
            "method": self.method,
            "unavailable_reason": self.unavailable_reason,
            "details": _json_copy(self.details, "details"),
        }


@dataclass(frozen=True)
class QMarketSnapshot:
    """Point-in-time Q vector with per-field source and availability."""

    underlying: str
    as_of: date | str
    partition: str
    fields: Mapping[str, QFieldObservation]

    def __post_init__(self) -> None:
        if not isinstance(self.underlying, str) or not self.underlying:
            raise ValueError("Q snapshot underlying must be non-empty")
        object.__setattr__(self, "as_of", as_date(self.as_of, "Q snapshot as_of"))
        if self.partition not in {"train", "dev", "evaluation_reused_non_private"}:
            raise ValueError("unexpected Q partition")
        fields = dict(self.fields)
        if set(fields) != set(Q_FIELD_NAMES) or not all(
            isinstance(value, QFieldObservation) for value in fields.values()
        ):
            raise ValueError("Q snapshot must contain the five frozen Q fields")
        for name, field in fields.items():
            if field.available and field.observation_date > self.as_of:
                raise ValueError(f"future-dated Q input: {name}")
        available_iv = [
            (IV_TARGET_DAYS[name], fields[name].value)
            for name in IV_TARGET_DAYS
            if fields[name].available
        ]
        total_variances = [
            float(value) ** 2 * days / 365.0 for days, value in available_iv
        ]
        tolerance = 1e-12 * max((1.0, *total_variances))
        if any(
            right + tolerance < left
            for left, right in zip(total_variances, total_variances[1:])
        ):
            raise ValueError("observed ATM total variance decreases with tenor")
        object.__setattr__(self, "fields", fields)

    @property
    def complete(self) -> bool:
        return all(self.fields[name].available for name in Q_FIELD_NAMES)

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "as_of": self.as_of.isoformat(),
            "partition": self.partition,
            "complete": self.complete,
            "fields": {name: self.fields[name].manifest for name in Q_FIELD_NAMES},
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "QMarketSnapshot":
        fields: dict[str, QFieldObservation] = {}
        try:
            for name, item in payload["fields"].items():
                fields[name] = QFieldObservation(
                    value=item["value"],
                    unit=item["unit"],
                    source=item["source"],
                    observation_date=item["observation_date"],
                    target_days=item["target_days"],
                    method=item["method"],
                    unavailable_reason=item["unavailable_reason"],
                    details=item["details"],
                )
            snapshot = cls(
                underlying=payload["underlying"],
                as_of=payload["as_of"],
                partition=payload["partition"],
                fields=fields,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid Q snapshot manifest") from exc
        return snapshot


@dataclass(frozen=True)
class QScenarioInput:
    """A complete Q snapshot selected from one immutable bundle."""

    snapshot: QMarketSnapshot
    q_cutoff: date | str

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, QMarketSnapshot) or not self.snapshot.complete:
            raise ValueError("scenario generation requires a complete observed Q snapshot")
        object.__setattr__(self, "q_cutoff", as_date(self.q_cutoff, "q_cutoff"))
        if self.snapshot.as_of > self.q_cutoff:
            raise ValueError("Q snapshot is after the frozen Q cutoff")

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": Q_SELECTION_SCHEMA,
            "q_cutoff": self.q_cutoff.isoformat(),
            "snapshot": self.snapshot.manifest,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "QScenarioInput":
        if payload.get("schema") != Q_SELECTION_SCHEMA:
            raise ValueError("unexpected Q selection schema")
        return cls(
            snapshot=QMarketSnapshot.from_manifest(payload["snapshot"]),
            q_cutoff=payload["q_cutoff"],
        )


@dataclass(frozen=True)
class QSideBundle:
    """Independent Q-side dataset with a separate P calibration cutoff."""

    p_history: Mapping[str, Any]
    q_data: Mapping[str, Any]
    split_policy: Mapping[str, Any]
    records: tuple[QMarketSnapshot, ...]
    schema: str = Q_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != Q_BUNDLE_SCHEMA:
            raise ValueError(f"Q bundle schema must be {Q_BUNDLE_SCHEMA!r}")
        p_history = _json_copy(dict(self.p_history), "p_history")
        q_data = _json_copy(dict(self.q_data), "q_data")
        split_policy = _json_copy(dict(self.split_policy), "split_policy")
        if "cutoff" not in p_history:
            raise ValueError("P history requires an explicit calibration cutoff")
        if "cutoff" not in q_data:
            raise ValueError("Q data requires an explicit cutoff")
        if split_policy.get("private_evaluation") != "unavailable":
            raise ValueError("Q split must not claim an unseen private evaluation set")
        records = tuple(self.records)
        if not records or not all(isinstance(item, QMarketSnapshot) for item in records):
            raise ValueError("Q bundle records must contain QMarketSnapshot values")
        ordered = tuple(sorted(records, key=lambda item: (item.as_of, item.underlying)))
        if records != ordered:
            raise ValueError("Q records must be sorted by date and underlying")
        keys = [(item.underlying, item.as_of) for item in records]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate Q record key")
        object.__setattr__(self, "p_history", p_history)
        object.__setattr__(self, "q_data", q_data)
        object.__setattr__(self, "split_policy", split_policy)
        object.__setattr__(self, "records", records)

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "p_history": _json_copy(self.p_history, "p_history"),
            "q_data": _json_copy(self.q_data, "q_data"),
            "split_policy": _json_copy(self.split_policy, "split_policy"),
            "records": [item.manifest for item in self.records],
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.manifest)

    def select(self, underlying: str, as_of: date) -> QScenarioInput:
        match = next(
            (item for item in self.records if item.underlying == underlying and item.as_of == as_of),
            None,
        )
        if match is None:
            raise ValueError(f"no Q bundle record for {underlying} {as_of.isoformat()}")
        if not match.complete:
            missing = [name for name in Q_FIELD_NAMES if not match.fields[name].available]
            raise ValueError(f"Q record is incomplete; unavailable fields: {', '.join(missing)}")
        return QScenarioInput(snapshot=match, q_cutoff=self.q_data["cutoff"])

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "QSideBundle":
        if payload.get("schema") != Q_BUNDLE_SCHEMA:
            raise ValueError("unexpected Q bundle schema")
        return cls(
            p_history=payload["p_history"],
            q_data=dict(payload["q_data"]),
            split_policy=payload["split_policy"],
            records=tuple(QMarketSnapshot.from_manifest(item) for item in payload["records"]),
            schema=payload["schema"],
        )


def load_q_side_bundle(path: str | Path) -> QSideBundle:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Q bundle must be a JSON object")
    return QSideBundle.from_manifest(payload)


def _partition(as_of: date) -> str:
    if as_of <= date(2022, 12, 31):
        return "train"
    if as_of <= date(2023, 12, 31):
        return "dev"
    return "evaluation_reused_non_private"


def _series_coverage(rows: Sequence[Mapping[str, str]], date_column: str) -> dict[str, Any]:
    values = sorted(row[date_column] for row in rows if row.get(date_column))
    cutoff = Q_CUTOFF.strftime("%Y%m%d")
    eligible = [value for value in values if value <= cutoff]
    return {
        "rows_through_cutoff": len(eligible),
        "min_date": eligible[0] if eligible else None,
        "max_date": eligible[-1] if eligible else None,
        "duplicate_dates_through_cutoff": len(eligible) - len(set(eligible)),
        "rows_after_cutoff_excluded": len(values) - len(eligible),
    }


def _field(
    value: float | None,
    *,
    unit: str,
    source: str,
    observation_date: str | None,
    target_days: int,
    method: str | None,
    unavailable_reason: str,
    details: Mapping[str, Any],
) -> QFieldObservation:
    return QFieldObservation(
        value=value,
        unit=unit,
        source=source,
        observation_date=observation_date if value is not None else None,
        target_days=target_days,
        method=method if value is not None else None,
        unavailable_reason=None if value is not None else unavailable_reason,
        details=details,
    )


def build_q_side_bundle(
    *,
    tushare_root: str | Path = "data/raw/tushare",
    cffex_path: str | Path = "data/interim/cffex/ic_im_mo_daily.csv",
) -> QSideBundle:
    """Build the Q bundle offline by reusing the formal panel and IV gates."""

    from .formal_market_builder import build_base_panel
    from .formal_option_iv import apply_full_phase
    from .tushare_batch import HISTORICAL_CALIBRATION_END_DATE, load_endpoint_rows

    tushare = Path(tushare_root)
    cffex = Path(cffex_path)

    index_rows = load_endpoint_rows(tushare, "index_daily")
    shibor_rows = load_endpoint_rows(tushare, "shibor")
    futures_rows = read_csv_rows(cffex)
    futures_master = load_endpoint_rows(tushare, "fut_basic")
    snapshots, provenance = build_base_panel(
        index_rows=index_rows,
        shibor_rows=shibor_rows,
        futures_rows=futures_rows,
        futures_master_rows=futures_master,
        panel_start=date(2022, 9, 1),
        panel_end=Q_CUTOFF,
    )
    fund_daily = load_endpoint_rows(tushare, "fund_daily")
    fund_adj = load_endpoint_rows(tushare, "fund_adj")
    fund_div = load_endpoint_rows(tushare, "fund_div")
    trade_calendar_rows = load_endpoint_rows(tushare, "trade_cal")
    exchange_open_dates = [
        datetime.strptime(row["cal_date"], "%Y%m%d").date()
        for row in trade_calendar_rows
        if row.get("is_open") == "1"
    ]
    snapshots, provenance, _, iv_stats = apply_full_phase(
        snapshots,
        provenance,
        sse_rows=load_endpoint_rows(tushare, "opt_daily"),
        mo_rows=futures_rows,
        option_master_rows=load_endpoint_rows(tushare, "opt_basic"),
        fund_rows=fund_daily,
        index_rows=index_rows,
        shibor_rows=shibor_rows,
        exchange_open_dates=exchange_open_dates,
    )
    provenance_by_key = {
        (item["episode_id"], item["round"]): item for item in provenance
    }
    records: list[QMarketSnapshot] = []
    rate_unit = "decimal_continuous_annualized"
    iv_unit = "decimal_annualized_implied_volatility"
    for snapshot in snapshots:
        row = provenance_by_key[(snapshot.episode_id, snapshot.round_num)]
        carry = row["carry"]
        option = row["option_iv"]
        option_date = option.get("selected_date")
        iv_values = {
            "atm_iv_1m": snapshot.atm_iv_1m,
            "atm_iv_3m": snapshot.atm_iv_3m,
            "atm_iv_6m": snapshot.atm_iv_6m,
        }
        fields: dict[str, QFieldObservation] = {
            "risk_free_rate": _field(
                snapshot.risk_free_rate,
                unit=rate_unit,
                source="Tushare SHIBOR eight-node curve",
                observation_date=row["rate"]["observation_date"],
                target_days=91,
                method="simple_deposit_discount_then_continuous_zero",
                unavailable_reason="no lag-compliant observed SHIBOR curve",
                details={"source_component": "shibor"},
            ),
            "carry_rate": _field(
                None if carry.get("fallback_zero") else snapshot.carry_rate,
                unit=rate_unit,
                source="CFFEX futures settlement",
                observation_date=carry.get("observation_date"),
                target_days=91,
                method="futures_implied_carry",
                unavailable_reason="quality_rejected:missing same-day active futures settlement; zero fallback rejected",
                details={
                    "product": carry["product"],
                    "contracts": carry.get("contracts", []),
                },
            ),
        }
        for name, value in iv_values.items():
            fields[name] = _field(
                value,
                unit=iv_unit,
                source=option["source_label"],
                observation_date=option_date,
                target_days=IV_TARGET_DAYS[name],
                method=option.get("tenors", {}).get(name, {}).get("method"),
                unavailable_reason="quality_rejected:observed settlement chain did not pass the formal IV gates",
                details={
                    "selected_versions": option.get("selected_versions", []),
                    "expiry_contracts": option.get("expiry_contracts", {}),
                    "staleness_trading_days": option.get("staleness_trading_days"),
                },
            )
        records.append(
            QMarketSnapshot(
                underlying=snapshot.underlying,
                as_of=snapshot.as_of,
                partition=_partition(snapshot.as_of),
                fields=fields,
            )
        )

    p_cutoff = parse_yyyymmdd(HISTORICAL_CALIBRATION_END_DATE).isoformat()
    q_data = {
        "start": COMMON_Q_START.isoformat(),
        "cutoff": Q_CUTOFF.isoformat(),
        "usage": "point_in_time_scenario_input_or_sensitivity",
        "point_in_time_semantics": (
            "observation-date-filtered post-hoc provider snapshot; not archived "
            "historical publication bytes"
        ),
        "listing_boundaries": {
            "IC": "2015-04-16",
            "IM": "2022-07-22",
            "MO": "2022-07-22",
            "510500_ETF_option": "2022-09-19",
        },
        "tenor_mapping_days": IV_TARGET_DAYS,
        "units": {
            "risk_free_rate": rate_unit,
            "carry_rate": rate_unit,
            "atm_iv": iv_unit,
        },
        "source_coverage": {
            "shibor": _series_coverage(shibor_rows, "date"),
            "fund_daily": _series_coverage(fund_daily, "trade_date"),
            "fund_adj": _series_coverage(fund_adj, "trade_date"),
            "fund_div": _series_coverage(fund_div, "ex_date"),
            "formal_iv": iv_stats,
        },
        "raw_redistribution": False,
    }
    split_policy = {
        "train": {"start": "2022-09-19", "end": "2022-12-31"},
        "dev": {"start": "2023-01-01", "end": "2023-12-31"},
        "evaluation_reused_non_private": {
            "start": "2024-01-01",
            "end": "2025-12-31",
            "qualification": "already used by the project; not unseen or private",
        },
        "private_evaluation": "unavailable",
    }
    return QSideBundle(
        p_history={
            "cutoff": p_cutoff,
            "source": "tushare index_daily (frozen historical calibration window)",
            "usage": "P_calibration_only",
        },
        q_data=q_data,
        split_policy=split_policy,
        records=tuple(sorted(records, key=lambda item: (item.as_of, item.underlying))),
    )


def write_q_side_bundle(bundle: QSideBundle, output_dir: str | Path) -> Path:
    """Write ``q_side_bundle.json`` into ``output_dir`` and return its path."""

    if not isinstance(bundle, QSideBundle):
        raise TypeError("bundle must be QSideBundle")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "q_side_bundle.json"
    atomic_write_text(target, bundle.canonical_json + "\n")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the frozen observed Q-side bundle")
    parser.add_argument("--tushare-root", type=Path, default=Path("data/raw/tushare"))
    parser.add_argument(
        "--cffex-path", type=Path, default=Path("data/interim/cffex/ic_im_mo_daily.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/derived/q_side")
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = build_q_side_bundle(tushare_root=args.tushare_root, cffex_path=args.cffex_path)
    artifact = write_q_side_bundle(bundle, args.output_dir)
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "records": len(bundle.records),
                "complete_records": sum(item.complete for item in bundle.records),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
