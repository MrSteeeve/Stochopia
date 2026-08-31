"""Frozen Phase-B option-chain parsing and implied-volatility construction."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence

from .benchmark import MarketSnapshot
from .market_data_math import (
    ExpiryPair,
    ExpiryPoint,
    ShiborCurve,
    build_expiry_point,
    interpolate_total_variance,
)


SSE_SYMBOL = re.compile(r"^510500([CP])(\d{4})([MA])(\d{5})$")
MO_SYMBOL = re.compile(r"^MO\d{4}-([CP])-(\d+(?:\.\d+)?)$")
TENORS = {"atm_iv_1m": 30, "atm_iv_3m": 91, "atm_iv_6m": 182}


def _day(value: str) -> date:
    return datetime.strptime(value, "%Y%m%d").date()


@dataclass(frozen=True)
class OptionContract:
    key: str
    symbol: str
    expiry: date
    strike: float
    call_put: str
    version: str
    per_unit: float
    list_date: date
    end_date: date


@dataclass(frozen=True)
class MasterBook:
    contracts: Mapping[str, OptionContract]
    excluded_adjusted: frozenset[str]
    other_underlying: frozenset[str]
    reject_counts: Mapping[str, int]


@dataclass(frozen=True)
class ChainResult:
    chain_date: date
    tenors: Mapping[str, float]
    tenor_methods: Mapping[str, str]
    points: tuple[ExpiryPoint, ...]
    point_contracts: Mapping[str, tuple[str, ...]]
    reject_counts: Mapping[str, int]


def parse_sse_master(rows: Iterable[Mapping[str, str]]) -> MasterBook:
    contracts: dict[str, OptionContract] = {}
    excluded: set[str] = set()
    other: set[str] = set()
    rejects: Counter[str] = Counter()
    for row in rows:
        if row.get("exchange") != "SSE":
            continue
        if row.get("opt_code") != "OP510500.SH":
            if row.get("ts_code"):
                other.add(row["ts_code"])
            continue
        key = row.get("ts_code", "")
        match = SSE_SYMBOL.fullmatch(row.get("symbol", ""))
        if match and match.group(3) == "A":
            excluded.add(key)
            rejects["A_excluded"] += 1
            continue
        try:
            strike = float(row.get("exercise_price", ""))
            encoded_strike = int(match.group(4)) / 1000 if match else -1
            per_unit = float(row.get("per_unit", ""))
            expiry, listed = _day(row["maturity_date"]), _day(row["list_date"])
            ends = [_day(row[column]) for column in ("delist_date", "last_edate") if row.get(column)]
            end = min(ends)
        except (KeyError, TypeError, ValueError):
            rejects["invalid_master"] += 1
            continue
        if not (
            key and match and match.group(1) == row.get("call_put")
            and strike > 0 and math.isclose(strike, encoded_strike, abs_tol=1e-9)
            and per_unit > 0 and listed <= expiry <= end
        ):
            rejects["invalid_master"] += 1
            continue
        contracts[key] = OptionContract(
            key, row["symbol"], expiry, strike, row["call_put"], "M", per_unit, listed, end
        )
    return MasterBook(contracts, frozenset(excluded), frozenset(other), dict(rejects))


def parse_mo_master(rows: Iterable[Mapping[str, str]]) -> MasterBook:
    contracts: dict[str, OptionContract] = {}
    rejects: Counter[str] = Counter()
    for row in rows:
        if row.get("exchange") != "CFFEX" or row.get("opt_code") != "OP000852.SH":
            continue
        symbol = row.get("symbol", "")
        match = MO_SYMBOL.fullmatch(symbol)
        try:
            strike = float(row.get("exercise_price", ""))
            encoded_strike = float(match.group(2)) if match else -1
            per_unit = float(row.get("per_unit", ""))
            expiry, listed = _day(row["maturity_date"]), _day(row["list_date"])
            ends = [_day(row[column]) for column in ("delist_date", "last_edate") if row.get(column)]
            end = min(ends)
        except (KeyError, TypeError, ValueError):
            rejects["invalid_master"] += 1
            continue
        if not (
            match and match.group(1) == row.get("call_put")
            and strike > 0 and math.isclose(strike, encoded_strike, abs_tol=1e-9)
            and per_unit > 0 and listed <= expiry <= end
        ):
            rejects["invalid_master"] += 1
            continue
        contracts[symbol] = OptionContract(
            symbol, symbol, expiry, strike, row["call_put"], "M", per_unit, listed, end
        )
    return MasterBook(contracts, frozenset(), frozenset(), dict(rejects))


def _tenor_method(points: Sequence[ExpiryPoint], target: int) -> str:
    days = sorted(point.dte for point in points)
    if target in days:
        return "exact"
    lower = [value for value in days if value < target]
    upper = [value for value in days if value > target]
    if lower and upper and upper[0] - lower[-1] <= 120:
        return "total_variance_bracket"
    return "nearest"


def build_chain_iv(
    rows: Iterable[Mapping[str, str]],
    *, chain_date: date, spot: float, rcurve: ShiborCurve,
    master: MasterBook, market: str,
) -> ChainResult:
    if rcurve.source_date != chain_date or rcurve.asof != chain_date:
        raise ValueError("spot, rate and options must use the same chain date")
    key_column = "ts_code" if market == "SSE" else "contract"
    settlement_column = "settle" if market == "SSE" else "settlement"
    oi_column = "oi" if market == "SSE" else "open_interest"
    rejects: Counter[str] = Counter()
    sides: dict[tuple[date, float, str, float], dict[str, tuple[OptionContract, float, float]]] = defaultdict(dict)
    for row in rows:
        key = row.get(key_column, "")
        if key in master.excluded_adjusted:
            continue
        contract = master.contracts.get(key)
        if contract is None:
            rejects["other_underlying_filtered" if key in master.other_underlying else "unmatched_master"] += 1
            continue
        if not contract.list_date <= chain_date <= contract.end_date:
            rejects["inactive_master"] += 1
            continue
        try:
            settlement, oi = float(row.get(settlement_column, "")), float(row.get(oi_column, ""))
        except (TypeError, ValueError):
            rejects["invalid_quote"] += 1
            continue
        if settlement <= 0:
            rejects["nonpositive_settlement"] += 1
            continue
        if oi <= 0:
            rejects["nonpositive_oi"] += 1
            continue
        group = (contract.expiry, contract.strike, contract.version, contract.per_unit)
        if contract.call_put in sides[group]:
            rejects["duplicate_side"] += 1
            continue
        sides[group][contract.call_put] = (contract, settlement, oi)

    grouped_pairs: dict[tuple[date, str, float], list[ExpiryPair]] = defaultdict(list)
    pair_contracts: dict[tuple[date, float], tuple[str, str]] = {}
    for (expiry, strike, version, per_unit), pair in sides.items():
        if set(pair) != {"C", "P"}:
            rejects["unpaired"] += 1
            continue
        call, put = pair["C"], pair["P"]
        grouped_pairs[(expiry, version, per_unit)].append(
            ExpiryPair(expiry, strike, version, per_unit, call[1], put[1], call[2], put[2])
        )
        pair_contracts[(expiry, strike)] = (call[0].key, put[0].key)

    points: list[ExpiryPoint] = []
    point_contracts: dict[str, tuple[str, ...]] = {}
    for (expiry, _, _), pairs in grouped_pairs.items():
        try:
            point = build_expiry_point(pairs, spot=spot, asof=chain_date, rcurve=rcurve)
        except ValueError:
            rejects["expiry_point_rejected"] += 1
            continue
        points.append(point)
        point_contracts[expiry.isoformat()] = tuple(
            key for strike in point.strikes for key in pair_contracts[(expiry, strike)]
        )
    points.sort(key=lambda point: point.dte)
    tenors, methods = {}, {}
    for name, target in TENORS.items():
        try:
            tenors[name] = interpolate_total_variance(points, target)
            methods[name] = _tenor_method(points, target)
        except ValueError:
            continue
    return ChainResult(chain_date, tenors, methods, tuple(points), point_contracts, dict(rejects))


def apply_full_phase(
    snapshots: Sequence[MarketSnapshot], provenance: Sequence[Mapping[str, Any]], *,
    sse_rows: Iterable[Mapping[str, str]], mo_rows: Iterable[Mapping[str, str]],
    option_master_rows: Iterable[Mapping[str, str]],
    fund_rows: Iterable[Mapping[str, str]], index_rows: Iterable[Mapping[str, str]],
    shibor_rows: Iterable[Mapping[str, str]],
) -> tuple[list[MarketSnapshot], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    masters = list(option_master_rows)
    books = {"CSI500": parse_sse_master(masters), "CSI1000": parse_mo_master(masters)}
    sse_by_date: dict[date, list[Mapping[str, str]]] = defaultdict(list)
    for row in sse_rows:
        sse_by_date[_day(row["trade_date"])].append(row)
    mo_by_date: dict[date, list[Mapping[str, str]]] = defaultdict(list)
    for row in mo_rows:
        if row.get("product") == "MO":
            mo_by_date[_day(row["trade_date"])].append(row)
    fund_spot = {_day(row["trade_date"]): float(row["close"]) for row in fund_rows if row.get("ts_code") == "510500.SH"}
    index_spot = {
        (row.get("ts_code", ""), _day(row["trade_date"])): float(row["close"])
        for row in index_rows if row.get("ts_code") == "000852.SH"
    }
    shibor = {
        _day(row["date"]): {days: float(row[column]) for days, column in {1:"on",7:"1w",14:"2w",30:"1m",91:"3m",182:"6m",274:"9m",365:"1y"}.items()}
        for row in shibor_rows
    }
    provenance_by_key = {(row["episode_id"], row["round"]): dict(row) for row in provenance}
    output, output_provenance, expiry_rows = [], [], []
    reject_total: Counter[str] = Counter(books["CSI500"].reject_counts)
    reject_total.update(books["CSI1000"].reject_counts)
    lag_counts: Counter[int] = Counter()
    coverage_by_underlying: Counter[str] = Counter()
    tenor_counts: Counter[str] = Counter()
    for snapshot in snapshots:
        underlying = snapshot.underlying
        label = "sse_510500_etf_iv_proxy" if underlying == "CSI500" else "cffex_mo_index_iv"
        chains = sse_by_date if underlying == "CSI500" else mo_by_date
        market = "SSE" if underlying == "CSI500" else "MO"
        available = sorted((day for day in chains if day <= snapshot.as_of and day.strftime("%Y%m") == snapshot.as_of.strftime("%Y%m")), reverse=True)[:3]
        selected: ChainResult | None = None
        selected_spot = None
        same_date_rejects = 0
        for lag, chain_day in enumerate(available):
            spot = fund_spot.get(chain_day) if underlying == "CSI500" else index_spot.get(("000852.SH", chain_day))
            rates = shibor.get(chain_day)
            if spot is None or rates is None:
                same_date_rejects += 1
                continue
            curve = ShiborCurve(chain_day, chain_day, rates)
            result = build_chain_iv(chains[chain_day], chain_date=chain_day, spot=spot, rcurve=curve, master=books[underlying], market=market)
            reject_total.update(result.reject_counts)
            if result.tenors:
                selected, selected_spot = result, spot
                lag_counts[lag] += 1
                break
        values = selected.tenors if selected else {}
        updated = replace(
            snapshot,
            atm_iv_1m=values.get("atm_iv_1m"),
            atm_iv_3m=values.get("atm_iv_3m"),
            atm_iv_6m=values.get("atm_iv_6m"),
            source=f"{snapshot.source}; {label if values else 'rv_fallback'}",
        )
        updated.pricing_volatility()
        output.append(updated)
        if values:
            coverage_by_underlying[underlying] += 1
            tenor_counts.update(values.keys())
        base = provenance_by_key[(snapshot.episode_id, snapshot.round_num)]
        selected_date = selected.chain_date if selected else None
        max_input = max(date.fromisoformat(base["max_input_date"]), selected_date) if selected_date else date.fromisoformat(base["max_input_date"])
        if max_input > snapshot.as_of:
            raise AssertionError("future-dated option input detected")
        base["option_iv"] = {
            "source_label": label,
            "selected_date": selected_date.isoformat() if selected_date else None,
            "staleness_trading_days": available.index(selected_date) if selected_date else None,
            "proxy_spot": {"code": "510500.SH" if underlying == "CSI500" else "000852.SH", "observation_date": selected_date.isoformat() if selected_date else None},
            "tenors": {name: {"method": selected.tenor_methods[name]} for name in values} if selected else {},
            "expiry_contracts": dict(selected.point_contracts) if selected else {},
            "selected_versions": ["M"] if selected else [],
            "reject_counts": dict(selected.reject_counts) if selected else {"same_date_missing": same_date_rejects},
            "A_excluded": books[underlying].reject_counts.get("A_excluded", 0),
            "max_input_date": max_input.isoformat(),
        }
        base["max_input_date"] = max_input.isoformat()
        output_provenance.append(base)
        if selected:
            for point in selected.points:
                expiry_rows.append({
                    "episode_id": snapshot.episode_id, "round": snapshot.round_num,
                    "underlying": underlying, "source_label": label,
                    "version": "M",
                    "selected_version": "M", "price_field": "settlement",
                    "chain_date": selected.chain_date.isoformat(), "expiry": point.expiry.isoformat(),
                    "dte": point.dte, "forward": point.forward, "implied_vol": point.implied_vol,
                    "spot": selected_spot, "forward_ratio": point.forward / selected_spot,
                    "mad_ratio": point.forward_mad_ratio, "range_ratio": point.forward_range_ratio,
                    "strikes": "|".join(strike.__format__("g") for strike in point.strikes),
                    "iv_observations": point.iv_observations,
                    "contracts": "|".join(selected.point_contracts.get(point.expiry.isoformat(), ())),
                })
    any_count = sum(any(value is not None for value in (row.atm_iv_1m,row.atm_iv_3m,row.atm_iv_6m)) for row in output)
    all_count = sum(all(value is not None for value in (row.atm_iv_1m,row.atm_iv_3m,row.atm_iv_6m)) for row in output)
    stats = {
        "any_iv": any_count, "all3_iv": all_count, "total": len(output),
        "any_rate": any_count / len(output), "all3_rate": all_count / len(output),
        "by_underlying": {
            name: {"count": coverage_by_underlying[name], "rate": coverage_by_underlying[name] / 36}
            for name in ("CSI500", "CSI1000")
        },
        "by_tenor": {
            name: {"count": tenor_counts[name], "rate": tenor_counts[name] / len(output)}
            for name in TENORS
        },
        "lag_counts": {str(key): value for key, value in sorted(lag_counts.items())},
        "reject_counts": dict(reject_total),
        "warnings": [message for threshold, value, message in ((0.8, any_count/len(output), "any-IV coverage below 80%"),(0.6, all_count/len(output), "all-three-IV coverage below 60%")) if value < threshold],
    }
    return output, output_provenance, expiry_rows, stats
