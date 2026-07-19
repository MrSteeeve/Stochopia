"""Synthetic tests for frozen SSE/MO option-IV rules."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from mirage.benchmark import MarketSnapshot
from mirage.formal_option_iv import apply_full_phase, build_chain_iv, parse_mo_master, parse_sse_master
from mirage.market_data_math import ShiborCurve, black76_price


NODES = (1, 7, 14, 30, 91, 182, 274, 365)


def _sse_master_and_chain(chain_date=date(2025, 1, 31)):
    expiry = chain_date + timedelta(days=91)
    curve = ShiborCurve(chain_date, chain_date, {node: 2.0 for node in NODES})
    rate, maturity, forward = curve.continuous_rate(91), 91 / 365, 5.05
    masters, chain = [], []
    for strike in (4.75, 5.0, 5.25):
        encoded = f"{round(strike * 1000):05d}"
        for side in ("C", "P"):
            key = f"{side}{encoded}"
            masters.append({
                "ts_code": key, "symbol": f"510500{side}2504M{encoded}",
                "exchange": "SSE", "opt_code": "OP510500.SH", "call_put": side,
                "exercise_price": str(strike), "per_unit": "10000",
                "maturity_date": expiry.strftime("%Y%m%d"),
                "list_date": (chain_date - timedelta(days=30)).strftime("%Y%m%d"),
                "delist_date": expiry.strftime("%Y%m%d"), "last_edate": expiry.strftime("%Y%m%d"),
            })
            chain.append({
                "ts_code": key, "settle": str(black76_price(forward, strike, maturity, rate, 0.25, side)),
                "oi": "10",
            })
    adjusted = dict(masters[0])
    adjusted.update(ts_code="adjusted", symbol="510500C2504A04750")
    masters.append(adjusted)
    return masters, chain, curve


def test_sse_parser_excludes_adjusted_and_builds_settlement_iv():
    masters, chain, curve = _sse_master_and_chain()
    book = parse_sse_master(masters)
    assert "adjusted" in book.excluded_adjusted
    assert book.reject_counts["A_excluded"] == 1
    result = build_chain_iv(chain, chain_date=curve.asof, spot=5.0, rcurve=curve, master=book, market="SSE")
    assert result.tenors["atm_iv_3m"] == pytest.approx(0.25, abs=1e-7)
    assert result.tenor_methods["atm_iv_3m"] == "exact"


def test_mo_parser_exact_symbol_and_no_future_master_use():
    chain_date = date(2025, 1, 31)
    valid = {
        "symbol": "MO2504-C-6000", "exchange": "CFFEX", "opt_code": "OP000852.SH",
        "call_put": "C", "exercise_price": "6000", "per_unit": "100",
        "maturity_date": "20250430", "list_date": "20250101",
        "delist_date": "20250430", "last_edate": "20250430",
    }
    future = {**valid, "symbol": "MO2504-P-6000", "call_put": "P", "list_date": "20250201"}
    book = parse_mo_master([valid, future])
    assert set(book.contracts) == {"MO2504-C-6000", "MO2504-P-6000"}
    curve = ShiborCurve(chain_date, chain_date, {node: 2.0 for node in NODES})
    result = build_chain_iv(
        [{"contract": "MO2504-P-6000", "settlement": "1", "open_interest": "2"}],
        chain_date=chain_date, spot=6000, rcurve=curve, master=book, market="MO",
    )
    assert result.reject_counts["inactive_master"] == 1
    assert not result.tenors


def test_chain_rejects_non_same_day_rate():
    masters, chain, _ = _sse_master_and_chain()
    chain_date = date(2025, 1, 31)
    stale_curve = ShiborCurve(chain_date, chain_date - timedelta(days=1), {node: 2.0 for node in NODES})
    with pytest.raises(ValueError, match="same chain date"):
        build_chain_iv(chain, chain_date=chain_date, spot=5, rcurve=stale_curve, master=parse_sse_master(masters), market="SSE")


def test_full_phase_falls_back_to_older_chain_with_fully_same_date_inputs():
    old_date, new_date = date(2025, 1, 30), date(2025, 1, 31)
    masters, old_chain, _ = _sse_master_and_chain(old_date)
    for row in old_chain:
        row["trade_date"] = old_date.strftime("%Y%m%d")
    new_chain = [{"ts_code": "not-in-master", "trade_date": new_date.strftime("%Y%m%d"), "settle": "1", "oi": "1"}]
    snapshot = MarketSnapshot(
        "CSI500_2025H1", 1, new_date, "CSI500", 5000, 0.02,
        realized_vol_20d=0.2, realized_vol_60d=0.2, source="base",
    )
    provenance = [{
        "episode_id": snapshot.episode_id, "round": 1, "as_of": new_date.isoformat(),
        "underlying": "CSI500", "max_input_date": new_date.isoformat(),
    }]
    shibor_rows = [
        {"date": day.strftime("%Y%m%d"), "on":"2","1w":"2","2w":"2","1m":"2","3m":"2","6m":"2","9m":"2","1y":"2"}
        for day in (old_date, new_date)
    ]
    updated, updated_provenance, _, stats = apply_full_phase(
        [snapshot], provenance, sse_rows=[*old_chain, *new_chain], mo_rows=[],
        option_master_rows=masters,
        fund_rows=[
            {"ts_code":"510500.SH", "trade_date":day.strftime("%Y%m%d"), "close":"5"}
            for day in (old_date, new_date)
        ],
        index_rows=[], shibor_rows=shibor_rows,
    )
    assert updated[0].atm_iv_3m == pytest.approx(0.25, abs=1e-7)
    assert "sse_510500_etf_iv_proxy" in updated[0].source
    assert updated_provenance[0]["option_iv"]["selected_date"] == old_date.isoformat()
    assert updated_provenance[0]["option_iv"]["staleness_trading_days"] == 1
    assert stats["lag_counts"] == {"1": 1}


def test_no_iv_row_is_explicit_rv_fallback():
    day = date(2025, 1, 31)
    snapshot = MarketSnapshot(
        "CSI500_2025H1", 1, day, "CSI500", 5000, 0.02,
        realized_vol_20d=0.2, realized_vol_60d=0.2, source="base",
    )
    provenance = [{"episode_id": snapshot.episode_id, "round": 1, "as_of": day.isoformat(), "underlying": "CSI500", "max_input_date": day.isoformat()}]
    shibor = [{"date":day.strftime("%Y%m%d"),"on":"2","1w":"2","2w":"2","1m":"2","3m":"2","6m":"2","9m":"2","1y":"2"}]
    updated, _, _, _ = apply_full_phase(
        [snapshot], provenance,
        sse_rows=[{"ts_code":"other", "trade_date":day.strftime("%Y%m%d"), "settle":"1", "oi":"1"}],
        mo_rows=[], option_master_rows=[],
        fund_rows=[{"ts_code":"510500.SH", "trade_date":day.strftime("%Y%m%d"), "close":"5"}],
        index_rows=[], shibor_rows=shibor,
    )
    assert updated[0].atm_iv_3m is None
    assert updated[0].source.endswith("rv_fallback")
