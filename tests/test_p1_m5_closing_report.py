"""Synthetic closing report: no data IO, fitted models, or remote requests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from experiments.p1_m4_report import ARMS, STOCKS
from experiments.p1_m5_closing_report import IMPACT, NLL, render_closing
from experiments.p1_m5_report import render

Json = dict[str, Any]


def _readout(value: float | None, interval: list[float] | None, days: int = 20) -> Json:
    return {"estimate": value, "interval": interval, "p05": None if interval is None else interval[0],
            "p10": None if interval is None else interval[0], "p90": None if interval is None else interval[1],
            "p95": None if interval is None else interval[1], "paired_days": days if interval else 0,
            "registered_days": days, "bootstrap_total": 2000, "bootstrap_valid": 2000 if interval else 0,
            "reason": "C2 undefined" if interval is None else None}


def _verdict(role: str, status: str, hit: bool, available: bool = True) -> Json:
    return {"role": role, "status": status, "point_available": available, "point_in_interval": hit}


def _trial(stock: str, arm: str, seed: int, index: int) -> Json:
    trial_id = f"{stock}-{seed}-{arm}-r{index:02d}"
    parent = {"model_id": "seed-gru", "source_sha256": "0" * 64}
    positive = seed == 17 and index == 1
    score_delta = 3.034029644 if positive else -1.0
    primary = _verdict("primary", "SUPPORTED" if positive else "REFUTED", False)
    status = "REFUTED"
    if arm == "metrics_gpt" and positive:
        status = "VERIFIED_GAIN"
    guard = {"finite_nll_by_seed": True, "delta_val_nll_total": 0.01, "delta_exhausted_windows": 0, "nll_passed": True,
             "exhaustion_passed": True, "drift_passed": True,
             "parent_free_abs_drift_ticks": 3.0, "child_free_abs_drift_ticks": 2.5}
    p10 = 0.1 if status == "VERIFIED_GAIN" else -2.8999842659077113
    return {
        "trial_id": trial_id, "round": index, "parent": parent,
        "candidate": {"model_id": trial_id, "source_sha256": "1" * 64},
        "verdict": status, "guardrails": guard, "mean_delta_score": score_delta,
        "p05_delta_score": -5.052364278688036, "p10_delta_score": p10,
        "has_benchmark_gain": status == "VERIFIED_GAIN", "candidate_metrics": {IMPACT: 30 + score_delta},
        "prediction_verdicts": [primary, _verdict("guardrail", "SUPPORTED", True)],
        "predictions": [{"role": "primary", "metric": "recovery", "direction": "decrease", "interval": [-0.12, -0.02]},
                        {"role": "guardrail", "metric": NLL, "direction": "equivalent", "interval": [-0.02, 0.02],
                         "equivalence_band": [-0.02, 0.02]}],
        "readouts": {IMPACT: _readout(score_delta, [-7.98, 15.09], 24),
                     NLL: _readout(0.007661599, [0.001134706, 0.014626206], 24),
                     "recovery": _readout(-0.604636546, [-1.35, 0.0553], 24)},
    }


def _result(stock: str, trial: Json, delta: float | None, status: str = "measured") -> Json:
    metrics = {IMPACT: None if delta is None else 20 + delta, NLL: 8.0,
               "claims": {"C2": "fail" if delta is None else "pass"}}
    score = _readout(delta, None if delta is None else [-3.0, 9.0])
    readouts = {IMPACT: score, NLL: _readout(0.002, [0.001, 0.003])}
    return {"stock": stock, "task_id": trial["trial_id"], "group": "synthetic", "candidate": trial["candidate"],
            "status": status, "metrics": metrics if status == "measured" else {}, "readouts": readouts,
            "seed_readouts": readouts, "assessment": {"prediction_verdicts": [_verdict("primary", "REFUTED", False)],
                                                       "has_benchmark_gain": False}}


def _fixture() -> tuple[Json, Json, Json]:
    dates = {"train_dates": list(range(1, 37)), "dev_dates": list(range(37, 61)), "test_dates": list(range(61, 81))}
    code = {"stochopia_commit": "fixture-stochopia", "openhyra_commit": "fixture-openhyra"}
    manifest: Json = {"schema": "stochopia-p1-m5-v1", "trajectories": {}, "code": code, "data": dates, "tasks": []}
    report: Json = {"schema": manifest["schema"], "mode": "m5-closing", "code": code, "test_dates": dates["test_dates"],
                    "scheduled_tasks": 14, "measured_tasks": 13, "scheduled_comparisons": 16, "measured_comparisons": 15,
                    "trajectories": {}, "audit_only_rejected_unverified": [], "secondary_best_guarded": [], "tasks": {},
                    "cross_regime": {"llm_best": {}, "no_llm_best": {}}}
    for stock in STOCKS:
        seed_trial = {"trial_id": f"{stock}-seed", "candidate": {"model_id": "seed-gru", "source_sha256": "0" * 64}}
        initial = _result(stock, seed_trial, 0.0)
        for arm in ARMS:
            for seed in (17, 29, 41):
                trials = [_trial(stock, arm, seed, index) for index in (1, 2, 3)]
                key = f"{stock}-{arm}-{seed}"
                manifest["trajectories"][key] = {"stock": stock, "arm": arm, "search_seed": seed, "trials": trials}
                final = initial
                if seed == 17:
                    trial = trials[0]
                    if arm == "metrics_ds" and stock == STOCKS[1]:
                        report["secondary_best_guarded"].append({"stock": stock, "arm": arm, "result": None})
                    else:
                        delta = None if arm == "metrics_ds" else 1.0
                        result = _result(stock, trial, delta)
                        report["tasks"][trial["trial_id"]] = result
                        report["secondary_best_guarded"].append({"stock": stock, "arm": arm, "trial_id": trial["trial_id"],
                            "candidate": trial["candidate"], "dev_score": trial["candidate_metrics"][IMPACT],
                            "dev_delta": trial["mean_delta_score"], "result": result})
                        if arm == "metrics_gpt":
                            final = result
                        if arm in ("metrics_gpt", "no_llm"):
                            category = "no_llm_best" if arm == "no_llm" else "llm_best"
                            target = next(x for x in STOCKS if x != stock)
                            report["cross_regime"][category][stock] = {"representative": trial["candidate"], "dev_score": 33.0,
                                "cells": {stock: result, target: _result(target, trial, -1.5)}}
                report["trajectories"][key] = {"stock": stock, "arm": arm, "search_seed": seed, "initial": initial,
                                                "first": {"status": "not_scheduled"}, "final": final}
    drows = []
    anchors = []
    for stock in STOCKS:
        for day in range(1, 81):
            drows.append({"symbol": stock, "date": day, "n_events": 100, "n_snapshots": 10,
                          "touch_match_rate": 0.8, "top10_log_vol_mae": 0.02})
            for session in (0, 1):
                anchors.append({"symbol": stock, "date": day, "session": session, "spread_ticks": 1})
    receipt = {"test_dates": dates["test_dates"], "excluded_test_dates": [],
               "reconciliation": [row for row in drows if row["date"] >= 61],
               "anchors": [row for row in anchors if row["date"] >= 61],
               "development_reconciliation": [row for row in drows if row["date"] < 61],
               "development_anchors": [row for row in anchors if row["date"] < 61],
               "remote_availability": {"market_wide_trade_day_files": 224, "by_stock": {stock: None for stock in STOCKS}}}
    return manifest, report, receipt


def _publish(root: Path, manifest: Json, report: Json, receipt: Json) -> None:
    (root / "verification").mkdir()
    for path, row in ((root / "study_manifest.json", manifest), (root / "verification/report.json", report),
                      (root / "verification/test-data-reconciliation.json", receipt)):
        path.write_text(json.dumps(row), encoding="utf-8")


def test_closing_report_contains_all_six_tables_and_undefined_scores(tmp_path: Path) -> None:
    manifest, report, receipt = _fixture()
    _publish(tmp_path, manifest, report, receipt)
    text = render(tmp_path, "基于冻结源码核验式：$h_t = f(h_{t-1},x_t)$")
    for index in range(1, 7):
        assert f"## 表 {index}" in text
    assert "3600 | 72/72 | 80.0000 | 0.020000" in text
    assert "-2.899984" in text
    assert "0.007662 [0.001135, 0.014626]" in text
    assert "SUPPORTED，点值未命中" in text
    assert "p10 未过零" in text
    assert "未证伪含 SUPPORTED 与 INCONCLUSIVE" in text
    assert "C2 undefined" in text
    assert "无护栏内代表" in text
    assert "13/14" in text
    assert "15/16" in text
    assert "仅指标组 | 点值未命中 | 3 | 3 | 1 | 0 | 2/2 (100.0%)" in text
    assert "36 | 36 | 4 | 32 | 0" in text
    assert "市场" in text and "224" in text


def test_closing_refuses_unfrozen_or_incomplete_inputs(tmp_path: Path) -> None:
    manifest, report, receipt = _fixture()
    incomplete = deepcopy(manifest)
    incomplete["trajectories"].popitem()
    with pytest.raises(ValueError, match="90 unique"):
        render_closing(tmp_path, incomplete, report, receipt, "source notes")
    report["secondary_best_guarded"].pop()
    with pytest.raises(ValueError, match="ten stock-arm cells"):
        render_closing(tmp_path, manifest, report, receipt, "source notes")


def test_remote_inventory_compact_with_full_bounds_and_metadata_scope() -> None:
    """A multi-year listing must not expand all 2,303 date records into prose."""
    from experiments.p1_m5_closing_report import _remote_availability

    # Deliberately large synthetic date arrays. Only endpoints and length matter.
    market_days = list(range(20170601, 20172904))
    months = [f"{year}-{month:02d}" for year in range(2017, 2027) for month in range(1, 13)
              if "2017-06" <= f"{year}-{month:02d}" <= "2026-11"]
    assert len(market_days) == 2303
    assert len(months) == 114
    first = {
        "trade_day_files": None, "confirmed_trade_day_files": 146,
        "possible_trade_day_files": 2302, "unresolved_dates": market_days[147:],
        "possible_market_months": months,
    }
    second = {
        "trade_day_files": None, "confirmed_trade_day_files": 137,
        "possible_trade_day_files": 2303, "unresolved_dates": market_days[137:],
        "possible_market_months": months,
    }
    scoped_first = {"trade_day_files": 146, "confirmed_trade_day_files": 146,
                    "possible_trade_day_files": 146, "absent_dates": [20171201],
                    "unresolved_dates": [], "confirmed_months": months[:7]}
    scoped_second = {"trade_day_files": None, "confirmed_trade_day_files": 137,
                     "possible_trade_day_files": 147, "absent_dates": [],
                     "unresolved_dates": market_days[137:147], "confirmed_months": months[:7]}
    receipt = {"remote_availability": {
        "market_wide_complete_dates": market_days, "market_wide_months": months,
        "market_wide_trade_day_files": 2303, "market_wide_parquet_files": 6909,
        "market_wide_partial_dates": [], "market_wide_partial_parquet_files": 0,
        "footer_audit": {"scope_from": 20170601, "scope_through": 20171231,
                         "complete_date_bundles_attempted": 147, "table_footers_attempted": 441,
                         "table_footers_failed": 0, "uninspected_after_scope": 2156,
                         "by_stock": {STOCKS[0]: scoped_first, STOCKS[1]: scoped_second}},
        "by_stock": {STOCKS[0]: first, STOCKS[1]: second}, STOCKS[0]: first, STOCKS[1]: second,
    }}
    text = _remote_availability(receipt)
    assert "2303 个" in text and "6909 个" in text
    assert "2017-06–2026-11（114 月）" in text
    assert "147 个完整日期组" in text and "441 个 Parquet footer" in text
    assert "| 000001.SZ | 146 | 146 | 1（20171201） | 0 |" in text
    assert "| 300001.SZ | 137 | 137–147（未确定） | 0 | 10 |" in text
    assert "146–2302（未确定）" in text and "137–2303（未确定）" in text
    assert "2156" in text and "2166" in text
    assert "verification/test-data-reconciliation.json" in text
    assert len(text.splitlines()) < 20 and len(text) < 2000
    assert str(market_days[700]) not in text
