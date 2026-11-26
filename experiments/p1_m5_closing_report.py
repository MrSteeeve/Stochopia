"""Render the six M5 closing tables without reading market data or rerunning evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from math import isclose, isfinite
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.p1_m4_report import ARMS, STOCKS, _number, _read, _table, _weighted

Json = dict[str, Any]
IMPACT = "lob_impact_score"
NLL = "val_nll_total"
MODES = {
    "数据证据组": ("evidence_gpt", "evidence_ds"),
    "仅指标组": ("metrics_gpt", "metrics_ds"),
    "无 LLM（固定占位）": ("no_llm",),
}
GUARDS = ("nll_passed", "exhaustion_passed", "drift_passed")


# --- Shared readout formatting and frozen-trial indexing ---


def _finite(value: object) -> bool:
    return type(value) in (float, int) and isfinite(value)


def _clean(value: object) -> str:
    return str(value).replace("|", "｜").replace("\n", " ")


def _count_ratio(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)" if denominator else "0/0（未定义）"


def _interval(bounds: object, digits: int = 4) -> str:
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2 and all(_finite(x) for x in bounds):
        return f"[{_number(bounds[0], digits)}, {_number(bounds[1], digits)}]"
    return "—"


def _estimate(readout: Mapping[str, Any], digits: int = 4) -> str:
    return f"{_number(readout.get('estimate'), digits)} {_interval(readout.get('interval'), digits)}"


def _prediction(row: Mapping[str, Any], role: str) -> Json | None:
    return next((item for item in row.get("prediction_verdicts", []) if item["role"] == role), None)


def _point(verdict: Json | None) -> str:
    if not verdict or not verdict.get("point_available"):
        return "未定义"
    return "命中" if verdict.get("point_in_interval") else "未命中"


def _guard_passed(trial: Json) -> bool:
    """Count measured hard guards, independently of whether C2 admits a score."""
    guards = trial.get("guardrails", {})
    if not all(guards.get(key) is True for key in (*GUARDS, "finite_nll_by_seed")):
        return False
    nll = guards.get("delta_val_nll_total")
    exhausted = guards.get("delta_exhausted_windows")
    parent_drift = guards.get("parent_free_abs_drift_ticks")
    child_drift = guards.get("child_free_abs_drift_ticks")
    if not all(_finite(value) for value in (nll, exhausted, parent_drift, child_drift)):
        return False
    nll_passed = nll <= 0.05 or isclose(nll, 0.05, rel_tol=0, abs_tol=1e-12)
    return nll_passed and exhausted <= 0 and child_drift < max(20.0, parent_drift)


def _trials(manifest: Json) -> list[Json]:
    rows = []
    for trajectory in manifest["trajectories"].values():
        for trial in trajectory["trials"]:
            row = dict(trial)
            row.update(stock=trajectory["stock"], arm=trajectory["arm"], search_seed=trajectory["search_seed"])
            rows.append(row)
    return rows


def _support(metrics: Json, readout: Json) -> str:
    """Support is paired day count and valid draw count, not window/seed count."""
    paired = readout.get("paired_days")
    registered = readout.get("registered_days")
    draws = readout.get("bootstrap_valid")
    total = readout.get("bootstrap_total")
    counts = f"日 {paired}/{registered}；重抽 {draws}/{total}" if paired is not None else "日/重抽未定义"
    reason = readout.get("reason") or metrics.get("lob_impact_score_reason")
    c2 = metrics.get("claims", {}).get("C2", "—")
    detail = []
    for side in ("parent", "candidate"):
        support = readout.get("bootstrap_support", {}).get(side)
        if support is not None:
            n_exec = len(support.get("incomplete_execution_dates", []))
            n_recover = len(support.get("incomplete_recovery_dates", []))
            undefined = support.get("undefined_draw_count", "—")
            detail.append(f"{side}缺执行/回落日 {n_exec}/{n_recover}，未定义重抽 {undefined}")
    extra = "；" + "；".join(detail) if detail else ""
    return f"C2={c2}；{counts}{extra}" + (f"；{_clean(reason)}" if reason else "")


def _model_id(result: Json) -> str:
    return result.get("model_id") or (result.get("candidate") or {}).get("model_id", "—")


def _score_text(result: Json) -> str:
    metrics = result.get("metrics", {})
    score = metrics.get(IMPACT)
    if _finite(score):
        return _number(score, 4)
    return "未定义" if result.get("status") == "measured" else "未测量"


def _metrics_table_count(artifact: Path) -> str:
    """Summarize the repaired prior V2 archive; never compare its old test as M5 holdout."""
    previous = artifact.parent / "p1-formal-20261124" / "verification"
    path = previous / "report.json"
    if not path.exists():
        return "本报告未获得补齐的 V2 独立复验账本，不报告其数字。"
    report = _read(path)
    retry = report.get("resource_retry", {})
    recovered = retry.get("recovered_task_ids", [])
    finite_recovered = 0
    recovered_records = 0
    for task_id in recovered:
        task = previous / "results" / f"{task_id}.json"
        if task.exists():
            recovered_records += 1
            if _finite(_read(task).get("metrics", {}).get(IMPACT)):
                finite_recovered += 1
    baseline_rows = []
    for stock in STOCKS:
        initial = next(row["initial"] for row in report["trajectories"].values() if row["stock"] == stock)
        baseline_rows.append(f"{stock} {_number(initial['metrics'].get(IMPACT), 4)} 分 / NLL {_number(initial['metrics'].get(NLL), 6)}")
    return (
        f"V2 独立复验完成 {report['measured_tasks']}/{report['scheduled_tasks']} 个登记任务，"
        f"资源补跑任务 {len(recovered)} 个，其中结果记录可核对 {recovered_records}/{len(recovered)}，有限综合分 {finite_recovered}/{recovered_records}，"
        f"测量完成不等于 C2 可识别。旧留出集种子：{'；'.join(baseline_rows)}。"
        "旧留出日已经计入 M5 的开发集，不用于 M5 独立泛化结论。"
    )


# --- Table 1: reconstruction and actual calendar ---


def _data_rows(artifact: Path, manifest: Json, receipt: Json) -> str:
    """Aggregate stock-day event sums, positive-spread anchors and snapshot-weighted error.

    The input is disjoint train/dev/test day records and anchor metadata, not
    the order book itself. Weight each date's match rate and log-depth MAE by
    its number of reference snapshots. If a split lacks complete records, the
    table reports the available coverage instead of imputing missing days.
    """
    dev_path = artifact / "development-data-reconciliation.json"
    dev_meta = _read(dev_path) if dev_path.exists() else {}
    dev_records = receipt.get("development_reconciliation", dev_meta.get("reconciliation", []))
    dev_anchors = receipt.get("development_anchors", dev_meta.get("anchors", []))
    rows = []
    for split in ("train", "dev", "test"):
        dates = set(receipt["test_dates"] if split == "test" else manifest["data"][f"{split}_dates"])
        records = receipt["reconciliation"] if split == "test" else dev_records
        anchors = receipt["anchors"] if split == "test" else dev_anchors
        for stock in STOCKS:
            units = [row for row in records if row["symbol"] == stock and row["date"] in dates]
            seen_dates = {row["date"] for row in units}
            if len(units) != len(seen_dates):
                raise ValueError(f"duplicate {split} reconciliation: {stock}")
            valid_anchors = [a for a in anchors if a["symbol"] == stock and a["date"] in dates and a.get("valid", True) and a.get("spread_ticks", 0) > 0]
            seen_anchors = {(a["date"], a["session"]) for a in valid_anchors}
            if len(seen_anchors) != len(valid_anchors):
                raise ValueError(f"duplicate {split} anchors: {stock}")
            match = _weighted(units, "touch_match_rate", "n_snapshots")
            depth = _weighted(units, "top10_log_vol_mae", "n_snapshots")
            rows.append((stock, split, f"{len(seen_dates)}/{len(dates)}",
                         sum(row["n_events"] for row in units) if units else "未核对",
                         f"{len(valid_anchors)}/{2 * len(dates)}" if anchors else "未核对",
                         _number(None if match is None else 100 * match, 4), _number(depth, 6)))
    return _table(("股票", "划分", "重构日/注册日", "已核对事件数", "有效双边窗口/理论数", "一档同时匹配 %", "十档对数深度 MAE"), rows)


def _month_span(months: object) -> str:
    """Preserve the observed span and month count without enumerating the calendar."""
    if not isinstance(months, list) or not months:
        return "—"
    first, last = min(months), max(months)
    return f"{first}–{last}（{len(months)} 月）"


def _metadata_value(value: object) -> str:
    return str(value) if value is not None else "未知"


def _count_list(values: object) -> str:
    return str(len(values)) if isinstance(values, list) else "未知"


def _membership_bounds(row: Json) -> str:
    exact = row.get("trade_day_files")
    if exact is not None:
        return str(exact)
    lower = row.get("confirmed_trade_day_files")
    upper = row.get("possible_trade_day_files")
    if lower is not None and upper is not None:
        return f"{lower}–{upper}（未确定）"
    return "未知"


def _remote_availability(receipt: Json) -> str:
    """Summarize remote metadata; preserve original date-level lists in the receipt.

    The complete market-wide files are stock bundles, not individual-stock
    evidence. Footer metadata can prove presence or absence in a limited
    audited period, and leave a symbol unresolved in the interior range.
    Do not infer either absence or positive-spread anchors for unknown dates.
    """
    available = receipt.get("remote_availability")
    if not available:
        return "20170601+ 远端库存未取得可核验的元数据；不外推单股可用日。"
    market_dates = available.get("market_wide_complete_dates", [])
    date_span = f"{min(market_dates)}–{max(market_dates)}" if market_dates else "未列明"
    market_summary = (
        f"20170601+ 远端仅盘点元数据。全市场完整交易日文件组 {_metadata_value(available.get('market_wide_trade_day_files'))} 个，"
        f"Parquet 表 {_metadata_value(available.get('market_wide_parquet_files'))} 个；"
        f"不完整日期 {_count_list(available.get('market_wide_partial_dates'))} 个，"
        f"不完整日期中的表 {_metadata_value(available.get('market_wide_partial_parquet_files'))} 个。"
        f"现有日期范围 {date_span}，月份 {_month_span(available.get('market_wide_months'))}。"
        "全市场文件组按日包含多只股票，不等同于任一单股的数据覆盖。"
    )
    footer = available.get("footer_audit") or {}
    audit_summary = "未取得按股票核验的 Footer 审计统计。"
    if footer:
        audit_summary = (
            f"在 {_metadata_value(footer.get('scope_from'))}–{_metadata_value(footer.get('scope_through'))} 范围，"
            f"核对 {_metadata_value(footer.get('complete_date_bundles_attempted'))} 个完整日期组、"
            f"{_metadata_value(footer.get('table_footers_attempted'))} 个 Parquet footer；"
            f"失败 {_metadata_value(footer.get('table_footers_failed'))} 个。"
            f"范围后另有 {_metadata_value(footer.get('uninspected_after_scope'))} 个完整日期组未作单股核对。"
            "仅读取 footer 统计及尾部定位字节，未读取任何盘口数据页；所有三种表的 `wind_code` 精确 min/max 端点是确证依据。"
            "区间内部不能确证的日期是未知，不是缺失。"
        )
    rows = []
    for stock in STOCKS:
        full = (available.get("by_stock") or {}).get(stock) or available.get(stock) or {}
        scoped = (footer.get("by_stock") or {}).get(stock) or {}
        scoped_absent = scoped.get("absent_dates")
        absent = _count_list(scoped_absent)
        if isinstance(scoped_absent, list) and 0 < len(scoped_absent) <= 3:
            absent += "（" + "、".join(map(str, scoped_absent)) + "）"
        rows.append((stock, _metadata_value(scoped.get("confirmed_trade_day_files")), _membership_bounds(scoped),
                     absent, _count_list(scoped.get("unresolved_dates")), _month_span(scoped.get("confirmed_months")),
                     _membership_bounds(full), _count_list(full.get("unresolved_dates")),
                     _month_span(full.get("possible_market_months"))))
    stock_summary = _table(("股票", "审计范围确证日期", "审计范围单股日期/上下界", "范围内确证缺席", "范围内未知",
                            "范围内确证月份", "20170601+ 单股总日期/上下界", "全范围未知日期", "全范围可能月份"), rows)
    receipt_ref = (
        "逐日清单、各月文件数、确证缺席/未决日期、完整 metadata 审计状态和源版本见"
        " [OpenHyra 的原始对账凭据](https://github.com/MrSteeeve/OpenHyra/blob/world-model/artifacts/p1-m5-formal-20261124/verification/test-data-reconciliation.json)"
        "（`remote_availability` 字段）；上述覆盖仅表示文件/代码元数据，不保证正价差双边窗口或交易日有效性。"
    )
    source = available.get("source_revision")
    if source is not None:
        receipt_ref += f" 数据源固定版本 `{source}`。"
    return "\n\n".join((market_summary, audit_summary, stock_summary, receipt_ref))


# --- Table 2: all 90 development trials ---


def _development_rows(trials: list[Json]) -> str:
    rows = []
    for stock in STOCKS:
        for arm in ARMS:
            units = [row for row in trials if row["stock"] == stock and row["arm"] == arm]
            finite = [unit for unit in units if _finite(unit.get("mean_delta_score"))]
            scores = [unit.get("candidate_metrics", {}).get(IMPACT) for unit in units]
            scores = [score for score in scores if _finite(score)]
            counts = Counter(unit["verdict"] for unit in units)
            compiled = sum(unit.get("candidate") is not None for unit in units)
            guarded = sum(_guard_passed(unit) for unit in units)
            gain = sum(unit["mean_delta_score"] > 0 for unit in finite)
            half = sum(unit["mean_delta_score"] >= 0.5 for unit in finite)
            p10 = sum(bool(unit.get("has_benchmark_gain")) for unit in units)
            p05 = sum(bool(unit.get("has_benchmark_gain")) and _finite(unit.get("p05_delta_score")) and unit["p05_delta_score"] > 0 for unit in units)
            rows.append((stock, arm, len(units), _count_ratio(compiled, len(units)), _count_ratio(guarded, len(units)),
                         len(scores), _count_ratio(gain, len(units)), _count_ratio(gain, len(finite)), half, f"{p10}/{p05}",
                         counts["VERIFIED_GAIN"], counts["UNVERIFIED_GAIN"],
                         _number(mean(scores) if scores else None, 4),
                         _number(mean(row["mean_delta_score"] for row in finite) if finite else None, 4)))
    return _table(("股票", "实验臂", "试验", "编译通过率", "硬护栏通过率", "有限候选分 n", "涨分/全试验", "涨分/有限配对 Δ",
                   "点 Δ≥0.50", "净涨分门槛 p10/p05", "VERIFIED_GAIN", "UNVERIFIED_GAIN", "有限候选分均值", "有限配对 Δ 均值"), rows)


# --- Table 3: endpoints and reasons for no promotion ---


def _primary_endpoints(report: Json) -> list[Json]:
    """Deduplicate common seed endpoints and repeat trajectory/audit references."""
    endpoints: dict[tuple[str, str], Json] = {}
    for trajectory in report["trajectories"].values():
        stock = trajectory["stock"]
        initial = trajectory["initial"]
        final = trajectory["final"]
        for role, row in (("seed-gru", initial), ("X_final", final)):
            if row is None or row.get("status") == "not_scheduled":
                raise ValueError("all preregistered initial/final endpoints must be included")
            key = stock, row["task_id"]
            if key not in endpoints:
                endpoints[key] = {**row, "stock": stock, "endpoint_roles": set(), "arm": trajectory["arm"]}
            endpoints[key]["endpoint_roles"].add(role)
    for result in report["audit_only_rejected_unverified"]:
        key = result["stock"], result["task_id"]
        if key not in endpoints:
            endpoints[key] = {**result, "endpoint_roles": set(), "arm": result.get("group", "—")}
        endpoints[key]["endpoint_roles"].add("UNVERIFIED_GAIN")
    return list(endpoints.values())


def _primary_rows(report: Json) -> str:
    rows = []
    for result in _primary_endpoints(report):
        study_roles = "+".join(sorted(result["endpoint_roles"]))
        parent_readouts = result.get("readouts", {})
        seed_readouts = result.get("seed_readouts", parent_readouts)
        seed_score = seed_readouts.get(IMPACT, {})
        seed_nll = seed_readouts.get(NLL, {})
        metrics = result.get("metrics", {})
        rows.append((result["stock"], study_roles, _model_id(result), result["status"], _score_text(result),
                     _estimate(seed_score), _number(seed_score.get("p05"), 4), _number(seed_score.get("p10"), 4),
                     _number(metrics.get(NLL), 6), _estimate(seed_nll, 6), _estimate(parent_readouts.get(IMPACT, {})),
                     _support(metrics, seed_score)))
    return _table(("股票", "A 集合端点", "模型", "状态", "test 得分", "对本股种子配对 Δ [95%]", "p05", "p10",
                   "test NLL", "对种子 ΔNLL [95%]", "对直接父模型 Δ [95%]", "样本支持"), rows)


def _prediction_details(trial: Json, role: str) -> str:
    registered = next((item for item in trial.get("predictions", []) if item["role"] == role), None)
    verdict = _prediction(trial, role)
    if not registered or not verdict:
        return "未登记或未判定"
    metric = registered["metric"]
    value = trial.get("readouts", {}).get(metric, {})
    direction = registered["direction"]
    band = _interval(registered.get("equivalence_band"), 6)
    return (f"{metric} {direction}；预测 {_interval(registered['interval'], 6)}；等效带 {band}；"
            f"实际 {_estimate(value, 6)}，p05={_number(value.get('p05'), 6)}，p10={_number(value.get('p10'), 6)}，"
            f"p90={_number(value.get('p90'), 6)}，p95={_number(value.get('p95'), 6)}；"
            f"{verdict['status']}，点值{_point(verdict)}")


def _bottlenecks(trial: Json) -> str:
    """List each failed *registered* admission condition; avoid treating point miss as veto."""
    reasons = []
    guard = trial.get("guardrails", {})
    for key in (*GUARDS, "finite_nll_by_seed"):
        if guard.get(key) is not True:
            reasons.append(key)
    delta = trial.get("mean_delta_score")
    p10 = trial.get("p10_delta_score")
    if not _guard_passed(trial) and all(guard.get(key) is True for key in (*GUARDS, "finite_nll_by_seed")):
        reasons.append("硬护栏数值与登记标志不一致或缺失")
    if not _finite(delta) or delta < 0.5:
        reasons.append("净增量不足 0.50 或未定义")
    if not _finite(p10) or p10 <= 0:
        reasons.append("p10 未过零或未定义")
    if trial["arm"] != "no_llm":
        if (_prediction(trial, "primary") or {}).get("status") != "SUPPORTED":
            reasons.append("主预测未获支持")
        if (_prediction(trial, "guardrail") or {}).get("status") == "REFUTED":
            reasons.append("护栏预测被推翻")
    return "；".join(reasons) if reasons else "无本轮晋升卡点"


def _dev_bottleneck_rows(trials: list[Json]) -> str:
    # All point-gain candidates at/above the registered magnitude threshold,
    # including explicit guardrail failures: no discretionary top-K truncation.
    heads = sorted((row for row in trials if _finite(row.get("mean_delta_score")) and row["mean_delta_score"] >= 0.5),
                   key=lambda row: (row["stock"], -row["mean_delta_score"], row["trial_id"]))
    rows = []
    for trial in heads:
        guard = trial.get("guardrails", {})
        values = trial.get("readouts", {})
        rows.append((trial["stock"], trial["trial_id"], trial["verdict"], _estimate(values.get(IMPACT, {}), 6),
                     _number(trial.get("p05_delta_score"), 6), _number(trial.get("p10_delta_score"), 6),
                     _estimate(values.get(NLL, {}), 6), _number(guard.get("delta_exhausted_windows")),
                     f"{_number(guard.get('parent_free_abs_drift_ticks'))}→{_number(guard.get('child_free_abs_drift_ticks'))}",
                     _prediction_details(trial, "primary"), _prediction_details(trial, "guardrail"), _bottlenecks(trial)))
    return _table(("股票", "开发试验", "实际终态", "开发 ΔScore [95%]", "p05", "p10", "配对 ΔNLL [95%]", "耗尽窗口 Δ",
                   "自由漂移 ticks 父→子", "事前主预测明细", "事前护栏预测明细", "未晋升瓶颈"), rows)


# --- Table 4: the pre-holdout selected secondary audit ---


def _secondary_rows(trials: list[Json], report: Json) -> str:
    rows = []
    by_id = {trial["trial_id"]: trial for trial in trials}
    for selection in report["secondary_best_guarded"]:
        stock, arm = selection["stock"], selection["arm"]
        result = selection.get("result")
        candidate = selection.get("candidate") or {}
        if result is None:
            rows.append((stock, arm, "无护栏内代表", "—", "—", "—", "—", "—", "—", "—", "—", "—", "未调度", "—"))
            continue
        readouts = result.get("readouts", {})
        seed_readouts = result.get("seed_readouts", readouts)
        score = readouts.get(IMPACT, {})
        seed_score = seed_readouts.get(IMPACT, {})
        metrics = result.get("metrics", {})
        verdict = _prediction(result.get("assessment") or {}, "primary")
        dev_verdict = _prediction(by_id[selection["trial_id"]], "primary")
        rows.append((stock, arm, selection["trial_id"], candidate.get("source_sha256", "—"),
                     _number(selection.get("dev_score"), 4), _number(selection.get("dev_delta"), 4), _score_text(result),
                     _estimate(score), _estimate(seed_score), _estimate(readouts.get(NLL, {}), 6),
                     f"{dev_verdict['status']} / {_point(dev_verdict)}" if dev_verdict else "未判定",
                     f"{verdict['status']} / {_point(verdict)}" if verdict else "未判定", result["status"], _support(metrics, score)))
    return _table(("股票", "实验臂", "开发试验", "源码 SHA256", "dev 分", "dev Δ", "test 分", "test Δ 对开发父架构 [95%]",
                   "test Δ 对本股种子 [95%]", "test ΔNLL 对父架构 [95%]", "dev 主预测方向 / 点值", "test 主预测方向 / 点值", "执行状态", "样本支持"), rows)


# --- Table 5: prediction calibration and score survival ---


def _forecast_rows(trials: list[Json], report: Json) -> str:
    """Compare registered predictions; a missing readout is inconclusive, not a miss.

    Input development units are unweighted proposal-trial records. The elite
    holdout subset is matched by trial ID, using its original parent and
    frozen predictions, not by model score ranking computed on holdout.
    """
    rows = []
    tests = report["tasks"]
    for group, arms in MODES.items():
        units = [row for row in trials if row["arm"] in arms]
        for role in ("primary", "guardrail"):
            for split in ("dev", "test"):
                forecasts = []
                applicable = units if split == "dev" else [unit for unit in units if unit["trial_id"] in tests]
                for unit in applicable:
                    if not any(p["role"] == role for p in unit.get("predictions", [])):
                        continue
                    outcome = unit if split == "dev" else tests[unit["trial_id"]].get("assessment") or {}
                    verdict = _prediction(outcome, role)
                    forecasts.append(verdict or {"status": "INCONCLUSIVE", "point_available": False})
                counted = Counter(f["status"] for f in forecasts)
                points = [f for f in forecasts if f.get("point_available")]
                hits = sum(bool(f.get("point_in_interval")) for f in points)
                rows.append((group, role, split, len(applicable), len(forecasts), counted["SUPPORTED"], counted["REFUTED"],
                             counted["INCONCLUSIVE"], _count_ratio(hits, len(points)), len(forecasts) - len(points)))
    return _table(("实验组", "预测角色", "数据划分", "涉及试验 n", "登记预测 n", "SUPPORTED", "REFUTED", "INCONCLUSIVE",
                   "点值命中 / 可评点", "点值不可评"), rows)


def _holdout_original_trials(trials: list[Json], report: Json) -> list[tuple[Json, Json]]:
    """One original trial per selected same-stock measured candidate forecast."""
    tests = report["tasks"]
    entries = []
    for trial in trials:
        if trial["arm"] == "no_llm":
            continue
        result = tests.get(trial["trial_id"])
        if result is not None:
            entries.append((trial, result))
    return entries


def _survival_rows(trials: list[Json], report: Json) -> str:
    """Survival uses candidate-minus-original-parent holdout Δ, strictly > 0.

    Restrict to positive development deltas for true score 'survival'. A
    separate stronger endpoint checks the same compound benchmark gate.
    Denominators exclude unknown score deltas, with coverage disclosed;
    failed evaluations and C2-undefined scores do not count as zero returns.
    The small selected sample is descriptive and not a causal comparison.
    """
    categories = ("点值命中", "点值未命中", "未证伪", "证伪", "方向获支持")
    selected = _holdout_original_trials(trials, report)
    positive_dev = [(t, r) for t, r in selected if _finite(t.get("mean_delta_score")) and t["mean_delta_score"] > 0]
    reports = []
    for group in ("数据证据组", "仅指标组", "全部 LLM"):
        arms = MODES.get(group, (*MODES["数据证据组"], *MODES["仅指标组"]))
        subsets: dict[str, list[tuple[Json, Json]]] = {label: [] for label in categories}
        for trial, result in positive_dev:
            if trial["arm"] not in arms:
                continue
            verdict = _prediction(trial, "primary")
            if verdict is None:
                continue
            if verdict.get("point_available"):
                key = "点值命中" if verdict.get("point_in_interval") else "点值未命中"
                subsets[key].append((trial, result))
            if verdict["status"] in ("SUPPORTED", "INCONCLUSIVE"):
                subsets["未证伪"].append((trial, result))
            elif verdict["status"] == "REFUTED":
                subsets["证伪"].append((trial, result))
            if verdict["status"] == "SUPPORTED":
                subsets["方向获支持"].append((trial, result))
        for category, pairs in subsets.items():
            measured = [result for _, result in pairs if result["status"] == "measured"]
            finite = [result for result in measured if _finite(result.get("readouts", {}).get(IMPACT, {}).get("estimate"))]
            positive = sum(result["readouts"][IMPACT]["estimate"] > 0 for result in finite)
            interval = [result for result in finite if result["readouts"][IMPACT].get("interval") is not None]
            robust = sum(bool((result.get("assessment") or {}).get("has_benchmark_gain")) for result in interval)
            reports.append((group, category, len(pairs), len(measured), len(measured) - len(finite), len(pairs) - len(measured),
                            _count_ratio(positive, len(finite)), _count_ratio(robust, len(interval))))
    return _table(("实验组", "开发主预测类别", "入选且开发涨分", "完成测量 n", "测量但配对分未定义", "未测量 n",
                   "test 点 Δ>0 / 有限配对 Δ", "test 净涨分门槛 / 区间可判定"), reports)


# --- Table 6: source-stock selection and target-stock training ---


def _cross_rows(report: Json) -> str:
    rows = []
    for category, label in (("llm_best", "LLM 最优代表"), ("no_llm_best", "no_llm 最优代表")):
        for source in STOCKS:
            transfer = report["cross_regime"][category][source]
            representative = transfer.get("representative")
            if isinstance(representative, dict):
                model_name = representative["model_id"]
            else:
                model_name = representative or "无护栏内代表"
            cells = []
            for target in STOCKS:
                item = transfer["cells"][target]
                values = item.get("readouts", {})
                score = values.get(IMPACT, {})
                nll = values.get(NLL, {})
                metrics = item.get("metrics", {})
                prefix = "本股" if target == source else "跨股重训"
                cells.append(f"{prefix}；{item['status']}；Δ分 {_estimate(score)}；ΔNLL {_estimate(nll, 6)}；{_support(metrics, score)}")
            rows.append((label, source, model_name, _number(transfer.get("dev_score"), 4), *cells))
    return _table(("源股代表种类", "来源股票", "开发代表架构", "dev 分", "000001.SZ 配对对种子 [95%]", "300001.SZ 配对对种子 [95%]"), rows)


# --- Public Markdown report; no changes to the legacy full-census path ---


def render_closing(artifact: Path, manifest: Json, report: Json, receipt: Json, architecture_notes: str) -> str:
    """Render an already-frozen M5 elite study using only detached evidence files."""
    if len(report["test_dates"]) != 20 or len(receipt["reconciliation"]) != 40 or len(receipt["anchors"]) != 80:
        raise ValueError("closing report requires 20 audited common trading days and two windows/day")
    if report["code"] != manifest["code"] or report["test_dates"] != receipt["test_dates"]:
        raise ValueError("frozen development code and effective holdout dates must agree")
    trials = _trials(manifest)
    if len(trials) != 90 or len({row["trial_id"] for row in trials}) != 90:
        raise ValueError("the frozen study must contain exactly 90 unique development trials")
    groups = Counter((row["stock"], row["arm"]) for row in trials)
    expected = {(stock, arm) for stock in STOCKS for arm in ARMS}
    if set(groups) != expected or any(count != 9 for count in groups.values()):
        raise ValueError("the 90 trials must cover every stock and arm at nine each")
    secondary = [(row["stock"], row["arm"]) for row in report["secondary_best_guarded"]]
    if len(secondary) != 10 or set(secondary) != expected:
        raise ValueError("secondary audit must list all ten stock-arm cells, including empty cells")
    if not architecture_notes.strip():
        raise ValueError("include a source-based analysis of the selected operators")
    counts = Counter(trial["verdict"] for trial in trials)
    valid_scores = sum(_finite(row.get("candidate_metrics", {}).get(IMPACT)) for row in trials)
    valid_deltas = sum(_finite(row.get("mean_delta_score")) for row in trials)
    positive = sum(_finite(row.get("mean_delta_score")) and row["mean_delta_score"] > 0 for row in trials)
    overview = (f"90 次开发试验全部封存；有限候选分 {valid_scores}/90，有限配对 Δ {valid_deltas}/90，"
                f"点涨分 {positive}/90，VERIFIED_GAIN {counts['VERIFIED_GAIN']}，UNVERIFIED_GAIN {counts['UNVERIFIED_GAIN']}。"
                f"精简留出复验完成独立重训/轨迹测量 {report['measured_tasks']}/{report['scheduled_tasks']} 个股票×源码任务，"
                f"配对比较完成 {report.get('measured_comparisons', '—')}/{report.get('scheduled_comparisons', '—')} 个；"
                "完成测量不保证 LOB-Impact 可定义或区间可判定。")
    code = manifest["code"]
    dates = "、".join(map(str, report["test_dates"]))
    exclusions = receipt["excluded_test_dates"]
    excluded = "无顺延。" if not exclusions else _table(("顺延剔除日", "登记理由"),
                                                            [(row["date"], _clean(row["reason"])) for row in exclusions])
    frozen_utc = manifest.get("frozen_at_utc") or manifest.get("frozen_at")
    if not frozen_utc:
        frozen_utc = report.get("closing_registration", {}).get("frozen_manifest_mtime_utc", "见冻结清单")
    methods = (
        "配对差定义为候选减去比较基线。每次使用同一交易日重抽索引同时聚合两模型，"
        "综合分由重抽面板重新汇总的物理读数计算而非逐日综合分简单平均。2,000 次交易日成组重抽，"
        "共同有效日起点不少于 max(6, 注册日数的一半)，有效重抽不少于 1,800 次。95% 区间为 p025/p975；"
        "预登记性能下界为 p10>0，同时保留 p05 敏感性。分数条件于完整面板 C2 可识别，"
        "不能把 C2 未定义视为零分或无影响。NLL 为 nats/事件，ΔNLL 是等权逐日配对均值；"
        "硬护栏要求各训练种子 NLL 有限、完整日 NLL 增量 ≤+0.05、耗尽窗口增量 ≤0、"
        "候选自由漂移低于 20 ticks（若父模型已 ≥20，则严格低于父模型）。"
        "主预测 increase：点 Δ 达登记区间近零端且 p10>0；decrease：点 Δ 达近零端且 p90<0。"
        "点方向不符或 95% 区间完全错过近零端时证伪。等价性要求 p05/p95 进入登记等效带（TOST），"
        "95% 区间与等效带不交才证伪。区间未定义或其余结果不可判定；点估计落入事前区间单独计分，不替代方向判定。"
        "晋升须通过硬护栏、Δ分≥0.50、p10>0；LLM 另需主方向 SUPPORTED 且护栏预测非 REFUTED。"
        "no_llm 使用固定预测占位，不受 LLM 机制预测晋升门槛约束。"
    )
    protocol = (
        f"两只股票各 36 日训练；旧 V2 的 12 日留出并入 12 日原开发，得 24 日开发。"
        f"五臂×两股票×三搜索种子×三轮。冻结开发代码为 Stochopia `{code['stochopia_commit']}`、"
        f"OpenHyra `{code['openhyra_commit']}`。冻结时间：{frozen_utc}（原清单未内嵌时间时取精简注册中的原清单 mtime）。报告遵循已冻结的开发期终态；"
        "留出名单按用户收尾协议精简为 A（预注册初始/最终与未证实涨分拦截审计）、B（每股票×臂最高护栏内 dev 分），"
        "C（每股 LLM 与无 LLM 最优两个源码分别跨股重训）。"
        "B/C 均标注为 Post-Freeze Secondary Audit，不把其留出涨分倒算为正式晋升，也不把未调度的首轮 X1 当作失效候选。"
    )
    summary = "; ".join(f"{kind}: {count}" for kind, count in sorted(counts.items()))
    parts = (
        "# P1-M5 深市 Level-2 高统计功效世界模型研究：正式收尾报告",
        protocol,
        "## 结果摘要", overview, f"开发终态：{summary}。", _metrics_table_count(artifact),
        "## 预登记统计和测量口径", methods,
        "## 表 1　三阶段数据对账与 P2 数据余量", f"实际独立留出日期：{dates}。", excluded,
        "按交易日汇总事件数，一档匹配率及十档对数深度 MAE 按参考快照数加权。"
        "训练窗口仅为回放起点可用性核查；训练不执行这批留出式生成实验。",
        _data_rows(artifact, manifest, receipt), _remote_availability(receipt),
        "## 表 2　90 次开发搜索全景统计", "编译通过与护栏通过分母为所有登记试验；涨分同时报告全试验和有限配对 Δ 分母。"
        "净涨分门槛另计 p10 与 p05；候选分均值只在有限综合分中计算，不能代表组整体期望。",
        _development_rows(trials),
        "## 表 3　预注册主终点与拦截审计（集合 A）", "所有共用种子和重复端点只列一次；若无晋升，X_final 读数复用种子。"
        "对初始固定种子的累积配对差与对直接父架构的局部配对差分开报告。", _primary_rows(report),
        "### 开发期头部候选未晋升的具体卡点", "列出全部达到点 Δ≥0.50 的开发候选，包括违反护栏的试验。"
        "点值命中与方向支持是两件事，点值未命中不是本轮独立否决项。以下数值保留六位小数；完整机器精度见冻结 JSON。",
        _dev_bottleneck_rows(trials),
        "## 表 4　冻结后次要审计：五组×两股 best_dev_guarded 代表（集合 B）",
        "Post-Freeze Secondary Audit。按全部三轮的最高有限开发综合分择一且须通过硬护栏，无候选时明确留空。"
        "test 主预测始终相对开发期的直接父架构，另报告可比种子配对差。", _secondary_rows(trials, report),
        "### 核心算子：基于冻结源码核验的数学方程", architecture_notes.strip(),
        "## 表 5　X → ΔY 事前预测校准度与样本外保分率",
        "证据组与指标组各为两供应商的合并试验数（各 36 项）；无 LLM 的 18 项是固定占位，不用于 LLM 率比较。"
        "每个角色以登记预测为方向分母，以点估计有限的预测为幅度分母；缺失判定单独记不可判定。"
        "test 列只包含 A/B 中实际调度的相同股票原有候选，未调度 X1 与跨股预测不计入。",
        _forecast_rows(trials, report),
        "### 开发主预测与留出正分存活率", "仅考察 A/B 中 dev 配对 Δ>0 的 LLM 原始试验，每个试验只计一次。"
        "未证伪含 SUPPORTED 与 INCONCLUSIVE，不等于获支持。点收益存活 Δ>0 与严格净涨分门槛分别报告；"
        "区间未定义不计入严格门槛分母。样本按固定开发排名有选择偏差，重复源码的预测也非独立样本；"
        "目标指标分布、历史反馈和任务复杂性可能不同，描述性率不证明证据引起机制理解改善。",
        _survival_rows(trials, report),
        "## 表 6　精简 2×2 强算子跨股票重训（集合 C）",
        "对角直接复用集合 B 的源码评估及相对本股种子配对差；另一股票的 36 日训练重新初始化权重，"
        "只迁移源码，不迁移权重。两列均为对目标股票种子在相同 20 日留出集的配对差，"
        "不把源股事前预测当成目标股机制陈述。", _cross_rows(report),
        "## 结论边界与证据包", "主终点以冻结 X_final 为准；B/C 高分与泛化读数仅作后续架构研究证据。"
        "20 日交易日成组区间不是因果效应的真值区间；历史真实盘口并未观察随机外生母单的个体反事实。"
        "综合分受 C2 识别约束，分项和 NLL 仍可测，须区分未定义、未调度与评估失败。"
        "不要以冻结后代表集替代全量 X1 普查，也不据此声称 LLM 改进了正式搜索晋升率。",
        "证据：OpenHyra `artifacts/p1-formal-20261124/`（已补齐的 V2）、"
        "`artifacts/p1-m5-formal-20261124/`（冻结清单、源码、开发台账、独立报告和 A/B/C CSV）。"
        "方法参照：[LOB-Impact](https://arxiv.org/html/2610.27100v1)、"
        "[留出集隔离](https://arxiv.org/html/2610.14063v1)。",
    )
    return "\n\n".join(parts) + "\n"
