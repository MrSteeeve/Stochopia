"""Render a completed M5 study from its frozen records and independent receipt."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.p1_m4_report import (
    ARMS, STOCKS, _audit_rows, _cross_rows, _data_checks, _development_rows,
    _holdout_rows, _number, _read, _table,
)


Json = dict[str, Any]
SCHEMA = "stochopia-p1-m5-v1"


# --- Prediction calibration and research trajectory ---


def _prediction_rows(manifest: Json, report: Json) -> str:
    """Report direction and point containment with explicit denominators."""
    rows = []
    for stock in STOCKS:
        for arm in ARMS:
            trials = []
            for trajectory in manifest["trajectories"].values():
                if trajectory["stock"] == stock and trajectory["arm"] == arm:
                    trials.extend(trajectory["trials"])
            for role in ("primary", "guardrail"):
                forecasts = []
                for trial in trials:
                    registered = any(row["role"] == role for row in trial.get("predictions", []))
                    if not registered:
                        continue
                    outcome = next(
                        (row for row in trial.get("prediction_verdicts", []) if row["role"] == role),
                        None,
                    )
                    if outcome is None:
                        # A sealed forecast without an executable or measurable
                        # candidate contributes unavailable direction/point coverage.
                        outcome = {"status": "INCONCLUSIVE", "point_available": False}
                    forecasts.append(outcome)
                counts = Counter(row["status"] for row in forecasts)
                points = [row for row in forecasts if row.get("point_available")]
                hits = sum(bool(row.get("point_in_interval")) for row in points)
                holdout = report["arm_summary"][stock][arm]["prediction_rates"][role]
                rows.append((
                    stock, arm, role, len(forecasts), counts["SUPPORTED"], counts["REFUTED"],
                    counts["INCONCLUSIVE"], f"{hits}/{len(points)}",
                    holdout["registered"], holdout["supported"], holdout["refuted"],
                    holdout["inconclusive"],
                    f"{holdout['point_in_interval']}/{holdout['point_evaluable']}",
                ))
    return _table(
        ("股票", "实验臂", "预测角色", "开发登记预测数", "方向支持", "证伪", "不可判定", "开发点值入带",
         "留出预测数", "方向支持", "证伪", "不可判定", "留出点值入带"), rows,
    )


def _sensitivity_rows(manifest: Json) -> str:
    """Compare candidate gain threshold support at the original and M5 quantiles."""
    rows = []
    for stock in STOCKS:
        for arm in ARMS:
            trials = []
            for trajectory in manifest["trajectories"].values():
                if trajectory["stock"] == stock and trajectory["arm"] == arm:
                    for trial in trajectory["trials"]:
                        if trial.get("has_benchmark_gain"):
                            trials.append(trial)
            legacy = sum(trial["p05_delta_score"] > 0 for trial in trials if trial.get("p05_delta_score") is not None)
            rows.append((stock, arm, len(trials), legacy, len(trials) - legacy))
    return _table(("股票", "实验臂", "通过 M5 涨分门槛", "其中 p05>0", "其中仅 p10>0"), rows)


def _trajectory_case(manifest: Json) -> str:
    """Choose the longest verified LLM trajectory with stable roster-order ties."""
    trajectories = [
        row for row in manifest["trajectories"].values() if row["arm"] != "no_llm"
    ]
    def promotions(trajectory: Json) -> int:
        return sum(trial["verdict"] == "VERIFIED_GAIN" for trial in trajectory["trials"])

    case = max(trajectories, key=promotions)
    rows = []
    for trial in case["trials"]:
        hypothesis = trial["hypothesis"] or {}
        candidate = trial["candidate"] or {}
        explanation = hypothesis.get("observed_bottleneck", "")
        prescription = hypothesis.get("structural_prescription", "")
        rows.append((
            trial["round"], trial["parent"]["model_id"], candidate.get("model_id", "—"),
            hypothesis.get("parent_hypothesis_id") or "种子", explanation.replace("|", "｜").replace("\n", " "),
            prescription.replace("|", "｜").replace("\n", " "), trial["verdict"],
            _number(trial.get("mean_delta_score")), _number(trial.get("p05_delta_score")),
            _number(trial.get("p10_delta_score")),
        ))
    title = (
        f"按开发期已证实晋升次数选择案例，同数按冻结顺序取首。"
        f"本例为 {case['stock']}、{case['arm']}、种子 {case['search_seed']}，"
        f"三轮共 {promotions(case)} 次已证实晋升。"
    )
    return title + "\n\n" + _table(
        ("轮次", "父架构", "候选", "父假设", "诊断 Z", "改动 X", "终态", "Δ分", "p05", "p10"), rows,
    )


# --- Public Markdown output ---


def render(artifact: Path, architecture_notes: str) -> str:
    manifest = _read(artifact / "study_manifest.json")
    report = _read(artifact / "verification" / "report.json")
    receipt = _read(artifact / "verification" / "test-data-reconciliation.json")
    if manifest["schema"] != SCHEMA or report["schema"] != SCHEMA:
        raise ValueError("M5 report requires its independently registered study")
    if manifest["code"] != report["code"] or report["test_dates"] != receipt["test_dates"]:
        raise ValueError("report must match frozen code and effective holdout calendar")
    if len(report["test_dates"]) != 20 or len(receipt["reconciliation"]) != 40 or len(receipt["anchors"]) != 80:
        raise ValueError("the report requires 20 audited common trading days")
    if sum(len(row["trials"]) for row in manifest["trajectories"].values()) != 90:
        raise ValueError("the frozen study must contain 90 development trials")
    if report["scheduled_tasks"] != len(manifest["tasks"]):
        raise ValueError("the holdout report must cover all frozen tasks")
    code = manifest["code"]
    dates = "、".join(map(str, report["test_dates"]))
    exclusions = receipt["excluded_test_dates"]
    excluded = "无顺延" if not exclusions else _table(
        ("剔除交易日", "登记理由"), [(row["date"], row["reason"]) for row in exclusions],
    )
    outcomes = Counter(
        trial["verdict"] for row in manifest["trajectories"].values() for trial in row["trials"]
    )
    audit = report["audit_summary"]
    overview = (
        f"90 次开发试验封存，已证实晋升 {outcomes['VERIFIED_GAIN']} 次，"
        f"未证实涨分 {outcomes['UNVERIFIED_GAIN']} 次。独立复验登记 {report['scheduled_tasks']} 项，"
        f"完成配对测量 {report['measured_tasks']} 项。"
    )
    opening = (
        f"两只深市股票共用预登记的 36 个训练日。将 V2 已开启的 12 个旧留出日并入原开发集，"
        f"组成 24 个开发日。五臂各有三个搜索种子和三轮演化，最后开启全新 20 个留出日。"
        f"锁码为 Stochopia `{code['stochopia_commit']}` 和 OpenHyra `{code['openhyra_commit']}`。"
    )
    methods = (
        "重抽以交易日为单位，共 2,000 次，共同有效日不少于 max(6, 注册日数的一半)，有效重抽不少于 1,800 次。"
        "性能门槛沿用 NLL、回放耗尽及自由漂移护栏。综合分增量至少 +0.50 且 p10>0 才通过涨分门槛。"
        "主方向检验同时要求点增量达到最小预测幅度及单侧 90% 区间过零。等价性采用 90% 区间进入登记带的 TOST 规则。"
        "点值是否进入给定区间单独统计，未定义的点增量不计入有效点预测分母。"
        "无 LLM 组采用固定预测占位。开发侧与留出侧分别计分。"
    )
    audit_summary = (
        f"冻结未证实涨分审计候选 {audit['candidates']} 项，留出可测 {audit['measured']} 项，"
        f"通过留出基准门槛 {audit['survived_benchmark_gate']} 项，"
        f"留出机制获支持 {audit['survived_mechanism_gate']} 项。"
    )
    return "\n\n".join((
        "# P1-M5 深市 Level-2 高统计功效世界模型研究", opening, "## 结果摘要", overview,
        "## 统计与试验协议", methods, "## 独立留出集对账", f"实际交易日为 {dates}。", excluded,
        _data_checks(receipt).replace("外生事件", "重构事件"),
        "## 开发三轮演化", _development_rows(manifest).replace("无净提升", "未通过晋升门槛"),
        "### 单侧门槛敏感性", _sensitivity_rows(manifest), "### 三轮诊断与谱系", _trajectory_case(manifest),
        "## 留出集 X0 / X1 / Xfinal", "NLL 单位为 nats/事件，— 表示读数不可用。", _holdout_rows(report),
        "## 预测证实与幅度校准", _prediction_rows(manifest, report),
        "## 未证实涨分候选审计", audit_summary, _audit_rows(report),
        "## 双股架构重训", "同一架构源码在目标股票训练。辅助行是开发期冻结的次选架构。", _cross_rows(report),
        "## 算子结构与结果解释", architecture_notes.strip(),
    )) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--architecture-notes", type=Path, required=True)
    args = parser.parse_args()
    notes = args.architecture_notes.read_text(encoding="utf-8")
    args.output.write_text(render(args.artifact, notes), encoding="utf-8")


if __name__ == "__main__":
    main()
