"""Render the registered SZSE M4 evidence into a Markdown results report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any


Json = dict[str, Any]
STOCKS = ("000001.SZ", "300001.SZ")
ARMS = ("evidence_gpt", "metrics_gpt", "evidence_ds", "metrics_ds", "no_llm")
SEEDS = (17, 29, 41)


def _number(value: object, digits: int = 3) -> str:
    if type(value) in (int, float) and isfinite(value):
        return f"{value:.{digits}f}"
    return "—"


def _table(columns: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    def line(values: Sequence[object]) -> str:
        return "| " + " | ".join(str(value) for value in values) + " |"

    return "\n".join([line(columns), line(["---"] * len(columns)), *(line(row) for row in rows)])


def _claim_bits(metrics: Json) -> str:
    claims = metrics.get("claims", {})
    mark = {"pass": "✓", "fail": "×", "inconclusive": "?"}
    return " ".join(mark.get(claims.get(f"C{index}"), "—") for index in range(1, 6))


def _estimate(metrics: Json, ci: list[float] | None = None) -> str:
    score = _number(metrics.get("lob_impact_score"), 2)
    if ci is not None:
        bounds = ", ".join(_number(value, 2) for value in ci)
        score += f" [{bounds}]"
    return score


def _weighted(rows: list[Json], metric: str, count: str) -> float | None:
    available = [(row[metric], row[count]) for row in rows if row.get(metric) is not None and row.get(count, 0) > 0]
    total = sum(weight for _, weight in available)
    return sum(value * weight for value, weight in available) / total if total else None


def _data_checks(receipt: Json) -> str:
    rows = []
    for stock in STOCKS:
        records = [row for row in receipt["reconciliation"] if row["symbol"] == stock]
        anchors = [row for row in receipt["anchors"] if row["symbol"] == stock]
        match = _weighted(records, "touch_match_rate", "n_snapshots")
        rows.append((stock, len(records), sum(row["n_events"] for row in records), len(anchors),
                     _number(None if match is None else 100 * match, 2),
                     _number(_weighted(records, "top10_log_vol_mae", "n_snapshots"), 4)))
    return _table(("股票", "交易日", "外生事件", "双边起点", "一档同时匹配 %", "十档对数深度 MAE"), rows)


def _requests(trials: Sequence[Json]) -> list[Json]:
    return [item for trial in trials for item in trial["requests"] if item["record_id"].endswith("-result")]


def _development_rows(manifest: Json) -> str:
    rows = []
    for stock in STOCKS:
        for arm in ARMS:
            strands = [manifest["trajectories"][f"{stock}-{arm}-{seed}"]["trials"] for seed in SEEDS]
            trials = [trial for strand in strands for trial in strand]
            statuses = Counter(trial["verdict"] for trial in trials)
            passed = sum(trial["candidate"] is not None for trial in trials)
            repairs = sum(sum(entry.get("attempt", 1) > 1 for entry in trial["compiler"]) for trial in trials)
            calls = _requests(trials)
            hit_costs = []
            for strand in strands:
                earliest = next((index for index, trial in enumerate(strand) if trial["verdict"] == "VERIFIED_GAIN"), None)
                if earliest is not None:
                    used = strand[:earliest + 1]
                    trained = sum(bool(row.get("training_budget_consumed")) for row in used)
                    hit_costs.append(f"{trained} 次训练 / {len(_requests(used))} 次 API")
            rows.append((stock, arm, f"{passed}/{len(trials)}", repairs, statuses["VERIFIED_GAIN"],
                         statuses["UNVERIFIED_GAIN"], statuses["GUARDRAIL_VIOLATION"], statuses["REFUTED"],
                         len(trials) - sum(statuses[key] for key in ("VERIFIED_GAIN", "UNVERIFIED_GAIN",
                                                                    "GUARDRAIL_VIOLATION", "REFUTED")),
                         len(calls), "；".join(hit_costs) or "—"))
    return _table(("股票", "实验臂", "有效编译", "修复次数", "已证实", "未证实涨分", "违反护栏", "无净提升",
                   "其他终态", "API 请求", "首次证实成本"), rows)


def _holdout_rows(report: Json) -> str:
    rows = []
    for stock in STOCKS:
        for arm in ARMS:
            for seed in SEEDS:
                trajectory = report["trajectories"][f"{stock}-{arm}-{seed}"]
                for phase in ("initial", "first", "final"):
                    endpoint = trajectory[phase]
                    if endpoint is None:
                        rows.append((stock, arm, seed, phase, "无合法首轮候选", "—", "—", "—", "—", "—", "—", "—", "—", "—"))
                        continue
                    metrics = endpoint["metrics"]
                    rows.append((stock, arm, seed, phase, endpoint["model_id"],
                                 _number(metrics.get("val_nll_total"), 4), _estimate(metrics, endpoint["impact_ci95"]),
                                 _number(endpoint["impact_delta_from_initial"], 2),
                                 _number(metrics.get("visible_over_blind_ratio")), _number(metrics.get("trade_sign_acf_exp_gap")),
                                 _number(metrics.get("end_over_peak_ratio")), _number(metrics.get("impact_delta")),
                                 _number(metrics.get("free_abs_drift_ticks")), _claim_bits(metrics)))
    return _table(("股票", "实验臂", "搜索种子", "端点", "模型", "NLL", "LOB-Impact [95%]", "相对种子 Δ分",
                   "可见/盲组", "符号 ACF 指数误差", "回落比", "冲击指数 δ", "自由漂移 ticks", "C1–C5"), rows)


def _rate(value: Mapping[str, object]) -> str:
    return f"{value['supported']}/{value['registered']}（{value['inconclusive']} 项不可判定）"


def _audit_rows(report: Json) -> str:
    rows = []
    for stock in STOCKS:
        for arm in ARMS:
            entry = report["arm_summary"][stock][arm]
            audit = entry["rejected_audit"]
            rates = entry["prediction_rates"]
            rows.append((stock, arm, _rate(rates["primary"]), _rate(rates["guardrail"]),
                         audit["candidates"], audit["measured"], audit["survived_benchmark_gate"],
                         audit["survived_mechanism_gate"]))
    return _table(("股票", "实验臂", "主预测命中", "护栏预测命中", "拦截候选", "留出可测", "留出净涨分", "留出机制获证实"), rows)


def _cross_rows(report: Json) -> str:
    rows = []
    for arm in ARMS:
        for source in STOCKS:
            strand = report["cross_regime"][arm][source]
            cells = []
            for target in STOCKS:
                cell = strand["cells"][target]
                metrics = cell["metrics"]
                cells.append(f"{_estimate(metrics, cell['impact_ci95'])} / {_number(metrics.get('val_nll_total'), 4)}")
            auxiliary = strand["auxiliary"]
            note = "种子" if strand["retains_seed"] else "已晋升"
            rows.append((arm, source, strand["representative"], note, *cells))
            if auxiliary is not None:
                target = next(stock for stock in STOCKS if stock != source)
                aux_cells = {source: auxiliary["diagonal"], target: auxiliary["off_diagonal"]}
                cells = []
                for stock in STOCKS:
                    cell = aux_cells[stock]
                    metrics = cell["metrics"]
                    ci = cell["readouts"].get("lob_impact_score", {}).get("candidate_interval")
                    cells.append(f"{_estimate(metrics, ci)} / {_number(metrics.get('val_nll_total'), 4)}")
                model_id = auxiliary["diagonal"]["candidate"]["model_id"]
                rows.append((arm, source, model_id, "辅助候选", *cells))
    return _table(("实验臂", "架构来源", "模型 ID", "开发集选择", "000001.SZ 得分 / NLL", "300001.SZ 得分 / NLL"), rows)


def _read(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def render(artifact: Path, architecture_notes: str) -> str:
    """Render only completed holdout evidence and separately supplied source analysis."""
    manifest = _read(artifact / "study_manifest.json")
    report = _read(artifact / "verification" / "report.json")
    receipt = _read(artifact / "verification" / "test-data-reconciliation.json")
    if report["code"] != manifest["code"] or receipt["test_dates"] != manifest["test_dates"]:
        raise ValueError("the report inputs must share the frozen code and test dates")
    dates = "、".join(str(date) for date in manifest["test_dates"])
    commits = manifest["code"]
    opening = (f"本研究在两只深市股票上比较五个实验臂。每只股票使用 36 日训练、12 日开发和 12 日独立复验。"
               f"各臂登记三个搜索种子，每个种子演化三轮。开发侧共 90 次候选试验。"
               f"实验代码锁定在 Stochopia `{commits['stochopia_commit']}` 和 OpenHyra `{commits['openhyra_commit']}`。")
    methods = ("三层验证器先检查候选张量契约及前缀因果，再核验验证 NLL 与自由盘口存续。"
               "最后要求 LOB-Impact 净涨分至少 0.50 且配对重抽的单侧 5% 下界大于零。"
               "LLM 候选还须有主预测获支持、护栏预测未被推翻。随机变异组用固定预测占位，按前两道成绩条件晋升。"
               "评分按交易日配对重抽 2,000 次。完整面板先判定 C2 是否可识别。对可评分模型，区间条件于该判定，"
               "每次重抽重新聚合物理读数并计算综合分。至少 6 个共同有效日及 1,800 次有效配对重抽时，"
               "在有效的配对重抽上报告区间与单侧下界，同时报告原始支持集和未定义次数。")
    version = ("`stochopia-p1-m4-v1` 在完成 40/90 次开发试验后因协议字段与指标极性歧义中止。"
               "累计 42 次 API 请求、74,560 个输入 token 和 26,650 个输出 token。留出集始终封存。"
               "工具链修复并完成冒烟核验后，独立登记 `stochopia-p1-m4-v2`，V1 账本留作预演诊断。")
    return "\n\n".join((
        "# P1-M4 深市 Level-2 开放算子演化实验", opening, "## 测量协议", version, methods,
        "本研究采用外生子单注入、盲组与回落读数，并沿用候选选择冻结后再开启留出集的审计原则。"
        " [LOB-Impact](https://arxiv.org/html/2610.27100v1) · [留出集审计](https://arxiv.org/html/2610.14063v1)",
        "## 留出集对账", f"交易日为 {dates}。", _data_checks(receipt), "## 开发期三轮轨迹", _development_rows(manifest),
        "## 留出集主结果", "NLL 单位为 nats/事件，C1–C5 按顺序用 ✓、×、? 标示。", _holdout_rows(report),
        "## 事前预测与伪涨分审计", "命中率以已登记预测为分母。`no_llm` 的读数来自固定占位预测。", _audit_rows(report),
        "## 跨股架构重训", "每个方向均只传递源码，在目标股票独立初始化权重。辅助行保留开发集规则下的次选架构。"
        " [LOB-Bench](https://arxiv.org/html/2502.09172v1)", _cross_rows(report),
        "## 胜出算子的结构与结果解释", architecture_notes.strip(),
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
