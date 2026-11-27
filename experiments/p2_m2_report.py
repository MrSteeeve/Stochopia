"""Render the five-table M2 research report from the sealed numerical evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import csv
import json
from math import isfinite
from pathlib import Path
from statistics import mean
from typing import Any

from experiments.p2_m2_design import (
    BURST_MODELS, CF, MEM_MODELS, REPORT_METRICS, SEEDS, THIN_MODELS, W1, W2, W3, W4, W6, roster,
)

Json = dict[str, Any]
WORLDS = (W1, W2, W3, W4, W6)
PROPERTIES = {
    "z_mem": ("sign_acf_sum_1_32", "sign_acf_log_slope", "flow_vr_32_4"),
    "z_thin": ("trade_to_top1_depth", "midprice_shift_rate", "cancel_to_limit_ratio"),
    "z_burst": ("zero_wait_prob", "interarrival_cv2"),
    "z_asym": ("book_imbalance_bias", "flow_sign_bias"),
}
MISSING = "—"


def _load(path: Path) -> Json:
    return json.loads(path.read_text(encoding="utf-8"))


def _number(value: object) -> str:
    if value is None or value == "":
        return MISSING
    number = float(value)
    if not isfinite(number):
        raise ValueError("report accepts only finite measurements or explicit nulls")
    return f"{number:.6g}"


def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render rectangular tables, rejecting missing columns instead of shifting cells."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError("table row does not align with its header")
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _effect(row: Json) -> str:
    """Estimate [day-bootstrap 95% CI], independent paired-day and draw support."""
    interval = row["interval"]
    ci = MISSING if interval is None else f"[{_number(interval[0])}, {_number(interval[1])}]"
    days = f"{row['paired_days']}/{row['registered_days']} 日"
    draws = f"{row['bootstrap_valid']}/{row['bootstrap_total']} 抽样"
    return f"{_number(row['estimate'])} {ci} ({days}; {draws})"


def _status(value: bool) -> str:
    return "通过" if value else "未通过"


# --- Evidence integrity and independent experimental units ---

def _validate(registration: Json, manipulation: Json, rows: list[dict[str, str]],
              daily: Json, summary: Json, laws: Json) -> None:
    """Verify the fixed complete study roster, seed/day axes and capacities before rendering."""
    expected = [(job.key, job.experiment, job.world, job.model, job.seed) for job in roster()]
    registered = []
    for job in registration["jobs"]:
        registered.append((job["job_id"], job["experiment"], job["world"], job["model"], job["seed"]))
    actual = []
    for row in rows:
        actual.append((row["job_id"], row["experiment"], row["world"], row["model"], int(row["train_seed"])))
        expected_parameters = registration["parameter_counts"][row["model"]]
        if int(row["parameters"]) != expected_parameters:
            raise ValueError("actual and registered parameter counts differ")
        if int(row["state_scalars"]) != registration["state_scalars"] or int(row["state_scalars"]) != 60:
            raise ValueError("all fits must have exactly 60 real state scalars")
        if int(row["model_k"]) != registration["model_k"]:
            raise ValueError("model rollout replica count differs from registration")
        if any(metric not in row for metric in REPORT_METRICS):
            raise ValueError("incomplete aggregate metric row")
    if Counter(registered) != Counter(expected) or Counter(actual) != Counter(expected):
        raise ValueError("report requires the exact registered 78-fit roster")
    if set(daily) != {job.key for job in roster()}:
        raise ValueError("daily panel does not align with the registered jobs")
    for job in roster():
        dates = [day["trading_date"] for day in daily[job.key]]
        expected_dates = manipulation["world_receipts"][job.world]["eval_days"]
        if dates != expected_dates or len(set(dates)) != registration["eval_days"]:
            raise ValueError("daily panel does not preserve the registered day axis")
    if summary["completed"] != len(expected) or summary["counts"] != {"E1": 36, "E2": 24, "E3": 18}:
        raise ValueError("execution summary does not match the fixed roster")
    if summary["api_calls"] != 0 or registration["api_calls"] != 0:
        raise ValueError("M2 is a zero-API experiment")
    if set(laws) != {"H1", "H2", "H3"}:
        raise ValueError("all three laws, including null or negative ones, are required")
    for law in laws.values():
        if law["status"] not in {"SUPPORTED_WITH_BOUNDARY", "INCONCLUSIVE", "REFUTED"}:
            raise ValueError("unrecognized frozen hypothesis status")


def _group_value(rows: list[dict[str, str]], metric: str) -> str:
    """Mean of available per-seed day means; missing seed support is visibly marked."""
    values = []
    for row in rows:
        if row[metric] not in ("", None):
            values.append(float(row[metric]))
    value = _number(mean(values)) if values else MISSING
    return value if len(values) == len(SEEDS) else f"{value} ({len(values)}/{len(SEEDS)} seeds)"


def _property(row: Json) -> str:
    return f"{_number(row['median'])} / {_number(row['intraday_iqr'])}"


# --- Five report tables ---

def _manipulation_table(registration: Json, result: Json) -> str:
    worlds = registration["worlds"]
    rows = []
    knobs = ("q_scale", "cancel_probability", "slow_persistence", "fast_persistence",
             "alpha_hawk", "mean_dt_ms", "beta_hawk")
    for knob in knobs:
        rows.append([f"λ · {knob}"] + [_number(worlds[world][knob]) for world in WORLDS] + [MISSING, MISSING])
    for knob in ("n_slots", "alpha", "max_len"):
        values = [_number(result["tier0"][label]["flow"][knob]) for label in ("short", "long")]
        rows.append([f"Tier 0 λ · {knob}"] + [MISSING] * len(WORLDS) + values)
    for dimension, names in PROPERTIES.items():
        for name in names:
            values = [_property(result["properties"][world][dimension][name]) for world in WORLDS]
            tier0 = [MISSING, MISSING]
            if name == "sign_acf_sum_1_32":
                tier0 = [_property(result["tier0"][label][name]) for label in ("short", "long")]
            rows.append([f"{dimension}.{name}"] + values + tier0)
    invalid = [_number(result["training_invalid_spread_rate"][world]) for world in WORLDS]
    rows.append(["训练事件无效价差率（物理质量，不属于 Z）"] + invalid + [MISSING, MISSING])
    table = _table(["配置或统计量（median / IQR）", *WORLDS, "Tier 0 short", "Tier 0 long"], rows)
    lines = ["## 表 1 · do(λ) → Z(λ) 干预核验", "", table, "",
             "IQR 为训练期各会话窗口 q75−q25；W5 为 W3 的别名，非第六个独立数据集。"
             "“—” 表示不适用或不可测。重构质量函数独立于 Z；合成盘口没有外部重构真值，"
             "此处只报告撮合路径实际的无效价差率。", ""]
    for key, accepted in result["checks"].items():
        lines.append(f"- `{key}` 干预/物理核验：**{_status(accepted)}**；预注册规则：{result['protocol'][key]}。")
    lines.extend(["", "**伴随属性漂移**（中位数差 / max(参照 IQR, 10⁻⁶)；RMS 不作因果效应）：", "",
                  "生成器旋钮是受控的，实测 Z 各维并不保证统计正交。例如盘口、拆单持久性可同时影响方向偏度，"
                  "不能将交互无条件归因于某一个 Z 标量。参照 IQR 为零时，标准化漂移使用 10⁻⁶ 下限，"
                  "很大的漂移数值不代表同等量级的经济效应。"])
    for pair, drift in result["property_drift"].items():
        details = []
        for name, value in drift.items():
            if name != "total_rms":
                details.append(f"`{name}`={_number(value)}")
        lines.append(f"- {pair}：RMS={_number(drift['total_rms'])}；" + "，".join(details))
    lines.extend(["", "**Tier 0 传播子交叉对照（仅机制与真值检查，不计额外训练）**："])
    for label, record in result["tier0"].items():
        fixed = record["fixed_kernel_truth"]
        independent = record["independent_kernel_truth"]
        lines.append(f"- {label}：固定传播核响应峰值/末值 {_number(max(fixed))}/{_number(fixed[-1])}；"
                     f"独立改变核 τ_max={_number(record['independent_kernel_tau_max'])} 后，"
                     f"峰值/末值 {_number(max(independent))}/{_number(independent[-1])}。")
    short_curve = result["tier0"]["short"]["fixed_kernel_truth"]
    long_curve = result["tier0"]["long"]["fixed_kernel_truth"]
    difference = max(abs(short - long) for short, long in zip(short_curve, long_curve, strict=True))
    lines.extend(["", f"固定传播核时短/长记忆真值曲线最大绝对差为 {_number(difference)}。"
                  "该对照将背景流记忆与价格核响应分开检查，不能替代 E1 的跨模型训练结论。", "",
                  "**已知生成器真值的报价支持**（逐窗口 × 时间点）："])
    for world, receipt in result["world_receipts"].items():
        lines.append(f"- {world}：K={receipt['truth_k']}，最少有效真值副本数 {receipt['truth_support_min']}，"
                     f"平均有效比例 {_number(receipt['truth_support_mean_fraction'])}，"
                     f"平均响应绝对峰值 {_number(receipt['truth_peak_abs_mean_ticks'])} tick。")
    return "\n".join(lines)


def _parent(experiment: str, model: str) -> str | None:
    if experiment == "E1" and not model.endswith("-base"):
        return model.split("-", maxsplit=1)[0] + "-base"
    if experiment == "E2":
        return {"matrix-noshift": "flat-gru", "matrix-advect": "matrix-noshift",
                "flat-readoutgate": "flat-gru"}.get(model)
    if experiment == "E3" and model != "gru-base":
        return "gru-base"
    return None


def _model_table(experiment: str, worlds: Sequence[str], models: Sequence[str], metrics: Sequence[str],
                 headers: Sequence[str], rows: list[dict[str, str]], effects: Json, capacity: bool = False) -> str:
    """Display raw three-seed means and match each variant to its registered parent."""
    result = []
    for world in worlds:
        for model in models:
            group = [row for row in rows if (row["experiment"], row["world"], row["model"]) == (experiment, world, model)]
            if sorted(int(row["train_seed"]) for row in group) != sorted(SEEDS):
                raise ValueError("model group must include all matched training seeds")
            values = [world, model]
            if capacity:
                values.extend([group[0]["parameters"], group[0]["state_scalars"]])
            values.extend([_group_value(group, metric) for metric in metrics])
            parent = _parent(experiment, model)
            if parent is None:
                values.extend(["基线", MISSING])
            else:
                key = f"{experiment}/{model}-vs-{parent}/{CF}"
                values.extend([parent, _effect(effects["comparisons"][key][world])])
            result.append(values)
    heading = ["世界", "模型"]
    if capacity:
        heading.extend(["参数数", "状态标量"])
    return _table(heading + list(headers) + ["配对父模型", "CF Δ [95% CI] / 支持"], result)


def _interactions(experiment: str, effects: Json) -> str:
    lines = ["**条件交互（高属性世界的候选−父模型差，再减去低属性世界的同一差）：**"]
    for key, interaction in effects["interactions"].items():
        if not key.startswith(experiment + "/"):
            continue
        name = f"{interaction['candidate']} vs {interaction['parent']} · {interaction['metric']}"
        lines.append(f"- {name}：{_effect(interaction)}。")
        if interaction["metric"] != CF:
            # These secondary within-world comparisons are not included as columns.
            for world, row in effects["comparisons"][key].items():
                lines.append(f"  - {world} 单世界配对效应：{_effect(row)}。")
    return "\n".join(lines)


def _laws_table(laws: Json) -> str:
    rows = []
    for name in ("H1", "H2", "H3"):
        law = laws[name]
        boundary = law["boundary"]
        if name == "H3":
            eq = "支持" if law["poisson_equivalence_supported"] else "未获支持（不等于证实有差异）"
            boundary += f"；泊松等效（±0.02 nats）：{eq}；{_effect(law['poisson_contrast'])}"
        rows.append([name, law["estimand"], law["status"], _status(law["manipulation_passed"]),
                     _effect(law["primary"]), boundary])
    return _table(["假设", "预注册主估计量", "冻结判定", "操作检查", "主效应 [95% CI] / 支持", "适用边界 / 反例"], rows)


def _generation_scope(registration: Json) -> str:
    calibration = registration["generation_calibration"]
    seen = "是" if calibration["formal_model_outcomes_seen"] else "否"
    return ("**模型训练前的生成器校准与真值边界**：\n\n"
            f"- 初步生成试验：{calibration['initial_generation_only']}\n"
            f"- 最终训练期锚点校准：{calibration['final_train_anchor_calibration']}\n"
            f"- 校准时是否看过正式模型结果：{seen}。\n"
            f"- 真值条件：{calibration['truth_condition']} "
            "从标准十档聚合盘口快照和生成器真实的隐符号/时钟状态启动反事实，保留共同随机标记；"
            "未建模原始订单队列的逐单账龄（cancel age）。这是特定合成生成机制的条件真值，"
            "不将其外推为真实交易所订单级反事实真值。")


def _coverage(daily: Json) -> str:
    """Count missing fit-day CF estimates without treating quote failures as zero."""
    totals: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    for job in roster():
        for row in daily[job.key]:
            totals[job.experiment] += 1
            if row[CF] is None:
                missing[job.experiment] += 1
    by_group = ", ".join(f"{key}: {missing[key]}/{totals[key]}" for key in ("E1", "E2", "E3"))
    return (f"**反事实误差可用性**：{sum(missing.values())}/{sum(totals.values())} 个拟合×评估日读数不可用"
            f"（{by_group}）。这些是逐拟合日计数，不是独立日数。若有一个窗口缺少完整报价响应，"
            "该拟合日的 CF 误差不可用，不填零，不以次级指标替代。配对分析进一步要求各训练种子和"
            "对照分支在同一日期共同有值；低于预注册门槛则不产生区间。可用样本的均值不能代表"
            "报价耗尽条件下的无条件性能；这种非随机缺失本身是模型局限。")


def _exhaustion_scope(rows: list[dict[str, str]], manipulation: Json) -> str:
    """Report descriptive mean exhaustion across model-seed cells, separate from CF."""
    lines = ["**生成轨迹的有效性和尺度限制**："]
    for experiment, worlds, models in (("E2", (W1, W3), THIN_MODELS), ("E3", (W3, W6), BURST_MODELS)):
        for world in worlds:
            rates = []
            for model in models:
                group = [row for row in rows if (row["experiment"], row["world"], row["model"]) == (experiment, world, model)]
                rates.append(mean(float(row["exhausted_replicate_rate"]) for row in group))
            lines.append(f"- {experiment} {world}：各模型耗尽副本率范围 {_number(100 * min(rates))}%–{_number(100 * max(rates))}%。")
    groups = []
    for experiment, worlds, models in (("E1", (W1, W2), MEM_MODELS), ("E2", (W1, W3), THIN_MODELS),
                                       ("E3", (W3, W6), BURST_MODELS)):
        for world in worlds:
            for model in models:
                values = []
                for row in rows:
                    if (row["experiment"], row["world"], row["model"]) == (experiment, world, model) and row[CF] != "":
                        values.append(float(row[CF]))
                if values:
                    groups.append((mean(values), experiment, world, model))
    if groups:
        value, experiment, world, model = max(groups)
        peak = manipulation["world_receipts"][world]["truth_peak_abs_mean_ticks"]
        lines.append(f"- 绝对 CF MAE 的最大三种子描述性均值：{experiment} {world} / {model}，"
                     f"{_number(value)} tick；该世界已知生成器平均响应绝对峰值 {_number(peak)} tick。"
                     "两者仅作量级检查，不是同一个估计量。")
    lines.append("高耗尽率与缺失 CF 提示局限：即使某个似然分项改善，也不表示生成模型已具备完整可用反事实能力。")
    return "\n".join(lines)


def _interpretation(laws: Json, effects: Json) -> str:
    """Retain primary frozen status and distinguish outcome and mechanism claims."""
    lines = ["**三条规律的实证解释**："]
    for name, law in laws.items():
        primary = law["primary"]
        interval = primary["interval"]
        explanation = "只在预注册主效应与适用边界内解释。"
        if not law["manipulation_passed"]:
            explanation = "目标属性的操作核验未通过。"
        elif interval is None:
            explanation = f"只有 {primary['paired_days']} 个共同配对日；无预注册有效区间，不作收益或无收益结论。"
        elif interval[0] <= 0 <= interval[1]:
            explanation = "区间跨零，不能把负方向的点估计解释为已证实收益，也不能据此宣称无效。"
        lines.append(f"- {name} **{law['status']}**：{_effect(primary)}。{explanation}")
    memory_interactions = []
    for key, row in effects["interactions"].items():
        if key.startswith("E1/") and key.endswith("/" + CF):
            memory_interactions.append(row["interval"])
    zero_crossings = sum(interval is not None and interval[0] <= 0 <= interval[1] for interval in memory_interactions)
    lines.append(f"- H1 的四个骨干×算子 CF 交互区间中，{zero_crossings}/{len(memory_interactions)} 个包括零；"
                 "各算子的方向 ACF 与 G_mem(64) 检查是次级证据，不自动升级为因果收益。")
    clock = effects["comparisons"]["E3/gru-hawkes-vs-gru-base/dt_nll"]
    h3_impact = effects["interactions"][f"E3/gru-hawkes-vs-gru-base/{CF}"]
    lines.append(f"- H3 分解：强簇聚 W6 的 dt NLL 配对差 {_effect(clock[W6])}；"
                 f"泊松 W3 为 {_effect(clock[W3])}。统计上的非零差异可以同时满足事前定义的实际等效带，"
                 "若泊松差值为正则表现小幅下降；等效不是逐次运行完全相同。时间头的条件收益不能替代冲击预测收益。"
                 f"Hawkes 对 CF 误差的交互为 {_effect(h3_impact)}。")
    return "\n\n".join(lines)


# --- Assemble and write only after all evidence files exist ---

def render_report(artifact: Path) -> str:
    """Read the complete M2 artifact directory; never recompute or revise frozen verdicts."""
    registration = _load(artifact / "registration.json")
    manipulation = _load(artifact / "manipulation_check.json")
    daily = _load(artifact / "daily_results.json")
    effects = _load(artifact / "paired_effects.json")
    laws = _load(artifact / "frozen_synthetic_laws.json")
    summary = _load(artifact / "execution_summary.json")
    with (artifact / "experiment_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    _validate(registration, manipulation, rows, daily, summary, laws)
    fit = registration["fit_protocol"]
    parts = [
        "# P2-M2 · 合成环境中数据属性 Z 与模型算子条件效应研究",
        f"**完成时间（UTC）**：{summary['completed_utc']}。**实验代码 SHA**：`{registration['code_commit']}`。",
        "## 实验口径\n\n"
        f"- 78 次独立固定预算拟合：E1 36、E2 24、E3 18；各元组固定三个训练初值 ({', '.join(str(seed) for seed in SEEDS)})。"
        "同架构跨实验独立拟合；未增加 Tier 0 训练。\n"
        f"- 五个合成逐笔世界：各 {registration['train_days']} 训练日、{registration['eval_days']} 评估日 × "
        f"{registration['sessions_per_day']} 会话窗口、每会话 {registration['events_per_session']} 事件；"
        f"历史上下文 {registration['context_len']} 步。固定四头似然；训练 {fit['steps']} 步、"
        f"subsequence={fit['subseq_len']}、burn-in={fit['burn_in']}、batch={fit['batch_size']}。\n"
        f"- 模型 K={registration['model_k']}、已知生成器 K={registration['truth_k']}：可见母单−无母单响应使用共同随机数，"
        "已知真实生成机制给出有限 Monte Carlo 均值估计，而非无抽样误差的数学期望；单位为 tick。"
        "`counterfactual_impact_mae`（CF）是该反事实目标误差，独立于 stylized impact MAE 或 C2 稳定性。\n"
        f"- 每日、每种子先聚合物理窗口和副本；配对对比先在日×训练种子内作差，后对三种子取均值；"
        f"按独立评估日索引重抽 {registration['bootstrap_draws']} 次。CI 是 95% percentile 区间，"
        "要求至少 6 个配对日与 1800 个有效重抽，不以训练种子或 Monte Carlo 副本扩大独立样本。\n"
        "- 模型绝对读数是三个训练种子的日均指标平均，允许的缺失值用 — 或明确标出种子支持；"
        "配对 Δ=候选−父模型（负误差差值为改善）；G_mem(L)=NLL_reset(L)−NLL_stateful，"
        "计算使用完全一致的监督掩码，G_mem 描述对历史的依赖，非越大越好。\n"
        f"- 数据、真值、副本、bootstrap 种子分别为 {registration['data_seed']}、{registration['truth_seed']}、"
        f"{registration['rollout_seed']}、{registration['bootstrap_seed']}；评估日配对与各世界生成内容哈希见 manipulation_check.json。\n"
        f"- 母单子单时刻：{registration['child_times_ms']} ms，每笔 {registration['child_shares']} 股；"
        f"时间网格 {registration['grid_ms']} ms。总执行墙钟 {_number(summary['wall_seconds'])} 秒；"
        f"各训练累计 {_number(summary['total_fit_seconds'])} 秒、评估累计 {_number(summary['total_eval_seconds'])} 秒。"
        "不使用私有数据，不调用 LLM API。",
        _generation_scope(registration),
        _manipulation_table(registration, manipulation),
        "## 表 2 · M2-E1 记忆强度条件效应\n\n" + _model_table(
            "E1", (W1, W2), MEM_MODELS,
            ("val_nll_total", "memory_gap_1", "memory_gap_4", "memory_gap_16", "memory_gap_64", "direction_acf_mae", CF),
            ("总 NLL", "G(1)", "G(4)", "G(16)", "G(64)", "方向 ACF MAE", "CF MAE (tick)"), rows, effects),
        _interactions("E1", effects),
        "## 表 3 · M2-E2 同状态容量的档位矩阵与坐标平移\n\n" + _model_table(
            "E2", (W1, W3), THIN_MODELS, ("rel_ticks_nll", "exhausted_replicate_rate", CF),
            ("价格档位 NLL", "耗尽副本率", "CF MAE (tick)"), rows, effects, capacity=True),
        "矩阵态预算 10×6 共 60 个实标量，其中两个槽保存因果已知的参考 bid/ask tick，58 个为可学习状态；"
        "NoShift 和 Advect 参数完全相同，仅改变因果平移机制。相同总状态容量不意味着各骨干可训练参数完全相等，"
        "表中记录精确参数量。耗尽副本率是报价缺失/盘口耗尽的窗口×副本轨迹比例，不把缺失报价编成价格。",
        _interactions("E2", effects),
        "## 表 4 · M2-E3 时间簇聚度与 Hawkes 强度态\n\n" + _model_table(
            "E3", (W3, W6), BURST_MODELS,
            ("val_nll_total", "dt_nll", "zero_wait_rate_error", "interarrival_cv2_error", "exhausted_replicate_rate", CF),
            ("总 NLL", "dt NLL", "零等待率绝对残差", "CV² 绝对残差", "耗尽副本率", "CF MAE (tick)"), rows, effects),
        _interactions("E3", effects),
        _coverage(daily),
        _exhaustion_scope(rows, manipulation),
        "## 表 5 · 可迁移条件规律与证伪反例冻结\n\n" + _laws_table(laws),
        "判定只读取预注册后的 `frozen_synthetic_laws.json`：干预核验通过且主效应置信区间完全低于零，支持所述条件规律；"
        "完全高于零则证伪；跨零、缺少区间或干预未通过均不可判定。H3 的泊松零增益仅在整个区间位于等效带内时成立。"
        "支持的结论仍限定于合成旋钮、所选骨干/算子、状态容量及母单协议，不代表已验证真实市场因果规律。"
        "次级指标和单一算子结果不自动继承主假设的判定。",
        _interpretation(laws, effects),
        "## 外部复现（无私有数据、无 API）\n\n"
        "从包含本 M2 报告及其生成器的 Stochopia 版本仓库根目录，使用 Python 3.12 环境。"
        "以下命令用独立 worktree 检出锁定的训练代码，在新的临时目录写入证据，避免覆盖已封存的原实验：\n\n"
        "```bash\n"
        'study_repo="$(git rev-parse --show-toplevel)"\n'
        'study_tmp="$(mktemp -d "${TMPDIR:-/tmp}/stochopia-p2-m2.XXXXXX")"\n'
        f'git -C "$study_repo" worktree add --detach "$study_tmp/locked-src" {registration["code_commit"]}\n'
        "(\n"
        '  cd "$study_tmp/locked-src"\n'
        "  uv sync --locked --python 3.12 --all-extras --dev\n"
        '  .venv/bin/python -m experiments.run_p2_m2_synthetic_study --artifact-root "$study_tmp/evidence" --work-root "$study_tmp/cache" --jobs 3\n'
        ")\n"
        'cd "$study_repo"\n'
        "uv sync --locked --python 3.12 --all-extras --dev\n"
        '.venv/bin/python -m experiments.p2_m2_report "$study_tmp/evidence" "$study_tmp/report.md"\n'
        "```\n\n"
        "机器可读证据位于 `OpenHyra/artifacts/p2-m2-synthetic/`：`registration.json`、`manipulation_check.json`、"
        "`experiment_results.csv`、`daily_results.json`、`paired_effects.json`、`frozen_synthetic_laws.json` 与 `execution_summary.json`。"
        "逐日结果和全部置信区间的完整精度以机器可读记录为准；本报告为只读摘要，不再做训练或改变冻结结论。",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = render_report(args.artifact_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
