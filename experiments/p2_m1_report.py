"""Render the retrospective P2-M1 evidence without recomputing any trajectory."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

Json = dict[str, Any]
LABELS = {
    "M0": "种子 GRU",
    "M1": "流动性读出门控（+3.03）",
    "M2": "循环状态门控（+0.18）",
    "M3": "有界输入残差（+8.13）",
}
CONTINUOUS = ("impact_curve_mae", "post_stop_relaxation_delta", "visible_minus_blind_auc")
HEADS = ("dt", "action", "rel_ticks", "volume")


def _number(value: float | None, digits: int = 6) -> str:
    return "不可判定" if value is None else f"{value:.{digits}f}"


def _interval(data: Mapping[str, Any]) -> str:
    points = data.get("interval")
    return "不可判定" if points is None else f"[{points[0]:.6f}, {points[1]:.6f}]"


def _delta(data: Mapping[str, Any], key: str = "estimate") -> str:
    return f"{_number(data.get(key))} {_interval(data)}"


def _width(data: Mapping[str, Any]) -> float | None:
    bounds = data.get("interval")
    return None if bounds is None else float(bounds[1] - bounds[0])


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    def format_row(row: tuple[object, ...]) -> str:
        return "| " + " | ".join(map(str, row)) + " |"

    return "\n".join((format_row(headers), format_row(tuple("---" for _ in headers)),
                      *(format_row(row) for row in rows)))


def _coverage(evidence: Mapping[str, Any]) -> str:
    support = evidence.get("available_quote_support")
    if support is None:
        return "未记录"
    return (f"{100 * support['visible_mean_fraction']:.2f}%/"
            f"{100 * support['joint_mean_fraction']:.2f}%")


def _impact_rows(report: Json) -> tuple[str, int, int, int, int]:
    rows = []
    available_points = 0
    available_intervals = 0
    for label, title in LABELS.items():
        key = f"{label}-k1"
        evidence = report["measurements"][key]["evidence"]
        pairs = report["pairs"][key]
        c2 = pairs["delta_c2_pass_prob"]
        for name in CONTINUOUS:
            available_points += int(pairs[name]["estimate"] is not None)
            available_intervals += int(pairs[name]["interval"] is not None)
        rows.append((
            label + " " + title, evidence["impact_identifiability_state"],
            _number(c2.get("candidate_bootstrap_mean")), _delta(c2, "mean"),
            _number(evidence.get("impact_curve_mae")), _delta(pairs["impact_curve_mae"]),
            _number(evidence.get("post_stop_relaxation_delta")), _delta(pairs["post_stop_relaxation_delta"]),
            _number(evidence.get("visible_minus_blind_auc")), _delta(pairs["visible_minus_blind_auc"]),
            _number(evidence.get("lob_impact_score")), _number(pairs["joint_c2_pass_share"]),
            _coverage(evidence),
        ))
    width = len(LABELS) * len(CONTINUOUS)
    return _table((
        "模型", "完整面板识别状态", "重抽 P(C2 pass)", "ΔP [95% 重抽区间]",
        "曲线 MAE (tick)", "Δ曲线 MAE [95%]", "停手回落 (tick)", "Δ回落 [95%]",
        "可见−盲组 AUC", "ΔAUC [95%]", "原条件总分", "联合 C2 通过比率",
        "有效报价率（可见/联合）",
    ), rows), available_points, width, available_intervals, width


def _nll_rows(report: Json) -> str:
    rows = []
    for label, title in LABELS.items():
        evidence = report["measurements"][f"{label}-k1"]["evidence"]
        gap = evidence["stateless_nll_gap_by_stage"]
        rows.append((label + " " + title, _number(evidence.get("val_nll_total")),
                     _number(evidence.get("stateless_nll_total")),
                     _number(evidence.get("stateless_nll_gap")),
                     *(_number(gap.get(stage)) for stage in HEADS)))
    return _table(("模型", "原 NLL", "失忆 NLL", "总差", "等待时间差", "动作差", "价位差", "数量差"), rows)


def _variance_rows(report: Json) -> str:
    rows = []
    reference = None
    for k in (1, 4, 8):
        left = report["measurements"][f"M0-k{k}"]
        right = report["measurements"][f"M3-k{k}"]
        readouts = report["pairs"][f"M3-k{k}"]
        continuous = _width(readouts["impact_curve_mae"])
        if k == 1:
            reference = continuous
        ratio = None if continuous is None or reference in (None, 0) else continuous / reference
        conditional = readouts["delta_lob_impact_score"]
        old_score = readouts["lob_impact_score"]
        left_time = left["timing"]["wall_seconds"]
        right_time = right["timing"]["wall_seconds"]
        windows = (left["evidence"]["paired_seed_windows"] + right["evidence"]["paired_seed_windows"]) * k
        pair_wall = report.get("schedule_timing", {}).get(f"M0-M3-k{k}", {}).get("wall_seconds")
        rows.append((
            k, _delta(readouts["impact_curve_mae"]), _number(continuous), _number(ratio, 4),
            _number(_width(conditional)), conditional["bootstrap_valid"], _number(_width(old_score)),
            f"{left['n_days']}/{right['n_days']}", _coverage(left["evidence"]) + "；" + _coverage(right["evidence"]),
            _number(pair_wall, 2),
            f"{left_time:.2f}/{right_time:.2f}", _number(1000 * (left_time + right_time) / windows, 2),
        ))
    return _table((
        "K", "M3−M0 Δ曲线 MAE [95%]", "95%宽度", "宽度/K=1",
        "条件分差分布宽度", "有限联合抽样数", "严格条件总分区间宽度",
        "父/子交易日簇", "父；子有效报价率", "并行组墙钟(s)", "父/子worker墙钟(s)", "ms/模型·训练种子·窗口·副本",
    ), rows)


def render(report: Json) -> str:
    registration = report["registration"]
    if registration["schema"] != "p2-m1" or len(registration["test_dates"]) != 20:
        raise ValueError("expected M1 20-day diagnostic panel")
    if any(row["n_days"] != 20 for row in report["measurements"].values()):
        raise ValueError("independent days changed with sampling repetitions")
    table1, n_point, points, n_interval, intervals = _impact_rows(report)
    rows = [
        "# P2-M1 连续冲击度量、复合状态与多副本方差验证",
        "## 实验定位与固定配置",
        "本轮重放同一批已经开启的 P1-M5 交易日，是新度量和回放协议的回顾性验证，"
        "不是新的留出测试，不重新选择 P1 最终模型，不得据此回写 P1 主终点。"
        "固定特锐德 300001.SZ、四份封存源码、训练种子 103/137/211 和已有 CPU 权重。"
        f"36 训练日仅用于标定/张量准备；20 日、每天两个窗口；重抽单位始终是交易日。"
        f"每个 K 使用相同的随机数种子 {registration['rollout_seed']} 和 2,000 次共享日重抽。"
        "零外部模型 API 调用、零新行情下载。P1 默认旧协议未改写；P2 使用显式 p2-m1 标识。",
        f"处理代码：Stochopia `{registration['code']['stochopia']}`、OpenHyra `{registration['code']['openhyra']}`。",
        "## 表 1　连续冲击指标及独立重抽 C2 恢复度",
        "冲击曲线是在固定时间网格上的买卖反对称响应（tick）。参考曲线采用训练历史日波动率 "
        "σ_ticks × sqrt(计划母单量/历史日均成交量)，执行期为平方根上升，停手期为平方根传播子的差。"
        "参照仅依赖固定历史标定和计划动作，不使用模型实际成交量；这里的 η 是日成交量归一化的计划量，"
        "并非决定子单间隔的 0.10 参与率。理论曲线误差为零不强制 C1–C5 原条件总分为满分，"
        "原有回落容差和流量/漂移门槛保持独立。"
        "可见−盲组 AUC 的时间轴以执行时长归一化。独立按日重抽后，父子分别重新判定 C2；"
        "ΔP 为配对 0/1 差的重抽均值，区间为该配对差的经验分位数，不把一次 C2 判定当作通过概率。"
        "条件 LOB-Impact 沿用原 C1–C5 的严格路径支持，不将不可识别结果补为零。",
        "连续指标则在同一窗口 K 副本内，按每一时间点有效的买/卖配对报价做算术均值；"
        "可见−盲组差要求同一副本四条轨迹均具有效报价。再按窗口/训练种子、交易日等权汇聚，"
        "记录有效报价覆盖率，保留完全缺失的时间点为不可用。该估计量条件于存在报价，不隐含缺失值为零；"
        "不同模型或 K 的覆盖率变化也可能影响比较。表中单模型绝对读数为各自有效报价，"
        "配对差则在双方共同可观测的日×时点上计算，因此当支持不同，差不必等于两列点值直接相减。",
        table1,
        f"K=1 四个模型对同股基线（含基线自对照）的三种连续指标："
        f"有限配对点估计 {n_point}/{points}，有限配对区间 {n_interval}/{intervals}。"
        "完整记录见独立证据 JSON；无效或者缺失的轨迹不会伪造读数。",
        "## 表 2　失忆似然差",
        "在历史验证样本上按相同的目标、掩码和事件数权重计算 NLL（nats/事件）。"
        "失忆模式保留每一步已经观测到的完整盘口和前步事件，将时间折叠为独立长度一的序列，"
        "切断跨步循环状态。正差表示该模型在其实际读取历史的条件下利用了跨步记忆，"
        "但差值受读出、编码、训练权重和状态共同影响，不能单凭差值判定某个机制的因果贡献。",
        _nll_rows(report),
        "源码对照：M1 保留 GRU 递推不变，令读出 $c_t=g_t h_t+(1-g_t)x_t$，"
        "$g_t=\\sigma(W[\\mathrm{spread}_t,\\mathrm{imbalance}_t]+b)$，因此改变历史信息在读出中的权重。"
        "M2 将 $|\\mathrm{spread}|-|\\mathrm{imbalance}|$ 追加至原始特征后再投影成 $x_t$，"
        "实际门值是 $g_t=\\sigma(x_{t,-1})$（投影后最后一个坐标，而不是直接用原始门输入），"
        "$h_t=g_t\\,\\mathrm{GRU}(x_t,h_{t-1})+(1-g_t)h_{t-1}$，并将混合后的状态反馈到下一步。"
        "M3 在输入投影加入 $0.5\\tanh(U[spread,imbalance,is\\_own]+d)$ 的有界残差，"
        "随后整体 tanh；循环结构与四头保持不变。每个模型独立训练，因此这里是不同架构的后验消融读数，"
        "不是将 M1/M2 的门互换而其他权重固定的实验。",

        "## 表 3　K 倍随机副本与方差、耗时",
        "每个窗口内先池化 K 个副本，再构造原训练种子×会话窗口以及日级配对簇；"
        "K=1/4/8 使用相同 P2 随机流前缀和日重抽索引。NLL 由观测历史决定，不作为随机回放副本扩增。"
        "条件分差分布宽度仅基于双方各自通过 C2 且总分有限的重抽；这不是无条件置信区间。"
        "严格区间仍需要原面板有限值及既有支持门槛。没有有效联合重抽时保留不可判定。"
        "CPU 两个并行 worker，均摊耗时的分母包括父、子两模型各三个训练种子和真实会话窗口。",
        _variance_rows(report),
        "## 证据边界",
        "保留原冻结证据与标签不变；P2 只对已有测试数据做回顾性工程验证。"
        "方差收窄与速度提升由表中实际测量决定，不假定 K 越大区间必然越窄。"
        "理论参考曲线是模型无关基准而非真实外生母单个体反事实。",
        "兼容性：冻结的 M5 72 个、V2 40 个可执行候选全部通过新的 B=1/B=5 科学契约；"
        "真实 000001.SZ 一日两个窗口的旧默认 p1/K=1 路径复现 M0a 数值哈希 "
        "`9652af47189e3885833fa8f4b2fa28d0ad3386167a8a340d52aafa124519440d`。",
        "机器可读的[实验登记与配对证据](https://github.com/MrSteeeve/OpenHyra/tree/world-model/artifacts/p2-m1-validation)。",
    ]
    return "\n\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.write_text(render(report), encoding="utf-8")


if __name__ == "__main__":
    main()
