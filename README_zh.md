# Stochopia

**由LLM研究循环持续修订的订单流市场世界模型**

[English README](README.md)

Stochopia是一个市场世界模型。它描述挂单、撤单、成交与价格变化之间的关系，由一个LLM研究循环提出假设、选择证据并提交修订。首个研究问题是市场冲击平方根律的适用范围。

## 状态

当前仓库保留可复用的世界模型、市场评估及合成实验基础设施。LLM 研究循环由 [OpenHyra](https://github.com/MrSteeeve/OpenHyra/tree/world-model) 维护。

| 路径 | 内容 |
|---|---|
| `stochopia/neural/` | 因果盘口世界模型接口、训练与参考算子 |
| `stochopia/harness/` | 合法事件回放与 JAX-LOB 撮合 |
| `stochopia/market/` | 市场窗口、订单流统计、闭环回放与深市评估 |
| `stochopia/measure/` | 数据属性、冲击曲线与评估指标 |
| `stochopia/sim/` | 可控传播子与队列反应合成世界 |
| `experiments/synthetic_tau_z/` | 条件算子合成研究的可复现入口和报告 |

P0 与 P1 历史报告及代码保存在 `tier0-p0-final` 与 `p1-szse-final` 标签下。现役测量协议及 P2 报告仍位于 `docs/research/`。

## 旧版：Benchmark v3

此前本仓库是结构化产品设计基准Benchmark v3，已停止维护。它的最终状态保存在标签`benchmark-v3-final`：

```bash
git checkout benchmark-v3-final
```

那一版的研究笔记也保存在该标签的`docs/research/`下，文中的文件路径只在该标签下存在。

## 开发

```bash
uv sync --locked --python 3.12 --all-extras --dev
```

```bash
.venv/bin/python -m pytest
```

合成实验不需要私有市场数据或 LLM API Key。
