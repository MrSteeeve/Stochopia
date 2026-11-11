# Stochopia

**由LLM研究循环持续修订的订单流市场世界模型**

[English README](README.md)

Stochopia是一个市场世界模型。它描述挂单、撤单、成交与价格变化之间的关系，由一个LLM研究循环提出假设、选择证据并提交修订。首个研究问题是市场冲击平方根律的适用范围。

## 状态

新代码正在搭建。仓库里目前只有通用工具模块：

| 文件 | 内容 |
|---|---|
| `stochopia/llm.py` | OpenAI兼容接口与Anthropic接口的客户端，模型在`config/models.yaml`登记 |
| `stochopia/llm_json.py` | 从模型回复中提取JSON |
| `stochopia/stats.py` | 种子派生、bootstrap置信区间、配对检验、Holm校正 |
| `stochopia/common.py` | 规范化JSON、功能性哈希ID、原子写入 |

研究笔记在`docs/research/`。

## 旧版：Benchmark v3

此前本仓库是结构化产品设计基准Benchmark v3，已停止维护。它的最终状态保存在标签`benchmark-v3-final`：

```bash
git checkout benchmark-v3-final
```

`docs/research/`下2026年11月11日之前的笔记描述的是那一版代码，文中的文件路径只在该标签下存在。

## 开发

```bash
uv sync --all-extras --dev
```

```bash
.venv/bin/python -m pytest
```

API key从环境变量读取，变量名见`.env.example`。
