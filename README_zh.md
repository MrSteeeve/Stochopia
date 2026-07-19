# MIRAGE：中证结构化产品设计 Benchmark

[English README](README.md)

MIRAGE 评估 LLM 能否在部分可观测、有限的交易台/客户/风控交互、以及跨多轮月末决策累积的组合风险约束下，反复设计出可执行的结构化产品——挂钩中证500/中证1000指数的vanilla/障碍期权、自动赎回票据、雪球结构。它不宣称开放式金融创新能力，也不宣称生产级市场定价精度。

完整协议——回合设计、确定性定价/结算引擎、LLM原生环境角色、离线裁判、统计方法——冻结在[docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)中。本 README 只是快速上手指引，不是协议正本。

> 早期的纳斯达克100“Structurer Playground”原型（`run.py`、`mirage.cli`、`mirage.engine`）已下线，这条入口在当前仓库中已不存在。`scenarios/structurer_nasdaq/`目录仅作为历史场景数据保留，无法通过任何现有命令运行。

## MIRAGE 评什么

- **设计一份合约，而不只是给它定价**：被测模型（`structurer`）可以查询客户、就交易台/风控/客户做定性咨询、请求确定性报价、提交设计方案——每轮受固定预算约束：3次查询、3次咨询、3次报价。
- **在部分可观测下工作**：Partial条件下，客户的硬性门槛（可投资金额、亏损容忍度、期限、产品白名单、保本要求）从不直接公开，只能通过`query_client`主动获取。
- **跨决策承担风险**：Dynamic轮次在一个episode的六个月末轮次之间保留未平仓名义本金、delta、vega、压力损失与客户信任度；Static轮次每轮重置。
- **不能靠固定加价率取巧**：`dealer_margin`是一个成本加成函数，取决于价内程度、vega敞口、路径依赖度、蒙特卡洛定价不确定性、压力损失与容量占用率，并再乘以一个客户适配度系数——因此“永远报最大名义本金的vanilla”不再是占优策略。精确公式见协议文档。

## 双层架构

```text
StructurerRole（被测 LLM）
   |  query_client / consult / request_quote / submit_design / skip_round
   v
LongHorizonEnvironment（中心路由）
   |-- ClientRole LLM      （可选；partial 信息披露、决策）
   |-- RiskRole LLM        （可选；咨询、决策）
   `-- DeskRole LLM        （可选；咨询、决策）
   v  只读、受事实约束的调用
DeterministicCore：定价 -> 硬约束检查 -> 客户合同门 -> 结算（从不调用 LLM）
   v
主榜结算（始终计算，确定性）：
  hard_executable、client_contract_pass、dealer_margin、hard_feasibility_rate、……
次级 WorkflowOutcome（仅当传入 --roles-config 且发生提交时）：
  workflow_deal、交易台/风控/客户各自的动作
离线：judge-runs（盲评、双裁判、批量跑；从不反馈进正式跑）
```

确定性核心——定价、硬约束检查、客户合同门与结算——是主榜的唯一权威，是episode数据与被测模型动作序列的纯函数，从不调用任何LLM。三个环境角色（客户、风控、交易台）是**可选**的：传入`--roles-config config/benchmark_roles.yaml`即可用各自独立的模型驱动定性对话与次级的工作流决策信号。不传`--roles-config`时，每一项主指标都与纯确定性路径逐字节相同。

## 快速开始

MIRAGE 需要 Python 3.10 或更高版本。

```bash
cd MIRAGE
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

真实调用模型前，先配置 API key：

```bash
cp .env.example .env
# 编辑 .env。默认的被测/环境模型走 DeepSeek，起步只需要填 DEEPSEEK_API_KEY；
# 全部已注册模型见 config/models.yaml。
```

命令行工具安装为`mirage-benchmark`（`python -m mirage.benchmark_cli`效果相同）：

```bash
# 校验市场快照 CSV（schema、来源、轮次连续性）。
mirage-benchmark validate-market scenarios/mirage_csi/market_snapshots.example.csv

# 用内置合成示例冒烟测试确定性交易台。
mirage-benchmark demo scenarios/mirage_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --product-json scenarios/mirage_csi/product.example.json --full

# 只在开发集 episode 上标定并冻结风险预算。
mirage-benchmark calibrate-budget scenarios/mirage_csi/market_snapshots.example.csv \
  --episodes SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/mirage_csi/client.example.json \
  --base-budget-json scenarios/mirage_csi/risk_budget.example.json \
  --report-output outputs/budget_report.json \
  --budget-output outputs/risk_budget.frozen.json

# 用一个已注册模型跑一个冻结 episode。加上
# --roles-config config/benchmark_roles.yaml 可开启 LLM 原生的
# 客户/风控/交易台角色（可选；不传则保持纯确定性）。
mirage-benchmark run-episode scenarios/mirage_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json outputs/risk_budget.frozen.json \
  --model deepseek-v4-flash --strategy ledger_archive \
  --output outputs/demo_run.json

# 冻结完整析因实验清单（Full/Partial × Static/Dynamic，n>=3 重复，
# partial_dynamic n=5），然后执行（重跑会自动跳过已完成作业）。
mirage-benchmark make-manifest --episodes SYNTHETIC_CSI500_DEMO \
  --models deepseek-v4-flash --output outputs/experiment_manifest.json
mirage-benchmark run-manifest scenarios/mirage_csi/market_snapshots.example.csv \
  --manifest outputs/experiment_manifest.json \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json outputs/risk_budget.frozen.json \
  --outputs-dir outputs/csi_benchmark

# 聚合：每个（模型、条件）单元格的聚类 bootstrap 置信区间，
# 配对的 dynamic/partial-observability 退化对比，含 Holm 校正。
mirage-benchmark aggregate outputs/csi_benchmark \
  --output-csv outputs/aggregate.csv --output-md outputs/aggregate.md

# 对冻结的 voluntary 提交做离线盲评裁判批跑。
mirage-benchmark judge-runs outputs/csi_benchmark \
  --judge-models deepseek-v4-pro qwen-max --salt "$(openssl rand -hex 16)"
```

运行测试：

```bash
.venv/bin/python -m pytest
```

## 目录结构

```text
MIRAGE/
|-- pyproject.toml                  # mirage-benchmark 控制台脚本
|-- config/
|   |-- models.yaml                 # 模型注册表：各模型的接入信息
|   `-- benchmark_roles.yaml        # v2 角色行为：structurer + 3 个环境 NPC + judges
|-- mirage/
|   |-- benchmark.py                # MarketSnapshot、RiskBudget、ProductDomainSpec、
|   |                               #   TradingDesk、HardConstraintEngine、结算
|   |-- benchmark_runner.py         # run_episode 主循环、compute_metrics
|   |-- benchmark_cli.py            # mirage-benchmark 命令行入口（见“快速开始”）
|   |-- pricing.py                  # Black-Scholes、障碍期权、蒙特卡洛、quote_economics
|   |-- products.py                 # ProductSpec、ClientProfile、MarketState
|   |-- env_agents.py               # FrozenEnvAgent、grounding 校验、响应缓存
|   |-- role_config.py              # benchmark_roles.yaml 加载与校验
|   |-- judge.py                    # 离线软性质量裁判 + 可靠性指标
|   |-- stats.py                    # 种子派生、bootstrap CI、Wilcoxon、置换检验、Holm
|   |-- experiment.py               # 析因实验清单 + 配对条件对比
|   |-- llm.py                      # OpenAI 兼容与 Anthropic 客户端、Mock 客户端
|   `-- market_builder.py、cffex_data.py、tushare_*.py、formal_*.py、raw_data_audit.py
|                                   # 市场数据获取与来源审计管线
|-- scenarios/
|   |-- mirage_csi/                 # v2 episode：快照、prompts/、benchmark.yaml
|   `-- structurer_nasdaq/          # 遗留 Playground 场景数据（无可运行入口）
|-- docs/
|   |-- BENCHMARK_PROTOCOL.md       # 协议正本
|   |-- DATA_AUDIT.md               # 数据来源审计与 go/no-go 门禁
|   `-- redesign/                   # v2 设计记录（REDESIGN_PLAN.md 与两份草案）
|-- data/                           # 冻结/派生标定产物（原始数据已 gitignore）
|-- outputs/                        # 运行产物（结果 JSON、清单、聚合结果）
|-- tests/
|-- README.md                       # 英文 README
`-- README_zh.md                    # 中文 README
```

## 数据来源与许可

正式实验用的市场数据必须携带明确的来源信息，并通过[docs/DATA_AUDIT.md](docs/DATA_AUDIT.md)中冻结的审计门禁；仓库内自带的`scenarios/mirage_csi/market_snapshots.example.csv`是合成数据，不能用于报告结果。只有抽象的工作流程、schema与合成规则可以取材自机构实践经验——任何私有文档、客户、机构、限额或可识别的元数据都不得进入公开数据或prompt。

代码以[MIT许可证](LICENSE)发布。
