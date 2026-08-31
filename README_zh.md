# Stochopia

**面向交易台 Agent 训练的状态化环境**

> 一个让 Agent 学会在不确定性中行动的构造世界。

[English README](README.md)

Stochopia 是一个状态化、部分可观测的金融决策环境。当前的中证 benchmark 评估 LLM 能否在有限的交易台/客户/风控交互、以及跨多轮月末决策累积的组合风险约束下，反复设计出可执行的结构化产品——挂钩中证500/中证1000指数的vanilla/障碍期权、自动赎回票据、雪球结构。它不宣称开放式金融创新能力，也不宣称生产级市场定价精度。

当前协议——回合设计、确定性定价/结算引擎、LLM原生环境角色、离线裁判、统计方法——定义在[docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md)中。本 README 只是快速上手指引，不是协议正本。

## 协议状态

- `stochopia.environment` 是新的 **v3-spine / Level-0** 训练接口：部分可观测与动态状态是环境不变量，动作和转移都有类型，`StochopiaStructurerEnv.reset()/step()` 内不调用 LLM，也不包含 forced prompt、策略规则或 oracle。默认不返回 evaluator 特权状态，并以有限的每轮 step 上限阻止 rollout 无限挂起。
- CLI 现在提供 v3 原生的`test-agent`入口；其余析因命令与`LongHorizonEnvironment`是为新 Stochopia v1 运行保留的 **v2 遗留引擎接口**。迁移前的轨迹、manifest、哈希和种子不兼容；Full/Static 不再是 v3 任务条件，v2 与 v3 引擎产物不得混合汇总。
- 每个 Agent 请求现在都携带完整、机器可读的动作契约。成交头寸会更新权威的客户/dealer账户，生命周期结果会进入 reward vector，episode 结束时所有未平仓合约都会按当前公允价值清算。
- v2 的动态头寸也已改为保存发行定盘价与绝对合约水平，处理障碍/敲入/自动赎回观察，并在下一市场快照重新计算 FV、delta、vega 与压力损失。

## Stochopia 评什么

- **设计一份合约，而不只是给它定价**：被测模型（`structurer`）可以查询客户、就交易台/风控/客户做定性咨询、请求确定性报价、提交设计方案——每轮受固定预算约束：3次查询、3次咨询、3次报价。
- **在部分可观测下工作**：Partial条件下，客户的硬性门槛（可投资金额、亏损容忍度、期限、产品白名单、保本要求）从不直接公开，只能通过`query_client`主动获取。
- **跨决策承担风险**：v3 环境始终动态；v2 遗留 runner 仍为新生成的 Stochopia 产物保留 Static 消融行为，但不重放迁移前的协议产物。
- **不能靠固定加价率取巧**：`dealer_margin`采用显式资金语义，取决于价内程度、vega敞口、路径依赖度、蒙特卡洛定价不确定性、静态delta对冲后的dealer压力损失与容量占用率；premium-paid 报价还会计入客户适配度。负margin不会被截成零。Level-0平价票据仍缺少payoff solve-for，因此在大规模训练前还需补完。精确公式见协议文档。

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
  hard_execution_rate、hard_execution_rate_given_submission、
  contract_acceptance_rate_given_hard_pass、settlement_acceptance_rate、dealer_margin、……
次级 WorkflowOutcome（仅当传入 --roles-config 且发生提交时）：
  workflow_deal、交易台/风控/客户各自的动作
离线：judge-runs（盲评、双裁判、批量跑；从不反馈进正式跑）
```

确定性核心——定价、硬约束检查、客户合同门与结算——是主榜的唯一权威，是episode数据与被测模型动作序列的纯函数，从不调用任何LLM。三个环境角色（客户、风控、交易台）是**可选**的：传入`--roles-config config/benchmark_roles.yaml`即可用各自独立的模型驱动定性对话与次级的工作流决策信号。不传`--roles-config`时，每一项主指标都与纯确定性路径逐字节相同。

## 快速开始

Stochopia 需要 Python 3.10 或更高版本。

```bash
cd Stochopia
uv sync --locked --all-extras --dev
```

`uv.lock`是开发与 CI 使用的可复现依赖集合。传统的
`python3 -m venv .venv && pip install -e ".[dev]"`仍可用于本地实验，但不属于
冻结环境。

真实调用模型前，先配置 API key：

```bash
cp .env.example .env
# 编辑 .env。默认的被测/环境模型走 DeepSeek，起步只需要填 DEEPSEEK_API_KEY；
# 全部已注册模型见 config/models.yaml。
```

v3 主接口是类型化的 partial-dynamic 环境。下面的冒烟运行完全确定性，
不需要 API key：

```python
import json
from pathlib import Path

from stochopia.benchmark import RiskBudget, load_market_snapshots
from stochopia.environment import EpisodeTask, StochopiaStructurerEnv, Skip
from stochopia.products import ClientProfile

root = Path.cwd()
snapshots = tuple(
    row for row in load_market_snapshots(
        root / "scenarios/stochopia_csi/market_snapshots.example.csv"
    )
    if row.episode_id == "SYNTHETIC_CSI500_DEMO"
)
client = ClientProfile(**json.loads(
    (root / "scenarios/stochopia_csi/client.example.json").read_text()
))
budget = RiskBudget(**json.loads(
    (root / "scenarios/stochopia_csi/risk_budget.example.json").read_text()
))
env = StochopiaStructurerEnv(EpisodeTask(snapshots, client, budget, task_seed=7))
observation, info = env.reset()
transition = env.step(Skip("typed v3 smoke test"))
print(observation.available_actions, transition.observation.round_num, info["seed_role"])
```

### v3 的可信结构：input -> interaction -> output -> evaluation

| 阶段 | 冻结或公开的对象 | 仅环境/evaluator 可见的真值 |
|---|---|---|
| Input | 带版本的`TaskSuite`、公开市场 observation、公开 action schema、固定公开名义本金网格 | 客户 mandate、风险限额、报价 policy 和完整 task manifest |
| Interaction | `ask_client`、`request_quote`、`submit_design`、`submit_product`、`skip` | 确定性定价、账户、约束和生命周期状态 |
| Output | 一条带哈希链的 trajectory，含动作、观察、reward vector、constraint signals 与运行来源 | 同一产物密封保存完整 task manifest 供重放，但绝不发送给策略 |
| Evaluation | 经重放验证的逐 task 指标，以及按 task 聚类的 policy 汇总 | evaluator 从密封输入独立重建精确经济状态 |

每个 v3 task 的最后一张快照是终端估值真值，不是新的决策轮。因此在最后
决策快照发行的合约，至少会经历一个市场区间，之后才到期或在 horizon
清算。训练代码必须通过`ScalarizedStochopiaEnv`和带版本的
`ScalarizationSpec`显式选择标量奖励；核心环境不会暗中决定权重。

真实 Agent 通过 v3 原生的`test-agent`命令测试。本地可执行程序每一步从
stdin 接收一个带版本的 JSON 请求，并向 stdout 输出一个动作 JSON：

```bash
stochopia-benchmark test-agent scenarios/stochopia_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/stochopia_csi/client.example.json \
  --risk-budget-json scenarios/stochopia_csi/risk_budget.example.json \
  --agent-command 'python my_agent.py' \
  --output outputs/v3-cli-agent.trajectory.json
```

API 模型既可以用`--model deepseek-v4-flash`从`config/models.yaml`选择，
也可以不修改注册表而直接接入。密钥只从指定的环境变量读取，不会进入 Agent
请求或轨迹：

```bash
export OPENAI_API_KEY='...'
stochopia-benchmark test-agent scenarios/stochopia_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/stochopia_csi/client.example.json \
  --risk-budget-json scenarios/stochopia_csi/risk_budget.example.json \
  --api-provider openai-compatible \
  --api-base-url https://api.openai.com/v1 \
  --api-model gpt-4o --api-key-env OPENAI_API_KEY \
  --output outputs/v3-api-agent.trajectory.json \
  --summary-output outputs/v3-api-agent.summary.json
```

正式评测不应只跑单个冒烟 episode。先冻结 task suite，再让同一策略跑完整
suite，最后独立经济重放与汇总：

```bash
stochopia-benchmark make-v3-suite market_snapshots.csv \
  --episodes CSI500_2024H1 CSI500_2024H2 \
  --client-json client.json --risk-budget-json risk_budget.json \
  --name stochopia-dev --version 1 --split dev \
  --output tasks/stochopia-dev.v1.json

stochopia-benchmark run-v3-suite tasks/stochopia-dev.v1.json \
  --agent-command 'python my_agent.py' \
  --replicates 3 \
  --output-dir outputs/stochopia-dev \
  --bootstrap-resamples 10000

stochopia-benchmark evaluate-trajectory \
  outputs/stochopia-dev/0000-r000-TASKHASH.trajectory.json \
  --suite tasks/stochopia-dev.v1.json \
  --output outputs/replayed.evaluation.json

stochopia-benchmark aggregate-v3 outputs/stochopia-dev \
  --output-json outputs/stochopia-dev.aggregate.json \
  --output-csv outputs/stochopia-dev.aggregate.csv
```

`run-v3-suite`只会复用 task、公开 run seed、环境配置和策略配置仍一致，并且
能够通过经济重放的完整轨迹；partial 或陈旧运行会重新执行。需要主动重跑
匹配产物时使用`--force`。

Python 侧使用`CommandAgentPolicy`、`LLMAgentPolicy`/
`create_api_agent_policy`和`run_agent_episode`走同一边界。模型输出的动作
格式错误会成为轨迹中可见的 invalid action；供应商、网络、命令退出与超时
故障则以`infrastructure_error`终止，不推进环境、不消耗 step，也不污染
invalid-action 计数。超时命令的整个进程组都会被清理。隐藏客户状态永远不会
发给策略。

Stochopia v1 轨迹包含公开动作 schema、环境配置、reset options、公开 run seed、
prompt/model 参数、依赖版本、实现哈希和 Git 工作树状态。哈希通过不等于经济
验证通过：`evaluate-trajectory`会重建 task，并要求每一个 transition 与
全新重放逐项一致。Agent summary 直接给出累计 reward/constraint 统计。
这不代表尚未实现的 TaskGenerator、repair teacher、counterfactual pair
exporter、payoff DSL/solve-for 或 fast/reference pricer 已经完成。

`stochopia-benchmark`还提供下面的 v2 遗留析因工作流，但所有新产物均使用
Stochopia v1 身份，迁移前产物不会被接受（`python -m stochopia.benchmark_cli`
调用的是同一个当前 CLI）：

```bash
# 校验市场快照 CSV（schema、来源、轮次连续性）。
stochopia-benchmark validate-market scenarios/stochopia_csi/market_snapshots.example.csv

# 用内置合成示例冒烟测试确定性交易台。
stochopia-benchmark demo scenarios/stochopia_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --product-json scenarios/stochopia_csi/product.example.json --full

# 只在开发集 episode 上标定并冻结风险预算。
stochopia-benchmark calibrate-budget scenarios/stochopia_csi/market_snapshots.example.csv \
  --episodes SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/stochopia_csi/client.example.json \
  --base-budget-json scenarios/stochopia_csi/risk_budget.example.json \
  --report-output outputs/budget_report.json \
  --budget-output outputs/risk_budget.frozen.json

# 用一个已注册模型跑一个冻结 episode。加上
# --roles-config config/benchmark_roles.yaml 可开启 LLM 原生的
# 客户/风控/交易台角色（可选；不传则保持纯确定性）。
stochopia-benchmark run-episode scenarios/stochopia_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/stochopia_csi/client.example.json \
  --risk-budget-json outputs/risk_budget.frozen.json \
  --model deepseek-v4-flash --strategy ledger_archive \
  --output outputs/demo_run.json

# 冻结完整析因实验清单（Full/Partial × Static/Dynamic，n>=3 重复，
# partial_dynamic n=5），然后执行（重跑会自动跳过已完成作业）。
stochopia-benchmark make-manifest --episodes SYNTHETIC_CSI500_DEMO \
  --models deepseek-v4-flash --output outputs/experiment_manifest.json
stochopia-benchmark run-manifest scenarios/stochopia_csi/market_snapshots.example.csv \
  --manifest outputs/experiment_manifest.json \
  --client-json scenarios/stochopia_csi/client.example.json \
  --risk-budget-json outputs/risk_budget.frozen.json \
  --outputs-dir outputs/csi_benchmark

# 聚合：每个（模型、条件）单元格的聚类 bootstrap 置信区间，
# 配对的 dynamic/partial-observability 退化对比，含 Holm 校正。
stochopia-benchmark aggregate outputs/csi_benchmark \
  --output-csv outputs/aggregate.csv --output-md outputs/aggregate.md

# 对冻结的 voluntary 提交做离线盲评裁判批跑。
stochopia-benchmark judge-runs outputs/csi_benchmark \
  --judge-models deepseek-v4-pro qwen-max --salt "$(openssl rand -hex 16)"
```

运行测试：

```bash
.venv/bin/python -m pytest
```

## 目录结构

```text
Stochopia/
|-- pyproject.toml                  # stochopia-benchmark 控制台脚本
|-- config/
|   |-- models.yaml                 # 模型注册表：各模型的接入信息
|   `-- benchmark_roles.yaml        # v2 角色行为：structurer + 3 个环境 NPC + judges
|-- stochopia/
|   |-- environment/                 # v3 reset/step、CLI/API Agent 适配与轨迹
|   |-- benchmark.py                # MarketSnapshot、RiskBudget、ProductDomainSpec、
|   |                               #   TradingDesk、HardConstraintEngine、结算
|   |-- benchmark_runner.py         # run_episode 主循环、compute_metrics
|   |-- benchmark_cli.py            # stochopia-benchmark 命令行入口（见“快速开始”）
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
|   `-- stochopia_csi/              # Stochopia v1 下的 v2 遗留引擎输入
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

正式实验用的市场数据必须携带明确的来源信息，并通过[docs/DATA_AUDIT.md](docs/DATA_AUDIT.md)中冻结的审计门禁；仓库内自带的`scenarios/stochopia_csi/market_snapshots.example.csv`是合成数据，不能用于报告结果。只有抽象的工作流程、schema与合成规则可以取材自机构实践经验——任何私有文档、客户、机构、限额或可识别的元数据都不得进入公开数据或prompt。

代码以[MIT许可证](LICENSE)发布。
