# DSL 合约设计科研实验：实现与验证

日期：2026-10-09。交接任务 A、B 已完成；C、D 已接通并完成离线流程实验，模型训练尚未开展。

> 2026-10-27 注：文中提到的轨迹哈希链、hash 记录与默认 `replay_verified` 已随《完整性机制收敛》移除；`artifacts/dsl-integration-20261009/` 只保留各夹具的 `summary.json` 与 `evaluation.json`，其 v1 轨迹文件已删除，新评测器不再读取 v1 格式。复现请重跑本节命令（`--output` 请指向新目录）。

## 最新研究收敛：股票需求驱动的 RSI 产品设计

用户已明确：弱化客户画像，直接用需求驱动 RSI 生成、评价和改进现金流产品；不以多智能体博弈为目标。当前限定股票标的，其他标的以后增加。本轮先研究“温和回撤保护并参与上涨”和“温和涨跌均获得收益”；每类分别设置要求保本和允许非保本的任务。定期收入等补齐真正中途付息后再做，不用到期累计支付替代。

已确认合成参数：实际投入 100 万元、期限 3 个月；回撤任务指定温和回撤路径收回投入、上涨 10% 路径三个月净收益至少 1%；双向任务指定正负 10% 路径净收益均至少 1%。保本任务的最终本金损失上限为 0%，允许非保本任务为 20%。非保本是允许承担本金风险，不强制候选产生本金损失。

OpenHyra 需求模块为 `tasks/stochopia_structuring/equity_demands.py`，需求协议与锚点见同目录 `TASK.md`。模型只生成 DSL 合约，评价器根据 Q 单位价格归一化名义本金，统一实际购买预算；先检查需求锚点、现有保守支付下界和完整环境成交，再按固定预算下的 P 侧期望净收益排序。没有改动 Stochopia 定价、结算或冻结 v3 指标，也没有修改 StructurerDSL。新的实验评分为 `equity-demand-expected-return.v1`，旧流程目标保留在显式 `--demand pipeline` 入口。

当前 P 侧是七个等权合成股票场景：上涨、横盘、温和下跌、强涨、深跌、回升、冲高回落；开发与留出使用不同完整路径。它是有限压力情景分布，未市场校准；不能将其期望净收益解释成真实市场收益预测，不能用它代替 Q 定价。锚点是明确的有限完整路径，也不构成所有温和回撤路径上的保证。保证下界沿用已有保守检查，没有新增通用证明、信息隔离或 hash 机制。

四份需求的离线控制实验各两轮、每轮两个候选，结果如下（均为三个月净收益率，不是年化）：

| 需求 | 开发期望净收益 | 留出期望净收益 | 可排序候选数（含种子） |
|---|---:|---:|---:|
| upside-protected | 0.6147% | 0.6298% | 2 / 5 |
| upside-risk20 | 0.6147% | 0.6298% | 3 / 5 |
| two-way-protected | 1.1179% | 1.1179% | 2 / 5 |
| two-way-risk20 | 1.1179% | 1.1179% | 4 / 5 |

四份任务均选择手写控制种子；允许非保本时额外的有界风险组合可以进入比较，但在当前等权场景下没有超过种子。所有离线候选仍为预设手写结构，不是模型设计，不证明 RSI 改进或结构创新。

结果保存在 `OpenHyra/artifacts/stochopia-equity-demands-20261009/offline/`。相关 OpenHyra 验证 **45 项通过（17.45 秒）**，覆盖固定实际预算、名义本金扩大不改变分数、保本选择、20% 最终损失门槛、评分现金分母、需求进入下一轮请求、API 格式和留出路由。独立进程 replay 所有 **109 份原始轨迹**及经验库内 **77 份副本**，全部通过；见该目录 `independent_replay.json`。当前新实验真实 API 调用数仍为 0；模型与调用预算正在等用户决定。

本轮修改文件：OpenHyra 的 `equity_demands.py`、`evaluator.py`、`generator.py`、`task.json`、`TASK.md`、`seed_solution/contract.json`、`experiments/run_stochopia_structuring.py`、`tests/test_stochopia_structuring.py`，以及本报告。未切换分支、提交、推送或覆盖其他工作；未运行全仓回归。

## 已确认的研究范围

聚焦合约设计。设计者提供现金流程序、连续参数和名义本金；环境计算未来现金流公允现值，
以该现值成交，不加价，不让候选选择发行价，不优化报价利润。
名义本金决定支付规模，实际发行收付款由估值决定；二者可以不同。
沿用现有公开信息边界及 hash 记录／回放，没有新增信息隔离或防篡改系统。

首轮支持单标的、固定月频网格上的终值支付、线性／分段线性组合、亚式、运行最值、
区间累计、条件支付和合约触发的自动提前终止。期权组合可以通过 clip、pl 等表达。
固定网格的“daily”别名在本实验中表示每个网格点，实际为月频，不是日频监测。

当前支付发生在到期或自动终止时；区间票息可以累计后支付。
多资产、按期中途付息、自主行权、发行后主动平仓／换仓暂未实现。
现有账户不处理客户后续追加保证金，因此子集要求保守估计的总支付下界非负；
允许存在被本金覆盖的有界负支付腿。支付下界按可能被终止抑制的支付腿分别估计，
不把可能取消的正支付当作保证金额。这不是通用 DSL 检查器的完整证明。

## A：原环境基线

固定合成市场：CSI500 月频 spot 为 6000、6100、6200、6300，波动率 20%、利率 2%、carry 0。
客户资金 1000 万元、最大损失比例 100%、年化收益门槛 2%、最低达标概率 25%。
客户和市场夹具是明确宽松的合成输入，仍经过现有约束，不属于正式 benchmark 数据。

通过 CommandAgentPolicy 执行询问、报价、提交与持有。原 Level-0 保本看涨结构
名义本金 100 万元、期限 3 个月、执行比例 0.95、参与率 1。
6 步，1 次有效成交，0 次无效动作；到期支付 110 万元，独立经济 replay 通过。
其按面额发行的报价利润为负；这项结果证明流程连通，不证明经济优势。

修复了外部命令入口的一个问题：Path.resolve() 把虚拟环境 Python 链接解析为系统 Python，
导致子进程丢失已安装依赖。现改为保留链接的绝对路径，并用真实虚拟环境子进程验证。

## B：DSL 闭环

入口为 `stochopia.environment.dsl.StochopiaDSLEnv`，显式启用，level 为 DSLExperimental。
代理使用 request_dsl_quote，再通过现有 submit_design 发行；询问和 skip 复用原接口。
旧 Level-0 解析器继续拒绝 DSL 动作及 custom 产品。

版本化载荷包含 Program JSON、参数、名义本金、客户别名和说明。
程序、参数通过 JSON 值快照保存；报价和发行都绑定到相应快照。
语言解析、原语解释与已有合法性检查来自 StructurerDSL；没有修改该仓库。
Stochopia 只增加本实验的适配范围和环境路由。

funded 根写为 `(funded 1 $issue_price body)`，其中 issue_price 是环境绑定的初始付款比例，
只能用于资金包装，不能用于未来收益函数。估值时排除 t=0 付款，成交时账户只扣款一次；
实际发行比例记录在已发行产品中。解释器按该实际比例执行整个交易时，得到的净现值
是“未来现金流现值减去发行付款”，不是存续资产价值。

Q 侧估值使用固定 seed 的 CPU float64 GBM，当前快照的利率、carry 与剩余期限波动率。
只生成未观察的未来，保留发行 fixing 和已发生的现货路径。
资产价值只包含当前时点之后的现金流，已结算部分不再估值或支付。
P 侧收益达标概率额外使用当前快照的 trend_alpha，不用于 Q 侧定价。
这是一个明确的合成定价模型，没有市场校准的波动率曲面、跳跃或复杂利率模型。

自动终止由解释器给出触发日期和支付，环境关闭持仓、更新账户并释放风险。
终止后的支付不会再次发生。期末尚存的合约按剩余现金流现值清算。
现有客户门槛、账户、持仓预算和提交接受逻辑继续由受信环境执行。

Greeks 使用共同随机数差分：相对现货扰动导数除以现货／现货平方，得到绝对 delta／gamma；
绝对 sigma 导数除以 100 得到每一个波动率百分点的 vega。存续期扰动只改变当前和未来现货，
历史观察与发行 fixing 保留。压力损失使用固定的现货正负 20% 与波动率正负 10 个百分点情景，
是本实验的模型风险量，不是完整市场风险评价。

最小实现位于 adapters/structurer_dsl.py 和 environment/dsl.py。
共用代码只增加必要的持仓构造、观察／结算和下界复用入口，以及代理解析、轨迹版本和 replay 路由。
原定价／结算函数保留；BENCHMARK_PROTOCOL.md 增加实验指针，正式统计继续冻结。
DSL 评价格式为 stochopia.dsl-experiment-evaluation.v1，原 v3 聚合器不会混入这类结果。

## 实际运行结果

以下五份手写夹具都通过外部策略入口运行，各完成一次有效成交、一次结算，
无无效动作；另一个进程独立 replay 全部通过。DSL 各次运行为 4096 条路径、seed 42、CPU 单线程。

| 夹具 | 成交价（元） | 结算支付（元） | 关闭原因 |
|---|---:|---:|---|
| 香草看涨 | 71,547.39 | 100,000.00 | 到期 |
| 看涨价差 | 49,289.74 | 100,000.00 | 到期 |
| 算术亚式 | 61,967.63 | 83,333.33 | 到期 |
| 区间累计票据 | 1,018,704.76 | 1,030,000.00 | 到期 |
| 自动赎回票据 | 1,005,619.24 | 1,010,000.00 | 首个观察日自动终止 |

自动赎回后继续经历后续市场，但没有重复支付。票据面额为 100 万元；
若成交价高于 100 万元，保证返还面额不能覆盖购买成本，因此不会自动标为保本。

相关测试共 **110 项通过（30.57 秒）**，覆盖新适配器、完整 replay、原代理入口／环境／评价、
benchmark 账户／约束与既有报价经济学。香草和价差与解析价格比较使用事先规定的 4 倍 MC 标准误，
Greeks 另用解析结果验证，未要求 MC 与解析逐字节相同。
测试还覆盖历史条件化、初始付款只计一次、自动终止、期末清算、报价快照和不支持的语义。
此处 A、B 验证未运行全仓所有数据测试或 StructurerDSL 全仓测试；C、D 的 OpenHyra 验证见下文。
不承诺任意平台或线程配置下逐位相同；独立 replay 在同一 CPU／依赖／线程设置下验证。

## 复现命令与工件

在 Stochopia 根目录，现有 .venv 已安装 editable StructurerDSL。
其他环境可先安装本仓库及 ../StructurerDSL；未添加安装器或新的依赖管理系统。

```sh
.venv/bin/python examples/structuring_roundtrip.py --output artifacts/dsl-integration-20261009/baseline-rerun
.venv/bin/python examples/dsl_roundtrip.py --case autocall --output artifacts/dsl-integration-20261009/autocall-rerun
.venv/bin/python examples/dsl_roundtrip.py --case asian --output artifacts/dsl-integration-20261009/asian-rerun
.venv/bin/python -m pytest tests/test_structurerdsl_adapter.py tests/test_v3_dsl_roundtrip.py tests/test_agent_runner.py tests/test_environment.py tests/test_v3_evaluation.py tests/test_benchmark.py tests/test_pricing_economics.py -q
.venv/bin/python -c 'import torch; torch.set_num_threads(1); from stochopia.environment import replay_and_evaluate; print(replay_and_evaluate("artifacts/dsl-integration-20261009/autocall/trajectory.json")["replay_verified"])'
```

每个夹具目录保存 trajectory.json、evaluation.json、summary.json。
artifacts/dsl-integration-20261009 下还保存 test_results.txt、independent_replay.json、runtime_dsl.json。
此前的固定面额／加价报价比较（pricing_example.json）已于 2026-10-27 删除，其公式不是本轮 DSL 的成交规则；需要时用 `examples/structuring_roundtrip.py --pricing-example` 重新生成。
初始 A 环境记录保留在 runtime.json。

检查到的 HEAD：Stochopia 6186a6343c32ce52a7b387658438bd16df29a2f6；
StructurerDSL b546db9fccf51f3beb180201f90660d9014ad219；
OpenHyra e998fd55369954f3a1da670b7df4fec512080b6d。
三仓库已有未提交／未跟踪内容，未覆盖；未切换分支、提交或推送。
冻结论文工件未触碰。

## C、D：OpenHyra 产品设计搜索与生成器接口

OpenHyra 新增 `tasks/stochopia_structuring/` 与 `experiments/run_stochopia_structuring.py`。
候选为固定的 DSL 合约工件，评价通过公开策略接口执行三次询问、一次报价、一次提交及后续持有。
复用既有 `AlgorithmDiscoveryLoop`、`DiscoveryLedger`、`ExperienceBank`，未改百慕大评价器或搜索核心。
同时提供原 Task 可加载的配置与标准 JSON evaluator；端到端验收使用专用 API／离线入口，
未运行通用 CLI 搜索或 macOS sandbox。

暂用 `accepted-client-return.v1` 做离线流程检查：所有开发场景合法且成交才可排序，
以客户资金流调整后的净清算财富变化／初始现金的场景均值排序。
完整奖励向量、约束、交互与评价成本保留。最终真实 API 实验的评分口径待用户决定；
这个暂用目标没有客户风险效用，也不是联合福利或合约创新性的证明。

开发场景 2 个，公开例子 1 个，留出场景 2 个，均为合成月频固定路径。
两轮各生成 2 个候选，连同手写价差种子共 5 个。先冻结同轮父代与开发反馈，
评价完成后才更新经验；先选择最佳，再单独评价留出。留出不进入生成请求或经验训练行。
CPU 单线程、4096 条定价路径、seed 42；已有 DSL 轨迹和 hash／replay 规则沿用。

离线流程中全部 5 个候选在两个开发场景均合法成交；最佳仍为手写种子。
其开发均值为 0.0024521073（0.2452%），留出均值为 -0.0003812260（-0.0381%）。
留出两场景也均成交。这个结果证明流程完成，没有证明搜索改进或泛化有效。

`ContractGenerator.generate(request) -> Generation` 是可替换入口：输入公开条件、父代、
开发反馈、目标和轮次，输出工件、原始回复、用量、时长与错误。
离线结构夹具和 OpenAI-compatible API 均实现。真实 API 接入优先于 CLI；
首轮模型与调用额度待用户决定，当前实际真实 API 调用数为 0。
API 固定每候选一次调用，无工具、无自动重试或截断升额；失败也占额度。
以后学习模型可替换生成器，当前未训练模型，没有新增模型管理框架。

相关 OpenHyra 测试 **38 项通过（4.78 秒）**：新增任务、已有 discovery、feedback、
experience events 和 generic task surfaces。包含真实 HTTP 请求格式的模拟响应测试、失败调用计数、
两轮父代／反馈、留出不回流、原 Task 加载、标准 evaluator 的 audit 路由及独立进程 replay。
API 模拟响应不等于供应商真实调用。OpenHyra 全仓测试未运行。

从 OpenHyra 目录运行：

```sh
../Stochopia/.venv/bin/python experiments/run_stochopia_structuring.py --generator offline --paths 4096 --output artifacts/stochopia-structuring-20261009/offline-final
../Stochopia/.venv/bin/python -m pytest tests/test_stochopia_structuring.py tests/test_algorithm_discovery.py tests/test_feedback.py tests/test_experience_events.py tests/test_generic_task_surfaces.py -q
```

详细协议见 `OpenHyra/tasks/stochopia_structuring/TASK.md`。最终离线工件在
`OpenHyra/artifacts/stochopia-structuring-20261009/offline-final/`，含实验配置、生成请求／输出、
经验训练行、搜索事件、EB、所有场景轨迹及评价、冻结选择和独立的留出结果。
另一个 CPU 单线程进程独立 replay 所有 13 份评价轨迹，全部通过；结果保存为 `independent_replay.json`。
API 可以显式限制预算和输入；公平比较仍要固定模型、提示、调用额度、交互、价格模型和评价划分。

## 四组模型官方 API 接口（2026-10-09）

新增 OpenHyra 股票需求任务的四组 API 预设与独立比较入口：Anthropic Haiku 5.5
原生 Messages、DeepSeek V4.1 Flash 与 StepFun Step 5 Preview 官方 Chat Completions，
以及 OpenRouter Nemotron 3 Super。用户选择按激活参数量考虑 Super，并使用各模型
官方默认推理、记录实际用量。配置省略采样和推理强度字段，不接入 CLI 模型后端。

每个模型／需求对建立独立生成器、调用额度、搜索和经验库；使用相同手写种子与评价口径，
第一轮设计请求相同，后续只用本组开发反馈。设计提示不包含本机日志路径、耗时或模型组名。
各模型结果在独立目录，模型汇总分别列出各需求；共同索引仅含路径，不混合分数或选统一赢家。
每次调用记录 UTC 时间、请求与返回模型、供应商（若返回）、完整响应、usage、状态与错误。
拒答、截断、网络或解析失败占一次额度，无重试、升额或模型替换。

比较入口默认只显示计划；完整默认设置为四组 × 四需求 × 两轮 × 两候选，最多 64 次生成调用，
每次最多 4000 输出 token。`--execute` 才执行生成。本轮没有生成 POST 请求：真实调用数仍为 0。
StepFun 官方模型列表真实 GET 验证返回 200，含 `step-5-preview`；其余生成接口与 StepFun 生成
格式用模拟 HTTP 响应测试，没有声称四家真实生成联通。当前本机缺 Anthropic 和 OpenRouter 凭据。

DeepSeek 官方披露总 552B、输入激活 8B／输出激活 16B，Super 为总 120B／激活 12B。
Haiku 与 Step 的参数量未从官方公开资料核实，保存未知；不能称四组已严格匹配参数规模。
默认推理、不同 tokenizer 及供应商服务也不构成等计算量。收益评分沿用用户已确认的股票需求
约束后最大化期望净收益，未擅自新增创新分数，收益提升不能单独证明结构创新。

相关 OpenHyra 测试 **57 项通过（48.02 秒）**，覆盖四家协议、原始响应／用量、失败额度、
计划与凭据预检、相同初始请求、两轮各组反馈及跨需求额度重置、独立 EB 和结果。
命令为 `../Stochopia/.venv/bin/python -m pytest tests/test_stochopia_api.py tests/test_stochopia_structuring.py tests/test_algorithm_discovery.py tests/test_feedback.py tests/test_experience_events.py tests/test_generic_task_surfaces.py -q`。
默认 CLI 计划已实际运行，存于 `OpenHyra/artifacts/stochopia-equity-demands-20261009/api-interface-plan.json`。
详细模型依据、改动文件清单、配置与执行命令见 `OpenHyra/tasks/stochopia_structuring/API.md`。
本轮 Stochopia 只补本文档；定价、结算、环境协议及既有记录／hash 行为沿用，未新增机制。

## 真实 API 实验与后续决策（2026-10-09）

上节的 4000 token、零真实生成和缺失凭据描述为接口初次验收状态。随后用户在本机
`/Users/williamqian/.config/stochopia/model-api.env` 填好凭据并授权执行；助手没有查看密钥内容。
首次 Haiku 的 7 个已返回请求均将 4000 token 用于思考，没有最终合约；第 8 个请求被中断，
用量未知。用户要求移除额外上限，代码现按供应商默认边界运行，Anthropic 的必填上限用
官方模型最大值 128000。非流式响应读取不再设 120 秒超时，连接和写入仍保留超时。
相关测试在修正后 **61 项通过（48.64 秒）**；没有新增重试、失败修复或安全隔离机制。

用户批准四组完整重跑 64 次，并另外批准 DeepSeek 的最大输出对照及正式 16 次新批次，
以及 Nemotron 切到固定 DeepInfra 的 16 次新批次。旧诊断、对照和新批次分别保存：

- Haiku 正式 16 次已完成；10 个模型候选通过需求，三个需求的开发收益超过手写种子。
  选择的四个产品都没有新增相对种子的 DSL 原语结构，主要为参与率、窄价差或票息变化。
  独立 CPU 单线程重放 130 条主轨迹，130/130 通过。报告在
  `OpenHyra/artifacts/stochopia-equity-demands-20261009/api-four-models-v2/claude-haiku-5.5/analysis.md`。
- DeepSeek 默认 65536 输出 token 用尽且无合约。最大值 393216 的单次对照成功后，
  用户批准按同一默认推理新跑 16 次；目录 `api-four-models-v2-deepseek`。16 次全部正常
  返回，原规则下 8 个模型候选达标，主轨迹独立重放 116/116 通过；对照未混入正式结果。
  双向保本选择为数字票息加看跌腿，非保本选择为两层数字票息，均有相对种子的组合变化；
  单次收益提升不构成金融创新证明。
- Step 一次请求达到服务最大输出 64000 且无最终文本。用户选择保留默认推理继续原批次，
  目录 `api-four-models-v2-step`，不降低推理或更改提示。供应商推理 token 报告为零，
  与大量思考文本不一致，应明确记录。
- Nemotron 原 DekaLLM 批次已返回 8 次，其中 5 次为上游共享池 HTTP429。用户批准的新
  DeepInfra 批次固定供应商、禁止 fallback，BF16、服务最大输出 16384；第一笔仍为上游
  `engine_overloaded` / `upstream_provider_shared_pool`。用户明确结束本次运行，分别保留
  `api-four-models-v2-nemotron` 和 `api-four-models-v3-nemotron-deepinfra`，不继续请求。

收益均为固定实际投入 100 万元、三个月、合成 P 情景的期望净收益；Q 定价仍为既有
4096 路径、seed 42 的 GBM 公允现值。没有市场校准或连续情景泛化证明。限流失败、
输出截断、金融需求不达标、参数改进与结构创新分别分析，不能汇成一个模型能力排名。
人工暂停期间的请求时长包含暂停，不是纯推理延迟；中断用量未知，不按零处理。

实时状态和后续各组完整报告以 OpenHyra 工件根目录 `live_run_state.json` 与组内
`analysis.md` 为准。主框架、定价、结算、DSL 语言和冻结论文工件没有因这些运行决策改动。

### 已批准、待原批次结束的本金下界修正

DeepSeek 两个 `premium → pay(..., maturity)` 候选没有条件或提前终止，支付表达式
最低为 0.85；同一 Q 成交价下损失上界为 14.8712%、14.8656%，均满足 20% 限制。
现有适配器将每次 pay 都视作可能缺席，给出零支付下界，导致原规则误拒。
用户批准先完成原批次，再针对直接无条件到期支付保留检查器已有表达式下界，
所有模型已有工件统一离线重评，原始结果保留，不新增 API 调用。
当前只保存修正补丁、回归检查和重评脚本，未在运行中改变评价器。
