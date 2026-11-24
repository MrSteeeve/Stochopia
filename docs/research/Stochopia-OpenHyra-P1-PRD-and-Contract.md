# Stochopia × OpenHyra P1 真实深市盘口学习、动态假设演化与开放算子搜索设计稿（PRD & Contract）

**状态**　2026-11-23，修订版（对齐真实沪深 Level-2 三表数据与动态 $Z_t$ 演化）。承接已封板基线 `tier0-p0-final`（[Stochopia `5010985`](https://github.com/MrSteeeve/Stochopia/commit/5010985779126bdb9be22272fabac93eb61f642c) · [OpenHyra `051a45e`](https://github.com/MrSteeeve/OpenHyra/commit/051a45e1c74ab87873ce22a50bd347deb9ca1e6e)），联动 [LOB-Impact-Scoring-Design.md](https://www.google.com/search?q=https://github.com/MrSteeeve/Stochopia/blob/main/docs/research/LOB-Impact-Scoring-Design.md)。

## 0. 设计原则与工程红线

P1 不再人为编造静态合成世界菜单，而是直接在**小切片真实深市 Level-2 三表数据 + `JAX-LOB` 闭环撮合**上运行开放式神经网络架构与算子搜索：
1. **$Z_t$ 由真实实验残差动态驱动，不做先验预设**：不预设固定数量的误差机制类别。每轮将当前模型在真实 Level-2 开发切片上的微观结构残差与母单冲击误差反馈给智能体，由其自主提出、细化、分叉或证伪假设 $Z_t$，再转化为可执行的算子与结构改动 $X_t$。
2. **钉死五元组完整订单生成契约，严禁退化为简单分类器**：模型输出必须是能够 1:1 送入 `JAX-LOB` 执行的完整外生订单事件分布 $(\Delta t, \text{动作类型}, \text{买卖方向}, \text{相对档位}, \text{委托手数})$。价格路径与大单冲击曲线必须由生成订单经撮合引擎自然涌现，严禁退化为仅预测 `{买, 卖, 撤}` 标签的分类器或直接回归中间价收益。
3. **P1 与 P2 按数据广度切分，接口一次成型**：P1 在 1–2 只代表性深市股票的连续竞价轻量切片上跑通“真实数据训练 + 开放 PyTorch 算子演化 + `JAX-LOB` 母单冲击评估”全闭环；P2 直接复用同一套数据与模型契约，扩展至大规模多股票、长周期样本与吞吐加速。
4. **开源级可读性与防过度防御铁律**：新增模块单文件严格控制在 `150–350` 行以内；安全与契约边界仅用**前缀因果不变性扰动测试 + 参数/状态维上限 + 子进程沙箱**（合计不足 50 行数学与资源检查）守住，严禁编写 AST 静态审查器或运行时类型反射框架。

## 1. 真实数据源规格与事件流重构契约

### 1.1 原始三表结构（Wind 格式沪深 Level-2 Parquet）

数据源采用库外存放的每日三表 Parquet 目录（对应 `venvoo/china-a-share-l2-level2-limit-order-book-tick-data`，按 `YYYYMMDD/{行情,逐笔委托,逐笔成交}.parquet` 组织，原始文件严格留在 Git 仓库外）：

| 表名 | 核心字段与类型 | 微观结构语义与解析铁律 |
| :--- | :--- | :--- |
| `行情.parquet` | `wind_code`, `date` (`uint32`), `time` (`uint32`), `price`, `volume`, `amount`, `nun_trades`（注：原始列名为 `nun_trades`）, `prev_close`, `ask_px1..10`, `ask_vol1..10`, `bid_px1..10`, `bid_vol1..10`, `wavg_ask_px`, `wavg_bid_px`, `tot_ask_vol`, `tot_bid_vol` | 3 秒级十档盘口快照。价格均为真实人民币价格 $\times 10^4$（A 股股票 `1 tick = 0.01 元 = 100` 原始单位）。用于窗口初始十档盘口锚定与回放对账校验。 |
| `逐笔委托.parquet` | `wind_code`, `date`, `time` (`uint32`), `order_id` (`uint64`), `ex_order_id` (`uint64`), `order_type` (`string`), `order_code` (`'B'`/`'S'`), `price` (`×10000`), `volume`（股） | 深交所（`*.SZ`）10ms 级逐笔新委托到达流。`ex_order_id` 为交易所权威委托序号，是跨表关联的唯一主键。 |
| `逐笔成交.parquet` | `wind_code`, `date`, `time` (`uint32`), `trade_id`, `trade_code` (`'0'`/`'C'`), `bs_flag`, `price`, `volume`, `ask_order_id` (`uint64`), `bid_order_id` (`uint64`) | **深市双义表**：`trade_code == 'C'`（`price == 0`）为**撤单事件**，非零侧的 `ask_order_id` 或 `bid_order_id` 指向被撤原委托的 `ex_order_id`；`trade_code == '0'` 为撮合成交明细。 |

### 1.2 清洗、时钟解析与防重复计数规则

数据加载器 `Stochopia/stochopia/data/szse_l2.py` 必须严格执行以下四条领域规则：
1. **标的与连续竞价时段过滤**：P1 仅选取具备完整逐笔委托的深交所标的（`*.SZ`，默认选厚盘口大盘股 `000001.SZ` 与薄盘口创业板股 `300001.SZ` 作对照；2017 年上交所 `*.SH` 无逐笔委托，显式拒绝加载）。每交易日从开盘前顺序重放 `逐笔委托` 与 `逐笔成交` 以建立准确的在簿挂单字典，但**模型训练与评估窗口严格限定在连续竞价时段 `09:30:00.000 <= t < 11:30:00.000` 与 `13:00:00.000 <= t < 14:57:00.000`**（剔除 `09:15–09:25` 及 `14:57–15:00` 集合竞价，以及涨跌停单边锁死窗口 `ask_px1 == 0` 或 `bid_px1 == 0`）。
2. **`HHMMSSmmm` 物理时钟转换（严禁整数直减）**：原始 `time` 列为十进制编码整数（如 `93000190` 表示 `09:30:00.190`）。必须先经专用函数解析为距当日零点的真实毫秒数 `ms_of_day`（午休时段跨段不拼合为单个窗口），再计算相邻事件真实时间间隔 $\Delta t_t = \text{ms}_t - \text{ms}_{t-1} \ge 0$，严禁对 `HHMMSSmmm` 原始整数直接做减法。
3. **单一外生动作流重构（严禁把主动单与成交碎片重复计数）**：一笔穿过对手最优价的主动委托在 `逐笔委托` 中出现 1 次，并在 `逐笔成交` 中产生 1 至多条同刻 `trade_code == '0'` 记录。重构事件流时，**只保留外生到达的“委托（被动挂单 / 主动跨价或市价吃单）”与“撤单（`trade_code == 'C'`）”作为生成模型的事件序列**；`trade_code == '0'` 仅用于核验撮合状态推进，绝不作为独立生成事件重复喂入序列。
4. **按交易日严格时序划分**：`train`、`dev`、`test` 按自然交易日严格前向切分（`max(train_dates) < min(dev_dates) < min(test_dates)`），杜绝跨窗口或同日早晚盘信息泄漏。

## 2. 世界模型输入输出契约与开放算子活动空间

### 2.1 钉死的输入与五元组生成输出契约 `LOBWorldModel(nn.Module)`

在 `Stochopia/stochopia/neural/lob_interface.py` 中固定模型的外部 I/O 协议。无论受测 API 模型内部如何改写网络结构，其前向训练接口 `loss_and_predictions(batch)` 与单步闭环采样接口 `sample_next_event(state, book_step, rng)` 必须严格遵循以下输入与五元组输出：

| 侧别 | 字段与张量表示 | 精确物理含义与取值约束 |
| :--- | :--- | :--- |
| **输入 1 · 历史事件流**<br>`EventTokens [B, T, ...]` | `dt_ms` (`float`), `action_type` (`0..5`), `rel_ticks` (`int`), `volume_lots` (`float`), `cancel_age_ms` (`float`), `is_own` (`0/1`) | `action_type` 为 6 类外生动作：`{0: 买被动限价, 1: 卖被动限价, 2: 买主动吃单, 3: 卖主动吃单, 4: 撤买单, 5: 撤卖单}`；`rel_ticks` 为相对事件前同侧/对手一档的整数 tick 偏移；`volume_lots` 以 `1 手 = 100 股` 归一化；`is_own` 标记是否为外生注入母单子单（盲组置 `0`）。 |
| **输入 2 · 事件前盘口**<br>`BookState [B, T, ...]` | `levels` (`FloatTensor [B, T, 2, 10, 2]`), `spread_ticks` (`FloatTensor [B, T]`), `imbalance_summary` (`FloatTensor [B, T, 4]`) | 对应买卖十档的相对中间价 tick 差与 $\log(1 + \text{手数})$，以及一档价差、前十档买卖总委量与加权均价偏移。训练时来自重构盘口，生成评估时由 `JAX-LOB` 实时撮合更新喂回。 |
| **输出 · 五元组联合条件分布**<br>$p_\theta(e_{t+1} \mid \mathcal{H}_t)$ | **1. 时间间隔头 $\Delta t_{t+1}$**：零膨胀概率 $\pi_0 = \mathbb{P}(\Delta t = 0)$（刻画同 `10ms` 切片内突发簇）+ 正间隔分布参数 $p(\Delta t > 0)$；<br>**2. 动作×方向头 $(k_{t+1}, s_{t+1})$**：6 维分类 Logits（由框架自动对空盘口/不可撤侧施加合法掩码）；<br>**3. 相对档位头 $\delta p_{t+1} \mid k, s$**：挂单输出距最优价档位偏移分布（`0..20` ticks + 深档桶），主动单输出最大穿透档位，撤单输出在当前有挂单的 `1..10` 档上的选择 Logits；<br>**4. 委托手数头 $v_{t+1} \mid k, s, \delta p$**：A 股整数手数离散混合分布（常用手数 `{1, 2, 3, 5, 10, 20, 50+}` 分桶 + 尾部分布，撤单量自动截断于该档可用挂单量）。 | 每次采样精确产出一个完整五元组 $(\Delta t_{t+1}, k_{t+1}, s_{t+1}, \delta p_{t+1}, v_{t+1})$。训练目标固定为五元组分解负对数似然之和 $\mathcal{L}_{\text{NLL}} = \mathcal{L}_{\Delta t} + \mathcal{L}_{\text{act}} + \mathcal{L}_{\delta p} + \mathcal{L}_v$，**严禁只保留动作分类头或添加直接预测未来价格的捷径头**。 |

**闭环撮合桥接规则**：在生成与母单冲击评估阶段，模型每采样出一个五元组，`lob_adapter` 将其立即转为 `JAX-LOB` 指令（对生成的撤单 `(cancel, side, level_px, vol)`，按价格档位匹配在簿挂单扣减 `vol`），撮合更新十档盘口 $B_{t+1}$；当模拟物理时钟到达母单子单预定注入时刻时，插入外生主动子单并按协议分别驱动可见组（`is_own=1`，输入真实受冲击盘口）与盲组（输入未注入盘口分支）。

### 2.2 受测 API 模型的自由创造空间与评估器硬边界

P1 通过 `OpenHyra` 沙箱允许受测 API 模型直接编写和迭代完整的 `CandidateLOBWorldModel(LOBWorldModel)` Python 源码：

- **完全开放的架构与算子空间**：
  - **盘口与事件表征**：自由设计十档盘口空间卷积/注意力/不平衡差分特征、$\Delta t$ 连续时间编码、撤单账龄与自身标记嵌入；
  - **状态转移方程与记忆算子**：自由组合或发明任意因果递推结构——内置 `stochopia.neural.primitives` 提供高速分块并行仿射扫描 `parallel_affine_scan`（支持实/复对角多尺度谱 SSM/LRU/S5 以及衰减率随输入与 $\Delta t$ 动态变化的 Mamba 式选择性扫描），模型也可自写门控循环、快慢双时间尺度记忆库、事件类型条件状态更新律或混合堆叠层；
  - **内部拓扑与四级生成头耦合结构**：自由分配各子流状态维度（如背景订单流状态 vs. 局部盘口韧性状态），自由设计自回归条件头之间的内部特征连接。
- **评估器锁死的四道纯数学与资源边界（零繁琐 AST 审查）**：
  1. **五元组完备性与概率归一检查**：随机喂入合法虚拟批次，验证四级分布的 `log_prob` 有限、采样五元组可无损转为合法 `JAX-LOB` 指令；
  2. **前缀因果不变性硬门控**：随机扰动序列后缀 `t+1..T`，断言所有输出头在 `0..t` 的最大绝对偏差 `<= 1e-6`；
  3. **公平容量与耗时上限**：总可学习参数量 $\le P_{\max}$（校准后固定，例如 `32,768` 参数）且单步隐状态总维数 $\le D_{\max}$、单种子训练墙钟时间 $\le T_{\max}$，防止靠无脑堆大模型宽度刷分；
  4. **固定外部训练与评测循环**：数据切分、优化器与学习率退火、随机种子及 `LOB-Impact` 评测全部由 `Stochopia` 主进程持有，候选代码在无网络、无测试集路径权限的子进程中运行。

## 3. 残差驱动的 $Z_t \leftrightarrow X_t$ 多轮动态演化循环

P1 的研究循环围绕**真实 Level-2 实测残差剖面 $E_t$** 展开多轮演化（`R = 3 ~ 4` 轮），连接 `OpenHyra` 的 [program\_search.py](https://www.google.com/search?q=https://github.com/MrSteeeve/OpenHyra/blob/world-model/program_search.py)、[island\_scheduler.py](https://www.google.com/search?q=https://github.com/MrSteeeve/OpenHyra/blob/world-model/island_scheduler.py) 与 [scoring.py](https://www.google.com/search?q=https://github.com/MrSteeeve/OpenHyra/blob/world-model/tasks/stochopia_tier0/scoring.py)：

1. **Round 0（基线种子诊断）**：
   - 在深市 `train` 切片上拟合干净的基础种子模型 $X_0$（如标准单尺度 `GRU` 基线），在 `dev` 切片上运行自由生成与母单冲击实验，生成首份结构化**微观结构诊断残差剖面 $E_0$**：
     - **事件与盘口生成残差（对应 C1 细拆）**：同刻突发率 $\mathbb{P}(\Delta t = 0)$ 偏差、正时间间隔分位数误差、6 类动作比例偏差、成交流方向自相关衰减曲线残差、十档价差与深度恢复误差、无注单自由漂移与扩散率；
     - **母单冲击与回落残差（对应 [LOB-Impact](https://www.google.com/search?q=https://github.com/MrSteeeve/Stochopia/blob/main/docs/research/LOB-Impact-Scoring-Design.md) 分项与 C2–C5）**：归一化冲击幅度 $Y$、规模幂律指数 $\delta$、可见/盲组峰值比（C3）、停手后回落比曲线 $r(\tau)$ 以及四级生成头各自的分解 NLL。
2. **Round $t$ 动态假设生成与代码合成（$E_{t-1} \to Z_t \to X_t + \Delta Y_t$）**：
   - **动态形成/更新 $Z_t$**：Planner 读取历史账本与最新残差剖面 $E_{t-1}$，提出或修正结构化假设记录 `DynamicHypothesis` $Z_t$——明确指认当前最突出的实测失真现象、给出关于“当前网络哪一处归纳偏置缺失导致该失真”的机制解释，并归入新建或既有的假设岛；
   - **编写算子源码与事前定量预测**：Proposer 据 $Z_t$ 编写新候选 `CandidateLOBWorldModel` 源码 $X_t$（在进入训练前允许最多 1 次零数据的形状/因果/参数上限快检），并同步封存事前预测 `StructuredPrediction` $\Delta Y_t$——必须预测本次架构改动对**目标残差读数**的改善方向与幅度区间，以及对非目标护栏读数的保持范围。
3. **基于证实/证伪的跨轮演化规则（$E_t \to Z_{t+1}$）**：
   - **证实深化（Verified Mechanism Gain）**：若 $X_t$ 在 `dev` 上既改善目标基准指标、且事前预测 $\Delta Y_t$ 获 `SUPPORTED`，则 $Z_t$ 标记为获支持，$X_t$ 晋升为该岛新父模，下一轮针对其剩余次生残差 $E_t$ 提出深层子假设 $Z_{t+1}$；
   - **证伪转向（Falsified / Unverified Gain）**：若 $X_t$ 退化，或虽总分微涨但 $\Delta Y_t$ 被判 `REFUTED`，账本将该次尝试记为反例，强制下一轮修正原假设 $Z_t$ 或将算力预算切换至竞争假设岛。
4. **对照组与一次性时序留出复验（`Verifier`）**：
   - **三类对照**：`evidence_llm`（接收完整微观结构残差剖面 $E_t$ 与分项冲击曲线）、`metrics_llm`（相同模型、相同 PyTorch 代码改写空间与参数预算，但仅接收总五元组 NLL 标量与总 `LOB-Impact` 标量分，屏蔽细粒度残差剖面）、`no_llm`（在内置算子积木库 `[GRU, LRU, S5, SelectiveScan]` 与拓扑模板上按固定预算做随机变异重组）；
   - **冻结复验四维评价**：在严格位于未来的留出交易日 `test` 上运行唯一一次 `Verifier`，分别报告：(1) **$Z_t \to X_t$ 有效架构发现率**（同时实现测试集涨分且对应机制预测获支持的“已证实机制改进”比例、首次有效改进成本及跨标的 `000001.SZ` $\leftrightarrow$ `300001.SZ` 迁移表现）；(2) **$X_t \to \Delta Y_t$ 定量预测命中率**；(3) **五元组生成对数似然**；(4) **[`LOB-Impact`](https://github.com/MrSteeeve/Stochopia/blob/main/stochopia/measure/impact.py) 百分制总分与 C1–C5 通过项变化**。

## 4. 模块划分与文件规模预算

| 仓库与模块路径 | 职责说明 | 单文件行数上限 |
| :--- | :--- | :--- |
| `Stochopia/stochopia/data/szse_l2.py` | 读取深市三表 Parquet，完成 `HHMMSSmmm` 时钟解析、连续竞价过滤、盘口重构对账与外生五元组事件切片导出。 | `<= 320` 行 |
| `Stochopia/stochopia/neural/lob_interface.py` | 定义 `EventTokens`、`BookState`、五元组条件自回归分布输出协议 `LOBWorldModel`、前缀因果门控与参数量计数。 | `<= 220` 行 |
| `Stochopia/stochopia/neural/primitives.py` | 提供通用 `parallel_affine_scan`（支持多尺度对角 SSM/LRU/S5 与输入/$\Delta t$ 选择性扫描）及标准种子模型 `seed_gru_lob`。 | `<= 260` 行 |
| `Stochopia/stochopia/harness/lob_rollout.py` | 将五元组采样器与 `JAX-LOB` 闭环连接，执行自由生成与可见/盲组母单注入轨迹并输出残差剖面 $E_t$ 与 `LOB-Impact` 读数。 | `<= 320` 行 |
| `OpenHyra/tasks/stochopia_p1/` | 封装沙箱代码候选加载、动态假设 `DynamicHypothesis` 账本、连接 `program_search.py` / `island_scheduler.py` 的多轮入口与独立 `Verifier`。 | 每文件 `<= 320` 行 |

## 5. 分阶段落地计划（逐棒验收，稳步推进）

1. **P1-M1 · 深市三表重构器与对账单测（纯数据层）**：实现 `szse_l2.py`，在本地小样本深市三表切片上验证 `HHMMSSmmm` 时钟解析、撤单 `trade_code == 'C'` 关联、主动单防重复计数，以及重构十档盘口与 `行情.parquet` 快照的一致性。
2. **P1-M2 · 五元组世界模型接口、扫描原语与 `JAX-LOB` 闭环生成（纯 `Stochopia` 模型与评测层）**：落地 `lob_interface.py`、`primitives.py` 与 `lob_rollout.py`；在真实深市小切片上跑通种子模型 $X_0$（`seed_gru_lob`）的训练、闭环订单生成、母单注入冲击曲线与残差剖面 $E_0$ 导出（为动态提出 $Z_1$ 提供第一手真实数据残差）。
3. **P1-M3 · 沙箱算子合成与动态 $Z_t$ 分岛演化联调（`OpenHyra` × `Stochopia`）**：接通子进程沙箱、前缀因果/参数量快检、`DynamicHypothesis` 动态岛账本与 `Verifier`，用本地确定性测试验证多轮“残差 $\to$ 假设更新 $\to$ 改 PyTorch 算子代码 $\to$ 证伪/深化”完整闭环。
4. **P1-M4 · P1 正式跨模型开放架构演化实验**：在冻结的深市连续竞价 `train / dev / test` 切片上运行 `evidence_llm`、`metrics_llm`、`no_llm` 多轮架构搜索与一次性留出交易日复验，发布 P1 正式报告并为 P2 大规模扩展封板。