# Stochopia 市场世界模型可行性复核与 V0 决策

日期：2026-08-31

迁移前的源代码提交：`cc75bb84bcd80428546bf5c7ab274046de06fd5a`。本研究复核发生在 Stochopia 命名空间迁移之前；该哈希证明的是继承实现的当时状态，不代表一个已发布的 Stochopia 版本。

## 结论

**Conditional Go：可以先做独立、离线、可种子复现的 Regime-Switching GBM 合成压力情景发生器；不可以把它称为“市场世界模型”，也不可以接入或替代 Benchmark v3 的正式评测。**

当前 Stochopia 最准确的定位是：**历史路径回放上的衍生品决策环境与经济结算核心**。它能在给定外生未来路径时计算行动后果，但市场不会因 Agent 行动改变，也没有可验证的预测型市场转移模型、hypothetical branch、向量化 rollout 或训练闭环。

V0 的研究对象因此被严格冻结为：

> 给定显式参数和随机种子，离线生成一条外生的 synthetic physical market path，并把月末观测映射为现有 `MarketSnapshot`；生成结果只用于压力测试和训练侧实验，不构成现实预测、反事实真值或 benchmark 证据。

## 主分析与独立复核

主 Codex 与两名未读取彼此草稿的 deep-reasoner，以及两名只读代码 explorer，结论均为 **Conditional Go**。共识是：生成必须发生在 `EpisodeTask` 之前；Benchmark v3 核心保持冻结；P/Q 分离、latent-state 隔离、可复现 provenance 和 synthetic/benchmark 分型都是硬门槛。

复核中有三处范围差异，本轮按更保守解释处理：

1. **月频直接生成还是日频 bridge**：最小 schema 接入可只生成月频，但这会伪造或缺失 20/60 日 RV 和回撤。本轮采用日频 weekday-only physical path，再取月末快照；同时明确它不是中国交易日历。
2. **是否已经是完整 MarketState generator**：显式 Q 侧 ATM IV 可以让快照被现有定价器消费，但它不是从数据中联合生成的 IV dynamics。因此 V0 定位降为 **spot-path baseline with supplied pricing inputs**。
3. **是否立即封装 TaskSuite**：技术上可以把路径冻结成 `EpisodeTask`，但现有 suite/aggregate 没有足够强的 track/schema 污染隔离。本轮只产出快照与独立 artifact，不增加 suite、CLI 或 evaluator wiring。

另一个未在本轮冒充解决的差距是：经验研究最终应建模 CSI500/CSI1000 的联合路径和共享/相关状态；当前单标的 API 只是可组合脚手架，不是双指数模型成立的证据。

## Artifact 边界

用于复核的完整 Claude 会话仅保留在本地，不进入 Git 仓库。本文只保留经过代码对照后的研究判断；完整私人对话及其页面标识不构成项目发布物。

## 对照判断

| Artifact 关键判断 | 当前仓库事实 | 复核结论 |
| --- | --- | --- |
| Stochopia 是“高保真 simulator” | 市场是预装月频快照；无流动性、价差、冲击、交易成本和月内路径；障碍/自动赎回按月末点监测 | **过度表述**。应称“确定性历史回放决策环境” |
| 世界模型与 Benchmark 可共享 Economic Core | 定价、约束、账户、持仓和结算可复用 | **部分成立**。共享核心不等于共享协议、数据、artifact 或 aggregate |
| World Model 不重视可复现性，可去掉哈希链/replay | 合成路径若不可重建，就无法排查训练差异、验证参数与审计结果 | **不成立**。实验轨同样必须保留 seed、规范化 manifest、source/data/path hash 和 exact reconstruction |
| 市场情景生成器是关键缺口 | 当前没有 TaskGenerator、市场发生器、校准协议或预测评分 | **成立**，但只是多个缺口中的第一个 |
| Regime-Switching GBM 与现有核心“完美兼容” | `MarketSnapshot` 需要 spot、r、carry、期限 ATM IV 和历史指标；现有 suitability 又把 Q 定价波动率用于 P 侧概率 | **只在 schema 层兼容**。必须分离 P 路径参数与 Q 定价输入，且不能声称已解决完整 P/Q 接口 |
| regime 标签可作为额外研究信号 | `MarketSnapshot.regime` 会进入公开 observation | **危险**。latent regime 直接写入快照会造成隐藏状态泄漏 |
| 12 个历史情景可变成无限参数化情景 | 数量可以无限，信息仍受参数假设和校准样本限制 | **数量成立、可信度不成立**。不能把模拟数量当有效样本量 |
| V0 需要 2–3 天 | 纯发生器和单元测试可在这个量级；独立 suite、evaluator、结算尾部、校准与科学验证不可能 | **只对最小模块成立** |
| Chronos-2 可“零工程”升级 | 它是通用概率预测模型，没有 Stochopia 的金融 schema、P/Q、无套利和路径依赖验证 | **不成立**。只能作为未来候选基线 |
| VolGAN 可自然成为 V2 | VolGAN 面向日频 SPX spot + 完整 IV surface；当前正式面板每个标的只有 36 个左右月度观测，且只有 ATM 期限代理 | **当前数据不兼容** |

## 当前代码边界

### 已有

- `EpisodeTask` 能封存完整快照序列、客户、预算、domain、quote policy 和 task seed，并形成 canonical task hash。
- `StochopiaStructurerEnv.step()` 是唯一经济状态转移入口；已有账户、合约、约束、报价与结算逻辑可复用。
- trajectory evaluator 能从冻结 manifest 重建环境，并逐 transition 精确 replay。
- 正式文档记录 72 行市场面板：CSI500/CSI1000 各 36 个月度观测，组成 12 个六行 episode；但 `data/derived/` 被忽略，本轮未把未提交的 derived bytes 当作冻结证据。

### 没有

- action-conditioned 市场转移模型；Agent 行动不影响未来 spot、IV、rate、carry。
- hypothetical clone/fork/branch API、向量化 rollout、Gym space 或训练器。
- 独立的 synthetic scenario schema、evaluator、aggregate 和污染隔离门。
- 日内/月内障碍真值、流动性、冲击、成本、完整 IV surface 和动态利率曲线。
- 足以从当前月频面板稳健估计多状态模型的样本量与 held-out 校准协议。

## 研究证据复核

- [Conditional Sig-Wasserstein GAN](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12423) 是中等维度条件时间序列生成方法，并有金融案例；这不自动意味着它在 Stochopia 的数据、状态 schema 和决策任务上“质量最好”。
- [Diffusion Factor Models](https://arxiv.org/abs/2504.06566) 面向高维多资产收益与因子结构，不直接解决当前单指数月频 ATM-IV 状态。
- [Chronos-2](https://github.com/amazon-science/chronos-forecasting) 支持概率预测与样本路径，但官方能力不包含金融无套利、衍生品状态映射或 action-conditioned simulator 验证。
- [VolGAN](https://www.tandfonline.com/doi/full/10.1080/1350486X.2025.2471317) 联合生成次日 SPX return 与日频 IV surface；其输入对象与当前 Stochopia 的月频、ATM-only 代理数据明显不同。
- [Controllable IVS VAE](https://arxiv.org/abs/2509.01743) 使用事后静态无套利修复；[Latent Flow Matching](https://arxiv.org/abs/2608.00616) 在固定 SPX 曲面网格上生成无条件 IV surface，并报告 90.8% 样本通过其静态无套利检查。后者论文仍把条件生成和时间演化预测列为未来工作，因此不能直接替代 Stochopia 的市场动态。
- [Synthetic Data for Portfolios](https://arxiv.org/abs/2501.03993) 的结论更精确地说是：生成更多 synthetic samples 不能消除初始有限样本造成的统计偏差，过量生成还可能围绕偏差估计过度集中；不能简化成“合成数据永远无用”。

## 放行的 V0

V0 只新增一个纯离线模块，不改 `benchmark.py`、`pricing.py`、`environment/core.py`、TaskSuite、CLI 和 evaluator。

### 数据与语义

1. 用两类独立随机流生成 latent regime 和 Gaussian innovation；root seed、分域 seed 和 RNG 版本全部封存。V0 的 transition matrix 明确定义为 **each-weekday**，Artifact 示例里的月频矩阵不能原样代入。
2. 在 weekday-only 日频网格上生成 physical GBM path，再派生月末 spot、20/60 日历史指标和六个月回撤；明确声明这不是中国交易所日历。
3. P 侧 `physical return/volatility` 只生成现实路径；Q 侧 `risk-free/carry/1m-3m-6m pricing IV` 只进入定价快照。改变 Q 参数不得改变 physical path。
4. latent regime 只留在 privileged synthetic artifact；公开 `MarketSnapshot.regime` 固定为 `synthetic`。
5. 输出 `decision_rounds + 1` 个快照，最后一个只承担 terminal valuation 语义。
6. 不提供“现实参数默认值”，不从当前 36 个点自动拟合，不生成客户，不接入正式 suite。

### Provenance

每条路径至少封存：schema/version、generator source hash、规范化 spec hash、参数来源、校准截止日与数据 hash（若存在）、root/domain seeds、RNG、calendar/monitoring grid、daily path hash、snapshot hash 和 root artifact hash。

当前 byte-identical 只在同一 Python/runtime 的两个干净进程中验证，尚未证明跨 Python/libm/平台逐位一致；同一 root seed 会有意复用 common random numbers，独立 episode 必须分配不同 root seed。

## 验收门槛

- **G0 重建**：相同 spec 和 seed 在两个干净进程中生成 byte-identical canonical artifact；任一参数、数据、seed 或实现变化必须改变对应 hash。
- **G1 P/Q 分离**：只改 Q 参数，physical daily path 精确不变；只改 P 参数，regime path 与同一 regime 的 Q 输入精确不变。
- **G2 不泄漏**：任何公开 snapshot 都不能包含 latent regime；synthetic source 和 schema 必须显式可见。
- **G3 隔离**：V0 不生成 Benchmark v3 suite/evaluation artifact，不进入 dev/test/private_test，不与真实面板成绩混报。
- **G4 核心不漂移**：现有 Economic Core 和 447 项基线测试必须保持通过。
- **G5 科学升级**：只有当 RS-GBM 相对单一 GBM 的 held-out predictive log-score 改善之 block-bootstrap 95% CI 下界大于 0，且波动聚集、尾部分位数、最大回撤、regime duration 至少三项进入 held-out 95% 区间，才可升级称为经过验证的市场模型。
- **G6 简单对手**：经验验证必须同时包含普通 GBM、moving-block bootstrap 和 Student-t GARCH/DCC 或 RS-GARCH；深度模型只有在冻结 private test 和至少五个训练 seed 上同时胜过这些基线，才有升级资格。
- **G7 任务有效性**：历史私测与合成任务的策略排序 Kendall τ 的 95% 下界应高于 0.5，且两两胜负方向一致率至少 80%；生成路径数不能充当独立历史样本量。

## 停止线

出现任一情况，停止扩展到 Chronos/VolGAN：

1. P/Q volatility 仍被混为同一生成参数，或模型结论依赖现有 suitability 中未拆开的 P/Q 近似。
2. latent regime 直接进入 Agent observation。
3. synthetic artifact 能混入 v3 aggregate，或替代正式历史面板。
4. 同 seed 无法重建同一路径，或 provenance 不能定位参数、实现和数据。
5. 当前月频样本被用于宣称已可靠识别 2–3 个 regime，而没有独立、哈希化的日频 calibration corpus。
6. 研究效果主要来自 horizon MTM、月末障碍近似或其他尚未验证的 core 简化。
7. 单一 GBM 的 held-out 表现不劣于 Regime-Switching GBM。
8. 任一状态样本占比低于 10%、训练期少于 10 次非重叠进入，或高低波动排序不能在至少 90% 的 block-bootstrap 拟合中稳定复现。

## 下一阶段顺序

1. 完成并验证 isolated V0 generator。
2. 另立任务设计独立 synthetic artifact/evaluator 和 train-only 门禁；不复用 v3 aggregate 格式。
3. 用独立日频 corpus 做预注册校准和 held-out 比较，并加入单一 GBM、历史 block bootstrap 基线。
4. 只有残差明确指向厚尾、波动聚集、spot-vol 联动或曲面动态，再分别评估 Chronos/SigWGAN/VolGAN/flow/diffusion；不按 Artifact 的时间表自动升级。

## 本轮落地状态

- 已新增 `stochopia/market_generator.py`：只实现 isolated V0 spec、日频 physical path、月末 `MarketSnapshot`、P/Q 参数分域、latent-state 隔离和完整 hash/provenance；没有修改现有 benchmark、pricing、environment、suite、CLI 或 aggregate。
- 已新增 `tests/test_market_generator.py`：覆盖同 seed 与跨进程 byte-identical、不同 seed 分叉、P/Q metamorphic、月末与 terminal 语义、20/60/126 日指标精确窗口、EpisodeTask schema 兼容、公开 observation 防直接泄漏、hash/provenance 自洽以及非法输入 fail-closed。
- 独立实现审查提出的 provenance 元数据可伪造、极端浮点消去和测试盲点均已修复；确定性 regime-IV emission 仍被保留为明确的科学限制，因此 V0 不适合用于 latent-state inference 结论。
- 修改前全量基线为 447 passed；最终集成后为 **488 passed in 57.75s**，新增目标测试为 **39 passed**。

这一结果只证明 V0 工程闭环和边界门禁成立，**不证明 RS-GBM 已通过经验有效性、市场真实性或训练增益验证**。
