# MIRAGE v2 重设计定案（综合稿）

日期：2026-07-17｜截稿：ICAIF 2026，8月2日（剩16天）
来源：Opus设计稿（draft_opus.md）与Codex设计稿（draft_codex.md）独立并行产出，本文是综合裁决。实现细节（函数签名、配置字段、公式推导）以两份原稿为准，本文只记录采纳了谁、为什么、以及分歧如何裁决。

---

## 一、两稿共识（直接定案，不再讨论）

两个模型在互不知情的情况下收敛到同一主线，以下九点直接采纳：

1. **结算层与对话层分离**。确定性引擎（定价、硬约束、结算、主指标）是唯一权威；LLM角色只做信息博弈与流程决策。环境LLM全部宕机时主指标仍可计算。
2. **四角色独立化**。structurer（被测）、client、risk_control、trading_desk各有独立system prompt文件、独立模型配置、独立对话history，温度0，seed按（episode，round，role，turn）派生。
3. **dealer_margin重定义**。废除恒定1%公式，对冲成本与加成率随结构风险变化，常数在开发集上一次性标定后冻结。
4. **oracle可行域对齐**。oracle候选与agent共用同一可行域与校验函数，补“attainment≤1”性质测试。
5. **删除环境替提交**。auto-best不再进入任何主指标，提交来源分列（voluntary／forced_prompt／none／imputed诊断）。
6. **judge离线接入**。episode trace封存后批跑，两个异构judge模型×3次重复×60个分层平衡样本，禁自评，只声称评审者间可靠性不声称专家效度。
7. **CLIENT_MAX_LOSS连续化**。`max(loss_frac, stress_loss/notional) ≤ client.max_loss_pct`，保本合同检查独立保留。
8. **统计**。以episode为聚类单位（12个episode是主复制维度），配对bootstrap 95% CI＋配对检验＋Holm校正，主格子n≥3。
9. **response缓存与transcript落盘**。相同请求哈希命中同一NPC回复，全部角色调用可审计。

## 二、分歧裁决

| 分歧点 | Opus方案 | Codex方案 | 裁决 |
|---|---|---|---|
| LLM角色的权力 | 纯咨询（consult），成交由确定性`hard_pass ∧ would_buy`裁决 | 双层结果：LLM角色有真实决策（desk发行/拒绝、risk批准/上报、client接受/拒绝），单列为`workflow_deal`次级指标，主榜仍确定性 | **采Codex**。双层结果让“全角色LLM化”名副其实（角色决策真实存在且被度量），又不污染主指标；Opus的consult机制并入交互协议作为对话动作 |
| 成交主口径 | `hard_pass ∧ would_buy`（adjudication可切换） | `hard_executable ∧ client_contract_pass`（确定性合同门） | **两者实质相同**（client_contract_pass即would_buy逻辑搬进合同门）。采Codex命名与纯函数`settle_submission`设计 |
| margin公式 | 加成率被suitability（客户适配度∈[0,1]）封顶 | r_c/r_h线性组合＋容量冲击Q²超线性项，MC不确定性带做spread proxy | **合并**：以Codex的r_c/r_h结构为骨架（含Q²容量项），r_c再乘Opus的suitability系数。适配度同时走两条路（价格端压加成＋合同门拦截），哑策略两头堵。标定后做0.8/1.0/1.2敏感性 |
| oracle表述 | 二分求最大可行名义，在约束绑定点取样 | 有限action lattice双方共享，myopic oracle改名one_step_attainment，hindsight DP为Should | **采Codex**（lattice共享从根上保证域相等，改名诚实），Opus的绑定点取样并入lattice的notional档位设计 |
| 防幻觉机制 | 带草案跳过LLM直接回确定性块 | grounding校验：narrative里的数字必须来自supplied facts，engine-signed facts | **两者都要**：跳过机制省成本，grounding校验兜底 |
| 工程节奏 | must（确定性修复，约6天）→should（LLM化+judge，约5天）→could；`env_agents.enabled=false`一键回退 | 7/24协议冻结、7/29结果冻结、Go/No-Go判据 | **合并**：Opus的增量叠加＋回退开关为实施策略，Codex的冻结日历为纪律约束 |

## 三、定案架构

```
StructurerRole（被测LLM）
   │  query_client / consult / request_quote / submit_design / skip_round
   ▼
RoleOrchestrator（只允许中心路由，无自由群聊）
   ├─ ClientRole LLM：披露/追问/accept-reject（数值由hidden ClientState经adapter填入）
   ├─ RiskRole  LLM：approve/revise/escalate（只能引用已有check_id，不可豁免PASS/FAIL）
   └─ DeskRole  LLM：issue/decline/revise（报价数字来自确定性quote，带草案跳LLM）
   ▼ 只读事实调用
DeterministicCore：Pricing → HardConstraintEngine → client_contract_pass → settle_submission（纯函数）
   ▼
双层结果：
  主榜   TechnicalSettlement：hard_executable、technical_margin、attainment（不依赖任何环境LLM）
  次级   WorkflowOutcome：workflow_deal、llm_action_vs_formal_truth分歧率
离线：judge批跑（不反馈给agent）
```

回退开关：`env_agents.enabled=false`时ClientRole退回确定性披露、Risk/Desk退回纯确定性块，主榜指标定义不变，实验不作废。

## 四、指标定案

- `dealer_margin = N·(r_c·suitability − r_h)`；r_c、r_h见draft_codex.md §3公式，suitability见draft_opus.md §3.2；系数在开发episode上按预注册目标（20–60%可行报价为正margin、结构排序非退化）标定一次后冻结。
- `one_step_attainment`（原oracle_margin_attainment改名）：只在voluntary提交且accepted上计算，性质测试断言≤1+1e-9。
- `CLIENT_LOSS_BUDGET_V2`：max(loss_frac, stress_loss/N)，文档称continuous loss proxy，不称数学worst-case。
- 主指标只算voluntary；forced_prompt（额度用尽后模型本人最后一次submit/skip）单列；environment_imputed仅作反事实诊断，不改portfolio。
- judge六维分开报告，不与经济指标合成总分。

## 五、时间表（16天）

| 日期 | 内容 | 对应 |
|---|---|---|
| 7/17–7/20 | MUST确定性修复：margin重定义＋标定、lattice共享oracle、强制提交口径、CLIENT_LOSS_BUDGET_V2、CI/配对检验；同时并行搭role_config＋RoleRuntime骨架 | 两稿迁移表顺序1–4 |
| 7/21–7/23 | 角色LLM化接线：orchestrator、四角色schema与grounding、consult动作、双层结算；judge离线CLI | 顺序5–8 |
| **7/24** | **协议冻结**：margin系数、lattice、prompt、stress网格、primary outcomes全部锁死，此后只修bug不调参 | Codex纪律 |
| 7/25–7/26 | development pilot（4个开发episode×2 repeats）：查故障率、成本、方差，不看效果 | |
| 7/27–7/29 | held-out正式跑（12 episodes，主格n≥3，partial_dynamic n=5），**7/29结果冻结** | |
| 7/30–7/31 | 聚合、CI、检验、图表、论文写作；BENCHMARK_PROTOCOL.md对齐实际口径 | |
| 8/1 | 只做复现检查与文字修正 | |

裁剪原则（时间不够时按序砍）：could全砍→角色LLM化降为仅desk定性层→judge降为单模型无重复。**绝不砍确定性指标修复**：审稿人先看指标能否被哑策略攻破，再看架构新颖度。

数据管线合并（market_builder与formal_market_builder二选一、tushare/cffex归子包）为COULD，可推迟到camera-ready。

## 六、风险与Go/No-Go

- 常数一旦用于评测必须冻结并在附录公示；oracle对齐后重跑`calibrate_risk_budget`确认可行率仍在20–40%目标带。
- fresh API调用不承诺位级复现：论文只主张economic_replay（冻结trace下指标完全确定）＋conversation_replay（artifact回放逐字节一致）。
- Go/No-Go（Codex判据）：若7/24前未完成margin/oracle/auto-submit/连续loss/CI五项修复，论文不得以“多智能体金融benchmark”为主贡献。
