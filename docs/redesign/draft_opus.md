# Stochopia 重设计方案（面向 ICAIF 2026，剩 16 天）

版本：opus draft v1｜作者：架构设计角色｜不含生产代码，只给到函数签名与配置字段级别。

---

## 0. 核心设计原则：结算层与对话层分离

所有后续设计围绕一条主线，这也是正面回应 orchestrator 关于“可复现性不能被破坏”的担忧：

> **确定性结算层是唯一权威，LLM 对话层只做信息博弈与定性协商，主指标在结构上不依赖任何环境 LLM 的输出。**

据此把系统切成两层：

| 层 | 组成 | 决定什么 | 可复现性来源 |
|---|---|---|---|
| 结算层（确定性） | `TradingDesk.quote`、`HardConstraintEngine`、`dealer_margin`、`ClientProfile.would_buy`、oracle | 报价数字、硬约束 PASS/FAIL、成交与否、margin、主指标 | 纯函数，无 LLM，固定 `MC_SEED` |
| 对话层（LLM） | 客户/风控/交易台三个 `EnvironmentAgent` | 自然语言披露/隐瞒、定性建议、语气、协商 | temperature=0＋固定 seed＋响应缓存＋transcript 落盘 |

关键推论：**即使全部环境 LLM 宕机，`request_quote`/`submit_design`/硬检查/margin 仍能算出**，因为它们只依赖结算层。环境 LLM 失败时对话层降级为固定兜底串，不影响任何主指标。这把“角色 LLM 化”与“可复现性”彻底解耦——可以增量加 LLM 角色而不动摇结算正确性。

---

## 1. 全角色 LLM 化架构

### 1.1 角色边界（谁决定什么）

三个环境角色都变成有独立 system prompt、可独立配模型的 LLM，但每个角色的“硬部分”留在确定性引擎：

**客户 Client**
- LLM 决定：在 partial 条件下如何用自然语言回答需求问题（可含糊、可分步披露、可表达偏好/情绪），软性兴趣表达。
- 确定性保留：真实阈值 `ClientProfile`（capital / max_loss_pct / max_maturity_months / allowed_product_types / min_hit_prob）；最终成交判断 `would_buy`（见 1.5）。LLM 不能推翻 would_buy，也不持有比 ClientProfile 更多的“真值”。
- 衔接点：`query_client(topic)` 在 partial 下从“字典查表返回精确值”改为“把 topic 转成自然语言问题发给 client LLM，返回自然语言回复”；full 下仍走确定性精确披露（`_full_client_payload`）。真值始终由 `ClientProfile` 持有，仅用于结算，不因 LLM 措辞而改变。

**交易台 Trading Desk**
- LLM 决定：定性流动性/买卖价差方向、对冲难度描述、结构调整建议。
- 确定性保留：`fair_value`、greeks、`client_price`、`dealer_margin`、stress_loss 全部由 `TradingDesk.quote` 计算。
- 衔接点：复用旧 `engine.py` 的 `_message_has_valid_draft` + `_build_deterministic_block` 模式——`consult` 消息里**带合法产品草案 JSON 时跳过 LLM，直接回确定性报价块**（省 token、防数字捏造）；纯文字咨询才走 LLM 定性回复。

**风控 Risk Control**
- LLM 决定：定性合规提示、临近边界预警（“这个名义已接近上限”）。
- 确定性保留：`HardConstraintEngine.evaluate` 的 13 条 PASS/FAIL 是权威，LLM judge 与风控 LLM 都无权豁免（`docs/BENCHMARK_PROTOCOL.md:63-66`）。
- 衔接点：带草案 → 附确定性硬检查预检块；纯文字 → LLM 定性回复。

### 1.2 交互协议（轮次结构与消息格式）

沿用 `benchmark_runner.py` 的“每轮单 JSON 动作”协议，动作集扩展一个 `consult`：

```
{"action":"query_client","topic":"..."}                      # partial 下路由到 client LLM
{"action":"consult","role":"trading_desk|risk_control|client","message":"...","draft":{...可选ProductSpec...}}
{"action":"request_quote","product":{...ProductSpec...}}     # 确定性结算报价，不变
{"action":"submit_design","quote_id":"Q-...","explanation":"..."}
{"action":"submit_product","product":{...},"explanation":"..."}
{"action":"skip_round"}
```

单轮流程（`run_episode` 主循环，改造后）：
1. 环境构造 `get_round_brief()`（确定性，不变）。
2. 被测 agent 循环最多 `max_actions_per_round` 步，发上述动作之一。
   - `consult`：调 1.3 的 `LongHorizonEnvironment.consult(role, message, draft)`；带 draft 且 role∈{desk,risk} 时附确定性块。
   - `request_quote`：确定性报价（含新 `dealer_margin`），入 `archive`。
   - `submit_*`：进入结算。
3. 结算：`adjudication` 模式决定 accept（见 1.5）。捕获 judge 数据（见 §6）。
4. `advance_round`：动态条件下 portfolio 演进（不变）。

`consult` 预算：新增 `max_consults_per_round`（默认 3），与 `query_client` 预算分开计。`request_quote` 预算不变（3 次）。

### 1.3 与现有数据结构的衔接（具体签名）

`LongHorizonEnvironment` 新增字段与方法：

```python
class LongHorizonEnvironment:
    def __init__(
        self, snapshots, client, risk_budget, condition, *,
        max_quotes_per_round=3, max_client_queries_per_round=3,
        max_consults_per_round=3,
        env_agents: dict[str, "FrozenEnvAgent"] | None = None,   # None → 对话层禁用，纯确定性回退
        adjudication: str = "hard",                              # "hard" | "hard_and_client"
    ): ...

    async def consult(self, role: str, message: str, draft: dict | None = None) -> dict:
        """LLM 定性回复；draft 合法且 role∈{trading_desk,risk_control} 时附确定性块。
        env_agents 为 None 或该角色缺失/LLM 失败时，返回 {"role":role,"reply":<兜底串>,"degraded":True}。
        绝不抛异常打断回合。"""
```

`consult` 的确定性块构造直接复用现有函数，无需新定价逻辑：
- desk 块 = `self.desk.quote(parse_product_draft(draft), snapshot, client, portfolio, state_version, idx).public_payload()`
- risk 块 = 同一 quote 的 `checks` 子集（只回 PASS/FAIL + observed/limit，不回价格）。

注意：`consult` 里的 desk quote **不占用** `request_quote` 预算、**不进 archive**、**不可用于 submit**（`state_version` 绑定仍要求正式 `request_quote`）。这样对话层拿到的报价是“咨询性”的，正式提交必须走结算层，避免对话层污染可提交报价集合。

### 1.4 可复现性保障（seed / 温度 / transcript / 缓存）

1. **温度与 seed**：所有环境 LLM 调用固定 `temperature=0.0`；seed 用 `seed_base ^ hash(episode_id, round_num, role, turn_index)` 派生，落到 `client.chat(..., seed=...)`（`llm.py:144` 已透传）。
2. **响应缓存（关键）**：`docs/BENCHMARK_PROTOCOL.md:57-58` 要求“NPC 正式回复生成一次、冻结、跨被测模型共享”。完全预生成对反应式回复不可行（回复依赖 agent 的话），故设计**按对话前缀哈希缓存**：

```python
class EnvResponseCache:
    # key = sha256(agent_id | model | temperature | seed | canonical_json(messages))
    def get(self, key: str) -> str | None: ...
    def put(self, key: str, value: str) -> None: ...
    # 持久化到 benchmark.yaml 的 env_cache_dir（jsonl，append-only，跨 run 复用）
```

   效果：相同对话前缀（例如同一份草案 consult）在不同被测模型间命中同一 NPC 回复，实现协议要求的“共享冻结”；replay 完全确定。缓存命中率与去重后的真实 LLM 调用数写入 usage，供成本核算。
3. **transcript 落盘**：`EpisodeTrace` 已存完整 `messages`/`actions`。新增每个 `consult` 的 `ActionTrace.response` 存 `{reply, deterministic_block, cache_hit, degraded}`；env agent 的完整 history 另存到 `outputs/.../env_transcripts/<job_id>.jsonl`，满足“NPC 回复可审计”。

### 1.5 成交裁决模式与 would_buy 复活

当前 benchmark 的 accept 只看 `quote.hard_pass`（`benchmark.py:613`），`would_buy` 的经济决策（min_hit_prob / hurdle_hit_prob / loss_frac）根本没接入正式链路。两档裁决：

- `adjudication="hard"`（默认、最低风险、兼容现状）：accept ⇔ `hard_pass`。
- `adjudication="hard_and_client"`（目标形态，配合 margin 重设计）：accept ⇔ `hard_pass` 且 `would_buy` 返回 True。`would_buy` 需要 `pricing["hurdle_hit_prob"]` 与 `loss_frac`，在 `TradingDesk.quote` 里补算（`hurdle_hit_prob` 用 `pricing.hurdle_hit_prob(product, market, client.min_return_pct, client_price)`，已存在）。客户 LLM 只管信息披露，would_buy 是确定性经济闸门。这让“客户适配度”真正进入成交判断，与 §3 的 margin 重设计形成合力（乱塞产品既压不出 margin，也过不了 would_buy）。

### 1.6 LLM 角色故障处理：降级不判失败

- 结算层无 LLM 依赖 → 主指标（feasibility、margin、oracle attainment）永远算得出。
- `consult`/`query_client(LLM)` 失败（API 错、解析错、超时耗尽）→ 返回固定兜底串 + `degraded=True`，记 `env_agent_error` 事件，**回合继续**。consult 是咨询性的，降级安全。
- 被测 agent 自身 LLM 失败 → 沿用 `llm.py` 现有重试；耗尽则该步记 `protocol_error`（已处理）。
- 落库统计 `degraded_consult_rate`，若某 run 降级率过高在 aggregate 打警示标记，供人工复核，但不作废该 run。

---

## 2. 角色配置 schema

复用 `scenario.py:36` 的 `AgentSpec`（已有 id/display_name/public_desc/prompt/model），扩两个字段：

```python
@dataclass
class AgentSpec:
    id: str
    display_name: str
    public_desc: str
    prompt: str
    model: str | None = None
    temperature: float = 0.0          # 新增
    role_kind: str = "npc"            # 新增："client" | "trading_desk" | "risk_control"
```

`scenarios/stochopia_csi/benchmark.yaml` 新增 `env_agents` 节（角色 prompt 统一放在 `scenarios/stochopia_csi/prompts/`，其中 risk_control.md / trading_desk.md 可直接复用其角色语义）：

```yaml
env_agents:
  enabled: true                 # false → 纯确定性回退（MUST 档可先 false）
  cache_dir: outputs/env_cache
  seed_base: 20260802
  roles:
    client:
      prompt_file: agents/client.md
      model: env_model          # 名字解析：优先 registry 显式模型名，否则用 defaults[env_model]
      temperature: 0.0
    trading_desk:
      prompt_file: agents/trading_desk.md
      model: env_model
      temperature: 0.0
    risk_control:
      prompt_file: agents/risk_control.md
      model: env_model
      temperature: 0.0
adjudication: hard_and_client   # 或 hard
```

`config/models.yaml` 的 `defaults` 增补用途键（`llm.py:42 BUILTIN_DEFAULTS` 已有 `env_model`/`judge_model`，只需补 client/desk/risk 或直接都用 `env_model`）：

```yaml
defaults:
  test_model: deepseek-v4-flash
  env_model: deepseek-v4-flash     # 三个环境角色共用便宜快的
  judge_model: deepseek-v4-pro     # judge 用更强的，避免与被测同模型
  judge_model_b: qwen-max          # 第二 judge，做 inter-judge 可靠性
```

加载：`load_scenario` 已能读 agent prompt_file（`scenario.py:456-468`）；给 benchmark.yaml 写一个轻量 `load_benchmark_env_agents(benchmark_dir, registry) -> dict[str, FrozenEnvAgent]`，内部 `create_client(resolve_model_name, registry)`。`FrozenEnvAgent` 包 `EnvironmentAgent`（`agents.py:104`，已有 history + inject_info + respond）再加缓存与 seed 派生。

---

## 3. dealer_margin 重定义

### 3.1 病因（精确）

`dealer_margin = client_price − hedging_cost`（`benchmark.py:480`），而 `client_price = MARKUP·fv = 1.03fv`（`pricing.py:739`），`hedging_cost = HEDGE_COST_RATIO·fv = 1.02fv`（`pricing.py:716`）。故 `dealer_margin ≡ 0.01·fair_value`，与结构风险、客户适配**零相关**。最大化 margin ⇔ 最大化 fv ⇔ 在预算内堆最大 fv/notional 的 vanilla。虽然 vega/delta 预算会兜住无限放大，但目标函数仍与产品质量无关。

### 3.2 新公式：成本加成 + 客户适配封顶

思路：交易台报价 = 公允价值 + 风险加载的加成；加成里**对冲成本随结构风险上升**，同时**加成率被客户能承受/愿付的上限封顶**，使乱塞产品无利可图。所有输入量都能从现有 `pricing.evaluate_product` 拿到。

```python
@dataclass(frozen=True)
class MarginConfig:
    kappa_vega: float      # 每单位 |vega_pct| 的对冲价差成本系数
    kappa_delta: float     # 每单位 |delta| 的再对冲换手成本系数
    kappa_fund: float      # 每单位压力损失占比的资金占用系数
    base_hedge: float      # 残余对冲成本（fv 占比），取代旧 0.02 常数
    markup_base: float     # 基础加成率
    markup_risk: float     # 风险溢价系数（vega + 尾概率）
    complexity: dict[str, float]  # {vanilla:1.0, barrier:1.4, autocallable:1.7, snowball:2.0}

def dealer_margin(product, pricing, market, client, *, cfg: MarginConfig) -> dict:
    """返回 {fair_value, client_price, hedging_cost, dealer_margin, breakdown}。"""
```

具体量（均来自 `pricing` 字典 / greeks / pricing_details）：

```
fv   = pricing["fair_value"]
gv   = abs(greeks["vega_pct"])                       # 每 1 vol 点、名义占比
gd   = abs(greeks["delta"])                          # 名义占比
sl   = stress_loss / notional                        # 由 _stress_loss 得，压力损失占比
kip  = pricing_details.get("knock_in_prob", 0.0)     # MC 产品尾概率
life = pricing_details.get("expected_life_months", maturity_months) / 12
cx   = cfg.complexity[family(product)]

# 对冲成本（名义占比），随 vega / delta 换手 / 资金占用 / 结构复杂度上升
hedge_frac    = (cfg.kappa_vega*gv*sqrt(life) + cfg.kappa_delta*gd*life + cfg.kappa_fund*sl*life) * cx
hedging_cost  = fv*cfg.base_hedge + notional*hedge_frac

# 加成率：风险溢价，但被客户适配度封顶
markup_rate   = (cfg.markup_base + cfg.markup_risk*(gv + kip)) * suitability(product, client)
client_price  = fv + notional*markup_rate

dealer_margin = client_price - hedging_cost
```

`suitability(product, client) ∈ [0,1]`：白名单命中、保本匹配、期限贴合、`hurdle_hit_prob` 相对 `min_hit_prob` 的余量，四项乘积或加权。不匹配 → 加成率被压到接近 0 → 乱塞产品 margin 为负或极小。

为什么哑策略失效：
- 堆最大 vanilla：`suitability` 低（对高定价敏感的 client_b 不买单，would_buy 也拦），`markup_rate` 被压；同时大名义撞 vega/delta 预算。margin/notional 变小。
- 复杂但贴合的 autocallable：`cx` 高抬对冲成本，但 `markup_rate` 因适配高、单位名义 vega 低而更优，且能在预算内做大名义 → margin 更高但需要真设计。
- margin 与 notional 不再单调线性，破坏“最大可行 vanilla 拿满分”。

### 3.3 常数标定

`MarginConfig` 的 7 个常数在**开发集**（warm-up 2022 + 若干 dev episode）上标定并冻结，标定目标可复用 `calibrate_risk_budget`（`benchmark.py:775`）的范式：选一组常数使 (a) oracle 最优产品的 margin/notional 落在合理区间（如年化 0.5%–3%），(b) 哑策略（max vanilla）margin 显著低于 oracle。写 `calibrate_margin_config(dev_cases) -> MarginConfig` 报告并冻结。**这是 protocol change，评测前必须冻结。**

---

## 4. oracle 修复

### 4.1 病因

oracle 网格名义上限 `ORACLE_NOTIONAL_FRACTIONS=(...,0.10)`（`benchmark.py:701`）= 资金 10%；但 agent 的 `CLIENT_CAPITAL` 检查允许 notional ≤ `client.capital`（100%，`benchmark.py:380`），且组合预算另有上限。agent 可行域远大于 oracle 网格 → oracle 不是上界，`oracle_margin_attainment` 可达约 10。

### 4.2 修复：把网格上界对齐到 agent 的真实可行域

对每个 (variant, maturity)，用二分法求在当前 portfolio 下能通过全部硬检查的**最大可行名义**，在该处（及少数内点）评估 margin：

```python
def _max_feasible_notional(template, snapshot, client, portfolio, budget, desk,
                           lo=0.0, hi=None) -> float:
    """二分 notional∈[0, client.capital]，返回仍 hard_pass 的最大名义。
    hi 默认 client.capital；对每个 budget 维（capital/vega/delta/stress/组合名义）
    的绑定点取 min 亦可解析给初值。"""

def oracle_candidate_grid(client, snapshot, portfolio, budget) -> list[ProductSpec]:
    # variants × maturities，每个在 max_feasible_notional 及其 {0.5, 0.75, 1.0} 处取样
```

因为新 margin 对 notional 非单调（vega 预算可能先于 capital 绑定），必须在**各约束绑定点**取样而非只取 capital 上限。`oracle_best_quote` 逻辑不变（`benchmark.py:757`），只是候选来自对齐后的网格。

### 4.3 兜底（若时间不够做二分）

退一步：把 `ORACLE_NOTIONAL_FRACTIONS` 扩到含 `1.0` 并加密（0.10,0.25,0.50,0.75,1.0），metric 改名 `reference_margin_ratio`，文中表述为“相对强参考策略的达成率”而非“相对上界”，并 clip 报告 `min(ratio, 收敛值)`。这是 COULD 档退路，MUST 档做 4.2。

---

## 5. 强制提交口径

### 5.1 病因

`benchmark_runner.py:263-270`：agent 未提交时环境自动提交 archive 中 margin 最大的可行报价，并写入 `trace.dealer_margin`；`compute_metrics`（`benchmark_runner.py:307`）把强制 margin 也计入 `total_dealer_margin`，污染主指标。

### 5.2 修复：默认关闭强制提交，主指标只算显式 submit

1. `run_episode` 增参 `force_best_on_timeout: bool = False`。**默认关闭**：agent 不提交则该轮 `submitted=False`、margin=0、动态下 portfolio 不新增。这让“未行动”成为真实的空结果。
2. 强制提交降级为**可选方法臂**（原本就是“method intervention”），仅在显式开启时运行，且始终打 `forced_submission=True`。
3. `compute_metrics` 拆口径：
   ```python
   explicit = [r for r in rounds if r.submitted and not r.forced_submission]
   result["total_dealer_margin"] = sum(r.dealer_margin for r in explicit if r.accepted)   # 主指标：仅显式
   result["forced_dealer_margin"] = sum(r.dealer_margin for r in rounds if r.forced_submission and r.accepted)  # 单列
   result["explicit_submission_rate"] = len([r for r in rounds if r.submitted and not r.forced_submission]) / n
   result["no_submission_rate"] = len([r for r in rounds if not r.submitted]) / n
   ```
4. `oracle_margin_attainment` 只在 `explicit` 且 accepted 上计算。

---

## 6. judge 接入

### 6.1 病因

`judge.py` 的 `judge_soft_quality`（`judge.py:94`）从未被 `run_episode`/`aggregate` 调用，但 `docs/BENCHMARK_PROTOCOL.md:78` 承诺“两 judge 模型 × 三次重复 × 60 样本”。

### 6.2 接入点：捕获 → 离线批判 → 聚合

不在 `run_episode` 内联跑 judge（成本与关注点分离）。三步：

1. **捕获**（`run_episode` 内，显式 submit 成功时）：`RoundTrace` 新增
   ```python
   submitted_product: dict | None = None       # public 化的 ProductSpec
   submitted_explanation: str = ""
   client_brief_snapshot: dict | None = None    # get_round_brief() 当轮快照
   ```
2. **离线批判**（新 CLI 子命令 `judge-runs`）：遍历 `outputs/.../*.json`，对每条显式 submission，用 2 个 judge 模型各 3 次重复调 `judge_soft_quality(client_brief=..., product=..., explanation=...)`，把 `JudgeResult.to_dict()` 写回结果 JSON 的 `judge` 块。judge 调用 temperature=0（`judge.py:112` 已固定）。
   ```python
   # benchmark_cli.py 新增
   jr = sub.add_parser("judge-runs")
   jr.add_argument("results_dir", type=Path)
   jr.add_argument("--judge-models", nargs="+", required=True)   # 两个，均≠被测模型
   jr.add_argument("--repeats", type=int, default=3)
   jr.add_argument("--sample", type=int, default=60)             # 分层抽样样本量
   ```
3. **聚合**（`_cmd_aggregate`）：加 judge 六维度均值列，并用 `judge.py` 现成的 `reliability_summary(left, right)`（两 judge 模型对齐）算 weighted Cohen's kappa / Spearman / exact agreement；如跑扰动臂再算 `score_flip_rate`。写入 aggregate markdown 一节。

### 6.3 judge 模型选择与自评偏差控制

- **禁止自评**：judge 模型必须≠被测模型；若某被测模型也在 judge 列表，跳过该配对的其自评样本。
- 两个异构 judge（如 `deepseek-v4-pro` + `qwen-max`）做 inter-judge 可靠性；不宣称专家效度，只报可靠性（`reliability_summary` 的 `claim` 字段已声明）。
- **样本策略**：60 个分层平衡样本（按 4 condition × 模型均衡抽），而非全量 submission，控成本。抽样用固定 seed 冻结样本清单。
- 扰动敏感性（可选 COULD）：对同一批样本做产品字段顺序/客户名替换/反事实扰动，重跑一遍 judge，用 `score_flip_rate` 报稳健性。

---

## 7. CLIENT_MAX_LOSS 修复

### 7.1 病因

`worst_case_payoff_ratio`（`benchmark.py:329-339`）对非保本一律返回 floor=0 → `max_loss_ratio=1.0`（100%），`CLIENT_MAX_LOSS`（`benchmark.py:387`）把任何非保本产品都按 100% 损失判，退化成布尔。

### 7.2 修复：用 pricing 已有的损失量替代 0/1

`evaluate_product`（`pricing.py:725`）已返回 `loss_frac`，语义天然正确：
- vanilla/barrier 非保本：`loss_frac = client_price/notional`（`pricing.py:750`）＝**权利金即最大终端损失**，经济上精确。
- 保本：`loss_frac = 0`（或按敲入 expected_loss）。

对 MC 产品（snowball/autocallable 非保本），`loss_frac` 是**期望**损失（`expected_loss_frac`），会低估最坏情形。客户 `max_loss_pct` 是最坏情形容忍度，故对 MC 产品应取尾部损失：

```python
# _mc_stats 一行增补：已逐路径算了 loss_fracs，只需多导出两个分位
"p95_loss_frac": sorted(loss_fracs)[int(0.95*n)],
"max_loss_frac": max(loss_fracs),

def client_max_loss_frac(product, pricing) -> float:
    if product.principal_protected:
        return 0.0
    if family in {vanilla, barrier}:
        return pricing["loss_frac"]                                  # = 权利金占比
    return pricing["pricing_details"]["p95_loss_frac"]              # MC 尾损失
```

`HardConstraintEngine.evaluate` 的 CLIENT_MAX_LOSS 改为：
```python
loss = client_max_loss_frac(product, pricing)
_check("CLIENT_MAX_LOSS", loss <= client.max_loss_pct + 1e-9, loss, client.max_loss_pct, ...)
```

保本相关检查（CLIENT_PROTECTION / PROTECTION_CLAIM / DESCRIPTION_PROTECTION）保持不变，它们校验的是“声明保本 vs 结构 floor”的诚实性，与 max_loss 数值检查正交。

**MUST 档低成本退路**：若不想动 `_mc_stats`，直接用现成的 `stress_loss / notional`（`_stress_loss` 已算好，`benchmark.py:409`，−20%/+20%/+10vol 压力网格的重定价损失）作为最坏损失代理，一行接入，零新增计算。推荐 `loss_frac`（终端语义更干净），`stress_loss` 为兜底。

---

## 8. 统计方案

### 8.1 病因

`experiment.py:47` 只有 `partial_dynamic` 有 3 seed，其余 n=1；`_cmd_aggregate`（`benchmark_cli.py:238-243`）只算均值，无 CI、无检验。

### 8.2 复制维度：以 episode 为主，而非 seed

temperature=0 使 seed 近乎冗余。把 **12 个 episode 当作主复制维度**（每 condition×model×strategy 天然 n=12，覆盖 CSI500/1000 × 6 个半年市场 regime），远比刷 seed 便宜且科学价值更高。seed 只对最难格子 `partial_dynamic` 保留 3 个估残余随机性，其余 seed=1。

### 8.3 CI 与配对检验（具体做法）

- **CI**：对每个 (model, condition)，在 12 个 episode 值上做 percentile bootstrap（10k 重采样）报 mean ± 95% CI。新增 `bootstrap_ci(values, n=10000, alpha=0.05) -> (lo, hi)`，接入 aggregate markdown。
- **condition 配对对比**：`experiment.py` 已有 `paired_condition_contrasts`（`experiment.py:69`）在 (episode,model,strategy) 内算匹配落差。加显著性：对 12 个 episode 级差值做 **Wilcoxon 符号秩检验** + 配对 bootstrap CI；跨对比族用 **Holm 校正**。落差方向即“full→partial 的可观测性退化”“static→dynamic 的长程退化”，是论文主结论。
- **模型对比**：同 episode 集配对（same episodes），Wilcoxon 符号秩 + 效应量（Cliff's delta 或 rank-biserial）。
- **judge 可靠性**：§6 的 weighted kappa / Spearman，样本 n=60。

需引入 `scipy.stats`（Wilcoxon）或自写符号秩（judge.py 已有零依赖 Spearman/kappa 的先例，可同法自写，避免加依赖）。推荐自写 `wilcoxon_signed_rank(diffs) -> (stat, p)` 保持零依赖。

### 8.4 成本核算

主表：12 episode × 4 condition × M 模型 × 1 strategy（选 `ledger_archive`）× 1 seed = 48M episode 运行；+ `partial_dynamic` 额外 2 seed = 12×M×2。M=4 → 192 + 96 ≈ 288 episode 运行，每 episode ~6 轮 ×~6 动作 ≈ 万级 LLM 调用量级，加 env 缓存去重后可控。strategy 消融（3 个 strategy）只在 4 个 episode 上跑，避免三倍全量。judge：60 样本 × 2 judge × 3 repeat = 360 次 judge 调用，成本极小。

---

## 9. 迁移计划（逐文件 + 依赖顺序 + 工作量 + must/should/could）

### 9.1 逐文件改动清单与依赖顺序

| 顺序 | 文件 | 改动 | 依赖 | 工作量 |
|---|---|---|---|---|
| 1 | `pricing.py` | 新增 `dealer_margin` + `MarginConfig` + `calibrate_margin_config`；`_mc_stats` 导出 `p95/max_loss_frac`；新增 `client_max_loss_frac` | 无 | 1.5d |
| 2 | `benchmark.py` | `TradingDesk.quote` 用新 `dealer_margin`；`HardConstraintEngine` CLIENT_MAX_LOSS 用 `client_max_loss_frac`；quote 补 `hurdle_hit_prob`；oracle 网格 `_max_feasible_notional` 对齐 | 1 | 2d |
| 3 | `benchmark_runner.py` | 强制提交默认关闭+单列；`compute_metrics` 拆 explicit/forced；捕获 judge 数据字段 | 2 | 1d |
| 4 | `experiment.py` + aggregate | `bootstrap_ci`、自写 `wilcoxon_signed_rank`、Holm；paired_contrasts 加 p 值；aggregate 输出 CI/检验 | 3 | 1.5d |
| 5 | `agents.py`/新 `env_agents.py` | `FrozenEnvAgent` + `EnvResponseCache`（包 `EnvironmentAgent`） | 无（并行） | 1.5d |
| 6 | `scenario.py` + `benchmark.yaml` + `agents/*.md` | `AgentSpec` 加字段；`load_benchmark_env_agents`；迁移旧 client/desk/risk prompt | 5 | 1d |
| 7 | `benchmark.py`+`benchmark_runner.py` | `LongHorizonEnvironment.consult`；`run_episode` 接 `consult` 动作 + `adjudication="hard_and_client"` | 2,3,5,6 | 2d |
| 8 | `judge.py` 接入 + CLI `judge-runs` | 捕获→离线批判→聚合 | 3 | 1d |
| 9 | 数据管线子包 `stochopia/data/` | 合并 `market_builder`+`formal_market_builder` 为一条；`tushare_*/cffex_data/formal_*/raw_data_audit/market_data_math` 归入子包；CLI 接 `build-market`/`audit`/`fetch` | 无（并行，隔离） | 2d |
| 10 | 测试 + 文档 | 更新 `tests/`；`BENCHMARK_PROTOCOL.md` 对齐新指标口径 | 全部 | 1.5d |

### 9.2 三档裁剪线（16 天，硬约束 8 月 2 日）

**MUST（信誉底线，缺则论文站不住；纯确定性、低风险、可单测）** — 约 6 天
- #3 dealer_margin 重定义（含标定）— 破解哑策略，最核心。
- #4 oracle 二分对齐（或至少 4.3 兜底改名）。
- #5 强制提交默认关闭 + 主指标只算 explicit。
- #7 CLIENT_MAX_LOSS 用 loss_frac/stress_loss。
- #8 CI + Wilcoxon 配对检验（至少 bootstrap CI）。
- 对应文件顺序 1→2→3→4。这一档做完，即使不上 LLM 角色，也是一个指标自洽、可发表的确定性 benchmark。

**SHOULD（用户选定的头条贡献：角色 LLM 化 + judge）** — 约 5 天
- #1/#2 三角色 LLM 化（顺序 5→6→7），`adjudication="hard_and_client"`。
- #6 judge 接入（顺序 8）。
- 关键：因结算层不依赖 env-LLM，这一档可**增量叠加**在 MUST 之上而不动摇已冻结的确定性指标；若中途出问题，`env_agents.enabled=false` 一键回退到纯确定性 MUST 版本，实验不作废。

**COULD（有余力再做）** — 约 3 天
- #9 数据管线合并归子包（与主链路解耦，可最后做或推迟到 camera-ready）。
- strategy 全矩阵消融、seed 复制扩展、judge 扰动敏感性、oracle 4.2 精确二分（若 MUST 用了 4.3 兜底）。

**若时间严重不足的取舍顺序**：先砍 COULD 全部 → 再把 SHOULD 的角色 LLM 化降到只保留 `trading_desk` 定性层（客户/风控仍走确定性）→ judge 降到单 judge 模型无重复。**绝不砍 MUST**：指标正确性优先于角色 LLM 化的完整度，因为审稿人先看指标能否被哑策略攻破，再看架构新颖性。

---

## 附：一句话风险提示

新 `dealer_margin` 与 `client_max_loss_frac` 的常数一旦用于评测就必须冻结并在论文附录公示；oracle 对齐后要重跑 `calibrate_risk_budget` 确认可行率仍在 20–40% 目标带内（`benchmark.yaml` 已声明该目标），否则难度漂移会让跨模型对比失真。
