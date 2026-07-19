# MIRAGE 全角色 LLM 化重设计方案（Codex 独立第二视角）

**日期：2026-07-17｜目标：ICAIF 2026，8 月 2 日截稿**

## 结论先行

推荐把 MIRAGE 重构为“**多 LLM 交互层 + 单一确定性真值层 + 双层结果**”。结构师、客户、风控、交易台都有独立 system prompt、模型、上下文和结构化动作；客户可表示接受/拒绝，风控可批准/退回，交易台可出具/拒绝报价。但正式价格、硬约束、技术可执行性、组合状态、结算算术、technical margin 和主指标只由确定性核心产生。LLM 角色共同形成的 `workflow_deal` 作为次级多智能体结果，不能替代主榜的 `hard_executable/technical_margin`。否则主榜测到的是“结构师 × NPC 阵容”的随机配对，而不是受测结构师能力。

主论文应表述为：LLM 能否在多角色、部分可观测、长周期工作流中合成可执行结构化产品；不要声称模拟真实交易台定价、真实流动性或真实客户成交。

## 0. 事实核验与行号校正

1. CLI 单跑链路在 `mirage/benchmark_cli.py:286-300`，manifest 链路在 `:149-186`。`run_episode` 只接收一个 `client`（`mirage/benchmark_runner.py:144-151`），唯一角色调用是 `client.chat(...)`（`:177-179`），唯一 system prompt 定义于 `:73-97`、装入消息于 `:155`。客户 query 是字典映射（`mirage/benchmark.py:573-589`），风控是 `HardConstraintEngine`（`:342-406`），交易台是确定性 `TradingDesk.quote`（`:434-487`）。审计结论成立。
2. `MARKUP=1.03`、`HEDGE_COST_RATIO=1.02` 位于 `mirage/pricing.py:13-16`；hedging cost 在 `:716-722`，client price 在 `:725-761`（赋值 `:739`），margin 取差在 `mirage/benchmark.py:475-483`，故严格为 `0.01×FV`。FV 对 notional 线性（`pricing.py:584-587,619-622,690-691`），最大名义策略会机械放大利润。
3. oracle 最大 fraction 常数确在 `mirage/benchmark.py:701`，实际生成在 `:745-754`；agent 的客户资金上限是 100% capital（`:380-381`，另受组合限额），attainment 未截断（`benchmark_runner.py:323-328`）。相同结构且其他约束不绑定时可约达 10；当前 oracle 不是 agent 域上界。
4. 六维 judge 入口在 `mirage/judge.py:94-113`，一致性/翻转工具在 `:169-195`；全仓生产 Python 无调用，只有 `tests/test_judge.py` 使用。`docs/BENCHMARK_PROTOCOL.md:72-87` 却承诺 judge 结果，审计成立。
5. 环境自动选择 archive 最佳可行报价并替交在 `mirage/benchmark_runner.py:263-270`。旧 playground 只是再要求模型 `submit_product|skip_round`（`mirage/agents.py:300-337`；`HEAD:mirage/engine.py:301-328`），并不替模型选最优。
6. `worst_case_payoff_ratio` 只有 0/1（`mirage/benchmark.py:329-339`），`CLIENT_MAX_LOSS` 使用 `1-floor`（`:361-388`），确已退化。行号措辞需纠偏：`_stress_loss` 实际在 `benchmark.py:409-431`，不在 `pricing.py`；连续 `loss_frac` 在 `pricing.py:725-762`，MC `expected_loss_frac` 由 `:287-358` 产出。
7. 仅 `partial_dynamic` 有 3 seeds，其他格子 n=1（`mirage/experiment.py:30-50`）；contrast 只求均值（`:69-105`），CLI aggregate 也仅求均值（`benchmark_cli.py:230-245`），无 CI。
8. 旧实现可复用：`AgentSpec` 有 prompt/model（`mirage/scenario.py:35-43`），`EnvironmentAgent` 有独立 history/system prompt（`mirage/agents.py:104-129`），历史引擎持有 `env_agents`（`HEAD:mirage/engine.py:57-78`）并按轮注入客户状态（`:85-119`）。但有合法草案时 desk/risk 会跳过 LLM、返回确定性块（`:253-264`）；新架构应把 LLM 表达与正式事实显式拆开，而非直接恢复旧文件。

---

## 1. 全角色 LLM 化架构

### 1.1 决策边界与结果分层

```text
RoleOrchestrator
  ├─ StructurerRole（被测模型）
  ├─ ClientRole（披露、追问、accept/reject）
  ├─ RiskRole（approve/revise/escalate）
  └─ DeskRole（issue/decline/revise、执行解释）
                ↓ 只读事实调用
DeterministicCore
  ProductDomain → Pricing → HardChecks → TechnicalSettlement
                ↓
ImmutableTrace → Primary Metrics / Workflow Metrics / Offline Judge
```

正式接口：

```python
@dataclass(frozen=True)
class SeedBook:
    pricing: int
    structurer: int
    client: int
    risk_control: int
    trading_desk: int
    judge: tuple[int, ...]

@dataclass(frozen=True)
class RoleRequest:
    protocol_version: str
    run_id: str
    episode_id: str
    round_num: int
    turn_id: int
    sender: str
    recipient: str
    kind: Literal[
        "client_query", "desk_review", "risk_review",
        "client_decision", "final_submission"
    ]
    state_version: str
    payload: dict

@dataclass(frozen=True)
class RoleResponse:
    request_id: str
    role_id: str
    status: Literal["ok", "abstain", "error"]
    action: str
    payload: dict
    narrative: str
    cited_fact_ids: tuple[str, ...]
    raw_hash: str

async def run_multi_role_episode(
    environment: LongHorizonEnvironment,
    roles: "RoleRegistry",
    *,
    strategy: str,
    replicate_id: int,
    seed_book: SeedBook,
    response_store: "ResponseStore",
    max_actions_per_round: int = 9,
) -> EpisodeTrace: ...

async def dispatch(
    request: RoleRequest,
    runtime: "RoleRuntime",
    facts: "FormalFacts",
) -> RoleResponse: ...
```

双层结果：

```python
@dataclass(frozen=True)
class TechnicalSettlement:
    hard_executable: bool
    quote_id: str
    client_contract_pass: bool
    technical_margin: float
    failure_ids: tuple[str, ...]

@dataclass(frozen=True)
class WorkflowOutcome:
    workflow_deal: bool
    desk_action: Literal["issue", "decline", "request_revision"]
    risk_action: Literal["approve", "request_revision", "escalate"]
    client_action: Literal["accept", "reject", "abstain"]

@dataclass(frozen=True)
class SettlementRecord:
    technical: TechnicalSettlement
    workflow: WorkflowOutcome

def settle_submission(
    product: ProductSpec,
    quote: Quote,
    hard_checks: tuple[CheckResult, ...],
    client_contract_pass: bool,
    desk: RoleResponse,
    risk: RoleResponse,
    client: RoleResponse,
) -> SettlementRecord: ...
```

`settle_submission` 是无 I/O 纯函数。主榜的成交/portfolio 更新仅依据 `technical.hard_executable && client_contract_pass`；`workflow_deal = hard_executable && desk.issue && risk.approve && client.accept` 单列为行为结果，不改变 technical settlement。另报 `llm_action_vs_formal_truth` 分歧率。Judge 永不参与结算。

`client_contract_pass` 确定性检查 capital、maturity、whitelist、protection、连续 loss proxy 和 `hurdle_hit_prob >= min_hit_prob`；旧 `ClientProfile.would_buy` 的相应逻辑在 `mirage/products.py:94-141`。客户 LLM 的态度不覆盖合同门。

### 1.2 交互协议

只允许 orchestrator 路由，禁止自由 role-to-role 群聊，避免不可控调用图和信息泄漏：

1. 引擎生成白名单 brief。
2. `client_query`：Client LLM 决定披露哪些允许字段、语气和追问；具体数值由 hidden `ClientState` 经 adapter 填入，LLM 不生成数值。
3. `desk_review(product)`：先做域校验与确定性定价，生成 engine-signed `QuoteFacts`；Desk LLM 只决定 issue/decline/revise 并解释执行难度。
4. `risk_review(product)`：先运行 hard checks；Risk LLM 只引用已有 `check_id` 并给 approve/revise/escalate，不得覆盖 PASS/FAIL。
5. `final_submission(quote_id)`：重新核验 state-bound quote，运行纯函数 settlement，原子更新组合。
6. episode trace 封存后再离线 judge，不把 judge 分数反馈给 agent。

```python
@dataclass(frozen=True)
class FormalFacts:
    state_version: str
    fact_ids: tuple[str, ...]
    public_client_fields: dict
    quote: dict | None
    checks: tuple[CheckResult, ...]
    allowed_numeric_strings: tuple[str, ...]

def validate_grounding(
    response: RoleResponse, facts: FormalFacts
) -> tuple[bool, tuple[str, ...]]: ...
```

四角色使用不同 response schema：

- Client：`{action: answer|counter|accept|reject|abstain, disclose_fields: [...], reason_codes: [...], narrative: str}`。
- Risk：`{action: approve|request_revision|escalate, check_refs: [...], suggestions: [...]}`。
- Desk：`{action: issue|decline|request_revision, quote_id: str|null, hedge_tags: [...], suggestions: [...]}`。
- Structurer：沿用 query/request_quote/submit/skip，但每次只允许一个 JSON。

`check_refs/fact_ids` 必须为 supplied facts 子集；narrative 中数字经规范化后必须出现在 `allowed_numeric_strings`，否则触发一次 schema repair。

### 1.3 可复现性

需注册两个不同主张：

- `economic_replay`：给定冻结动作/角色响应 trace，定价、硬检查、技术结算和指标完全一致；这是论文可主张的确定性复现。
- `conversation_replay`：保存 canonical request/response、真实模型 ID、provider、system prompt SHA256、generation config、usage、时间和 response SHA256；从 artifact 回放逐字节一致。重新 live 调 API 只称同配置重复，不能称精确复现。

现 OpenAI-compatible client 只是尽力传 seed（`mirage/llm.py:123-146`），Anthropic 路径明确忽略 seed（`:243-272`），所以不能承诺 fresh-call 位级复现。

```python
def derive_seed(namespace: str, *parts: str | int) -> int:
    """SHA256 canonical tuple -> uint32；禁止使用 Python hash()."""

def evaluate_product(
    product: ProductSpec,
    market: MarketState,
    *,
    mc_paths: int,
    mc_seed: int,
) -> PricingResult: ...
```

同一 episode/round/stress_id 下所有候选共享 pricing seed（common random numbers）。trace 冻结 protocol version、domain、stress grid、MC paths、prompt/model config hash、git commit、Python/依赖版本和 rounding policy。当前 `MC_PATHS=10000, MC_SEED=42`（`pricing.py:15-16`）改为显式协议字段。

### 1.4 故障、超时和格式错

```python
@dataclass(frozen=True)
class RetryPolicy:
    transport_retries: int = 2
    format_retries: int = 1
    backoff_s: tuple[float, ...] = (1.0, 2.0)

@dataclass(frozen=True)
class RoleFailure:
    role_id: str
    error_class: Literal["timeout", "provider", "format", "grounding"]
    attempts: int
    request_hash: str
```

- timeout/429/5xx：同 request hash、同 seed 重试 2 次；格式/grounding 错只 repair 1 次。
- 环境角色仍失败：job 标 `environment_failure`，不把它算结构师 0 分，也不静默回退成字典回答；同 job id 补跑。另报 failure rate，并做排除 provider outage 的敏感性。
- 结构师格式错：消耗 action；用尽后 `no_valid_submit`，不能由环境替交。
- Judge 失败：经济结果保留，soft score 为 missing，绝不填 0，可离线补跑。
- fail-closed 的 desk/risk/client 行为可分别记 decline/escalate/abstain，但不得污染 technical settlement 主榜。

---

## 2. 角色配置 schema

当前 `ModelConfig` 只允许 provider/base_url/model/api_key/temperature/max_tokens/timeout（`mirage/llm.py:27-38,58-88`）。连接信息继续放 `config/models.yaml`，角色行为放新 `config/benchmark_roles.yaml`：

```yaml
protocol_version: mirage-csi-v2.0
main_npc_lineup_id: npc-fixed-v1
roles:
  structurer:
    role: structurer
    model_ref: ${job.model}
    system_prompt_file: scenarios/mirage_csi/prompts/structurer.md
    temperature: 0.0
    max_tokens: 1800
    timeout_s: 60
    seed_policy: derived
    seed_offset: 1000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: structurer_action_v1
    tools: [query_client, review_desk, review_risk, submit_design, skip_round]
    max_calls_per_round: 9
    history_scope: episode
    failure_policy: no_action
  client_main:
    role: client
    model_ref: deepseek-v4-flash
    system_prompt_file: scenarios/mirage_csi/prompts/client_main.md
    private_state_file: scenarios/mirage_csi/client.example.json
    temperature: 0.2
    max_tokens: 500
    timeout_s: 30
    seed_policy: derived
    seed_offset: 2000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: client_response_v1
    tools: []
    max_calls_per_round: 4
    history_scope: episode
    numeric_authority: supplied_facts_only
    failure_policy: abstain
  risk_control:
    role: risk_control
    model_ref: deepseek-v4-flash
    system_prompt_file: scenarios/mirage_csi/prompts/risk_control.md
    temperature: 0.0
    max_tokens: 600
    timeout_s: 30
    seed_policy: derived
    seed_offset: 3000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: risk_response_v1
    tools: []
    max_calls_per_round: 3
    history_scope: round
    numeric_authority: supplied_facts_only
    failure_policy: escalate
  trading_desk:
    role: trading_desk
    model_ref: qwen-max
    system_prompt_file: scenarios/mirage_csi/prompts/trading_desk.md
    temperature: 0.0
    max_tokens: 600
    timeout_s: 30
    seed_policy: derived
    seed_offset: 4000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: desk_response_v1
    tools: []
    max_calls_per_round: 3
    history_scope: round
    numeric_authority: supplied_facts_only
    failure_policy: decline
judges:
  models: [deepseek-v4-pro, qwen-max]
  system_prompt_file: scenarios/mirage_csi/prompts/judge.md
  repeats: 3
  temperature: 0.0
  max_tokens: 1200
  blind_model_identity: true
  exclude_same_model_family: true
```

```python
@dataclass(frozen=True)
class InferenceSpec:
    model_ref: str
    temperature: float
    max_tokens: int
    timeout_s: float
    seed_policy: Literal["derived", "fixed", "none"]
    seed_offset: int

@dataclass(frozen=True)
class RoleSpecV2:
    id: str
    role: Literal["structurer", "client", "risk_control", "trading_desk", "judge"]
    system_prompt_file: Path
    system_prompt_sha256: str
    inference: InferenceSpec
    retry: RetryPolicy
    output_schema: str
    tools: tuple[str, ...]
    max_calls_per_round: int
    history_scope: Literal["round", "episode"]
    numeric_authority: Literal["none", "supplied_facts_only"]
    failure_policy: str
    private_state_file: Path | None = None

def load_role_specs(path: str | Path, registry: ModelRegistry) -> dict[str, RoleSpecV2]: ...
def create_role_registry(
    specs: dict[str, RoleSpecV2], registry: ModelRegistry, store: ResponseStore
) -> RoleRegistry: ...
```

loader 必须拒绝未知字段、重复 role、缺 prompt、模型不存在、seed offset 重复、越权 tools，以及 client/risk/desk 的 `numeric_authority != supplied_facts_only`。工程上允许每角色独立配模型，但正式主榜冻结一个 NPC lineup，只改变 structurer；第二 lineup 只能作为 robustness。否则不能归因。

---

## 3. dealer_margin 重定义

### 3.1 可计算量与扩展点

当前 `MarketSnapshot` 无真实 bid/ask（`mirage/benchmark.py:42-61`），不能把 proxy 冒充实盘买卖价差。截稿版使用“MC model spread proxy”，论文明确其含义；真实 bid/ask 校准放 Should。

```python
@dataclass(frozen=True)
class MCDiagnostics:
    pv_mean_frac: float
    pv_std_frac: float
    pv_se_frac: float
    expected_loss_frac: float
    event_probs: dict[str, float]
    expected_life_months: float

@dataclass(frozen=True)
class QuotePolicy:
    a_f: float = 0.030
    a_v: float = 0.25
    a_p: float = 0.002
    a_b: float = 0.75
    b_f: float = 0.020
    b_v: float = 0.75
    b_p: float = 0.004
    b_b: float = 1.00
    b_l: float = 0.003
    b_q: float = 0.015
    client_cap: float = 0.08
    hedge_cap: float = 0.10
    diagnostic_paths: int = 4096

def mc_diagnostics(
    product: ProductSpec,
    market: MarketState,
    *,
    n_paths: int = 4096,
    seed: int,
) -> MCDiagnostics: ...

def quote_economics(
    pricing: PricingResult,
    diag: MCDiagnostics,
    *,
    stress_loss: float,
    post_notional: float,
    capacity_notional: float,
    policy: QuotePolicy,
) -> QuoteEconomics: ...
```

`_mc_stats` 已逐路径积累 fair value、KI/KO、life、loss（`pricing.py:287-358`），只需增加 `sum(pv²)` 和任一触发计数；vanilla/barrier 加同一 4096-path payoff sampler，barrier 可复用 `_mc_hit_prob` 的月度触碰逻辑（`:864-982`）。diagnostics 按去掉 notional 的结构键缓存，oracle 不重复烧算力。`_stress_loss` 改用 base/stressed fair value 而非新 client price，避免 quote 公式循环。

令

\[
N=\text{notional},\quad F=\text{fair value},\quad f=F/N,
\]
\[
V=|\text{vega\_pct}|,
\quad P=\begin{cases}0,&\text{vanilla}\\
2\sqrt{\operatorname{mean}_{j\in events}p_j(1-p_j)},&\text{path-dependent}
\end{cases},
\]

将 P clip 到 [0,1]；events 为 touch/KI/KO。`vega_pct` 当前定义是每 1 vol point 的名义占比（`pricing.py:549-558,698-710`）。模型双边 spread proxy：

\[
B=2\times1.96\times \text{pv\_se\_frac},\quad
L=\text{stress\_loss}/N,\quad
Q=\text{post\_notional}/\text{capacity\_notional}.
\]

B 只能称“MC 模型不确定性双边带”，不能称真实流动性价差。除 N 与确定性 capacity 外，V/P/B/L 均由 `pricing.py` MC 或其低成本扩展产出。

具体公式：

\[
r_c=\operatorname{clip}(a_f f+a_vV+a_pP+a_bB,0,0.08),
\]
\[
r_h=\operatorname{clip}(b_f f+b_vV+b_pP+b_bB+b_lL+b_qQ^2,0,0.10),
\]
\[
\text{client\_price}=F+Nr_c,
\quad \text{hedging\_cost}=F+Nr_h,
\quad \text{dealer\_margin}=N(r_c-r_h).
\]

`Q²` 是组合容量/库存冲击项，使 hedging cost 随 N 超线性增长，直接破除“永远上满 100% notional”哑策略；负 margin 不得强制归零。系数只能在 development episodes 按预注册目标（20–60% feasible quotes 为正 margin、结构排序非退化）做一次校准后冻结，并报告系数 0.8/1.0/1.2 sensitivity。若不同初值表现不稳，应降级为 sensitivity，不在 held-out 后调参。

另分列：

```python
margin_rate = dealer_margin / notional
risk_adjusted_margin = dealer_margin / max(stress_loss, 0.01 * notional)
total_technical_margin = sum(executed technical margins)
```

绝对 margin、margin rate、risk-adjusted margin、hard feasibility 不合成单分。

---

## 4. oracle 与 agent 可行域对齐

正式 track 采用有限 action lattice；open track 允许连续参数，但 denominator 只能叫 `best_known_grid`，不得叫 oracle upper bound。

```python
@dataclass(frozen=True)
class ProductDomainSpec:
    product_types: tuple[str, ...]
    notional_fractions: tuple[float, ...] = (
        .005, .01, .02, .05, .10, .20, .30, .40,
        .50, .60, .70, .80, .90, 1.00
    )
    maturities: tuple[int, ...] = (3, 6, 12)
    strikes: tuple[float, ...] = (.95, 1.00, 1.05)
    barriers: tuple[float, ...] = (.75, .85, 1.10, 1.20)
    coupons: tuple[float, ...] = (.04, .08)
    participations: tuple[float, ...] = (.5, 1.0)
    principal_protected: tuple[bool, ...] = (False, True)
    version: str = "csi-domain-v1"

def validate_domain(
    product: ProductSpec, client: ClientProfile, domain: ProductDomainSpec
) -> tuple[CheckResult, ...]: ...

def enumerate_domain(
    client: ClientProfile, domain: ProductDomainSpec
) -> Iterator[ProductSpec]: ...

def oracle_best_quote(
    domain: ProductDomainSpec,
    snapshot: MarketSnapshot,
    client: ClientProfile,
    portfolio: PortfolioState,
    risk_budget: RiskBudget,
    *,
    state_version: str,
    pricing_seed: int,
) -> tuple[ProductSpec, Quote] | None: ...
```

`request_quote` 和 oracle 必须共用 `validate_domain/enumerate_domain`，禁止 oracle-only variants；notional 都按 `round(client.capital*fraction)`。测试枚举每个 agent 合法动作都属于 oracle 域，并断言 `one_step_attainment <= 1+1e-9`，超出是 protocol error，不能 clip 掩盖。

当前 oracle 是逐轮 myopic best（`benchmark_runner.py:160-168`），dynamic 条件下不是全 episode 上界，因为本轮占用改变未来机会。Must 改名 `one_step_frontier_margin/one_step_attainment`。Should 增加：

```python
def hindsight_episode_oracle(
    snapshots: Sequence[MarketSnapshot],
    initial_state: PortfolioState,
    domain: ProductDomainSpec,
    client: ClientProfile,
    budget: RiskBudget,
) -> OraclePolicyResult: ...
```

它可看整个冻结市场路径，只能标为 hindsight upper bound。

---

## 5. 强制提交改为单列口径

删除环境从 archive 选 best 并成交。额度用尽可再向**模型**发一次 `submit_or_skip`；该动作来源单列：

```python
SubmissionOrigin = Literal[
    "voluntary", "forced_prompt", "environment_imputed", "none"
]

@dataclass(frozen=True)
class RoundFinalization:
    origin: SubmissionOrigin
    actual: TechnicalSettlement | None
    imputed_counterfactual: TechnicalSettlement | None
    imputed_executed: Literal[False] = False

def finalize_round(
    environment: LongHorizonEnvironment,
    trace: RoundTrace,
    archive: CandidateArchive,
    *,
    allow_forced_prompt: bool = True,
) -> RoundFinalization: ...
```

- `voluntary`：action budget 内模型主动提交。
- `forced_prompt`：用尽额度后模型本人响应最后一次 submit/skip；进入 `all_model_submission_rate`，但与 voluntary 分列。
- `none`：仍无合法提交，technical margin=0，portfolio 不更新。
- `environment_imputed`：仅兼容/诊断。archive best 可生成 `imputed_counterfactual_margin`，但不得调用 `environment.submit_design`、不得改 client memory/portfolio、不得进入 hard feasibility、margin 或 attainment 主口径。

报告 `voluntary_submission_rate/all_model_submission_rate/forced_prompt_rate/imputed_rate`。旧 v1 结果不可与 v2 拼表。

---

## 6. judge 接入与 LLM 自评偏差

### 6.1 接入位置

judge 放在 episode trace 封存后、aggregate 前离线跑；不在 action loop 内，不反馈给角色：

```python
@dataclass(frozen=True)
class JudgeInput:
    submission_id: str
    blind_id: str
    client_brief: dict
    product: dict
    explanation: str
    public_fact_sheet: dict   # 无 model/vendor/strategy/margin/oracle/PASS 标签

@dataclass(frozen=True)
class JudgeSpec:
    model_ref: str
    model_family: str
    prompt_sha256: str
    repeat: int
    permutation: str

async def judge_submission(
    item: JudgeInput,
    judges: Sequence[JudgeSpec],
    clients: dict[str, BaseLLMClient],
    *,
    seed_book: SeedBook,
) -> list[JudgeResult]: ...

async def run_judge_batch(
    trace: EpisodeTrace,
    judge_specs: Sequence[JudgeSpec],
    *,
    concurrency: int = 4,
) -> dict[str, list[JudgeResult]]: ...

def aggregate_judge_bundle(results: Sequence[JudgeResult]) -> dict:
    """每维 median/IQR/missing；不生成总排名。"""
```

在 `benchmark_cli._cmd_run_manifest` 当前写 trace/metrics 的位置（`benchmark_cli.py:179-186`）先原子写经济结果，再写独立 `*.judges.json`，aggregate 用 submission_id join。Judge 故障不会使 episode 失败。

### 6.2 偏差控制

1. 两个不同模型家族 × 3 repeats；测试模型与 judge 同家族时 leave-one-family-out，主结果不允许“模型评自己”。
2. canonical input 删除模型名、provider、策略、token、margin、oracle、输出顺序线索；待评文本作为 untrusted JSON data，防 prompt injection。
3. 三 repeats 预注册 rubric order、匿名 client label、paraphrase permutation；报告 `score_flip_rate`（已有 `judge.py:191-195`）。
4. 每维报告 quadratic weighted kappa、总分 Spearman、exact agreement和 missing coverage；现 `reliability_summary` 已明确只代表 inter-judge reliability，非专家效度（`judge.py:169-188`）。
5. evidence 必须是 explanation 的真实子串；在现有 score/evidence/reason 校验（`judge.py:60-91`）上增加 span validation。
6. 60 个按 underlying×condition×hard-pass×model 平衡的样本，两个 judge×三 repeats。至少 30、最好 60 个由两名金融背景专家盲评并仲裁 20% 分歧；没有专家标签时六维只能作为 exploratory reliability，不能声称 validity。
7. 六维分开报，不与 hard/economic 指标合成，不参与成交。

---

## 7. CLIENT_MAX_LOSS 连续化

保留 `CLIENT_PROTECTION/PROTECTION_CLAIM` 合同地板，新增连续 proxy：

```python
@dataclass(frozen=True)
class ClientLossMeasure:
    expected_loss_frac: float
    premium_at_risk_frac: float
    stress_loss_frac: float
    observed_loss_frac: float
    worst_stress_id: str

def client_loss_measure(
    product: ProductSpec,
    pricing: PricingResult,
    stress_loss: float,
    *,
    worst_stress_id: str,
) -> ClientLossMeasure:
    expected = float(pricing.pricing_details.get("expected_loss_frac", 0.0))
    premium = float(pricing.loss_frac)
    stress = stress_loss / product.notional
    return ClientLossMeasure(
        expected, premium, stress, max(expected, premium, stress), worst_stress_id
    )

class HardConstraintEngine:
    def evaluate(
        self,
        product: ProductSpec,
        pricing: PricingResult,
        client: ClientProfile,
        portfolio: PortfolioState,
        *,
        delta_dollars: float,
        vega_dollars: float,
        portfolio_stress_loss: float,
        client_loss: ClientLossMeasure,
    ) -> list[CheckResult]: ...
```

MUST 口径：

\[
\texttt{CLIENT\_LOSS\_BUDGET\_V2 passes}
\iff \max(\texttt{pricing.loss\_frac},\texttt{stress\_loss}/N)
\le \texttt{client.max\_loss\_pct}+10^{-9}.
\]

实现可把 `expected_loss_frac` 也放入 max，字段分别输出，便于解释。`observed` 为连续值，reason 同时列 expected/premium/stress；`PORTFOLIO_STRESS_LOSS` 继续用人民币，不混用。

当前 `_stress_loss` 是 `-20%/+20%/vol+10pp` 三情景（`benchmark.py:409-430`），迁移为带 `stress_id` 的冻结配置，截稿前不扩 grid。当前非保护 vanilla/barrier 的 `loss_frac=client_price/N`，coupon 产品取 MC expected loss（`pricing.py:742-759`），语义不统一；文档必须称 `continuous loss proxy`，不能称数学 worst-case。Should 再从 MC 路径增加 `loss_var95/loss_es95`。

---

## 8. 统计方案：seed、重复、CI、配对检验

### 8.1 Job 与 seed

```python
@dataclass(frozen=True)
class ExperimentJob:
    episode_id: str
    test_model: str
    strategy: str
    condition: str
    replicate_id: int
    npc_lineup_id: str
    seed_book: SeedBook
    protocol_version: str

def build_experiment_manifest(
    episode_ids: list[str],
    models: list[str],
    *,
    strategies: tuple[str, ...],
    replicates_by_condition: Mapping[str, int],
    npc_lineup_id: str,
    master_seed: int,
) -> list[ExperimentJob]: ...
```

seed 用 SHA256 分 namespace 派生。不同测试模型在相同 episode/condition/strategy/replicate 下共享 pricing 和 NPC role seed；记录 provider 是否支持 seed、真实模型 snapshot、时间和 response hash。6 rounds 不是独立样本。

### 8.2 成本受控重复

- development pilot：4 个开发 episodes × 全格子 × 2 repeats，只查 failure、成本和方差，不看测试效果，不进主表。
- held-out：12 episodes，所有 model×strategy×condition 至少 n=3；不再只重复 partial_dynamic。
- 预注册主条件 `partial_dynamic` 可 n=5（预算足再 8），其余不按结果方向自适应补跑。
- 正式主榜固定便宜 NPC lineup；第二 lineup 只做少量 robustness。
- judge 只对已提交样本，60 balanced samples×2 models×3 repeats；经济指标评全量。
- response cache 仅重放完全相同 request hash/故障恢复，不能把不同模型的不同自然语言问题映射成同一答复。

### 8.3 CI 与检验

```python
def aggregate_with_ci(
    rows: list[dict],
    metric: str,
    *,
    cluster: str = "episode_id",
    stratify: str = "underlying",
    bootstrap_reps: int = 10_000,
    seed: int = 20260802,
) -> dict: ...

def paired_permutation_test(
    rows: list[dict],
    metric: str,
    contrast: ContrastSpec,
    *,
    unit: tuple[str, ...] = ("episode_id", "replicate_id"),
    n_permutations: int = 100_000,
    seed: int = 20260802,
) -> dict: ...

def holm_adjust(p_values: dict[str, float]) -> dict[str, float]: ...
```

以 episode 为最高 cluster，按 CSI500/CSI1000 分层重采样 episode，再在 episode 内重采样 replicate；模型、策略、Full/Partial、Static/Dynamic 比较始终基于 matched pair 差值。报告 point estimate、episode-cluster paired bootstrap 95% CI、paired sign-flip/permutation p-value、paired effect size。对两个预注册 primary outcomes（hard feasibility、margin rate）做 Holm；其余 exploratory。bounded rate 也用 cluster bootstrap，不用把每轮当独立 Bernoulli 的 Wald CI。

`environment_failure` 不算模型 0 分，同 job 补跑；结构师自身 format/no-submit 记模型结果。单列 role call/token/cost、timeout、format、fail-closed、no-submit、judge missing 及其 CI，并给出排除 provider outage 的敏感性表。

---

## 9. 逐文件迁移计划与 8 月 2 日裁剪线

### 9.1 逐文件计划

| 文件 | 具体迁移 | 验收 |
|---|---|---|
| `mirage/role_config.py`（新） | `RoleSpecV2/InferenceSpec/RetryPolicy/load_role_specs`、prompt hash、严格 schema | 四角色独立 prompt/model；坏配置 fail fast |
| `mirage/agents.py` | generic `RoleRuntime`、四类 strict parser、grounding validator、独立 history | 环境 LLM 不能越权生成数字；故障有类型 |
| `mirage/scenario.py` | `AgentSpec` 迁至 RoleSpecV2；scenario 只引用 roles config/private state | 客户私有字段不进 public brief |
| `mirage/pricing.py` | `PricingResult/MCDiagnostics/QuotePolicy/quote_economics`；扩 `_mc_stats` 二阶矩与事件；拆 loss 字段 | margin 非 1%FV；同 seed CRN；负 margin 可见 |
| `mirage/benchmark.py` | `ProductDomainSpec`、共享 validate/enumerate、ClientLossMeasure、engine-signed Quote、纯 settlement | agent=oracle 域；loss 连续；settlement 无 I/O |
| `mirage/benchmark_runner.py` | 多角色 orchestrator/event log、SeedBook/failure、submission origin；删除 auto-best | no-submit 不成交；trace 可 replay |
| `mirage/judge.py` | blind input、multi-judge batch、span validation、missing/median/IQR | judge 离线且不污染经济结果 |
| `mirage/experiment.py` | 每格 repeats、分角色 seed、cluster CI、paired tests | 全主格 n≥3；CI 可固定 seed 重算 |
| `mirage/benchmark_cli.py` | `--roles-config --npc-lineup --replicate --response-store --replay-trace`；经济/judge 分文件 | resume 不重复收费；failure 分类补跑 |
| `config/models.yaml` | 只保留 provider connection；记录 seed support | 不误称所有 provider 支持 seed |
| `config/benchmark_roles.yaml`（新） | 本文角色 schema、固定主 NPC lineup | run artifact 保存完整配置/hash |
| `scenarios/mirage_csi/prompts/*.md`（新） | 四角色+judge 独立 system prompt；移植历史角色设定和数字禁令 | 角色私有知识/工具权限不串线 |
| `docs/BENCHMARK_PROTOCOL.md` | 双层结果、proxy spread/loss 限定、两 track oracle、judge/CI 实际口径 | 文档承诺均有运行产物 |
| `tests/test_roles.py`（新） | parser/grounding/timeout/repair/history/prompt injection | 环境故障不造 fallback 答案 |
| `tests/test_pricing_economics.py`（新） | 公式 golden cases、CRN、容量凹性、结构差异 | 不再恒等；最大 N 非必然最优 |
| `tests/test_benchmark.py` | 域相等、attainment≤1、连续 loss、signed quote、pure settle | 主 benchmark validity gates 通过 |
| `tests/test_benchmark_runner.py` | 四角色路由、submission origins、imputed 不改 portfolio、replay | 删除替交后主指标正确 |
| `tests/test_judge.py` | blind/span/same-family/missing/batch join | judge 可复算且不影响 accepted |
| `tests/test_experiment.py` | 全格 repeats、seed 派生、paired bootstrap deterministic | CI 固定 seed 可重算 |

不要恢复 `HEAD:mirage/engine.py` 到正式路径；只移植多 agent map、独立 history、按轮注入思想。

### 9.2 Must / Should / Could

**Must（7 月 24 日协议冻结，7 月 29 日结果冻结；缺任一项就不应把 v2 当论文核心）：**

1. 四角色独立 prompt/model/config、统一事件协议、独立 history；固定主 NPC lineup。
2. LLM facts 双通道；engine-signed quote/check；纯 technical settlement 与完整 replay artifact。
3. 新 margin 公式、MC diagnostics、容量冲击；明确 spread 只是 proxy；golden tests 证明非恒等。
4. agent/oracle 共享 lattice，覆盖 100% notional；one-step attainment 改名并断言 ≤1。
5. 删除环境 auto-best；voluntary/forced-prompt/imputed/none 分列，imputed 不改组合。
6. `CLIENT_LOSS_BUDGET_V2=max(loss_frac,stress/N)`；保护地板独立保留。
7. judge 真正离线接线、六维分列、2 judges×3 repeats 的 60-sample reliability；无专家标签就不声称 validity。
8. 所有主格至少 n=3；episode-cluster paired CI/检验；failure/cost flow 表。
9. 协议文档、CLI artifact、关键测试一致。

建议：7/17–7/20 core/角色/config；7/21–7/23 runner/no-submit/oracle/loss；7/24 冻结 protocol/domain/formula/prompt/stress/outcomes；7/25–7/26 development smoke/成本；7/27–7/29 held-out；7/30–7/31 统计与论文；8/1 只做复现检查和文字修正。7/24 后不得调系数或 prompt。

**Should：**

- 加真实 bid/ask 字段和数据审计，用观测 spread 替换 MC proxy并做敏感性。
- 30–60 个双专家盲评，建立 judge validity。
- dynamic finite-domain hindsight DP oracle。
- 第二 NPC lineup、风险预算和 margin 系数 0.8/1.0/1.2 robustness。
- 主条件 n=5/8。

**Could（截稿后）：**

- nested delta-hedging MC、CVaR/ES、更细路径监控和真实成交冲击。
- 连续域全局优化/open-track oracle、角色自由议价、全角色 cross-play。
- 多客户同时竞价、coalition、可学习 reputation、开放 payoff DSL、人类结构师大规模基线。

### 9.3 Go / No-Go

若 7 月 24 日前只能给三个确定性工具套 LLM 话术，却未修 oracle 域、auto-submit、恒定 margin、连续 loss 与 CI，就不要在 ICAIF 稿件中以“多智能体金融 benchmark”作为主贡献；审稿人会视为昂贵角色扮演包装。只有当 LLM 交互复杂性增加而确定性经济真值不退化，且主结果有共同机会集、重复和 CI，v2 才值得成为论文核心。
