# Stochopia v2 设计公式参考

2026-07-17 的两份独立设计稿（Codex 稿 `draft_codex.md`、Opus 稿 `draft_opus.md`）已于 2026-10-27 删除。
两稿的综合裁决稿 REDESIGN_PLAN.md 也已于 2026-11-08 作为过时计划删除，现行口径以 [BENCHMARK_PROTOCOL.md](../BENCHMARK_PROTOCOL.md) 为准；本文只保留代码 docstring 仍在引用的章节，**原文照录、不作修订**，
章节号前缀 `C` 表示 Codex 稿、`O` 表示 Opus 稿（如 `C§3` 即原 Codex 稿第 3 节）。
两稿中的 16 天排期、针对 2026-07 旧代码的行号核验、逐文件迁移计划和 Must/Should/Could 裁剪线已随原稿删除。

文中出现的 `stochopia/xxx.py:行号` 是 2026-07-17 时的旧行号，仅作历史定位，不再对应当前代码。

---

## Codex 稿保留章节

## C1. 全角色 LLM 化架构

### C1.1 决策边界与结果分层

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

`client_contract_pass` 确定性检查 capital、maturity、whitelist、protection、连续 loss proxy 和 `hurdle_hit_prob >= min_hit_prob`；旧 `ClientProfile.would_buy` 的相应逻辑在 `stochopia/products.py:94-141`。客户 LLM 的态度不覆盖合同门。

### C1.2 交互协议

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


---

## C2. 角色配置 schema

当前 `ModelConfig` 只允许 provider/base_url/model/api_key/temperature/max_tokens/timeout（`stochopia/llm.py:27-38,58-88`）。连接信息继续放 `config/models.yaml`，角色行为放新 `config/benchmark_roles.yaml`：

```yaml
protocol_version: stochopia.csi.legacy.v1
main_npc_lineup_id: stochopia.npc-lineup.fixed.v1
roles:
  structurer:
    role: structurer
    model_ref: ${job.model}
    system_prompt_file: stochopia/resources/structurer.md  # 实现：打包 prompt；structurer 不在 benchmark_roles.yaml 配置（见 BENCHMARK_PROTOCOL.md "LLM-native roles"）
    temperature: 0.0
    max_tokens: 1800
    timeout_s: 60
    seed_policy: derived
    seed_offset: 1000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: stochopia.role.structurer-action.v1
    tools: [query_client, review_desk, review_risk, submit_design, skip_round]
    max_calls_per_round: 9
    history_scope: episode
    failure_policy: no_action
  client_main:
    role: client
    model_ref: deepseek-v4-flash
    system_prompt_file: scenarios/stochopia_csi/prompts/client_main.md
    private_state_file: scenarios/stochopia_csi/client.example.json
    temperature: 0.2
    max_tokens: 500
    timeout_s: 30
    seed_policy: derived
    seed_offset: 2000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: stochopia.role.client-response.v1
    tools: []
    max_calls_per_round: 4
    history_scope: episode
    numeric_authority: supplied_facts_only
    failure_policy: abstain
  risk_control:
    role: risk_control
    model_ref: deepseek-v4-flash
    system_prompt_file: scenarios/stochopia_csi/prompts/risk_control.md
    temperature: 0.0
    max_tokens: 600
    timeout_s: 30
    seed_policy: derived
    seed_offset: 3000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: stochopia.role.risk-response.v1
    tools: []
    max_calls_per_round: 3
    history_scope: round
    numeric_authority: supplied_facts_only
    failure_policy: escalate
  trading_desk:
    role: trading_desk
    model_ref: qwen-max
    system_prompt_file: scenarios/stochopia_csi/prompts/trading_desk.md
    temperature: 0.0
    max_tokens: 600
    timeout_s: 30
    seed_policy: derived
    seed_offset: 4000
    retry: {transport_retries: 2, format_retries: 1}
    output_schema: stochopia.role.trading-desk-response.v1
    tools: []
    max_calls_per_round: 3
    history_scope: round
    numeric_authority: supplied_facts_only
    failure_policy: decline
judges:
  models: [deepseek-v4-pro, qwen-max]
  system_prompt_file: scenarios/stochopia_csi/prompts/judge.md
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


---

## C3. dealer_margin 重定义

### C3.1 可计算量与扩展点

当前 `MarketSnapshot` 无真实 bid/ask（`stochopia/benchmark.py:42-61`），不能把 proxy 冒充实盘买卖价差。截稿版使用“MC model spread proxy”，论文明确其含义；真实 bid/ask 校准放 Should。

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


---

## C5. 强制提交改为单列口径

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


---

## C6. judge 接入与 LLM 自评偏差

### C6.1 接入位置

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

### C6.2 偏差控制

1. 两个不同模型家族 × 3 repeats；测试模型与 judge 同家族时 leave-one-family-out，主结果不允许“模型评自己”。
2. canonical input 删除模型名、provider、策略、token、margin、oracle、输出顺序线索；待评文本作为 untrusted JSON data，防 prompt injection。
3. 三 repeats 预注册 rubric order、匿名 client label、paraphrase permutation；报告 `score_flip_rate`（已有 `judge.py:191-195`）。
4. 每维报告 quadratic weighted kappa、总分 Spearman、exact agreement和 missing coverage；现 `reliability_summary` 已明确只代表 inter-judge reliability，非专家效度（`judge.py:169-188`）。
5. evidence 必须是 explanation 的真实子串；在现有 score/evidence/reason 校验（`judge.py:60-91`）上增加 span validation。
6. 60 个按 underlying×condition×hard-pass×model 平衡的样本，两个 judge×三 repeats。至少 30、最好 60 个由两名金融背景专家盲评并仲裁 20% 分歧；没有专家标签时六维只能作为 exploratory reliability，不能声称 validity。
7. 六维分开报，不与 hard/economic 指标合成，不参与成交。

---


---

## C7. CLIENT_MAX_LOSS 连续化

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


---

## C8. 统计方案：seed、重复、CI、配对检验

### C8.1 Job 与 seed

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

### C8.2 成本受控重复

- development pilot：4 个开发 episodes × 全格子 × 2 repeats，只查 failure、成本和方差，不看测试效果，不进主表。
- held-out：12 episodes，所有 model×strategy×condition 至少 n=3；不再只重复 partial_dynamic。
- 预注册主条件 `partial_dynamic` 可 n=5（预算足再 8），其余不按结果方向自适应补跑。
- 正式主榜固定便宜 NPC lineup；第二 lineup 只做少量 robustness。
- judge 只对已提交样本，60 balanced samples×2 models×3 repeats；经济指标评全量。
- response cache 仅重放完全相同 request hash/故障恢复，不能把不同模型的不同自然语言问题映射成同一答复。

### C8.3 CI 与检验

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


---

## Opus 稿保留章节

## O3. dealer_margin 重定义

### O3.1 病因（精确）

`dealer_margin = client_price − hedging_cost`（`benchmark.py:480`），而 `client_price = MARKUP·fv = 1.03fv`（`pricing.py:739`），`hedging_cost = HEDGE_COST_RATIO·fv = 1.02fv`（`pricing.py:716`）。故 `dealer_margin ≡ 0.01·fair_value`，与结构风险、客户适配**零相关**。最大化 margin ⇔ 最大化 fv ⇔ 在预算内堆最大 fv/notional 的 vanilla。虽然 vega/delta 预算会兜住无限放大，但目标函数仍与产品质量无关。

### O3.2 新公式：成本加成 + 客户适配封顶

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

### O3.3 常数标定

`MarginConfig` 的 7 个常数在**开发集**（warm-up 2022 + 若干 dev episode）上标定并冻结，标定目标可复用 `calibrate_risk_budget`（`benchmark.py:775`）的范式：选一组常数使 (a) oracle 最优产品的 margin/notional 落在合理区间（如年化 0.5%–3%），(b) 哑策略（max vanilla）margin 显著低于 oracle。写 `calibrate_margin_config(dev_cases) -> MarginConfig` 报告并冻结。**这是 protocol change，评测前必须冻结。**

---


---

## O6. judge 接入

### O6.1 病因

`judge.py` 的 `judge_soft_quality`（`judge.py:94`）从未被 `run_episode`/`aggregate` 调用，但 `docs/BENCHMARK_PROTOCOL.md:78` 承诺“两 judge 模型 × 三次重复 × 60 样本”。

### O6.2 接入点：捕获 → 离线批判 → 聚合

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

### O6.3 judge 模型选择与自评偏差控制

- **禁止自评**：judge 模型必须≠被测模型；若某被测模型也在 judge 列表，跳过该配对的其自评样本。
- 两个异构 judge（如 `deepseek-v4-pro` + `qwen-max`）做 inter-judge 可靠性；不宣称专家效度，只报可靠性（`reliability_summary` 的 `claim` 字段已声明）。
- **样本策略**：60 个分层平衡样本（按 4 condition × 模型均衡抽），而非全量 submission，控成本。抽样用固定 seed 冻结样本清单。
- 扰动敏感性（可选 COULD）：对同一批样本做产品字段顺序/客户名替换/反事实扰动，重跑一遍 judge，用 `score_flip_rate` 报稳健性。

---
