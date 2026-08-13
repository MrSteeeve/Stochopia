# MIRAGE 路线 A：形式化隐藏类型博弈设计

日期：2026-08-13

状态：书面规格已完成独立复核，等待用户批准后进入实现计划

适用基线：`ac8cd3c49235c58b4b651ec92756a7599c2c5f46`

本文冻结路线 A 的总架构和分阶段门禁。它不是一次性实现整个路线的授权；
第一个、也是唯一立即进入实现计划的子项目是 A0 Mechanism Probe。

## 1. 决策摘要

MIRAGE 的下一阶段不增加自由角色扮演式 NPC，也不直接启动 PPO、GRPO
或双方共同学习。首个研究对象冻结为：

> Desk-side Harness 已知自身 Book 和 Dealer type，但不知道 Client type；
> 它在有限查询、提案和修订预算下，选择可执行结构化产品，并在硬风控与
> Client 个体理性约束下最小化合约选择 regret。

Client 在第一阶段由版本化的形式响应核控制，不使用 LLM。外生市场继续使用
现有真实来源的月度回放；定价、Greeks、压力测试、硬约束、账户与生命周期
继续由确定性金融内核控制。

本设计将 MIRAGE 定位为：

> 真实市场回放上的、可审计的有限时域隐藏类型博弈 benchmark。

第一阶段不声称模拟真实客户或真实交易员，不称多智能体世界模型，也不支持
生产报价或适当性决策。

## 2. 备选方案与选择

### 2.1 方案 A1：Desk-side Harness，采用

被测策略代表 desk-side structurer。它知道自己的 Dealer type、Book、账户和
允许查看的报价成本，但不知道 Client type。Client 使用固定形式响应核。

采用理由：被测主体、效用归属、信息集和 regret 都可明确；有限类型和有限
产品域可求精确参考策略；也能直接承接 MIRAGE 当前的 structurer/Harness
接口。

### 2.2 方案 A2：中立结构师，不采用

被测策略作为 Client 与 Dealer 之间的中立 mediator，优化双方联合效用或
Nash surplus。该方案需要证明双方效用可进行基数比较，并引入额外的公平性
选择，不适合作为首个可证伪闭环。

### 2.3 方案 A3：双方共同学习，不采用

Client 和 Dealer 都由学习策略控制。该方案需要 equilibrium、population、
cross-play 和非平稳训练协议；当前数据和评测无法区分机制问题与对手训练
问题。

## 3. 研究问题与成功条件

首个研究问题为：

> 在同一公开市场和 Book 状态下，有限交互是否帮助 desk-side policy 推断
> 决策相关的隐藏 Client type，并相对无交互策略降低 held-out 合约选择 regret？

必须同时满足以下条件，结果才支持“策略互动有用”：

1. 改变密封的 Client type 会改变可达历史上的响应或最优成交集合。
2. 查询与反报价不会直接完整泄露类型，而是提供有成本的部分信息。
3. Client 的响应进入结算前权威因果链。
4. 策略互动相对静态完整历史基线和薄 FSM 基线产生可重复的增益。
5. 所有硬约束、账户守恒、状态权限和经济重放保持正确。

若类型没有决策价值，或薄 FSM 达到相同结果，则保留形式博弈作为独立
benchmark 的可能性，但删除 MIRAGE 框架必要性与世界代理主张。

## 4. 系统边界

### 4.1 本阶段包含

- `premium_paid` 的 `vanilla_call` 与 `vanilla_put`。
- 两至四个决策轮次；终端快照只作估值，不作决策轮次。
- 形式化 Client type、Dealer type、双边效用和 outside option。
- 有序、分阶段、结算前绑定的博弈协议。
- 同公开上下文、单变量干预的配对任务生成。
- 精确参考策略、静态/FSM/贪心等基线和 type-aware 评测。
- 密封完整轨迹、公开脱敏轨迹、精确经济重放。

### 4.2 本阶段排除

- Barrier、snowball、autocallable 和 funded-note 产品。
- 自由自然语言谈判和 LLM Client/Desk。
- Risk 作为第三个策略玩家。
- 双方共同学习、equilibrium 与 cross-play。
- PPO、GRPO、reward shaping 和 teacher hierarchy。
- 实际对冲成交、滑点、交易成本和 after-cost Dealer P&L。
- LOB、市场冲击和行动驱动的市场路径。
- 人类行为模仿、生产可用性和现实世界模型主张。

排除项不是永久路线图承诺；只有本设计的 Go 门通过后才重新评估。

## 5. 形式博弈

一个任务定义有限时域 Bayesian game：

\[
G=(x_{1:H}, B_1, M_C, E_C, \theta_C, \theta_D, p(\theta_C), A_D, A_C,
T, U_C, U_D, \Gamma)
\]

- `x_{1:H}`：预载外生市场路径；任何角色动作都不改变它。
- `B_t`：权威 Book、ClientAccount、DealerAccount、持仓和生命周期状态。
- `M_C`：公开且冻结的 ClientMandate，由权威合同门执行。
- `E_C`：公开且冻结的 ClientEndowment，即 Client 在不买产品时的基础资产/负债
  暴露；它只进入 Client 增量效用，不进入报价。
- `θ_C`：密封 Client utility type，仅 Client kernel 与 evaluator 可见。
- `θ_D`：Dealer utility type，desk-side policy 与 evaluator 可见。
- `p(θ_C)`：公开且版本化的 Client type prior。
- `A_D`、`A_C`：双方合法动作。
- `T`：确定性、有序、版本化的状态转移。
- `U_C`、`U_D`：版本化双边效用。
- `Γ`：不可权衡的产品域、适当性和硬风险约束。

`ClientMandate` 与 `ClientUtilityType` 必须分离：前者定义不可越过的合同门，
后者定义在合法合同之间的偏好。`QuotePolicy` 与 `DealerUtilityType` 也必须
分离：前者定义统一的确定性报价机制，后者定义 Dealer 对合法报价的接受和
选择偏好。

## 6. 类型与效用

### 6.1 ClientMandate

从现有 `ClientProfile` 中收窄出独立的硬字段：可用资金、产品白名单、最大
期限、保护要求和不可超过的损失限制。它只进入路线 A 新建的
`RouteAContractGate`，不直接作为效用权重。现有 `client_contract_pass` 还包含
`min_return_pct`/hurdle 等偏好门，不能直接复用；这些字段不得同时充当 HARD
门和效用偏好。首版将 return hurdle 移出 HARD gate，若保留只能经版本化
Client utility 表达。

A0 中 `ClientMandate` 与 `ClientEndowment` 都对 Desk 公开并在任务开始时冻结；
唯一密封变量是 `ClientUtilityType`。这样 contract gate 或基础暴露不会成为
第二套未建模的隐藏类型，
策略差异也不能偷换成“猜客户是否有资格”。A2 若要研究 mandate discovery，
必须另立协议版本和单独的因果实验。

首版 `ClientEndowment` 只允许“现金 + 对同一 underlying 的有符号线性暴露”，
不含已有期权或路径依赖资产。对产品到期时点的情景价值固定为
$W_s=\text{cash}+\beta D_t(T)S_{T,s}$；$\beta$、现金、underlying id 和估值日期
均公开并写入 manifest。它不是客户画像标签，而是使 call/put 的风险对冲价值
可以被明确计算的基础状态。

### 6.2 ClientUtilityType

首版只保留三个决策相关参数：

- `risk_aversion`：终值分布的风险厌恶。
- `liquidity_cost`：锁定资金时间成本。
- `reservation_utility`：不成交的总价值 $r_C$，单位为 CNY。

所有现金流先折现到当前决策时点。对情景 $s$，令 $W_s(E_C)$ 为 Client 不成交
时的基础 endowment 价值，$Y_s$ 为产品到期 payoff，$D_t(T)$ 为公开、版本化
的折现因子。产品对 Client 的增量 certainty-equivalent 价值为：

\[
\Delta CE_C=CE_{\alpha_C,P_C}
\left(W+D_t(T)Y-\text{client price}\right)
-CE_{\alpha_C,P_C}(W)
\]

Client 的成交总价值为：

\[
V_C^{deal}=\Delta CE_C-\lambda_L(\text{client price}\times\tau)
\]

其中首版采用 CARA certainty equivalent：

\[
CE_{\alpha,P}(Z)=
\begin{cases}
\sum_s p_s Z_s,&\alpha=0\\
-\alpha^{-1}\log\sum_s p_s e^{-\alpha Z_s},&\alpha>0
\end{cases}
\]

实现必须使用 log-sum-exp 数值稳定形式。`risk_aversion` 的单位为 CNY⁻¹；
`liquidity_cost` 的单位为 year⁻¹；τ 是从决策时点到产品到期的 ACT/365 年数。
由于首版只有 vanilla，`complexity_cost` 不进入 schema；它在产品复杂度确有
变化前没有决策意义，提前加入只会制造决策等价类型。

Client 用于决策和报告的净剩余定义为：

\[
S_C^{deal}=V_C^{deal}-r_C,\qquad S_C^{no\ deal}=0
\]

因此 `reservation_utility` 只出现一次；“Client accept”严格指
$S_C^{deal}\ge 0$，不得再把 $r_C$ 当作额外惩罚或把总价值和净剩余混报。

`P_C` 是版本化、仅使用 time-t 信息的有限场景分布。A0 objective manifest
必须冻结情景支持、概率、ClientEndowment 到各情景 $W_s$ 的映射、折现曲线、
日计数和小数舍入；概率之和必须为 1。
不得从未来市场快照
倒推 Client type 或私有信号。首版 `P_C` 是所有 Client type 共享的公共信念，
只允许风险厌恶、流动性和 outside option 随 `θ_C` 改变；不得把隐藏
主观市场方向混入 Client type。若 `risk_aversion=0`，`CE` 退化为期望净收益。

总价值和净剩余均以 CNY 计量，所有权重必须声明单位。优化使用净剩余；报告时另外给出
per-cash-outlay 和 per-face-value 的归一化诊断，但不得用归一化值替代真实
notional 的决策效用。

### 6.3 DealerUtilityType

首版只保留四个参数：

- `capital_shadow_price`：压力资本占用的影子价格。
- `inventory_penalty`：Book 集中度惩罚。
- `reservation_utility`：不成交的总价值 $r_D$，单位为 CNY。
- `decision_horizon`：资本占用计算的冻结期限 $H_D$，单位为 years。

令 $K(B)$ 为现有 HARD stress engine 在同一市场状态下计算的 Book stress
capital，定义：

\[
K_{inc}=\max(0,K(B^{post})-K(B^{pre})),\qquad
KTime=K_{inc}\min(\tau,H_D)
\]

`KTime` 的单位是 CNY-year。首版 concentration 使用 objective manifest 冻结的
underlier × maturity bucket 和非负公开权重 $w_b$：

\[
I_{inc}=\sum_b w_b\max(0,|q_b(B^{post})|-|q_b(B^{pre})|)
\]

$q_b$ 是该 bucket 的 CNY-equivalent stress exposure，因此 $I_{inc}$ 的单位
为 CNY。若现有内核不能为某产品确定地产出 $q_b$，该候选 fail closed，不能
用任意常数补值。

协议另有公开、固定、非类型相关的 `ProtocolCostSpec`：每次 query、proposal、
counter 和 actor transition 分别产生 $c_q,c_p,c_c,c_t$ CNY 成本。到终局累计
为 $C_{protocol}$。Dealer 的成交总价值和净剩余固定为 ex-ante 口径：

\[
V_D^{deal}=\text{client price}-\text{economic cost}
-\lambda_K KTime-\lambda_I I_{inc}-C_{protocol}
\]

\[
S_D^{deal}=V_D^{deal}-r_D,\qquad
S_D^{no\ deal}=-C_{protocol}
\]

`capital_shadow_price` 的单位为 year⁻¹，`inventory_penalty` 为无量纲。无动作
直接 no-deal 的净剩余为 0；发生查询或提案后 no-deal 仍承担已经发生的协议
成本。`reservation_utility` 只在成交净剩余中减一次。

其中：

\[
\text{economic cost}=\text{fair value}+\text{incremental hedge cost}
\]

现有 `hedging_cost` 已包含 fair value，适配时必须拆出 incremental 部分，或直接
使用 `client_price - hedging_cost`；严禁再次减 fair value 造成双重扣减。

首版不得把 ex-ante margin 与 terminal lifecycle P&L 相加。由于当前没有真实
交易成本模型，首版不得称 after-cost P&L。

### 6.4 决策等价类型

若两个类型在所有可达交互历史上产生相同响应，并具有相同最优成交集合，则
视为决策等价：

\[
\theta \sim \theta'
\]

机制包（A0 为 ProbeTaskEnumerator，A2 为 TaskGenerator）必须合并或拒绝决策
等价类型。每个保留类型类都必须存在至少
一个合法查询、提案或修订，使其响应或最优动作与另一类型类不同。

### 6.5 跨轮价值与信念

`θ_C`、`θ_D`、公开 mandate 和 public endowment 在一个任务的 2–4 个决策轮次内保持不变；Book、
账户和公开市场按权威 transition 演化。已经披露的 Client 类别和公共响应历史
跨轮保留，动作预算在每个决策轮开始时重置。reference solver 对确定性 Client
kernel 使用精确 Bayes 更新：

\[
p_{t+1}(\theta_C)\propto p_t(\theta_C)
\mathbf 1\{R(h_t,a_t,\theta_C)=r_t\}
\]

零似然观察视为 kernel/manifest 错误并 fail closed，不做平滑。任务终局
$S_C^{terminal}$ 与 $S_D^{terminal}$ 是各轮已成交净剩余和所有已发生协议成本
的和；同一笔 protocol cost 不得同时记入单轮和终局两次。Book 影响后续
`economic cost`、stress capital 和 concentration，因此 reference policy 必须在
完整 state 上求解，不能逐轮独立贪心。

## 7. 报价与信息去耦

现有报价中的 Client suitability 不能继续参与首版价格形成，因为它读取隐藏
客户字段并可能通过价格泄漏类型。首版报价必须满足：

1. 对给定产品、公开市场、Book 和冻结报价机制，价格不读取 `θ_C`。
2. 未披露的 ClientUtilityType 不进入 fair value、hedge cost 或 markup。
3. Client mandate 不参与价格；公开 mandate 仍须由权威 contract gate 复核，
   Desk 不能自行声明 pass。
4. Client 只收到 `ClientQuoteView`：产品条款、对客价格、标准化收益/风险说明。
5. Desk policy 收到 `DeskQuoteView`：对客价格、允许的成本与风险分解、Book
   占用；不收到 `θ_C`、Client utility、regret 或参考策略动作。
6. Evaluator 收到完整 `EvaluatorQuoteView`，但该视图永不进入 policy request。

A0 必须新建独立的 `ProbeQuoteAdapter`。其输入仅为 `ProductSpec`、公开市场、
公开报价配置、公开 notional base 和 Dealer Book；不得接收 `ClientProfile`、
`ClientMandate`、`ClientEndowment` 或 `ClientUtilityType`。它可以调用底层的确定性 pricing/stress
数学原语，但不得调用当前 `TradingDesk.quote()`、`solve_quote_equilibrium()`
或 `client_contract_pass()`，因为这些路径读取 suitability、capital、
`min_return_pct` 等 Client 字段。A0 的价格、成本和风险 fixture 必须能在不构造
任何 Client 对象的条件下复算。

首版报价固定为公开的 cost-plus 机制，而不是按 Client hurdle 反求最高可接受价：

\[
P_{raw}=\text{economic cost}+\mu_F\,\text{face value}
+\mu_K K_{inc}+\mu_I I_{inc}
\]

\[
\text{client price}=\left\lceil P_{raw}/\delta_P\right\rceil\delta_P
\]

$\mu_F$、$\mu_K$ 和 $\mu_I$ 均为无量纲公开参数，$\delta_P$ 为 CNY 报价 tick；
全部在 A0 objective manifest 中冻结。所有候选共用同一组参数，报价不读取
Dealer type 的 $H_D$ 或效用权重。`ProbeQuoteAdapter` 不做“是否值得成交”的
判断：负 Dealer 净剩余由 Desk policy 自己避免，Client 是否接受由 Client
kernel 决定。

首版只协商产品条款，不协商任意连续价格。Client counter 只能从协议允许的
离散 term delta 中选择；任何期限、strike、notional 或其他条款变化都必须
生成新 `ProductSpec`、重新报价并重新运行全部检查。

## 8. 有序交互协议

### 8.1 NegotiationState

每轮唯一的博弈状态至少包含：

- `phase`
- `actor_to_move`
- `proposal_revision`
- `current_product_hash`
- `current_quote_id` 与 `quote_state_version`
- `proposal_commitment`：`review_after_accept` 或 `issue_if_accept`
- `client_response`
- `desk_authorization`
- `queries_left`、`proposals_left`、`counters_left`
- `pending_trade_authorization`

### 8.2 阶段

1. `DESK_DISCOVERY_OR_PROPOSAL`
   - Desk 可 `ask_client`、`propose_product`、`skip_round`。
   - 每个 `propose_product` 必须由 Desk policy 显式选择
     `review_after_accept` 或 `issue_if_accept`。后者是在提交 proposal 时对该
     product/quote/revision/state version 预签不可撤销的发行授权，不是 evaluator
     自动代签。
2. `CLIENT_DISCLOSURE`
   - Desk query 后，Client kernel 以独立 transition 返回该主题的离散类别，
     随后回到 `DESK_DISCOVERY_OR_PROPOSAL`。
3. `CLIENT_RESPONSE`
   - Client kernel 对新鲜报价执行 `accept`、`reject` 或 `bounded_counter`。
   - 对原报价 `accept` 且已有有效 `issue_if_accept` 时，直接进入
     `COMMIT_OR_END` 消费预签授权；否则按下述规则转移。
   - `reject` 在 proposal 预算尚存时回到 `DESK_DISCOVERY_OR_PROPOSAL`；否则
     进入 `COMMIT_OR_END` 并 no-deal。
4. `DESK_REVIEW`
   - 只允许在 Client 已绑定接受当前新鲜 quote 后进入；Desk 只能 `issue` 或
     `decline`，不得再修改条款。
5. `COMMIT_OR_END`
   - 一次性授权校验成功后原子结算；否则以明确的 no-deal 原因结束。

Client kernel 的动作必须作为独立 transition 记录，不能只作为 Desk action 的
隐藏副作用。

`issue_if_accept` 只绑定 Desk 原 proposal 的精确报价。Client reject、
`bounded_counter`、任何重新报价、Book/市场变化或离开当前 response 都使其
失效；尤其不能把它迁移到 Client counter 的替代产品。预签模式下 Client 对原
报价 accept 后，环境只能校验并消费既有 Desk 授权，不能替 Desk 新生成动作。

首版每轮预算冻结为：最多 2 次 Client query、2 次 Desk proposal、1 次 Client
counter。Client counter 是一个完整的替代 `ProductSpec`，只允许相对当前产品
改变一个离散条款；环境对其重新报价和检查。`bounded_counter` 的协议语义是
Client 对“该替代产品按重新计算出的精确价格”作出绑定接受：只有新 quote
通过 HARD/contract gate 且 $S_C^{deal}\ge0$ 时，环境才原子设置
`client_accept=true` 并进入 `DESK_REVIEW`。Desk 此后只能 `issue` 或 `decline`；
若想提出不同条款，必须先 `decline` 并结束本轮，不能复用该接受授权。每轮最多
12 个 actor transition；超限以明确的 no-deal 结束，不自动选择产品。

### 8.3 Client disclosure 与形式响应

Client query 只允许三个版本化主题：`risk_preference`、
`liquidity_preference`、`outside_option`。每次响应只返回对应参数的离散类别，
不返回数值权重。类型格和 prior 必须保证：任何仅由最多 2 次 query 形成的历史
都至少保留两个决策不等价类型的正后验概率；否则该 manifest 无效。接受、拒绝
和 counter 历史本身可以进一步收缩后验，因为这正是待测的策略互动信号。

对每个新鲜且 HARD-pass 的报价，Client kernel 按以下固定顺序响应：

1. ClientMandate 不通过：`reject(reason=contract_gate)`。
2. $S_C^{deal} \ge 0$：`accept`。
3. $S_C^{deal} < 0$ 且 counter 预算尚存：枚举所有合法的一条款邻居，使用重新报价
   后的 ClientQuoteView 计算净剩余；在 $S_C^{deal} \ge 0$ 的候选中选择净剩余最高者，
   再以 canonical product hash 作确定性 tie-break，返回 `bounded_counter`。
4. 不存在可接受 counter：`reject(reason=utility_below_outside_option)`。

HARD-fail 的 proposal 不发送给 Client，记录确定性拒绝后返回
`DESK_DISCOVERY_OR_PROPOSAL`。Client kernel 不做策略性 holdout：当前报价一旦
满足 $S_C^{deal} \ge 0$ 即接受。该限制使首版保持单一学习主体；更复杂的议价策略
属于 A3 或后续协议版本。

### 8.4 成交授权

`TradeAuthorization` 必须绑定：

- product hash
- quote id
- quote state version
- proposal revision
- `hard_pass`
- `client_contract_pass`
- `client_accept`
- `desk_issue`

最终提交条件为：

\[
commit = quote\_fresh \land hard\_pass \land client\_contract\_pass
\land client\_accept \land desk\_issue
\]

角色效用不是硬成交门。Desk 可以错误地发行负效用交易；这应当成交并作为
policy failure 计入 regret，而不是被环境替它纠正。Client 由形式响应核控制，
因此只在自身 mandate 合法且 $S_C^{deal}\ge0$ 时接受。

`client_accept` 只能由版本化 Client kernel 签发；`desk_issue` 只能由当前被测
Desk policy 的显式 `issue` 动作，或 proposal 中显式预签的
`issue_if_accept` 承诺签发；evaluator 只校验，不代签。两者都绑定同一
product/quote/revision/state version 及 commitment mode。任一产品、价格、
Book、市场或报价配置变化，任何新 proposal、`decline`、reject、counter、市场
推进，或非法离开当前 phase，都会使对应授权立即失效。发行授权只可消费一次；
成功 commit 后不得重放。

任何 stale quote、越权 actor、错误 phase、非法 counter 或重复 authorization
均 fail closed，不写账户、不推进市场。

## 9. Actor-specific observation

所有视图使用 allowlist，不使用“删除若干私有字段”的 denylist。

### 9.1 DeskObservation

包含：公开市场、Dealer Book/账户允许视图、自身 `θ_D`、公开 prior、完整公开
ClientMandate 与 ClientEndowment、已披露的 Client utility 类别、当前
DeskQuoteView、预算、公共历史和合法动作。

不包含：`θ_C`、ClientUtility、参考策略动作、未来市场、task hash 预像或
evaluator-only score。

### 9.2 ClientObservation

包含：自身 mandate 与 `θ_C`、自身账户、公开市场、ClientQuoteView、公共历史
和合法 Client 动作。

不包含：`θ_D`、Dealer Book、内部 margin、hedge-cost breakdown、参考策略动作
或未来市场。

### 9.3 EvaluatorState

包含所有类型、账户、报价内部事实、转移状态和参考策略结果。只用于密封重放
和离线评测。

## 10. TaskGenerator 与数据切分

TaskGenerator 以现有真实来源的市场 episode 为外生父上下文，组合：

- 初始 Book/账户状态。
- ClientMandate。
- ClientEndowment。
- ClientUtilityType 与公开 prior。
- DealerUtilityType。
- 冻结产品域、报价机制和博弈预算。

每个任务必须记录 `context_family_id`、`type_intervention_id`、generator version、
seed、父市场 episode、所有 transformation 和类型 provenance。

生成器必须产生配对反事实任务：公共市场、公开 Book、公开 prior 和初始观察
完全相同，只改变一个密封类型。为防止“先选出有互动价值的样本，再证明互动
有价值”，A0 必须冻结两套互不重叠的 `context_family_id`：

1. `mechanism_calibration`：只用于选择类型格、效用常数、产品格和协议成本；
   可以检查 full-information gap、类型差异和策略反转，但其结果永不作为 Go
   证据或置信区间样本。
2. `mechanism_acceptance`：在 type/objective/protocol manifest 完全冻结后，按
   预注册的父市场 episode 与 Book 抽样规则一次性生成；不得依据隐藏类型、
   full-information gap、Client response、策略排序或 VOI 进行纳入/排除。

Acceptance family 的纳入只能依赖生成前可知的 provenance 和 schema 有效性。
若某个冻结 family 不存在 HARD-feasible 产品、出现信息泄漏、类型等价或 solver
失败，该 family 不能被静默删除或替换；它作为相应 gate 的失败计入报告。所有
acceptance 统计使用完整冻结清单。数值“严格为正”统一定义为超过
$\epsilon_f=10^{-4}K_0$，其中 $K_0$ 是该 family 初始 Dealer risk-capital limit；
该阈值相当于 1 bp，既排除浮点噪声，也冻结最小经济量级。

Train/dev/test/private_test 必须按 `context_family_id` 分割；同一市场路径的类型
变体不得跨 split。统计也按 `context_family_id` 聚类，不能把 type twins 当作
独立市场样本。

A0 实现一套与现有 v3/A2 解耦、但对 probe domain 语义完整的机制包：
`ProbeTaskEnumerator`、类型盲 `ProbeQuoteAdapter`、`RouteAContractGate` 和
`ProbeTransitionModel`。三类 reference policy 必须只调用这一个完整 transition
model，不能各自复制响应或结算逻辑。A0 不写入 `TaskSuite`，也不承诺 curriculum、
训练 split 或公开 runner 接口。通过 A0 后，A1 将同一协议移植为 production-
quality authoritative kernel，并要求对 A0 的 canonical fixtures 逐 transition
一致；A2 才把已接受的 manifest 固化成正式版本化 TaskGenerator。

## 11. Reference policies 与评测

本设计中的“oracle”只表示已冻结形式博弈内的参考求解器，不代表现实金融
最优或唯一正确答案。公开材料统一使用 `reference_policy` 命名。

必须实现并冻结：

- `full_information_reference`：Desk 在 t=0 免费知道 `θ_C`，除此以外与
  interactive policy 使用相同动作、预算、协议成本和状态转移。
- `bayesian_interactive_reference`：Desk 只知道 prior，在完整交互预算下的
  Bayes-optimal 策略。
- `no_interaction_reference`：Desk 不查询，提交恰好一个 proposal；Client 的
  counter budget 固定为 0，并在 proposal 中显式预签 `issue_if_accept`。Client
  accept 时环境只消费该既有授权并原子 commit，reject 时授权失效并 no-deal；
  Desk 不能观察 response 后再行动。
- random、greedy、memoryless。
- 当前 v3 动态 Book、无策略 Client 的基线。
- 脱离完整 MIRAGE runner 的极薄 FSM benchmark。
- 完整历史条件化的静态强基线。

若有限状态空间无法精确求解，则先缩小类型、产品或轮次数量；首版不得用近似
RL policy 冒充 reference policy。

对一个公开 context family $f$，市场路径和初始 Book 固定，隐藏类型按公开
prior $p_f(\theta_C)$ 抽取。策略价值唯一地定义为终局 Dealer 净剩余的期望：

\[
J_D^f(\pi)=\sum_{\theta_C}p_f(\theta_C)
\,\mathbb E_{\omega_\pi}[S_D^{terminal}(f,\theta_C,\pi;\omega_\pi)]
\]

首版 Client kernel、quote 和 transition 都是确定性的，唯一允许的随机性
$\omega_\pi$ 来自 policy；其 seed 清单必须冻结。跨 family 的 $J_D$ 是预注册
family 权重的加权平均。不得把某一密封类型上的 realized value 当作 Bayes
expected value，也不得在求解后重加权 prior 或 market family。

### 11.1 主指标

主指标是同信息集、同预算下相对 `bayesian_interactive_reference` 的
Dealer utility regret：

\[
Regret_D = J_D(\pi^*_{Bayes}) - J_D(\pi)
\]

约束违规不与 utility 加权；单独作为硬失败报告。

### 11.2 次级指标

- Client utility 与 Dealer utility。
- 双边个体理性率。
- Pareto-efficient deal rate。
- worst-type regret 与类型分位数。
- query/proposal/counter 成本。
- no-deal 原因分解。
- 类型识别校准；仅评估决策等价类，不要求恢复原始连续参数。
- simulator exploit gap。

### 11.3 互动必要性

“策略排序反转”不是指 action label 不同，而是一个有经济量级的双向
cross-evaluation。对同一 public context 中的类型对 $(\theta,\theta')$，令
$\pi^*_{\theta}$、$\pi^*_{\theta'}$ 为各自 full-information reference。只有同时
满足

\[
J_D^{f,\theta}(\pi^*_{\theta})-J_D^{f,\theta}(\pi^*_{\theta'})>\epsilon_f
\]

和

\[
J_D^{f,\theta'}(\pi^*_{\theta'})-J_D^{f,\theta'}(\pi^*_{\theta})>\epsilon_f
\]

才计一次 reversal。tie、仅产品 hash 不同但价值差不足、或单向 dominance 都
不计。该定义证明“知道类型会改变正确策略”，但本身不证明 MIRAGE 框架必要。

在完整冻结的 `mechanism_acceptance` family 上报告 pooled full-information gap
恢复率：

\[
VOI_{pool}(\pi)=
\frac{\sum_f w_f[J_D^f(\pi)-J_D^f(\pi_{no})]}
{\sum_f w_f[J_D^f(\pi_{full})-J_D^f(\pi_{no})]}
\]

不能为了使分母为正而删除 family。若 pooled 分母不超过
$\sum_f w_f\epsilon_f$，路线 A 直接 No-Go。单 family 分母不超过
$\epsilon_f$ 时，该 family 的 ratio 记为 `NA` 但仍进入 pooled 分子、分母和
gap prevalence；不得静默丢弃。95% CI 按 `context_family_id` cluster bootstrap，
使用预注册的 resample 数、seed 和 percentile/BCa 方法，不把同 family 的 type
twins 当作独立样本。

## 12. 轨迹、重放与私有验收

保留现有单一 `step()`、前后 state hash、record hash 和经济重放原则，并扩展为
多 actor 事件流。每个 transition 必须记录：

- actor 与 phase。
- actor-specific observation hash。
- action、quote/revision/state version。
- 类型、效用、博弈协议和 policy configuration hashes。
- state hash before/after。

保存两类 artifact：

1. `sealed_trajectory`：含完整 task manifest、类型和 evaluator state，仅私有
   evaluator 可读取。
2. `public_trajectory`：去除类型、内部账户/报价事实和可反推 private task 的
   标识，用于公开审计。

本地 command policy 的 private-test 运行必须在受限工作目录、环境变量
allowlist、网络/文件访问边界下执行；仅把 split 字段写成 `private_test` 不构成
私有验收。

新增博弈模块必须进入 implementation hash 清单，否则轨迹不能声称承诺完整
运行语义。

## 13. 错误处理与 fail-closed 规则

以下错误不改变经济状态：

- actor 与 `actor_to_move` 不一致。
- action 不属于当前 phase。
- quote stale、quote id 不存在或 revision 不匹配。
- counter 超出离散协议域。
- Client kernel、reference solver 或 utility feature 不可用。
- task manifest、type schema、objective 或 protocol version 不匹配。
- 观察投影检测到私有字段。

Policy 的非法动作计入 policy failure；provider、进程、超时和 evaluator 故障
继续作为 `infrastructure_error`，不得记成 policy 零分。任何账户写入必须通过
唯一 commit 路径完成。

## 14. 测试设计

### 14.1 机制与权限

- 未获得 Client accept 或 Desk issue 时不能写账户。
- Hard/contract fail 永远不能被角色动作覆盖。
- 负 $S_D^{deal}$ 的错误 issue 仍按协议成交并产生负效用/regret。
- counter 后旧 quote 必须失效，新 ProductSpec 必须重新报价和检查。
- `issue_if_accept` 只可由 Desk proposal 预签，只能消费原报价一次；reject、
  counter 或状态变化后必须失效，evaluator 不得代签。
- 重复或 stale authorization 必须被拒绝。

### 14.2 信息隔离

- DeskObservation 不含 `θ_C`、Client utility 或未来市场。
- ClientObservation 不含 `θ_D`、Book 或内部 margin。
- Client type 变化不能通过未披露的价格输入旁路泄漏。
- 所有 observation projection 使用 allowlist property tests。
- public trajectory 不能重建 sealed type 或 task preimage。

### 14.3 经济与重放

- Client/Dealer/settlement 现金流守恒。
- 所有 game transition 可逐步精确重放。
- 相同 task、policy 和 seed 产生相同形式响应与经济结果。
- 任一 action、type、objective 或 protocol mutation 都改变相应 hash。

### 14.4 科研有效性

- 决策等价类型检测。
- 配对任务仅改变指定隐藏变量。
- full-info、Bayes、no-interaction reference 的单调关系检查。
- `mechanism_calibration` 与 `mechanism_acceptance` 的 family 完全不重叠。
- Acceptance family 不因 gap、response、策略排序或 solver 结果被删除或替换。
- 至少 20% 的 acceptance 配对 family 出现预注册的策略排序反转。
- 薄 FSM、静态强基线和完整环境使用相同信息、动作与预算。

## 15. Go / No-Go

进入 Harness 训练前必须同时通过：

1. 私有 acceptance 配对 family 中至少 20% 出现隐藏类型导致的策略排序反转。
2. `VOI_pool` 的 cluster-bootstrap 95% 置信区间下界大于 0，并至少恢复 10%
   的 pooled full-information gap；不得删除零 gap family。
3. 完整环境相对静态强基线和薄 FSM 的 held-out regret 有预注册的实用增益。
4. HARD/contract authority 错误、账户不守恒、私有信息泄漏和 replay divergence
   均为 0。
5. 独立金融专家对效用排序与机制边界的盲审达到至少 90% 一致或系统正确拒答；
   用于设计/校准的专家和案例不得进入私有验收。

任一条件失败时：

- 隐藏类型没有决策差异：退回静态 benchmark。
- 薄 FSM 与完整框架等价：保留形式 game/protocol，删除 MIRAGE 框架必要性。
- 专家不认可效用排序：只称合成算法 benchmark，不称金融行为代理。
- 存在泄漏、越权或重放错误：停止所有训练与结果发布。

### 15.1 分阶段门禁

- **A0 → A1**：只在冻结的 `mechanism_acceptance` 上判断：reference solver 对
  全部状态精确收敛；不存在未合并的决策等价类型；至少 20% 配对 family 出现
  策略排序反转；至少 80% family 的 full-information gap 超过各自
  $\epsilon_f$。任何失败都按预注册清单报告，不能换样；任一门槛失败即停止
  路线 A。若要重新设计类型格，必须生成新的 manifest/version 和全新 acceptance
  family，旧结果只保留为 calibration，不得继续累计显著性。
- **A1 → A2**：authority、allowlist leakage、账户守恒和 replay 测试全部零
  错误；任何失败均禁止生成训练任务。
- **A2 → Harness training**：执行本节五项完整门槛，包括静态/FSM held-out
  比较、VOI 置信区间和独立专家盲审。

## 16. 分阶段实施边界

本设计拆为三个顺序子项目；每个子项目单独设计、计划和验收：

1. **A0 Mechanism Probe**
   - 类型、效用、utility feature、决策等价检查。
   - 类型盲 ProbeQuoteAdapter、RouteAContractGate、完整 ProbeTransitionModel、
     小型产品/类型枚举和三类 reference policy。
   - 独立生成 calibration family 和至少 48 个 acceptance 配对 family；只用
     后者验证策略排序反转与 full-information gap，不生成正式 TaskSuite。
   - 不接 LLM，不训练，不改现有 v3 主协议。
   - 固定输出为：版本化 type/objective/protocol manifest、两套 family 清单、
     reference-policy value table、决策等价报告、VOI/策略排序反转报告和
     Go/No-Go 结论。A0 不输出宣传性模型成绩。
2. **A1 Authoritative Game Kernel**
   - NegotiationState、actor-specific observation、有序动作、TradeAuthorization。
   - 密封/公开轨迹与精确重放。
   - 只在 A0 Go 后开始。
3. **A2 Task/Evaluation Harness**
   - 版本化 TaskGenerator、split、聚类统计、基线矩阵和私有验收。
   - 只在 A1 authority/leakage/replay 全部通过后开始。

首个实现计划只覆盖 A0。A0 失败时不实施 A1/A2。

## 17. 现有组件处置

### 17.1 分阶段复用

- A0 可复用 `ProductSpec` 数据结构，以及不接收 Client 对象的底层定价、Greeks
  和 stress 数学原语；产品格、报价 adapter、contract gate、utility 和 transition
  model 均为路线 A 独立版本。
- 当前 `TradingDesk.quote()`、`solve_quote_equilibrium()`、
  `client_contract_pass()` 不得在 A0 reference path 上直接复用。
- `ClientAccount`、`DealerAccount`、`PortfolioState`、生命周期账本，以及 v3
  typed transition、state hash、TrajectoryRecorder 和 replay evaluator，只作为
  A1 移植候选；复用前必须证明不读取旧 Client 偏好或旧 reward 语义。
- `TaskSuite`、agent runner 的基础协议和数据审计模式只在 A2 评估复用，不进入
  A0 证据链。

### 17.2 保留但不进入路线 A 主结果

- legacy `FrozenEnvAgent`、prompt、response cache。
- `WorkflowOutcome` 与 `workflow_deal`。
- LLM offline judge。
- Full/Static × Partial/Dynamic 的 v2 factorial 结果。
- `trust`、`relationship_delta`、`communication_faithfulness`。

这些组件不物理删除，以保持 v2/v3 历史复现；路线 A 使用新的协议、任务和指标
版本，严禁与旧结果混合汇总。

## 18. 专家数据的角色

第一阶段不要求专家逐条标注 rollout。专家工作限定为：

- 审查类型参数范围和效用排序是否具备金融合理性。
- 审查硬约束与权限边界。
- 对少量密封成对决策进行独立盲审。
- 标记规格支持域之外的情形，要求系统拒答。

若专家成本仍随 rollout 数量线性增长，或每个反事实分支都需专家给答案，
则 MIRAGE 没有形成可复用环境价值，应停止扩张。
