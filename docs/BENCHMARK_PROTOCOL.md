# Stochopia CSI Protocol: v2 Legacy Benchmark and v3 Environment Spine

> **Version boundary.** The factorial CLI commands and
> `LongHorizonEnvironment` described below are inherited **v2 engine surfaces**
> that now emit Stochopia v1 artifacts; they do not accept pre-migration
> trajectories, manifests, hashes or seeds. `stochopia.environment` and
> `stochopia-benchmark test-agent` are the
> **v3-spine / Level-0** synchronous interface, in which partial observability
> and dynamic state are invariants rather than experimental conditions. v2
> and v3 engine artifacts generated under Stochopia v1 are versioned separately
> and must never be pooled in one result table.

## Research claim

Stochopia evaluates whether an LLM can repeatedly synthesize executable structured
contracts under partial observability, limited desk/client/risk interaction and
portfolio risk carried across decisions. It does not claim open-ended financial
invention or production-grade market pricing.

v2 separates the benchmark into two layers. A deterministic core (pricing, hard
constraints, the client contract gate and settlement) is the sole authority for
the primary leaderboard and is a pure function of the episode data and the
tested model's action sequence; it never calls an LLM. Three environment roles
(client, risk control, trading desk) may optionally be backed by their own
LLMs for qualitative dialogue and a secondary workflow-decision signal, but
none of them can override a deterministic check, and the primary metrics are
computed identically whether or not any environment LLM is wired in. This
document only describes behaviour that is implemented and covered by
`tests/`; features named in `docs/redesign/` but not yet wired into the CLI or
the runner are marked "not enabled / future work".

The legacy v2 pre-registered design is Full/Partial information crossed with
Static/Dynamic state. Static rounds reset portfolio and client history. Dynamic
rounds retain issue fixing, absolute contract levels, barrier/knock-in/autocall
path state, remaining maturity and client history; FV, delta, vega and stress
loss are recomputed from the next market snapshot. Barrier monitoring remains
a monthly-snapshot approximation; the closed-form barrier price applies the
Broadie–Glasserman–Kou discrete-monitoring shift (`exp(±0.5826·σ·√(1/12))`)
so pricing and lifecycle settlement use the same monthly convention. Touched
status comes only from the observed monthly path; a delta-bumped or stressed
spot is just the start of the future path and counts as knocked only beyond
the shifted barrier. BGK is an approximation with a material residual against
monthly Monte Carlo (see the 2026-11-09 changelog). A
knocked-in, non-called unprotected autocall redeems `min(S_T/S0, 1)` with no
coupon in both pricing and settlement. Full/Static are not v3 task conditions.

## v3-spine / Level-0 boundary

`stochopia.environment.StochopiaStructurerEnv` exposes typed `reset`/`step`
transitions around the deterministic core. Its actions are `AskClient`,
`RequestQuote`, `SubmitDesign`, `SubmitProduct`, `Skip` and `InvalidAction`;
its output keeps an unscalarised `RewardComponents` vector and separate
`ConstraintSignals`. The policy-visible `Observation` never contains the full
client profile. By default `info` contains only version/hash/run metadata.
Evaluator-only truth appears as `info["privileged_state"]` only when the
harness explicitly opts into `expose_privileged_info=True`; it must never be
passed to a policy. Partial quote and settlement payloads expose client check
IDs/statuses but redact their exact observed values, limits and reasons.

The core contains no LLM call, prompt, strategy, forced completion or oracle.
`RewardComponents` stores measured or explicitly unavailable terms with
provenance, units and normalization metadata; downstream scalarization requires
an explicit versioned `ScalarizationSpec`. `TrajectoryRecorder` takes a canonical
JSON snapshot of each transition at record time, embeds the complete task
manifest, records the environment configuration payload, public action schema,
policy/prompt configuration, dependency versions and the Git commit, and writes
one atomic JSON artifact (`stochopia.trajectory-json.v2`). This is an architectural
spine, not yet a claim that the planned payoff DSL, solve-for tools, task
generator, teacher hierarchy, Pareto evaluator, transaction-cost model or
fast/reference pricer are complete. Until a versioned TaskGenerator exists,
`reset(options={"curriculum": ...})` fails closed instead of accepting a label
that would not alter the task distribution.

Typed product actions are canonicalised through the same parser as JSON
actions; policies cannot set `reference_spot`, barrier-history or elapsed-time
fields. Quote storage snapshots the mutable legacy `ProductSpec`, preventing
post-quote mutation before submission. Invalid actions cannot run forever:
`max_steps_per_round` ends the rollout with `truncated=True`. Reward/termination
configuration and reset options are recorded verbatim in the trajectory.

### v3 external Agent interface

`stochopia.environment.agent_runner` is the only implemented policy-facing
rollout layer for v3. It does not move LLM calls into the deterministic core:
it converts an external response into a typed action and calls the same
`StochopiaStructurerEnv.step()` entrypoint used by a training loop.

Two adapters share the versioned `stochopia.agent-request.v1` request:

- `CommandAgentPolicy` invokes a local executable directly, without a shell.
  Each step sends one JSON object on stdin and requires one action JSON on
  stdout. The request carries the packaged action contract, current public
  observation and a bounded public interaction history.
- `LLMAgentPolicy` accepts any existing `BaseLLMClient`.
  `create_api_agent_policy` constructs an OpenAI-compatible or Anthropic
  client directly from provider/base URL/model/API-key-environment-variable
  arguments, without requiring a `models.yaml` edit.

The CLI exposes both through `stochopia-benchmark test-agent`: use
`--agent-command`, `--model` (registered), or the direct
`--api-model/--api-base-url/--api-key-env` path. These choices are mutually
exclusive. Literal secret values are not accepted; only the environment
variable name is configured, and its value is neither serialized nor sent to
the model.

Every Agent request contains the authoritative machine-readable
`action_schema`: the nine exact client topics, product-domain version, target
client, allowed types and scalar grids, legal field combinations and funding
rules. It contains only a public task-family identifier and a public run seed;
it never includes `task_hash`, source episode/client identifiers, the task
manifest, hidden client profile or evaluator-only `privileged_state`. The
complete task manifest remains in the evaluator's saved trajectory so the run
can be replayed. Malformed model output, unknown fields and unavailable actions
become explicit `InvalidAction` transitions and consume the ordinary step limit.
v3 also replaces internal quote keys with round-local public aliases and strips
account snapshots plus internal position/event/cashflow identifiers from tool
results; those values remain available only inside the sealed evaluator state.
Provider, transport, command-exit and timeout failures instead stop the run as
`infrastructure_error` without stepping the environment or incrementing Agent
invalid-action counts. Command timeout/cancellation terminates its whole POSIX
process group. Raw policy output is recorded by default;
`--redact-raw-output` omits `policy_metadata.raw_output` and, for unparseable
output, replaces the action with an `InvalidAction` without `raw` before
`step()`, so the text never reaches the trajectory, the environment's
`last_event` or the agent's history (the parser error message is kept).

### v3 input, interaction, output and evaluation contract

- **Input**: `TaskSuite` freezes its split (`train/dev/test/private_test`), full
  task manifests, task hashes and generator provenance under one suite hash.
  The Level-0 importer builds suites from selected market episodes; it is not
  yet a stochastic curriculum generator.
- **Interaction**: only the public `Observation`, bounded public history and
  action schema cross the policy boundary. The environment remains
  partial-observation and dynamic for every task.
- **Output**: `TrajectoryRecorder` writes the sealed task manifest,
  environment configuration, reset options, public run seed, action schema,
  transitions and policy provenance into one atomic JSON artifact.
- **Evaluation**: `evaluate-trajectory` scores the recorded transitions. With
  `--replay` it additionally reconstructs the task and environment and requires
  every transition to match a fresh economic replay exactly; this doubles the
  cost and is off by default. `aggregate-v3` accepts Level-0 evaluation
  artifacts from one suite, first averages replicates within each task, and
  reports equal-task-weighted bootstrap intervals by policy.

`run-v3-suite` supports explicit replicates and resumes only complete
trajectories whose task, public run seed, environment configuration and policy
configuration still match. Partial or stale runs are rerun; `--force` disables
reuse.

The final market snapshot is terminal valuation truth, not a policy decision
round. Each accepted contract therefore carries through at least one market
interval. At the terminal snapshot, natural lifecycle events are processed
first and any remaining position is liquidated at current deterministic fair
value. This removes the degenerate strategy of issuing a product and
immediately liquidating it in the same final action.

The v3 evaluator reports termination, truncation, partial/infrastructure
failure, step and invalid-action rates; submission, HARD-pass, conditional
contract-acceptance and settlement-acceptance rates; lifecycle closure count;
flow-adjusted client return; accepted ex-ante margin-per-face sum; realised
hedged lifecycle P&L over dealer risk capital; capital-time and stress-time
ratios; and explicit interaction costs. Quote-only HARD checks are not counted
as submissions. Aggregation refuses mixed suite hashes and refuses to pool one
policy name across different policy-configuration hashes.

## Frozen legacy v2 episode design

- Price-only warm-up: 2022.
- Evaluation: 2023--2025.
- Underlyings: CSI500 and CSI1000.
- Twelve half-year episodes, six month-end rounds each.
- Per round: up to 3 `query_client` calls, up to 3 `consult` calls, up to 3
  `request_quote` calls, then a submission or a skip.
- Frozen finite product action lattice `stochopia.product-domain.csi-action-contract.v1` (below),
  shared by the tested agent and the oracle.

The released example snapshot (`market_snapshots.example.csv`) is synthetic and
cannot be used for reported results. A production snapshot row must carry
explicit source provenance. Raw Tushare/CFFEX files are collected by
`stochopia.tushare_batch` / `stochopia.cffex_data` with a JSON sidecar per
request; the research-only licence boundary is recorded in
`data/reference/license_decision.json`. Preferred data are contract-level CFFEX MO
history for CSI1000 and SSE 510500 ETF option history for CSI500. Full option
surfaces are optional: the frozen fallback is 3m ATM IV, 1m ATM IV, 6m ATM IV,
20d realized volatility, then 60d realized volatility
(`MarketSnapshot.pricing_volatility`).

When paired calls and puts are available, the environment uses

\[
F(K,T)=K+e^{rT}(C-P)
\]

so dividend and carrying-cost expectations are absorbed by the market-implied
forward. ETF fees are not deducted
again from ETF spot/NAV. Dividend-related contract adjustments must be
normalized or the affected row excluded.

## Legacy v2 two-layer architecture

```text
StructurerRole (tested LLM)
   | query_client / consult / request_quote / submit_design / submit_product / skip_round
   v
LongHorizonEnvironment (central router; no free-form multi-agent chat)
   |-- ClientRole LLM      (partial-info disclosure, consult, submission decision)
   |-- RiskRole LLM        (consult, submission decision; can only cite existing check_id)
   `-- DeskRole LLM        (consult, submission decision; numbers come from a real quote)
   v  read-only, fact-gated calls
DeterministicCore: TradingDesk (pricing + HardConstraintEngine) -> client_contract_pass
                    -> settlement (accepted = hard_executable AND client_contract_pass)
   v
Primary settlement (deterministic, always computed):
  hard_executable, client_contract_pass, accepted, dealer_margin, hard/contract failures
Secondary WorkflowOutcome (only when env roles are wired and a submission happened):
  workflow_deal, desk_action, risk_action, client_action, degraded
Offline: judge-runs batch over frozen voluntary submissions (never feeds back into the run)
```

There is no `TechnicalSettlement`/dialogue class hierarchy in code; the primary
settlement is a plain dict returned by `LongHorizonEnvironment.submit_design`
(and mirrored onto `RoundTrace` / `compute_metrics`), and the secondary layer is
`stochopia.benchmark.WorkflowOutcome` produced by the pure function
`settle_submission`.

**Fallback.** Passing no `--roles-config` to `run-episode` / `run-manifest`
leaves `env_agents=None`: `query_client`, hard checks, the contract gate and
`dealer_margin` behave byte-for-byte as if no environment LLM existed, `consult`
returns a fixed degraded-fallback string, and `workflow_review` is never called.
(The redesign draft names this switch `env_agents.enabled=false`; the shipped
mechanism is the presence or absence of `--roles-config` on the CLI, not a
config field.)

## Level-0 lattice and the legacy v2 oracle

`stochopia.benchmark.ProductDomainSpec`
(`version="stochopia.product-domain.csi-action-contract.v1"`) is a frozen
finite grid: 6 product families (`vanilla_call/put`, `barrier_call/put`,
`autocallable`, `snowball`), 14 notional fractions (0.5% to 100%), maturities
{3, 6, 12} months, strikes {0.95, 1.00, 1.05}, barriers
{0.75, 0.85, 1.10, 1.20}, coupons {4%, 8%}, participations {0.5, 1.0} and
principal-protection {False, True}. `enumerate_domain` materializes every
structurally valid combination (barrier direction is inferred from the barrier
level; snowball always uses a lower knock-in barrier and never claims
protection). A protected autocall has no knock-in barrier. An unprotected
autocall must use a lower knock-in barrier so its downside claim and payoff
agree. `custom` remains a legacy parser value but is not in the Level-0 finite
action grammar.

Legacy v2 interprets the notional fractions against client capital. Every v3
`EpisodeTask` instead freezes an explicit public notional base (CNY 10 million
by default), so the action schema cannot leak hidden client capital through its
allowed values.

A single `DOMAIN` HARD check (`validate_domain`) rejects any quote request
whose structural signature is not an exact element of this lattice. Because the
oracle (`oracle_best_quote`, via `enumerate_domain`) enumerates exactly the
same lattice through the same `TradingDesk`, every quotable agent action is by
construction an oracle candidate; `tests/test_benchmark.py` asserts this
symmetry and a dedicated property test
(`test_one_step_attainment_never_exceeds_one`,
`test_voluntary_best_submission_attainment_le_one`) asserts the resulting
attainment ratio can never exceed 1 + 1e-9.

This oracle is retained only for newly generated runs through the inherited v2
engine workflow. It is not a compatibility layer for pre-migration artifacts.
The v3 task-suite evaluation does not treat it as a teacher or a unique correct
answer; v3 reports replay-derived outcome, constraint, efficiency and lifecycle
metrics.

`oracle_best_quote` is a **one-step** frontier: at the start of each round,
before the agent acts, it prices every lattice candidate against the *current*
portfolio state and returns the maximum-`dealer_margin` feasible one. A lattice
point whose quote equilibrium does not converge is not quotable (the agent's
`request_quote` fails closed on it too), so the frontier skips it and records
the count as `RoundTrace.frontier_nonconverged_skipped`. It is not
an episode-level (hindsight/DP) upper bound and is not claimed as one; a
hindsight oracle is future work. The runner passes the environment's exact
`domain` and `quote_policy`; it no longer silently rebuilds a default grid or
default pricing policy.

## Legacy v2 information boundary, tools and per-round budgets

The tested model sees `get_round_brief()` (including the public routing field
`client_id`, required by `ProductSpec.target_client`) and can call:

- `query_client(topic)` -- deterministic in Full-information rounds (already
  disclosed) and in Partial-information rounds without an env role; routed
  through the client LLM (`query_client_llm`) when one is wired and the round
  is Partial. Budget: 3 per round.
- `consult(role, message, draft=None)` -- qualitative, non-binding chat with
  `trading_desk` / `risk_control` / `client`. With a `draft` ProductSpec and
  role in `{trading_desk, risk_control}`, a real (but not archived, not
  budget-consuming) consultative quote/check block is priced and attached so
  the role's numbers are grounded in fact. The draft passes the same
  target-client / lifecycle-state / `DOMAIN` gate as `request_quote`; a draft
  outside the lattice is not priced and the reply carries `draft_error`. Consult never spends the
  `request_quote` budget, never enters the candidate archive, and its output
  cannot be submitted directly. Budget: 3 per round. Requires `--roles-config`;
  without it every `consult` call returns a fixed degraded-fallback string.
- `request_quote(ProductSpec)` -- up to 3 per round; returns a full quote
  bound internally to the complete economic state. v3 exposes only a public
  state label and a round-local quote alias; the hidden episode-derived binding
  key never crosses the Agent boundary. A stale quote cannot be submitted.
  Structurally out-of-domain requests are rejected before the quote counter
  increments, and the `DOMAIN` error lists the allowed lattice values; an
  unquotable request (`QuoteNotConvergedError`) does not consume the budget
  either. In Partial information the quote omits `suitability`. Each v2 round
  message carries the frozen lattice as `product_domain`
  (`policy_action_schema()`; Partial v2 lists notional fractions of client
  capital, not absolute notionals, until `capital` is disclosed that round --
  the capital reply then carries `allowed_notionals`, and the schema and
  `DOMAIN` errors list them for the rest of the round).
- `submit_design(quote_id, explanation)` -- settles an already-issued quote.
- `submit_product(ProductSpec, explanation)` -- shorthand for
  `request_quote` immediately followed by `submit_design`.
- `skip_round` -- ends the round with no submission.

A submission or skip ends the round. `one_shot` is now literally one action:
only `submit_product` or `skip_round` is legal, with no protocol-repair turn and
no forced prompt. The interactive `quote_and_revise` and `ledger_archive`
strategies retain the normal action budget (`max_actions_per_round`, default 9)
and one last-chance forced-prompt turn after exhaustion; that turn accepts only
`submit_design` against an already-issued quote or `skip_round`. See Metrics
below for how this is scored.

The model never receives a raw option chain, future observations, hidden
client fields, cleaning rules or an unrestricted pricing calculator. Any
environment-role LLM is grounding-checked: its narrative may only contain
numbers copied from the facts it was handed (`FormalFacts`,
`validate_grounding`) and may only cite `check_id`/fact ids it was actually
given; a violation triggers one repair turn, then degrades to the role's
`failure_policy` action.

## Deterministic checks

Two independently-evaluated check families gate a submission; both are
required for acceptance.

**Quote-time HARD checks** (`Quote.checks`, computed by `TradingDesk.quote`;
`quote.hard_pass` is true iff every check with `severity == "HARD"` passes):
`DOMAIN` (lattice membership), `TARGET_CLIENT_MATCH`, `CLIENT_ACCEPTING`, `CLIENT_CAPITAL`,
`CLIENT_MATURITY`, `CLIENT_PRODUCT_WHITELIST`, `CLIENT_LOSS_BUDGET_V2`
(continuous loss proxy, see below), `CLIENT_PROTECTION` (a
principal-protection-seeking client requires a deterministic bond floor of 1),
`PROTECTION_CLAIM` (a declared `principal_protected=true` must match the
structural payoff floor within `description_tolerance_bp`, default 1 bp),
`DESCRIPTION_PROTECTION` (a natural-language protection claim in `pitch`, e.g.
"保本"/"本金保障", must match the structural floor), and five post-trade
portfolio budget checks (`PORTFOLIO_NOTIONAL`, `PORTFOLIO_NET_DELTA`,
`PORTFOLIO_GROSS_DELTA`, `PORTFOLIO_NET_VEGA`,
`PORTFOLIO_DEALER_STRESS_LOSS`) against the frozen exposure limits and the
authoritative dealer risk-capital account. Dealer stress capacity is cumulative
with the active portfolio and prior realised hedged P&L. Any HARD failure makes
the quote non-executable; an LLM (structurer or any env role) cannot waive it.

**Settlement-time CONTRACT checks** (`client_contract_pass`, `severity ==
"CONTRACT"`, evaluated only at `submit_design`/`submit_product` against the
already-priced quote): `CONTRACT_TARGET_CLIENT_MATCH`, `CONTRACT_ACCEPTING`, `CONTRACT_CAPITAL`,
`CONTRACT_MATURITY`, `CONTRACT_WHITELIST`, `CONTRACT_PROTECTION`,
`CONTRACT_LOSS` (the same continuous loss proxy as `CLIENT_LOSS_BUDGET_V2`,
re-derived at settlement) and `CONTRACT_HURDLE` (`hurdle_hit_prob >=
client.min_hit_prob`). This is the deterministic client gate; no LLM's stated attitude overrides it.

Settlement: `accepted = hard_executable AND client_contract_pass`, where
`hard_executable = quote.hard_pass`. `submit_design`'s return dict carries
`hard_executable`, `client_contract_pass`, `accepted`, `hard_failures`,
`contract_failures` and `dealer_margin` (0 when not accepted) individually --
they are never collapsed into one boolean before being recorded.

## Continuous client loss proxy (`CLIENT_LOSS_BUDGET_V2` / `CONTRACT_LOSS`)

`stochopia.pricing.client_loss_measure` reports
`observed_loss_frac = max(expected_loss_frac, premium_at_risk_frac,
stress_loss_frac)` against `client.max_loss_pct`. All three legs use the
client's actual cash funding denominator once a quote exists:

- `expected_loss_frac`: Monte Carlo expected loss for coupon-style structures
  (`pricing_details.expected_loss_frac`), converted from face-value units to
  cash-outlay units for funded notes.
- `premium_at_risk_frac`: `max(cash_outlay - protected_amount, 0) /
  cash_outlay`. A standalone option can therefore lose 100% of its premium;
  a client paying 106 for a minimum redemption of 100 has a 6/106 contractual
  funding gap rather than a false zero.
- `stress_loss_frac`: `client_value_loss / cash_outlay`, where
  `client_value_loss` is the worst client-long value decline on the frozen
  three-scenario grid.

This is a **continuous loss proxy**, explicitly not a mathematical worst-case
bound: it blends an expectation, a premium-at-risk figure and a 3-point stress
grid rather than enumerating the true worst path. `ClientLossMeasure` names
this contract in code.

## Funding, bilateral stress and lifecycle accounting

`ProductSpec` separates risk exposure from client funding:

```text
notional                         # risk/participation exposure
funding_style                    # premium_paid | funded_note
face_value
issue_price_pct
protected_amount                 # deterministic payoff floor
```

Every quote additionally reports `cash_outlay`, `premium` and `dealer_fee`.
For a standalone premium-paid option, `cash_outlay = premium = quoted price`.
For a Level-0 funded note, issuance is at par
(`issue_price_pct = 1`, `cash_outlay = face_value`) and `premium = null`.
`CLIENT_CAPITAL` and `CONTRACT_CAPITAL` check cash outlay; portfolio risk limits
continue to check exposure notional. A protection claim must cover actual cash
outlay and be supported by the deterministic payoff floor.

The current Level-0 grammar does **not** yet solve participation or coupon to a
target par-note economics. Consequently a par funded note can have negative
dealer inception margin. That value is retained, never clamped or relabelled
as profit; a payoff solve-for is required before this can be treated as a
production training objective.

Stress quantities are directionally separate:

```text
client_pnl                  = stressed_fair_value - base_fair_value
dealer_liability_pnl        = -client_pnl
dealer_hedged_pnl           = dealer_liability_pnl + static_delta_hedge_pnl
client_value_loss           = max(-client_pnl)
dealer_unhedged_liability_loss = max(-dealer_liability_pnl)
dealer_hedged_stress_loss   = max(-dealer_hedged_pnl)
```

The client loss gate uses `client_value_loss`. Dealer quote/risk-budget inputs
use `dealer_hedged_stress_loss`, whose provenance is explicitly
`static_delta_v1`. The scenario ledger enforces
`client_pnl + dealer_liability_pnl == 0`; the static-delta proxy is not called a
realised hedge result.

Accepted dynamic trades debit `ClientAccount.available_cash`, credit
`locked_cash`, credit dealer issue proceeds and record an issuance cashflow.
Maturity, knock-out and autocall closures release client capital, pay the
contract settlement, update both accounts and append a `LifecycleEvent`.
The lifecycle ledger reports client realised P&L, equal-and-opposite dealer
liability P&L, and a monthly frozen-delta hedge P&L before transaction costs.
Dealer equity is cash less the active liability fair value; realised hedged
P&L adjusts future dealer risk-capital capacity. Explicit client-capital
schedule changes are external account cashflows and do not erase prior P&L.

Every transition measures client net-liquidation-wealth change, locked-capital
release and dealer hedged-stress-capital release. A closure also makes
`terminal_lifecycle_pnl` available using dealer realised hedged P&L before
transaction costs. The last episode round liquidates every still-active
contract at its current deterministic fair value; no position is silently
dropped as `open_at_horizon`. `info["episode_summary"]` and the Agent run
summary contain cumulative raw/normalized rewards and constraint counts.
Actual hedge trades, slippage, transaction costs and `dealer_total_pnl` after
costs remain unavailable and are not imputed as zero.

## Risk budget calibration

`stochopia-benchmark calibrate-budget` runs `calibrate_risk_budget`: it scales a
base `RiskBudget` by a grid of factors (default 0.1 to 3.0 step 0.1, or an
explicit `--factors` list), prices every lattice candidate against each
development episode, and selects the factor whose feasible-candidate rate
falls closest to the midpoint of the target band `[0.20, 0.40]`; non-converged
candidates are excluded from the rate and counted per grid row as
`skipped_nonconverged`. The full grid
and a `"freeze this result before evaluating held-out episodes"` warning are
written to `--report-output`; the selected budget is written to
`--budget-output`. Both budget and margin calibration resolve the client's
`round_overrides` separately for every snapshot and enumerate the domain for
that resolved client.

`calibrate-budget`, `run-episode` and `run-manifest` all accept an optional
`--quote-policy-json` (a JSON-serialised `stochopia.pricing.QuotePolicy`, e.g.
`data/derived/quote_policy.v2.candidate.json`): when given, quotes are priced
with that policy instead of `QuotePolicy()`'s code defaults (`calibrate-budget`
passes it through to `calibrate_risk_budget`'s new `policy` kwarg;
`run-episode`/`run-manifest` pass it to `LongHorizonEnvironment`).

All calibration numbers created before the funding, bilateral-stress and quote
equilibrium migration are stale and must not be used for a result claim.
`data/derived/*v2*` files are historical inputs, not a current frozen v3
calibration bundle.

## `dealer_margin` economics

`stochopia.pricing.quote_economics` replaces the old constant 1% markup with a
cost-plus structure. For a quote of notional `N` and fair value `F` (`f =
F/N`):

- `V = |vega_pct|` (vega per 1 vol point, as a fraction of notional).
- `P` = path dependence: 0 for vanilla; otherwise `clip(2*sqrt(mean(p*(1-p))),
  0, 1)` over the structure's MC touch/knock-in/knock-out event probabilities.
- `B = 2 * 1.96 * pv_se_frac`, an approximate 95% Monte Carlo uncertainty band
  on the price estimate, as a fraction of notional.
- `L = dealer_hedged_stress_loss / N`.
- `Q = post_notional / capacity_notional` (post-trade notional utilization;
  `capacity_notional = risk_budget.notional`).

```
r_c = clip(a_f*f + a_v*V + a_p*P + a_b*B,                         0, client_cap)
r_h = clip(b_f*f + b_v*V + b_p*P + b_b*B + b_l*L + b_q*Q^2,        0, hedge_cap)
suitability = clip(s_type^w_type * s_protect^w_protect
                    * s_maturity^w_maturity * s_hurdle^w_hurdle,   0, 1)
suggested_price = F + N * r_c * suitability
cash_outlay     = suggested_price                    # premium_paid
cash_outlay     = face_value * issue_price_pct       # funded_note
hedging_cost   = F + N * r_h
dealer_fee     = cash_outlay - F
dealer_margin  = cash_outlay - hedging_cost          # can be negative; never truncated
```

`suitability` (docs/redesign/DESIGN_REFERENCE.md O§3.2) is a four-factor product of client fit: product
whitelist hit (`s_type`, 1.0 or 0.20), protection-preference match
(`s_protect`, 1.0 or 0.10), maturity fit (`s_maturity`, 1.0 within mandate,
linear decay beyond it), and hurdle-probability margin (`s_hurdle`, 1.0 once
`hurdle_hit_prob >= min_hit_prob`, otherwise the ratio). It both compresses the
client-side markup and (independently) feeds the `CONTRACT_*` gate, so a
mis-sold product is penalised on price and can still be hard-blocked at
settlement. The `b_q*Q^2` term makes "always quote the maximum notional" a
strictly dominated strategy as utilization approaches the risk budget.

For premium-paid products, price, hurdle probability and hurdle suitability
form a fixed point. `solve_quote_equilibrium` uses a bounded bisection plus a
canonical refinement and returns price/probability residuals. Runtime fails
closed on non-convergence. `evaluate_quote_policy` calls the same solver and
counts non-converged candidates rather than silently using the old two-pass
approximation. Funded notes pass through the same function at their fixed par
cash price. `MarketSnapshot.trend_alpha` is propagated into `MarketState` and
is part of the hurdle cache key.

The coefficients currently frozen in code (`stochopia.pricing.QuotePolicy`
defaults) are: `a_f=0.030, a_v=0.25, a_p=0.002, a_b=0.75`,
`b_f=0.020, b_v=0.75, b_p=0.004, b_b=1.00, b_l=0.003, b_q=0.015`,
`client_cap=0.08, hedge_cap=0.10`, `suit_whitelist_miss=0.20,
suit_protection_miss=0.10`, all `suit_w_* = 1.0`.

`stochopia.pricing.calibrate_quote_policy` grid-searches an overall scale on the
`a_*` markup coefficients (holding `b_*` fixed) against a pre-registered
target: 20-60% of development candidates have positive margin, and the margin
ranking across structures is non-degenerate (the largest-notional vanilla is
not always optimal). It is covered by
`tests/test_pricing_economics.py::test_calibrate_quote_policy_hits_target_band`
and wired to `stochopia-benchmark calibrate-margin`, which samples a fixed-seed
subset of the shared `ProductDomainSpec` lattice per (episode, round)
development snapshot (`--candidates-per-snapshot`, default 30 -- the full
lattice is tens of thousands of candidates, far more MC than a few minutes of
wall time affords), builds dev cases from those candidates against the real
market snapshots, runs `calibrate_quote_policy`, and additionally reports
0.8x/1.0x/1.2x sensitivity of the *selected* policy (scaling its `a_*` a
second time on top of the calibrated factor).

No post-migration calibration result is frozen in this revision. A new bundle
must include the input manifest, per-round
client resolution, selected policy/budget, the full grid and sensitivity
report before held-out evaluation begins.

## LLM-native roles

Four roles, each with an independent system prompt, model and conversation
history. The tested structurer is not configured in `config/benchmark_roles.yaml`:
its model is `run-episode --model` / the manifest job model, its prompt is the
packaged `stochopia/resources/structurer.md`, it runs at temperature 0 with the
`config/models.yaml` `max_tokens`, and its seed is `--seed` / the job seed. The
three env NPCs are configured in `config/benchmark_roles.yaml`
(`protocol_version: stochopia.csi.legacy.v1`, loaded by `stochopia.role_config` with
fail-fast validation; a `structurer` role there is rejected):

| role id | role | model | temperature | history scope | numeric authority | failure policy |
|---|---|---|---|---|---|---|
| `client_main` | env NPC | `deepseek-v4-flash` | 0.2 | episode | `supplied_facts_only` | `abstain` |
| `risk_control` | env NPC | `deepseek-v4-flash` | 0.0 | round | `supplied_facts_only` | `escalate` |
| `trading_desk` | env NPC | `qwen-max` | 0.0 | round | `supplied_facts_only` | `decline` |

`deepseek-v4-flash` is both the `client_main` and `risk_control` NPC and a
documented tested model: when it is the structurer under test, its
counterparties are the same model, so its results are not an independent
comparison and must be flagged separately in reports.
`risk_control` and `trading_desk` must use `history_scope: round`
(`role_config` rejects `episode`): the submission review hands them the
unredacted quote, and round scope keeps it out of later consult replies.

The three env-NPC roles are pinned as `main_npc_lineup_id: stochopia.npc-lineup.fixed.v1`: only
the structurer's model varies across a benchmark run, so a tested model never
also gets a different negotiation counterparty. `numeric_authority:
supplied_facts_only` is enforced both at config-load time (`role_config`
rejects any client/risk/desk role that isn't `supplied_facts_only`) and at
call time (`validate_grounding`): these three roles can never invent a number
not present in the `FormalFacts` handed to them for that call. The structurer
has no such restriction (`numeric_authority: none`) since it is the model under
test.

`FrozenEnvAgent` (`stochopia.env_agents`) wraps one role: strict JSON-schema
parsing (`stochopia.role.client-response.v1` / `stochopia.role.risk-response.v1` / `stochopia.role.trading-desk-response.v1`, one
of `answer/counter/accept/reject/abstain`,
`approve/request_revision/escalate`, `issue/decline/request_revision`
respectively) with one format-repair retry, then grounding validation with one
repair retry; any remaining failure returns a `degraded=True` response whose
`action` is the role's `failure_policy`. Only provider/transport failures
(`LLMError`, `httpx.HTTPError`, timeouts) degrade; programming errors raise.
A `consult("client", ...)` runs in its own conversation scope, never sharing
history with `query_client_llm` / `workflow_review` turns that carry the hidden
profile; the `client_decision` review in `workflow_review` has a third scope, so
later topic queries are not answered from a history holding the full profile. Seeds derive
deterministically from `(episode_id, round_num, role_id, turn_id)` via
`stats.derive_seed` plus the role's `seed_offset`, so replaying a frozen
episode reproduces the same seed sequence. An `EnvResponseCache`
(`--env-cache-dir`, append-only JSONL) keys on
`sha256(inference_contract|protocol_version|role_id|model|temperature|seed|
max_tokens|output_schema|canonical(messages))`, so an identical
request from a different tested model hits the same frozen NPC reply, and
every accepted role call is written to the cache for audit/replay. The cached
value (`stochopia.env-response.v2`) is every history turn after the request,
including format/grounding repair turns, so a replay rebuilds the same history
and later keys still hit.

Submitted content handed to any env role or the judge is wrapped as an
explicit untrusted-data block with an instruction to treat it as text, not
instructions, guarding against prompt injection from a tested model's
`explanation`/`pitch` fields.

## Two-tier outcomes

**Primary (deterministic, always computed):** `hard_executable`,
`client_contract_pass`, `accepted`, `dealer_margin`, `hard_failures`,
`contract_failures`. Independent of whether any env-role LLM is wired in.

**Secondary (`WorkflowOutcome`, only when env roles are wired and a submission
happened):** `workflow_review` asks the desk, risk and client roles to react to
the settled quote; `settle_submission` (a pure function) sets
`workflow_deal = hard_executable AND desk_action=="issue" AND
risk_action=="approve" AND client_action=="accept"`. A role that could not be
reached (`None`) or returned a degraded response counts as non-affirmative and
sets `degraded=True`. `workflow_deal` never feeds back into the primary
settlement or the portfolio; it is reported as `workflow_deal_rate` and
`llm_action_vs_formal_truth` (the rate at which the workflow decision disagrees
with the deterministic `accepted` outcome) -- a divergence-rate diagnostic, not
a correctness label, since neither side is ground truth for the other.

## Legacy v2 canonical metrics (`compute_metrics`, `stochopia.benchmark_cli.CANONICAL_METRICS`)

Every round's submission has exactly one `submission_origin`:
**voluntary** (submitted within the normal per-round action budget),
**forced_prompt** (submitted only on the last-chance prompt after the budget
was exhausted), or **none** (explicit skip, exhausted budget with no valid
forced-prompt response, or an invalid action). There is no environment
auto-submission path; a round with no valid model action settles as `none`
with `dealer_margin=0`.

- `hard_execution_rate`: hard-executable submissions divided by all rounds;
  this always-defined value is the pre-registered execution primary, so skips
  and protocol failures count as non-execution rather than creating
  condition-dependent missing data.
- `hard_execution_rate_given_submission`: hard-executable submissions divided
  by all submissions (`None` when there was no submission).
- `contract_acceptance_rate_given_hard_pass`: contract-passing submissions
  divided by hard-executable submissions (`None` when none passed HARD).
- `settlement_acceptance_rate`: final accepted rounds divided by all rounds,
  pooling voluntary and forced-prompt settlements. The old, misleading
  `hard_feasibility_rate` key is retained only as a deprecated serialized alias
  of this value.
- `total_dealer_margin` and
  `mean_dealer_margin_per_voluntary_accepted_trade`: sum/mean of
  `dealer_margin` over **voluntary**-accepted rounds only (the mean is `None`
  when there is no such round).
  `mean_dealer_margin` remains a deprecated alias; `forced_prompt_margin` is
  reported separately and never pooled into these.
- `voluntary_submission_rate`, `forced_prompt_rate`, `no_submission_rate`:
  the three-way partition of `submission_origin` over all rounds (sums to 1).
- `one_step_attainment` (formerly `oracle_margin_attainment`): mean of
  `dealer_margin / one_step_frontier_margin` over rounds that are voluntary and
  accepted and have a positive frontier; `None` if no such round exists. A
  ratio above `1 + 1e-9` is a protocol error, not silently reported. The
  serialized `oracle_margin` field remains a deprecated trace alias. This is
  not an episode-level upper bound -- see the oracle definition above.
- `quote_failures`, `repeated_hard_violations`, `revision_success_rate`,
  `failure_counts`: request_quote-level diagnostics, independent of
  `submission_origin`.
- `imputed_counterfactual_margin` (per round, in the trace only, **not**
  aggregated by `compute_metrics`): the best archived feasible quote's margin
  from the round's `request_quote` calls, whether or not the model ever
  submitted it. Read-only diagnostic -- never calls `submit_design`, never
  touches portfolio/client state, never enters any primary or secondary
  metric.
- `workflow_deal_rate`, `llm_action_vs_formal_truth`, `degraded_consult_rate`:
  secondary dialogue-layer metrics, `None` when no env roles were wired /
  no consult happened (see Two-tier outcomes above).

`aggregate`'s per-(model, condition) table reports mean +/- a cluster
(episode-level) bootstrap `1 - --alpha` CI (default 95%) for canonical metrics including the three
separated execution/contract/settlement rates, `total_dealer_margin`,
`one_step_attainment` and the submission-origin rates; a metric missing from
every row in a results directory is reported as fully missing rather than
silently defaulting. A cell with one episode (or a contrast with one pair) has
no estimable bootstrap interval: `bootstrap_ci` / `cluster_bootstrap_ci` return
`None` bounds and the markdown prints `CI n/a` instead of `[mean, mean]`
(`aggregate-v3` writes `null` / empty `ci_low`, `ci_high`).

## Offline judge protocol

Judging never feeds back into a run: `stochopia-benchmark judge-runs` reads
already-frozen `*.json` results and writes independent `<file>.judges.json` +
`judge_manifest.json`, never touching the original result files.

- **Population**: only rounds with `submission_origin == "voluntary"` and a
  non-empty `submitted_product` in result files with `complete: true` are
  eligible (`_collect_voluntary_samples`) -- forced_prompt and unsettled rounds
  are never judged.
- **Sampling**: a deterministic `(condition, model)`-stratified,
  round-robin sample of `--sample` (default 60) submissions
  (`_stratified_sample`), seeded from `--seed` so re-running with the same
  inputs reproduces the same picks regardless of filesystem globbing order.
- **Blinding**: `build_judge_input` whitelists only fields the tested model
  itself could see (`CLIENT_BRIEF_ALLOWED_KEYS`) or produced
  (`PRODUCT_ALLOWED_KEYS`) -- model/vendor identity, `dealer_margin`,
  `oracle_margin`, hard/contract PASS-FAIL and `submission_origin` are dropped
  by construction, never copied into the judge prompt. A `blind_id` is a
  truncated SHA-256 of `submission_id + --salt`, one-way and never reversible
  from a leaked `judges.json`.
- **No self-judging**: an entry is skipped (`status: "skipped_self_judge"`)
  whenever `judge_model == item["model"]` (exact name match). Two judge
  models are required (`--judge-models`, exactly two distinct names), each
  called `--repeats` times (default 3) per non-self-judged submission.
- **Evidence-span guard**: `parse_judge_result` requires exactly the 6
  dimensions (`client_understanding`, `soft_suitability`, `risk_explanation`,
  `non_misleading`, `hedging_rationale`, `commercial_communication`), each a 0-4
  integer with non-empty `evidence`/`reason`; `validate_evidence_spans`
  downgrades any dimension whose `evidence` is not a literal
  (whitespace-normalized) substring of one submitted text the judge was shown
  (explanation, `product.pitch` or `product.hedging_plan`) to `score=None`
  (missing), guarding against fabricated evidence. Only provider/transport
  errors and malformed judge output become `missing` after retries;
  programming errors abort the batch.
- **Reliability, not validity**: `aggregate`'s markdown output appends a
  "Judge" section (median/IQR/missing-rate per dimension via
  `aggregate_judge_bundle`, no composite score) from the `*.judges.json` files
  of the latest `judge-runs` only (source file selected in `judge_manifest.json`
  and the same seed/salt/judge models/repeats; stale files are ignored), plus inter-judge reliability between the two judge models on
  aligned, fully-scored `(submission_id, repeat)` pairs: exact total
  agreement, Spearman correlation of totals, and per-dimension quadratic
  weighted Cohen's kappa. `reliability_summary`'s output literally carries the
  claim string `"inter-judge reliability only; not expert-grounded validity"`
  -- Stochopia does not claim expert-grounded validity for the judge, only
  agreement between the two configured judge models.

**Config-file defaults**: `config/benchmark_roles.yaml`'s `judges:` block
(`models`, `repeats`, `temperature`, `max_tokens`) is
loaded by `stochopia.role_config.load_judges_config` into a `JudgesConfig`
(unknown fields, model-count and registry-membership checks all fail fast).
The shipped judges are `deepseek-v4-pro` and `qwen-max`; `qwen-max` is also the
`trading_desk` NPC, so the judge pair is not independent of the NPC lineup (no
registry model is outside both the NPC lineup and the documented tested-model
lineup).
`judge-runs --roles-config config/benchmark_roles.yaml` reads it to fill in
`--judge-models`/`--repeats` when those flags are omitted; an explicit
`--judge-models` and/or `--repeats` always overrides the file's value, and
`--judge-models` is still required if `--roles-config` is not given (or its
`judges.models` doesn't have exactly two entries). `judge_soft_quality` still
hardcodes `temperature=0.0` with no `max_tokens` cap regardless of the file's
`temperature`/`max_tokens` values -- those two fields are recorded for
documentation but not yet applied to the judge call. There is no
`exclude_same_model_family` field: the self-judge skip in `_run_judge_runs`
compares `judge_model == item["model"]` by exact name, since inferring model
family from a name string is unreliable; `load_judges_config`/`load_role_specs`
reject that key as unknown.

## Statistics

- **Seeding**: every stochastic routine (bootstrap resampling, permutation
  sign-flips, sample selection) derives its seed from `stats.derive_seed`, a
  SHA-256-based deterministic derivation over an explicit namespace and parts
  -- never Python's built-in `hash()`, which is per-process randomized.
- **Replicate design**: `stochopia-benchmark make-manifest` builds every
  (episode, model, strategy, condition) cell with `n>=3` replicates, and
  over-samples `partial_dynamic` (full observability loss plus market drift,
  the hardest cell) to `n=5` (`DEFAULT_REPLICATES_BY_CONDITION`); replicate
  seeds derive from `(episode_id, model, strategy, condition, replicate)`.
- **Cell CIs**: `aggregate` reports each (model, condition) cell's mean with a
  cluster (episode-level, two-stage) percentile bootstrap 95% CI
  (`cluster_bootstrap_ci`, default 10,000 resamples), respecting within-episode
  correlation instead of treating rounds/replicates as independent.
- **Paired condition contrasts**: `paired_condition_contrasts` first collapses
  replicates to one mean per (episode, model, strategy, condition) cell, forms
  matched differences across conditions within each (episode, model,
  strategy) base, then averages every correlated model/strategy/information-
  level contrast sharing an episode to exactly one value per episode before
  inference. The two pre-registered contrasts are
  `dynamic_degradation` (static minus dynamic, within each information level)
  and `partial_observability_degradation` (full minus partial, within each
  horizon) -- and reports a percentile bootstrap CI, a two-sided Wilcoxon
  signed-rank test (exact DP null distribution for `n<=25` non-zero
  differences, normal approximation with tie correction otherwise) and a
  paired sign-flip permutation test for each.
- **Multiple comparisons**: Holm-Bonferroni step-down correction is applied
  within each model x metric's 2-contrast family (`holm_adjust`); `aggregate`'s
  markdown flags `holm_p < alpha` with `*`.
- **Pre-registered primary outcomes**:
  `["hard_execution_rate", "settlement_acceptance_rate",
  "total_dealer_margin"]` (`manifest_payload`'s
  `protocol.primary_outcomes`).
  Every other metric/contrast is exploratory.

## Reproducibility: two separate Stochopia v1 claims

These claims apply only to artifacts created with the current Stochopia v1
schemas and seed namespaces; pre-migration artifacts are incompatible.
Stochopia does not claim bit-level reproducibility from a fresh API call to a
live model endpoint (provider/model updates and non-determinism persist even
at temperature 0). It makes two narrower, implemented claims instead:

- **`economic_replay`**: given a frozen episode trace (the recorded action
  sequence and the market snapshots), the deterministic core -- pricing, hard
  checks, the contract gate, `dealer_margin`, the separated execution/
  contract/settlement rates,
  `one_step_attainment` -- is a pure function of that input and recomputes
  identically every time, with no LLM call involved.
- **`conversation_replay`**: env-role dialogue is reproducible from the
  `EnvResponseCache` JSONL artifact (keyed by `inference_contract|
  protocol_version|role_id|model|temperature|seed|max_tokens|output_schema|
  canonical(messages)`):
  replaying the same request sequence against the same cache file reproduces
  the exact same NPC replies byte-for-byte, without re-calling any model.

## Reproduction

```bash
# Convert a public daily close file into leakage-safe month-end features.
python -m stochopia.benchmark_cli build-market daily.csv market_snapshots.csv

# Validate schema, provenance and contiguous episode rounds.
python -m stochopia.benchmark_cli validate-market market_snapshots.csv

# Freeze complete v3 inputs. The final row of every selected episode is the
# terminal valuation snapshot rather than a policy decision.
python -m stochopia.benchmark_cli make-v3-suite market_snapshots.csv \
  --episodes CSI500_2024H1 CSI500_2024H2 \
  --client-json client.json --risk-budget-json risk_budget.json \
  --name stochopia-dev --version 1 --split dev \
  --output tasks/stochopia-dev.v1.json

# Run one external policy over every frozen task. Each trajectory is
# scored from its recorded transitions before aggregation.
python -m stochopia.benchmark_cli run-v3-suite tasks/stochopia-dev.v1.json \
  --agent-command 'python my_agent.py' \
  --replicates 3 \
  --output-dir outputs/stochopia-dev \
  --bootstrap-resamples 10000

# A separately received trajectory can be scored against the same suite;
# add --replay to re-run every action and require exact economic equality.
python -m stochopia.benchmark_cli evaluate-trajectory \
  outputs/stochopia-dev/0000-example.trajectory.json \
  --suite tasks/stochopia-dev.v1.json \
  --output outputs/example.evaluation.json

# Aggregate evaluations from one suite.
python -m stochopia.benchmark_cli aggregate-v3 outputs/stochopia-dev \
  --output-json outputs/stochopia-dev.aggregate.json \
  --output-csv outputs/stochopia-dev.aggregate.csv

# The remaining commands run inherited v2 engine workflows and emit Stochopia v1 artifacts.
# Smoke-test the deterministic desk with the synthetic example.
python -m stochopia.benchmark_cli demo \
  scenarios/stochopia_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --product-json scenarios/stochopia_csi/product.example.json --full

# Calibrate and freeze the risk budget on development episodes only.
# --quote-policy-json is optional (e.g. data/derived/quote_policy.v2.candidate.json);
# omit it to price candidates with the default QuotePolicy.
python -m stochopia.benchmark_cli calibrate-budget market_snapshots.csv \
  --episodes CSI500_2023H1 CSI500_2023H2 \
  --client-json scenarios/stochopia_csi/client.example.json \
  --base-budget-json scenarios/stochopia_csi/risk_budget.example.json \
  --report-output data/derived/budget_calibration_report.json \
  --budget-output data/derived/risk_budget.frozen.json

# Calibrate and freeze the QuotePolicy markup scale (stochopia.pricing.calibrate_quote_policy)
# on development episodes only, then report 0.8x/1.0x/1.2x sensitivity of the
# selected policy.
python -m stochopia.benchmark_cli calibrate-margin market_snapshots.csv \
  --episodes CSI1000_2023H1 CSI500_2023H1 \
  --client-json scenarios/stochopia_csi/client.example.json \
  --risk-budget-json data/derived/risk_budget.frozen.json \
  --report-output data/derived/quote_policy_calibration_report.json \
  --policy-output data/derived/quote_policy.v2.candidate.json

# Run one registered model against one frozen episode; --roles-config and
# --quote-policy-json are both optional (omit --roles-config to stay
# pure-deterministic; omit --quote-policy-json to use the default QuotePolicy).
python -m stochopia.benchmark_cli run-episode market_snapshots.csv \
  --episode CSI500_2023H1 \
  --client-json scenarios/stochopia_csi/client.example.json \
  --risk-budget-json data/derived/risk_budget.frozen.json \
  --model deepseek-v4-flash --strategy ledger_archive \
  --roles-config config/benchmark_roles.yaml \
  --quote-policy-json data/derived/quote_policy.v2.candidate.json \
  --env-cache-dir outputs/env_cache \
  --output outputs/csi500_2023h1.json

# Freeze the complete factorial job manifest (n>=3, partial_dynamic n=5)
# without spending API budget.
python -m stochopia.benchmark_cli make-manifest \
  --episodes CSI500_2023H1 CSI500_2023H2 CSI500_2024H1 CSI500_2024H2 \
             CSI500_2025H1 CSI500_2025H2 CSI1000_2023H1 CSI1000_2023H2 \
             CSI1000_2024H1 CSI1000_2024H2 CSI1000_2025H1 CSI1000_2025H2 \
  --models gpt-4o claude-sonnet deepseek-v4-flash minimax \
  --output outputs/experiment_manifest.json

# Execute a frozen manifest sequentially. Each result records its `inputs`
# (resolved market CSV path, client profile, risk budget, QuotePolicy, env-role
# model/temperature/seed/prompt text, the tested model's registry resolution);
# reruns skip outputs already marked complete for the same job_id and identical
# inputs; anything else is overwritten. Writes are atomic. A failed job
# (provider error, metric computation, ...) is written as `complete: false,
# status: "failed"` with its error and partial trace and does not abort the
# batch. The structurer prompt and code are not recorded, so change the output
# directory after editing them.
# --quote-policy-json is optional (default QuotePolicy if omitted).
python -m stochopia.benchmark_cli run-manifest market_snapshots.csv \
  --manifest outputs/experiment_manifest.json \
  --client-json scenarios/stochopia_csi/client.example.json \
  --risk-budget-json data/derived/risk_budget.frozen.json \
  --roles-config config/benchmark_roles.yaml \
  --quote-policy-json data/derived/quote_policy.v2.candidate.json \
  --env-cache-dir outputs/env_cache \
  --outputs-dir outputs/csi_benchmark

# Aggregate results into CSV + markdown (cell CIs, paired contrasts, Holm).
# --manifest adds per-cell complete/failed/missing job counts.
python -m stochopia.benchmark_cli aggregate outputs/csi_benchmark \
  --manifest outputs/experiment_manifest.json \
  --output-csv outputs/aggregate.csv --output-md outputs/aggregate.md

# Offline blind judge batch over frozen voluntary submissions (never
# touches the original result files). --judge-models/--repeats are optional
# when --roles-config is given: they then default to config/benchmark_roles.yaml's
# judges.models / judges.repeats; an explicit flag always overrides the file.
python -m stochopia.benchmark_cli judge-runs outputs/csi_benchmark \
  --roles-config config/benchmark_roles.yaml \
  --salt "$(openssl rand -hex 16)"
```

Only abstract workflow, schemas and synthetic rules may be derived from private
institutional material. No private document, client, institution, limit or
identifying metadata may enter released data or prompts.

## 2026-11-10：世界模型生成器门槛改为层级命名

StructurerDSL `docs/SPEC.md` 11.2 的生成器门槛不再用 G5–G7 编号，改为四个累进层级：草稿层（`draft`）、
自洽层（`coherent`）、预测层（`predictive`）、决策层（`decision_grade`）。原 G5 与 G6 合并为预测层，原 G7 为
决策层，各自的阈值未改；自洽层和留出数据的计次规则为新增。作为声明证据的要求因此变严：多了自洽层，且必须
按层级顺序通过。自洽层的检验和计次台账都尚未实现，在此之前没有生成器能越过草稿层。下文更早条目里的"G5–G7"
按此对应。

`stochopia.research.world_models` 注册表的 `generator_gates` 字段改为 `tier_checks`，新增 `tier` 属性；
`world_model_generator` 的 manifest 相应以 `tier`、`tier_checks` 取代 `generator_gates`，`development_only`
保留，schema 标识与 seed 派生不变，生成的路径逐位不变。`r02s01` 与 `ewma_gbm_reference` 都在草稿层，仍是
development-only。此项只涉及世界模型研究线，不修改 Benchmark v3 的定价、结算、约束或指标，已有 episode 结果不受影响。

## 2026-11-09（三）：v2 隐藏画像泄露、格点披露与结果管线修复

Level-0 定价改动（MC 不把 t=0 计为已观测敲入、公允价值 ≤ 0 的 premium-paid 结构不可报价）见下一节
"2026-11-09（二）"的 `PRICING_VERSION` 说明，本节只列 v2 与共用后端的其余改动。已有 v2 结果：partial 条件下的结论受
下列泄露污染、`one_step_attainment` 与依赖 MC 敲入 delta/压力的 HARD 结论受定价改动影响，应重跑；
`inputs` 结构改变，旧结果不再匹配续跑，会被重跑。

- **partial 条件不再泄露隐藏画像**：公开报价（`public_quote_payload`，v2/v3、forced prompt、consult
  共用）在 partial 条件下删去 `suitability`（`12/(2−suitability)` 可还原 `max_maturity_months`，白名单、
  保本、`min_hit_prob` 也可由其因子解码）；`submit_design` 返回的 `client_account` 删去
  `initial_cash`、`available_cash`、`book_wealth`、`wealth`、`net_liquidation_wealth` 与外部资金流字段；
  `policy_action_schema()` 在 v2 无公开名义基数时不再给出 `client.capital` 及其绝对名义金额，只给比例
  （`notional_rule.kind = fraction_of_client_capital`），直到本轮 capital 被披露（见下条）。已知残差：premium-paid 结构的 `client_price`、
  `dealer_margin`、`margin_rate` 经均衡依赖 `suitability`（`client_price − F = N·r_c·suitability`），
  知道 `QuotePolicy` 系数的策略可近似反推；funded note 的现金价固定，margin 不含 `suitability`。
  这是定价经济学本身的信号，本轮不改定价。
- **partial 名义格点可构造**：v2 的 `query_client("capital")` 返回可用现金，而名义格点按当轮计划资金
  缩放，二者在动态条件有成交后不等（此前第 2 轮起按 可用现金×比例 构造的名义全被 DOMAIN 拒绝，
  `partial_observability_degradation` 因此混入一个与信息量无关的构造障碍）。现在本轮 capital 一经披露
  （确定性查询，或客户 LLM 的 `disclose_fields` 含 capital），回复附 `allowed_notionals`（当轮格点的
  绝对名义金额），之后本轮的 `policy_action_schema()` 与 DOMAIN 错误也列出它们；每轮重置。capital 的
  含义（可用现金）不变。旧 partial_dynamic 结果中第 2 轮起的提交率与成交率被压低，应重跑。
- **交易台/风控只允许 round 历史**：`workflow_review` 把未脱敏报价（`suitability`、客户检查的
  observed/limit）交给 trading_desk / risk_control；`role_config` 现在拒绝这两类角色的
  `history_scope: episode`（缺省值也是 episode，配置须显式写 round），保证这些内容不进入之后的 consult
  回复历史。冻结配置本就是 round，行为不变。
- **客户 NPC 会话隔离**：`workflow_review` 的 `client_decision`（携带完整画像）使用独立会话，
  不再进入 `query_client_llm` 的 episode 历史。客户角色缓存键随历史变化，旧缓存对这类请求不再命中。
- **consult 不再是无限制定价器**：带 draft 的 consult 与 `request_quote` 走同一 target / 生命周期状态 /
  `DOMAIN` 校验，格点外草案不定价并返回 `draft_error`；交易台块与正式报价同为条件感知载荷。
- **报价额度**：不可报价请求（`QuoteNotConvergedError`，含零价值 premium-paid）不消耗 `request_quote`
  额度；`oracle_best_quote`、`calibrate_risk_budget`、`calibrate-margin` 跳过并计数。
- **格点披露**：v2 每轮消息附 `product_domain`（`policy_action_schema()`），`DOMAIN` 错误附允许值；
  `structurer.md` 不再写 `maturity_months：1–60`。被测模型的 prompt 改变，旧 v2 结果与新结果不可比。
- **v2 指标**：`mean_dealer_margin_per_voluntary_accepted_trade`（及别名 `mean_dealer_margin`）在无自愿
  成交时为 `None`（此前 0.0，拉低均值）；`aggregate` 的缺失提示改为"条件未发生或字段缺失"，不再推断为
  旧 schema。
- **运行失败可见**：被测模型调用的供应商/传输错误（升额后仍截断、不可重试的 4xx、重试耗尽）使
  `run_episode` 返回 `status: "failed"` 的部分轨迹；`run-manifest` / `run-episode` 对任何失败写出
  `complete: false, status: "failed"` 的结果（含 error；`run-episode` 写出后重新抛出该异常），
  `run-episode` 改为原子写。
- **`aggregate` 汇总口径**：跳过无 `schema_version` 的旧文件；拒绝合并 `inputs`（除 `tested_model`）不同的
  结果，以及同一模型名 registry 解析不同的结果；报告失败结果（没有可汇总结果时也先写出该报告再退出），给 `--manifest` 时按 (model, condition)
  报告 expected/complete/failed/missing；`--alpha` 同时决定单元格 CI 与配对对比 CI 的水平和表头。
- **`calibrate-margin`**：`--risk-budget-json` 改为必填，容量恒取 `risk_budget.notional`（与运行时一致）；
  此前缺省时用 `client.capital`（示例 50M 对运行时 200M），校准出的 markup 尺度与运行时不一致，需要重跑。
- **结果 `inputs`**：`market_csv` 记为绝对路径；新增 `tested_model`（provider、模型 id、base_url、
  max_tokens、timeout）与 `env_roles`（各 NPC 的模型解析、temperature、seed 策略与 offset、history scope、
  prompt 路径与全文）。只做记录与续跑比较，不做哈希。
- **judge**：证据片段可取自 judge 看到的 explanation、`pitch` 或 `hedging_plan`（此前只认 explanation，
  引用 pitch 的合法证据被判缺失）；`parse_judge_result` 使用共用的 `llm_json` 解析；重试只吞供应商错误与
  格式错误。已有 judge 文件的缺失率偏高，需重跑 `judge-runs`。
- **角色配置说明**：`deepseek-v4-flash` 同时是 `client_main` 与 `risk_control` NPC，又在文档被测模型名单中；
  它作为被测模型时对手方是同一模型，结果不是独立对照，报告应单独标注，不换模型。
- **定价整理（数值不变）**：`evaluate_product` 不再计算无人读取的 `expected_payoff`（少一次 MC）；
  `fair_value_stress_profile(..., base=...)` 复用调用方已算的基准价；雪球 1.03 / autocall 1.05 触发水平
  收为 `SNOWBALL_KNOCK_OUT_PCT` / `AUTOCALL_TRIGGER_PCT`。

## 2026-11-09（二）：v3 环境泄露、不可报价与评测口径修复

环境版本升为 `v3-spine-level0.5-quote-unavailable`（策略可见历史、无效动作口径与 reset 选项变化）；
DSL 实验环境继承同一 `step()`，版本升为 `v3-spine-dsl-experimental.2`（DSL 定价版本不变），评测器按
`v3-spine-dsl-experimental.` 前缀识别 DSL 轨迹；
`PRICING_VERSION` 升为 `benchmark-pricing-v3.3-mc-unobserved-t0-unquotable`，对应同轮 Level-0 定价改动：
MC 不再把 t=0 计为已观测的敲入，公允价值 ≤ 0 的 premium-paid 结构不可报价。旧 v3 轨迹的 `--replay`
因版本不一致被拒；不带 `--replay` 的计分仍可用，但反映旧定价与旧历史内容。

- **策略历史不含归一化奖励**：`run_agent_episode` 交给策略的 `history[].reward_components` 删去各项
  `normalized_value`。归一化分母是隐藏量（客户初始资金、交易商风险资本），`value / normalized_value`
  可直接还原客户资金。原始值不变，轨迹与评测中的完整奖励向量不变。`run_agent_episode` 不把 `info`
  交给策略；但 `ScalarizedStochopiaEnv` 的 `info` 会交给 RL 智能体，因此它返回的 `reward_components`
  同样删去 `normalized_value`，`episode_summary.reward` 删去 `normalized_totals`（内层环境的 `info`
  与轨迹不变）。已知残余：标量奖励本身由归一化值加权得到，若权重选中按客户资金或风险资本归一化的分量，
  智能体用标量奖励与 `info` 中的原始值仍可反解分母；需要严格隐藏时只对 identity 归一化的分量加权。
  旧 v3 运行中策略能从历史读到客户资金，partial-information 结论可能受污染，应重跑。
- **不可报价**：`step()` 把 `QuoteNotConvergedError` 与定价层 `QuoteUnavailableError` 记为
  `quote_unavailable` 事件：`action_valid=True`、`ConstraintSignals.quote_unavailable=True`，不收
  `invalid_action_cost` 与 `quote_cost`，不消耗报价预算，不计入 `invalid_action_rate`，由
  `episode_summary.constraints.unavailable_quotes` 单独计数；仍占本轮步数上限，`submit_product` 也不结束本轮。
  DSL 实验环境中公允价值 ≤ 0 的合约同样改为 `QuoteUnavailableError`（此前为 `BenchmarkError`，记为无效动作），
  且 DSL 后端只在定价成功后才计入报价预算。
  此前不收敛的均衡被记为无效动作（−1）并消耗预算，零价值 premium-paid 结构（如 3 个月、H=0.75、
  低波动的下敲入看涨）以 `PricingError` 逃出 `step()`、成为 `infrastructure_error`。含这类步的旧结果，
  无效动作率、成本与失败计数口径不同。
- **条件比率空分母**：`hard_execution_rate_given_submission`、`contract_acceptance_rate_given_hard_pass`、
  `settlement_acceptance_rate` 分母为 0 时为 `None`（此前 0.0）；`aggregate-v3` 跳过非数值，这些指标的
  `n_runs` 只计有定义的运行。旧聚合把无提交运行按 0 计入，均值偏低。口径说明（名称不改）：v3
  `submission_rate` 是提交步数 / 总步数（按步，不是按轮）；v3 `settlement_acceptance_rate` 是被接受的提交 /
  提交数，而 v2 同名指标是被接受轮数 / 全部轮数，二者不可比。
- **聚合版本一致**：`aggregate_evaluations` 拒绝 `environment_version` 或 `pricing_version` 不一致的评测，
  并在聚合结果中写出这两个版本。
- **删除加载时自校验**：`evaluate-trajectory` 不再拒绝内嵌 manifest 重算哈希与 `metadata.task_hash` 不一致的
  轨迹（与 2026-11-08 删除 `TaskSuite.load` 校验同类）；评测结果的 `task_hash` 取自实际计分的 manifest，
  仍是分组 ID。
- **reset 选项**：`reset(options=...)` 拒绝任何键（Level 0 未实现任何 reset 选项）；此前除 `curriculum`
  外的键会被静默记录为无操作标签。
- **策略上下文窗口**：`policy_run_metadata` 增加 `history_records`、`history_chars`、`record_raw_output`
  （`stochopia.environment.policy_run_metadata`），`policy_config_hash` 因此区分不同窗口与脱敏设置，
  `aggregate-v3` 不再把它们并为同一策略，`run-v3-suite` 续跑按同一组字段匹配。旧轨迹缺这些键，
  其 `policy_config_hash` 与新轨迹不同，不会被续跑复用。
- **DSL 适配器**：删除 `_payment_bounds` 中永不触发的 NaN 检查（`check` 已把 NaN 界映射为 ±∞），行为不变。

## 2026-11-09：v2 定价/结算与评测管线复核修复

以下条目改变 Level-0 定价、结算或报告口径。已有 v2 episode 结果（含 `run-manifest` 输出、
预算/margin 校准报告）中的 barrier 报价、knocked-in autocall 结算与依赖它们的 margin、
attainment、HARD/CONTRACT 结论均已过时，需要重跑；v3 旧轨迹的 `--replay` 因 `PRICING_VERSION` 升级被拒（见下节），不带 `--replay` 的计分反映旧定价。

- **障碍期权离散监控**：`barrier_option(..., monitoring_dt=1/12)` 在 Reiner–Rubinstein 闭式解上
  应用 Broadie–Glasserman–Kou 修正（障碍位向远离现货方向乘 `exp(±0.5826·σ·√Δt)`），
  `price_product`、希腊值、压力情景与 P 测度期望统一使用；生命周期仍按月度快照监控与结算。
  已触碰状态只来自已观测的月度路径（`barrier_touched`）；delta 扰动或压力冲击后的现货只是未来
  路径起点，越过原障碍但未越过平移障碍时仍按 BGK 定价，不再被当作已敲出/敲入（否则
  K=100、H=110、S=109、剩 2 个月的上敲出看涨 delta 为 −0.79）。`delta_dollars`、净 delta
  HARD 检查与对冲 P&L 都经此路径。BGK 在只有 3–12 个月度监控日时残差明显，不应视为精确：
  σ∈[0.15, 0.3] 的发行时格点上，价值 ≥ 0.5% 名义本金的结构相对月度 MC 最大偏差约 +34%
  （K=1.0、H=1.10、3 个月、σ=0.3 的上敲出看涨：1.02% vs 0.76%），更小的价格可达 +70%；
  存续期现货离障碍 3% 时可达约 2 倍（0.146 vs 0.065 点）；上例 delta 为 −0.27，而月度 MC
  有限差分约 −0.12。连续公式的偏差更大（如 1.10 上敲出 3 个月看涨 0.40 vs 0.97 点）。
- **autocall knocked-in 结算**：结算对齐定价合约——非保本 autocall 触发敲入且未敲出时
  到期支付 `face·min(S_T/S0, 1)`，即使期末回到期初价之上也不付票息（此前结算付
  `face·(1+coupon·T)`）。雪球结算与定价原本一致，未改。
- **报价不收敛**：`TradingDesk.quote` 对不收敛的均衡抛 `QuoteNotConvergedError`；
  `oracle_best_quote` 与 `calibrate_risk_budget` 跳过这些格点并计数
  （`RoundTrace.frontier_nonconverged_skipped`、网格行 `skipped_nonconverged`），
  此前 `run-episode` 在示例数据第 1 轮即崩溃。可行率分母不再包含不收敛候选。
- **动作 schema**：`funding_constraints` 改为描述格点实际接受的组合：`premium_paid` 仅用于
  非保本 vanilla/barrier，`funded_note` 用于保本 vanilla/barrier 与全部 autocallable/snowball。
  格点与 DOMAIN 检查不变；schema 版本升为 `stochopia.environment.action-schema.v2`。
- **v2 结果文件**：`run-manifest` / `run-episode` 结果增加 `inputs`（市场 CSV 路径、客户画像、
  风险预算、QuotePolicy、环境角色模型），`run-episode` 结果也写 `schema_version` 与
  `complete: true`；续跑要求 `job_id` 与 `inputs` 都一致（旧结果无 `inputs`，会被重跑）。
  `compute_metrics` 失败只记该作业失败。
- **环境角色**：`consult("client")` 使用独立会话，不再读取含隐藏画像的 query/决策历史；
  `EnvResponseCache` 值改为请求之后的全部历史轮次（含 repair），合约升为
  `stochopia.env-response.v2`，旧缓存不再命中；只有供应商/传输错误降级，编程错误抛出；
  `RoleResponse.raw_hash` 删除。LLM 客户端截断升额后实际生效的 `max_tokens` 记入
  `usage.max_tokens_escalations`。
- **角色配置**：`config/benchmark_roles.yaml` 删除从未生效的 `structurer` 角色（被测模型始终使用
  打包 prompt，见 LLM-native roles），`role_config` 不再接受 `structurer`；
  `scenarios/stochopia_csi/prompts/structurer.md` 删除。judges 仍为 `deepseek-v4-pro, qwen-max`：
  `qwen-max` 同时是 trading_desk NPC，judge 与 NPC lineup 并不独立；`config/models.yaml` 中没有
  既不在 NPC lineup、也不在文档被测模型名单（gpt-4o、claude-sonnet、deepseek-v4-flash、minimax）
  里的第二个模型，换成被测模型会让该模型的提交只剩一个 judge。
- **统计与 judge 汇总**：单值/单 cluster 的 bootstrap 区间返回 `None` 并显示为不可估计；
  `aggregate` 只汇总最近一次 `judge-runs` 的 judge 文件，judge 抽样只取 `complete: true` 的结果。
- **正式行情构建**：full phase 把 `trade_cal` 开市日传给 IV 陈旧度计算（此前按期权链文件日计数）；
  `carry_complete` 门槛改为检查真实的 carry 回退状态（任何 0 回退都使其失败）。
  当前冻结数据重建结果不变（69 个 IV 观测的陈旧度均为 0，无 carry 回退）。

## 2026-11-09：环境边界、泄露与无效动作口径修复

本节各项不改 Level-0 定价、结算与约束规则。环境版本升为 `v3-spine-level0.4-public-terminal-snapshot`；
v3 定价版本 `PRICING_VERSION` 升为 `benchmark-pricing-v3.2-discrete-barrier-autocall-ki`，对应上节
同日的障碍期权离散监控修正与 autocall 敲入结算修正；DSL 实验定价版本升为
`dsl-monthly-gbm-future-cashflows.v2`。旧轨迹的 `--replay` 会因版本不一致被拒，
不带 `--replay` 的 `evaluate-trajectory` 仍可计分。

- **合成路径公开快照**（`stochopia.market_generator`）：`trend_alpha` 改为转移矩阵平稳分布加权的
  P 漂移，整段 episode 恒定，不再逐月等于当前潜在 regime 的漂移（此前可经 suitability、
  `client_price`、margin 反推 regime）；快照 `source` 去掉 `episode=…;seed=…`，二者只留在
  privileged provenance。合成路径本就不进入正式评测；用旧生成器产生的合成任务报价会变，需重新生成。
  真实数据 episode 不受影响。
- **终局快照**：`last_event["terminal_snapshot"]` 改为 `MarketSnapshot.public_brief()`
  （再去掉 `episode_id`），不再含 `source`、`trend_alpha`、`risk_free_rate`、`carry_rate`。
  终局后没有决策，经济数字不变。
- **任务校验**：`EpisodeTask` 要求快照逐个落在连续的日历月上（原来只要求日期严格递增），
  与持仓每轮老化一个月、DSL 月度网格一致；DSL 环境原有的同一检查并入此处。现有正式 episode
  为月末快照，不受影响；日频或跳月任务现在构造时即报错，而不是让 3 个月产品 3 天到期。
- **无效动作口径**：`step()` 只把 `BenchmarkError` / `ProductError`（预算耗尽、未知 quote、域外产品等
  策略可负责的校验失败）记为无效动作并收取 `invalid_action_cost`；`PricingError` 等内部错误不再
  被吞成策略的 −1，而是向上抛出；`run_agent_episode` 捕获后记为 `infrastructure_error`，照常写出
  截至出错前的轨迹，`run-v3-suite` 将其计为失败并继续后续任务。不支持的动作对象与
  含 NaN 等非 JSON 值的 `InvalidAction.raw` 改为可记录的 JSON 视图（必要时为 `repr`）。DSL 动作的
  格式错误（如 `contract` 不是对象、程序树畸形）记为解析错误而非 `infrastructure_error`。
  旧结果中若有内部定价错误被计为无效动作的步，其无效动作率与成本口径与新结果不同。
- **DSL 适配器 bump**：`conditional_paths` 的 `spot_factor` 只作用于生成的未来列，当日已观测
  价格不再被 bump。此前在已过期的观察日附近、bump 越过自动终止价位时会抹掉整条路径
  （1M 名义本金的 autocall delta 曾达 −51M）。公允价值不变，delta、gamma、stress 损失与依赖它们的
  HARD 检查改变；已有 DSL 实验结果需重跑，Level-0 不受影响。
- **评测指标**：`contract_acceptance_rate_given_hard_pass` 的分子只计 HARD 通过的提交，
  此前分子计所有合约通过的提交、可大于 1。对已有轨迹重新 `evaluate-trajectory` 即得新值。
- **`ScalarizedStochopiaEnv`**：事件型分量（`query_cost`、`quote_cost`、`dealer_economics`、
  `operational_cost`）在没有对应事件的步按 0 计，不再在内层环境已经前进后抛错；Level-0 从不测量的
  `relationship_delta`、`communication_faithfulness` 在构造时即拒绝非零权重。

## 2026-11-08：市场世界模型前推预测研究线

新增独立研究线 `stochopia.market-world-model-forecast.v1`：`stochopia.research.market_panel`
从本地 Tushare / CFFEX 缓存构建日频 P/Q 公开面板（指数收盘、SHIBOR、IC/IM 隐含 carry、MO 30 天 ATM IV），
`stochopia.research.forecast_protocol` 定义逐期前推起点、因果历史前缀、DSL payoff 探针与 CRPS skill 评分。
OpenHyra `tasks/stochopia_world_model` 以 API 基座模型修订世界模型程序并在沙箱中执行。规则见
[研究协议](research/2026-11-08-market-world-model.md)。该研究线不修改 Benchmark v3 的定价、结算、约束或指标，
已有 episode 结果不受影响，其产物不得与 v3 聚合混用。

同日，循环选出的 `r02s01` 以冻结拷贝接入 `stochopia.research.world_models` 注册表，由
`stochopia.research.world_model_generator` 作为 P 侧情景生成器使用。它尚未通过 StructurerDSL SPEC 11.2 的
G5–G7，标为 development-only；它不进入 `step()`，也不替代任何 Q 侧定价。

## 2026-11-08：完整性机制收敛（续）

延续 2026-10-27 的裁定，删除当时作为例外保留、但只承担校验或防篡改责任的哈希与自校验。
定价、结算、约束、指标口径与 `economic_replay` / `conversation_replay` 主张不变；
已有 episode 结果的经济数字不受影响，只是旧产物里的下列字段不再写出或不再被检查。

- **v2 `run-manifest`**：删除 `run_fingerprint` / `job_fingerprint` / `fingerprint_inputs`、
  失效结果的 quarantine 与 `aggregate` 对混合 fingerprint 的拒收。续跑只看同一 `job_id`
  的结果是否 `complete`；不同输入配置请使用不同输出目录。`experiment` manifest 删除
  `manifest_fingerprint`。结果里的 `roles_config_sha256` 删除，保留 `npc_lineup_id`。
- **v3 评测**：`TaskSuite.load` 不再校验 `suite_hash` / `task_hashes` / `task_count`，
  编辑过的 suite 按编辑后内容加载并得到新的 `suite_hash`；`aggregate_hash` 删除。
  `suite_hash`、`task_hash`、`policy_config_hash` 继续作为分组 ID，聚合仍拒绝混合 suite，
  也不把同名但配置不同的策略混在一起。
- **策略元数据**：删除命令参数文件的 `sha256`、`argv_sha256`、`policy_sha256`、
  `system_prompt_sha256` 与 `raw_output_sha256`；`--redact-raw-output` 省略 `policy_metadata.raw_output`，
  且在 `step()` 前把无法解析输出的 `InvalidAction.raw` 置空，原文不再进入轨迹、环境 `last_event`
  与智能体历史（解析错误信息保留）；因此开启脱敏时智能体也看不到自己上一步的畸形原文。命令策略的身份只看 argv 与参数文件路径、大小：
  修改策略脚本后请用 `--force` 或新的输出目录，否则 `run-v3-suite` 可能复用旧轨迹。
  命令策略名仍带 argv 短摘要以区分不同命令。角色配置 `RoleSpecV2.system_prompt_sha256` 删除。
- **环境**：`reset` info 删除 `options_hash`（`reset_options` 仍原样记录）；
  `ScalarizedStochopiaEnv` 删除 `scalarization_hash`（`scalarization` 原样随 info 返回）；
  `TrajectoryRecorder` 不再把记录冻结为只读结构，仍做 canonical JSON 快照。
- **数据层**：tushare 与交易日历缓存 sidecar 不再写、也不再校验 CSV `sha256`，保留
  加载时的 `status`、`row_count`、`columns` 检查（交易日历构建另查日期边界与 pretrade 链；
  抓取器命中缓存只看 `status` 与 `cache_key`，CSV 先于 sidecar 原子写入）；交易日历 artifact 删除
  `artifact_sha256` 与加载时的逐字段自校验。CFFEX 归档 sidecar 由内容哈希命名改为
  `metadata/<YYYYMM>.json`，删除 `archive_sha256` 与内容冲突报错；`data/raw/cffex/metadata/`
  下已有的哈希命名文件未改动，也没有代码读取它们。
- **规则文档**：`CLAUDE.md` / `AGENTS.md` 不再要求为可复现性保留哈希；可复现靠 seed、
  规范化 manifest 与按需 `--replay`，v3 轨迹另记 `git_sha`；v2 结果不记录 `git_sha`，也不再
  记录角色 prompt 指纹，prompt 或代码改动后请换输出目录。世界模型可行性文档（含 G0 哈希门槛）作为
  过时计划删除，其 G5–G7 门槛移入 StructurerDSL `docs/SPEC.md` 11.2。

保留的哈希均为功能性 ID：`task_hash`、`suite_hash`、`policy_config_hash`、seed 派生、
judge 盲评 ID、`EnvResponseCache` 键、
tushare 请求缓存键、报价与事件 ID、`git_sha`。

## 2026-10-27：完整性机制收敛

本仓库是科研实验代码，不是对抗篡改的发布系统。本次把只为"防篡改"存在、且在
运行时反复计算哈希的机制全部移除；定价、结算、约束与指标口径完全不变，
`economic_replay` 主张不变（改为按需执行）。

- **轨迹格式升级为 `stochopia.trajectory-json.v2`**：删除逐条 `record_hash` /
  `previous_record_hash` 链、`root_hash`、`genesis_hash`、每步 `state_hash`、
  `implementation_hash`、工作树 diff 哈希与各类配置哈希。元数据只保留版本串、
  `task_hash`、`package_version` 与 `git_sha`。`StepTransition` 不再携带
  `state_hash_before/after`。v1 轨迹与新评测器不兼容。
- **评测默认不重放**：`evaluate-trajectory` 直接按记录的 transition 计分，
  `--replay` 时才重建环境并逐项比对。`aggregate-v3` 不再要求 `replay_verified`
  与 `evaluation_hash`。`run-v3-suite` 的续跑判定不再做哈希校验与重放。
- **合成路径 provenance 瘦身**（`stochopia.synthetic-regime-scenario.v2` 字段
  变化）：删除生成器源码哈希、spec/Q 选择/日历/日路径/快照各哈希、计数字段与
  seed domain 字段；保留 normalized spec、seeds、Q 选择 manifest 与 horizon。
  `GeneratedMarketPath` 不再对照 provenance 自校验。
- **派生输入**：日历 artifact 与 Q 侧 bundle 改为固定文件名
  （`<calendar_id>.json`、`q_side_bundle.json`），不再写 `manifest.sha256`；
  删除对原始 manifest 的重验、构建器源码哈希、许可证文件哈希与 P/Q 语料哈希。
- **原始数据层**：删除 append-only 预检与 `manifest.sha256` 写出；缓存 sidecar
  仍记录 `sha256`，加载时校验一次以识别不完整下载（已于 2026-11-08 删除，见上节）。
- **审计层整体删除**：`raw_data_audit`、`historical_index_audit`、
  `formal_gate_audit`、`data/audit/*`、`data/reference/*_protocol.json` 与四份数据审计文档；
  同日随后整体删除了无人消费的 `multi_asset_data` 采集模块及其原始数据。

## 2026-10-09：DSL 合约设计科研实验

新增显式启用的 `StochopiaDSLEnv`，实验规则和复现记录见
[DSL 集成说明](research/2026-10-09-dsl-integration-progress.md)。实验以未来现金流公允现值成交，
采用单标的月频 DSL 和自动终止条款；评价格式为 `stochopia.dsl-experiment-evaluation.v1`。
现有 Level-0 产品域、定价、结算及奖励口径保留。DSL 工件不能进入原 v3 聚合统计。
这一实验不解除原 benchmark 的冻结，也不修改 StructurerDSL 的语言规范。

OpenHyra 的独立 `stochopia_structuring` 任务已经接入上述实验环境；它使用固定 DSL 工件，
单独的合成开发／留出场景与生成器接口。搜索目标由实验配置决定，保留原奖励向量；
结果、经验数据和预算记录仍归实验类型，不混入原 v3 统计。详情及运行证据见同一集成说明的 C、D 节。

### 独立的路径需求研究协议（2026-10-12）

`stochopia.research` 新增有限月度树和路径现金流实验环境，版本
`stochopia-path-cashflow-tree.v2`。该模块不修改冻结 benchmark v3 的
定价、结算或指标，也不与正式 episode 聚合。完整研究规则见 OpenHyra
`docs/stochopia_path_research_protocol.md`。

实验采用同一波动率输入的 P/Q 市场、P=Q 对照、12 月历史状态树，支持定日
支付、现金需求和公允价减折价的提前退出。默认不部分出售；可部分出售为
单独对照。条件价值使用精确树递推，没有逐路径嵌套蒙特卡洛。理论最优
仅限同一有界有限树，不称连续 GBM 最优。复现记录保存三仓库版本、完整
科学源码快照和配置，包括未追踪的适配源码；本轮不增加信息隔离机制。

### 独立的股票共同流动性预实验（2026-10-13）

`stochopia.research.liquidity_market` 与 `liquidity_execution` 提供合成5分钟
共同流动性、目标权重市价执行和原生DSL现金流研究接口。协议为
`equity-liquidity-mechanism-discovery.v1`，规则见
[环境说明](research/2026-10-13-liquidity-pilot.md)与OpenHyra
`docs/stochopia_liquidity_pilot.md`。资金、部分对冲报价和观点响应门槛仅用于
该研究；原benchmark v3、历史月度DSL/路径实验及已有episode口径保持不变。
