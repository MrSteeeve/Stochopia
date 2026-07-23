# MIRAGE CSI Protocol: v2 Legacy Benchmark and v3 Environment Spine

> **Version boundary.** The factorial CLI commands and
> `LongHorizonEnvironment` described below are the reproducible **v2 legacy
> benchmark**. `mirage.environment` and `mirage-benchmark test-agent` are the
> **v3-spine / Level-0** synchronous interface, in which partial observability
> and dynamic state are invariants rather than experimental conditions. v2
> and v3 trajectories are versioned separately and must never be pooled in
> one result table.

## Research claim

MIRAGE evaluates whether an LLM can repeatedly synthesize executable structured
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
a monthly-snapshot approximation. Full/Static are not v3 task conditions.

## v3-spine / Level-0 boundary

`mirage.environment.MirageStructurerEnv` exposes typed `reset`/`step`
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
an explicit versioned `ScalarizationSpec`. `TrajectoryRecorder` canonicalizes
and deep-freezes each transition at record time, embeds the complete task
manifest, records package/git/pricing implementation provenance, verifies its
own chain before saving and writes atomically with an end-of-file root hash.
This is an architectural spine, not yet a claim that the planned payoff DSL,
solve-for tools, task generator, teacher hierarchy, Pareto evaluator or
realised hedge-P&L reward are complete.

Typed product actions are canonicalised through the same parser as JSON
actions; policies cannot set `reference_spot`, barrier-history or elapsed-time
fields. Quote storage snapshots the mutable legacy `ProductSpec`, preventing
post-quote mutation before submission. Invalid actions cannot run forever:
`max_steps_per_round` ends the rollout with `truncated=True`. Reward/termination
configuration and reset options are committed into the state hash, and the
trajectory metadata commits an environment-configuration hash.

### v3 external Agent interface

`mirage.environment.agent_runner` is the only implemented policy-facing
rollout layer for v3. It does not move LLM calls into the deterministic core:
it converts an external response into a typed action and calls the same
`MirageStructurerEnv.step()` entrypoint used by a training loop.

Two adapters share the versioned `mirage.agent-request.v1` request:

- `CommandAgentPolicy` invokes a local executable directly, without a shell.
  Each step sends one JSON object on stdin and requires one action JSON on
  stdout. The request carries the packaged action contract, current public
  observation and a bounded public interaction history.
- `LLMAgentPolicy` accepts any existing `BaseLLMClient`.
  `create_api_agent_policy` constructs an OpenAI-compatible or Anthropic
  client directly from provider/base URL/model/API-key-environment-variable
  arguments, without requiring a `models.yaml` edit.

The CLI exposes both through `mirage-benchmark test-agent`: use
`--agent-command`, `--model` (registered), or the direct
`--api-model/--api-base-url/--api-key-env` path. These choices are mutually
exclusive. Literal secret values are not accepted; only the environment
variable name is configured, and its value is neither serialized nor sent to
the model.

The Agent request includes `task_hash` but never the task manifest, hidden
client profile or evaluator-only `privileged_state`. The complete task
manifest remains in the evaluator's saved trajectory so the run can be
verified. Transport failures, malformed JSON, unknown fields and unavailable
actions become explicit `InvalidAction` transitions and consume the ordinary
step limit rather than receiving hidden retries. Raw policy output is hashed
and recorded by default; `--redact-raw-output` retains only its hash.

## Frozen episode design

- Price-only warm-up: 2022.
- Evaluation: 2023--2025.
- Underlyings: CSI500 and CSI1000.
- Twelve half-year episodes, six month-end rounds each.
- Per round: up to 3 `query_client` calls, up to 3 `consult` calls, up to 3
  `request_quote` calls, then a submission or a skip.
- Frozen finite product action lattice `csi-domain-v1` (below), shared by the
  tested agent and the oracle.

The released example snapshot (`market_snapshots.example.csv`) is synthetic and
cannot be used for reported results. A production snapshot row must carry
explicit source provenance. The source audit and go/no-go rules are frozen in
[DATA_AUDIT.md](DATA_AUDIT.md). Preferred data are contract-level CFFEX MO
history for CSI1000 and SSE 510500 ETF option history for CSI500. Full option
surfaces are optional: the frozen fallback is 3m ATM IV, 1m ATM IV, 6m ATM IV,
20d realized volatility, then 60d realized volatility
(`MarketSnapshot.pricing_volatility`).

When paired calls and puts are available, the environment uses

\[
F(K,T)=K+e^{rT}(C-P)
\]

so dividend and carrying-cost expectations are absorbed by the market-implied
forward (`mirage.benchmark.option_implied_forward`). ETF fees are not deducted
again from ETF spot/NAV. Dividend-related contract adjustments must be
normalized or the affected row excluded. Carry is perturbed by plus/minus 25 bp
as a sensitivity check (`mirage.benchmark.carry_sensitivity`).

## Two-layer architecture

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
`mirage.benchmark.WorkflowOutcome` produced by the pure function
`settle_submission`.

**Fallback.** Passing no `--roles-config` to `run-episode` / `run-manifest`
leaves `env_agents=None`: `query_client`, hard checks, the contract gate and
`dealer_margin` behave byte-for-byte as if no environment LLM existed, `consult`
returns a fixed degraded-fallback string, and `workflow_review` is never called.
(The redesign draft names this switch `env_agents.enabled=false`; the shipped
mechanism is the presence or absence of `--roles-config` on the CLI, not a
config field.)

## Frozen product action lattice and the oracle

`mirage.benchmark.ProductDomainSpec` (`version="csi-domain-v1"`) is a frozen
finite grid: 6 product families (`vanilla_call/put`, `barrier_call/put`,
`autocallable`, `snowball`), 14 notional fractions of client capital (0.5% to
100%), maturities {3, 6, 12} months, strikes {0.95, 1.00, 1.05}, barriers
{0.75, 0.85, 1.10, 1.20}, coupons {4%, 8%}, participations {0.5, 1.0} and
principal-protection {False, True}. `enumerate_domain` materializes every
structurally valid combination (barrier direction is inferred from the barrier
level; snowball always uses a lower knock-in barrier and never claims
protection).

A single `DOMAIN` HARD check (`validate_domain`) rejects any quote request
whose structural signature is not an exact element of this lattice. Because the
oracle (`oracle_candidate_grid` / `oracle_best_quote`) enumerates exactly the
same lattice through the same `TradingDesk`, every quotable agent action is by
construction an oracle candidate; `tests/test_benchmark.py` asserts this
symmetry and a dedicated property test
(`test_one_step_attainment_never_exceeds_one`,
`test_voluntary_best_submission_attainment_le_one`) asserts the resulting
attainment ratio can never exceed 1 + 1e-9.

`oracle_best_quote` is a **one-step** frontier: at the start of each round,
before the agent acts, it prices every lattice candidate against the *current*
portfolio state and returns the maximum-`dealer_margin` feasible one. It is not
an episode-level (hindsight/DP) upper bound and is not claimed as one; a
hindsight oracle is future work. The runner passes the environment's exact
`domain` and `quote_policy`; it no longer silently rebuilds a default grid or
default pricing policy.

## Information boundary, tools and per-round budgets

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
  the role's numbers are grounded in fact. Consult never spends the
  `request_quote` budget, never enters the candidate archive, and its output
  cannot be submitted directly. Budget: 3 per round. Requires `--roles-config`;
  without it every `consult` call returns a fixed degraded-fallback string.
- `request_quote(ProductSpec)` -- up to 3 per round; returns a full quote
  bound to a `state_version` hash of `(episode_id, round_num,
  portfolio.revision, condition.id)`. A stale quote (state changed since it was
  issued) cannot be submitted.
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
`PORTFOLIO_DEALER_STRESS_LOSS`) against
the frozen `RiskBudget`, cumulative with the current dynamic-condition
portfolio. Any HARD failure makes the quote non-executable; an LLM (structurer
or any env role) cannot waive it.

**Settlement-time CONTRACT checks** (`client_contract_pass`, `severity ==
"CONTRACT"`, evaluated only at `submit_design`/`submit_product` against the
already-priced quote): `CONTRACT_TARGET_CLIENT_MATCH`, `CONTRACT_ACCEPTING`, `CONTRACT_CAPITAL`,
`CONTRACT_MATURITY`, `CONTRACT_WHITELIST`, `CONTRACT_PROTECTION`,
`CONTRACT_LOSS` (the same continuous loss proxy as `CLIENT_LOSS_BUDGET_V2`,
re-derived at settlement) and `CONTRACT_HURDLE` (`hurdle_hit_prob >=
client.min_hit_prob`). This is the deterministic replacement for the old
`ClientProfile.would_buy` NPC rule; no LLM's stated attitude overrides it.

Settlement: `accepted = hard_executable AND client_contract_pass`, where
`hard_executable = quote.hard_pass`. `submit_design`'s return dict carries
`hard_executable`, `client_contract_pass`, `accepted`, `hard_failures`,
`contract_failures` and `dealer_margin` (0 when not accepted) individually --
they are never collapsed into one boolean before being recorded.

## Continuous client loss proxy (`CLIENT_LOSS_BUDGET_V2` / `CONTRACT_LOSS`)

`mirage.pricing.client_loss_measure` reports
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

Accepted dynamic trades record an issuance cashflow. Maturity, knock-out and
autocall closures move the contract into `closed_positions`, append a
settlement cashflow and a `LifecycleEvent`, and calculate client realised P&L
and the equal-and-opposite dealer-liability P&L. Actual hedge trades,
transaction costs and `dealer_total_pnl` remain unavailable. Accordingly the
v3 reward schema represents `terminal_lifecycle_pnl` as `value=null,
available=false`, not as a measured zero.

## Risk budget calibration

`mirage-benchmark calibrate-budget` runs `calibrate_risk_budget`: it scales a
base `RiskBudget` by a grid of factors (default 0.1 to 3.0 step 0.1, or an
explicit `--factors` list), prices every lattice candidate against each
development episode, and selects the factor whose feasible-candidate rate
falls closest to the midpoint of the target band `[0.20, 0.40]`
(`scenarios/mirage_csi/benchmark.yaml: risk_budget_calibration`). The full grid
and a `"freeze this result before evaluating held-out episodes"` warning are
written to `--report-output`; the selected budget is written to
`--budget-output`. Both budget and margin calibration resolve the client's
`round_overrides` separately for every snapshot and enumerate the domain for
that resolved client.

`calibrate-budget`, `run-episode` and `run-manifest` all accept an optional
`--quote-policy-json` (a JSON-serialised `mirage.pricing.QuotePolicy`, e.g.
`data/derived/quote_policy.v2.candidate.json`): when given, quotes are priced
with that policy instead of `QuotePolicy()`'s code defaults (`calibrate-budget`
passes it through to `calibrate_risk_budget`'s new `policy` kwarg;
`run-episode`/`run-manifest` pass it to `LongHorizonEnvironment`).

All calibration numbers created before the funding, bilateral-stress and quote
equilibrium migration are stale and must not be used for a result claim.
`data/derived/*v2*` files are historical inputs, not a current frozen v3
calibration bundle.

## `dealer_margin` economics

`mirage.pricing.quote_economics` replaces the old constant 1% markup with a
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

`suitability` (draft_opus §3.2) is a four-factor product of client fit: product
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

The coefficients currently frozen in code (`mirage.pricing.QuotePolicy`
defaults) are: `a_f=0.030, a_v=0.25, a_p=0.002, a_b=0.75`,
`b_f=0.020, b_v=0.75, b_p=0.004, b_b=1.00, b_l=0.003, b_q=0.015`,
`client_cap=0.08, hedge_cap=0.10`, `suit_whitelist_miss=0.20,
suit_protection_miss=0.10`, all `suit_w_* = 1.0`.

`mirage.pricing.calibrate_quote_policy` grid-searches an overall scale on the
`a_*` markup coefficients (holding `b_*` fixed) against a pre-registered
target: 20-60% of development candidates have positive margin, and the margin
ranking across structures is non-degenerate (the largest-notional vanilla is
not always optimal). It is covered by
`tests/test_pricing_economics.py::test_calibrate_quote_policy_hits_target_band`
and wired to `mirage-benchmark calibrate-margin`, which samples a fixed-seed
subset of the shared `ProductDomainSpec` lattice per (episode, round)
development snapshot (`--candidates-per-snapshot`, default 30 -- the full
lattice is tens of thousands of candidates, far more MC than a few minutes of
wall time affords), builds dev cases from those candidates against the real
market snapshots, runs `calibrate_quote_policy`, and additionally reports
0.8x/1.0x/1.2x sensitivity of the *selected* policy (scaling its `a_*` a
second time on top of the calibrated factor).

No post-migration calibration result is frozen in this revision. A new bundle
must include the input manifest, implementation/run fingerprint, per-round
client resolution, selected policy/budget, the full grid and sensitivity
report before held-out evaluation begins.

## LLM-native roles

Four roles, each with an independent system prompt file, model and
conversation history, configured in `config/benchmark_roles.yaml`
(`protocol_version: mirage-csi-v2.0`, loaded by `mirage.role_config` with
fail-fast validation):

| role id | role | model | temperature | history scope | numeric authority | failure policy |
|---|---|---|---|---|---|---|
| `structurer` | tested model | `${job.model}` (resolved per job) | 0.0 | episode | `none` | `no_action` |
| `client_main` | env NPC | `deepseek-v4-flash` | 0.2 | episode | `supplied_facts_only` | `abstain` |
| `risk_control` | env NPC | `deepseek-v4-flash` | 0.0 | round | `supplied_facts_only` | `escalate` |
| `trading_desk` | env NPC | `qwen-max` | 0.0 | round | `supplied_facts_only` | `decline` |

The three env-NPC roles are pinned as `main_npc_lineup_id: npc-fixed-v1`: only
the structurer's model varies across a benchmark run, so a tested model never
also gets a different negotiation counterparty. `numeric_authority:
supplied_facts_only` is enforced both at config-load time (`role_config`
rejects any client/risk/desk role that isn't `supplied_facts_only`) and at
call time (`validate_grounding`): these three roles can never invent a number
not present in the `FormalFacts` handed to them for that call. The structurer
has no such restriction (`numeric_authority: none`) since it is the model under
test.

`FrozenEnvAgent` (`mirage.env_agents`) wraps one role: strict JSON-schema
parsing (`client_response_v1` / `risk_response_v1` / `desk_response_v1`, one
of `answer/counter/accept/reject/abstain`,
`approve/request_revision/escalate`, `issue/decline/request_revision`
respectively) with one format-repair retry, then grounding validation with one
repair retry; any remaining failure returns a `degraded=True` response whose
`action` is the role's `failure_policy` and never raises. Seeds derive
deterministically from `(episode_id, round_num, role_id, turn_id)` via
`stats.derive_seed` plus the role's `seed_offset`, so replaying a frozen
episode reproduces the same seed sequence. An `EnvResponseCache`
(`--env-cache-dir`, append-only JSONL) keys on
`sha256(role_id|model|temperature|seed|canonical(messages))`, so an identical
request from a different tested model hits the same frozen NPC reply, and
every accepted role call is written to the cache for audit/replay.

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

## Canonical metrics (`compute_metrics`, `mirage.benchmark_cli.CANONICAL_METRICS`)

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
  `dealer_margin` over **voluntary**-accepted rounds only.
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
(episode-level) bootstrap 95% CI for canonical metrics including the three
separated execution/contract/settlement rates, `total_dealer_margin`,
`one_step_attainment` and the submission-origin rates; a metric missing from
every row in a results directory is reported as fully missing rather than
silently defaulting.

## Offline judge protocol

Judging never feeds back into a run: `mirage-benchmark judge-runs` reads
already-frozen `*.json` results and writes independent `<file>.judges.json` +
`judge_manifest.json`, never touching the original result files.

- **Population**: only rounds with `submission_origin == "voluntary"` and a
  non-empty `submitted_product` are eligible (`_collect_voluntary_samples`) --
  forced_prompt and unsettled rounds are never judged.
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
  (whitespace-normalized) substring of the actual explanation to `score=None`
  (missing), guarding against fabricated evidence.
- **Reliability, not validity**: `aggregate`'s markdown output appends a
  "Judge" section (median/IQR/missing-rate per dimension via
  `aggregate_judge_bundle`, no composite score) whenever `*.judges.json` files
  are present, plus inter-judge reliability between the two judge models on
  aligned, fully-scored `(submission_id, repeat)` pairs: exact total
  agreement, Spearman correlation of totals, and per-dimension quadratic
  weighted Cohen's kappa. `reliability_summary`'s output literally carries the
  claim string `"inter-judge reliability only; not expert-grounded validity"`
  -- MIRAGE does not claim expert-grounded validity for the judge, only
  agreement between the two configured judge models.

**Config-file defaults**: `config/benchmark_roles.yaml`'s `judges:` block
(`models`, `repeats`, `temperature`, `max_tokens`, `blind_model_identity`) is
loaded by `mirage.role_config.load_judges_config` into a `JudgesConfig`
(unknown fields, model-count and registry-membership checks all fail fast).
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
- **Replicate design**: `mirage-benchmark make-manifest` builds every
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

## Reproducibility: two separate claims

MIRAGE does not claim bit-level reproducibility from a fresh API call to a
live model endpoint (provider/model updates and non-determinism persist even
at temperature 0). It makes two narrower, implemented claims instead:

- **`economic_replay`**: given a frozen episode trace (the recorded action
  sequence and the market snapshots), the deterministic core -- pricing, hard
  checks, the contract gate, `dealer_margin`, the separated execution/
  contract/settlement rates,
  `one_step_attainment` -- is a pure function of that input and recomputes
  identically every time, with no LLM call involved.
- **`conversation_replay`**: env-role dialogue is reproducible from the
  `EnvResponseCache` JSONL artifact (keyed by `role_id|model|temperature|
  seed|max_tokens|output_schema|inference_contract|canonical(messages)`):
  replaying the same request sequence against the same cache file reproduces
  the exact same NPC replies byte-for-byte, without re-calling any model.

## Reproduction

```bash
# Convert a public daily close file into leakage-safe month-end features.
python -m mirage.benchmark_cli build-market daily.csv market_snapshots.csv

# Validate schema, provenance and contiguous episode rounds.
python -m mirage.benchmark_cli validate-market market_snapshots.csv

# Smoke-test the deterministic desk with the synthetic example.
python -m mirage.benchmark_cli demo \
  scenarios/mirage_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --product-json scenarios/mirage_csi/product.example.json --full

# Calibrate and freeze the risk budget on development episodes only.
# --quote-policy-json is optional (e.g. data/derived/quote_policy.v2.candidate.json);
# omit it to price candidates with the default QuotePolicy.
python -m mirage.benchmark_cli calibrate-budget market_snapshots.csv \
  --episodes CSI500_2023H1 CSI500_2023H2 \
  --client-json scenarios/mirage_csi/client.example.json \
  --base-budget-json scenarios/mirage_csi/risk_budget.example.json \
  --report-output data/derived/budget_calibration_report.json \
  --budget-output data/derived/risk_budget.frozen.json

# Calibrate and freeze the QuotePolicy markup scale (mirage.pricing.calibrate_quote_policy)
# on development episodes only, then report 0.8x/1.0x/1.2x sensitivity of the
# selected policy.
python -m mirage.benchmark_cli calibrate-margin market_snapshots.csv \
  --episodes CSI1000_2023H1 CSI500_2023H1 \
  --client-json scenarios/mirage_csi/client.example.json \
  --report-output data/derived/quote_policy_calibration_report.json \
  --policy-output data/derived/quote_policy.v2.candidate.json

# Run one registered model against one frozen episode; --roles-config and
# --quote-policy-json are both optional (omit --roles-config to stay
# pure-deterministic; omit --quote-policy-json to use the default QuotePolicy).
python -m mirage.benchmark_cli run-episode market_snapshots.csv \
  --episode CSI500_2023H1 \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json data/derived/risk_budget.frozen.json \
  --model deepseek-v4-flash --strategy ledger_archive \
  --roles-config config/benchmark_roles.yaml \
  --quote-policy-json data/derived/quote_policy.v2.candidate.json \
  --env-cache-dir outputs/env_cache \
  --output outputs/csi500_2023h1.json

# Freeze the complete factorial job manifest (n>=3, partial_dynamic n=5)
# without spending API budget.
python -m mirage.benchmark_cli make-manifest \
  --episodes CSI500_2023H1 CSI500_2023H2 CSI500_2024H1 CSI500_2024H2 \
             CSI500_2025H1 CSI500_2025H2 CSI1000_2023H1 CSI1000_2023H2 \
             CSI1000_2024H1 CSI1000_2024H2 CSI1000_2025H1 CSI1000_2025H2 \
  --models gpt-4o claude-sonnet deepseek-v4-flash minimax \
  --output outputs/experiment_manifest.json

# Execute a frozen manifest sequentially. Reruns skip only complete outputs
# whose run/job fingerprints still match. Stale or partial files are
# quarantined and rerun; writes are atomic. A single failed job does not abort
# the batch, and aggregate refuses mixed run fingerprints.
# --quote-policy-json is optional (default QuotePolicy if omitted).
python -m mirage.benchmark_cli run-manifest market_snapshots.csv \
  --manifest outputs/experiment_manifest.json \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json data/derived/risk_budget.frozen.json \
  --roles-config config/benchmark_roles.yaml \
  --quote-policy-json data/derived/quote_policy.v2.candidate.json \
  --env-cache-dir outputs/env_cache \
  --outputs-dir outputs/csi_benchmark

# Aggregate results into CSV + markdown (cell CIs, paired contrasts, Holm).
python -m mirage.benchmark_cli aggregate outputs/csi_benchmark \
  --output-csv outputs/aggregate.csv --output-md outputs/aggregate.md

# Offline blind judge batch over frozen voluntary submissions (never
# touches the original result files). --judge-models/--repeats are optional
# when --roles-config is given: they then default to config/benchmark_roles.yaml's
# judges.models / judges.repeats; an explicit flag always overrides the file.
python -m mirage.benchmark_cli judge-runs outputs/csi_benchmark \
  --roles-config config/benchmark_roles.yaml \
  --salt "$(openssl rand -hex 16)"
```

Only abstract workflow, schemas and synthetic rules may be derived from private
institutional material. No private document, client, institution, limit or
identifying metadata may enter released data or prompts.
