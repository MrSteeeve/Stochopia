# MIRAGE: CSI Structured-Product Design Benchmark

[Chinese README](README_zh.md)

MIRAGE evaluates whether an LLM can repeatedly synthesize executable
structured products -- vanilla/barrier options, autocallables, snowballs --
linked to the CSI500/CSI1000 indices under partial observability, limited
desk/client/risk interaction and portfolio risk carried across a run of
month-end decisions. It does not claim open-ended financial invention or
production-grade market pricing.

The full protocol -- episode design, the deterministic pricing/settlement
engine, the LLM-native environment roles, the offline judge, and the
statistics -- is frozen in [docs/BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md).
This README is a quick-start pointer, not the protocol of record.

## Protocol status

- `mirage.environment` is the new **v3-spine / Level-0** training interface:
  partial observability and dynamic state are invariants, actions and
  transitions are typed, and `MirageStructurerEnv.reset()/step()` contains no
  LLM calls, forced prompt, strategy logic or oracle. Privileged evaluator
  state is off by default, and finite per-round step limits prevent hanging
  rollouts.
- The CLI now exposes the v3-native `test-agent` entry point. Its other
  factorial commands and `LongHorizonEnvironment` remain the explicitly
  legacy **v2 benchmark** so frozen v2 runs can still be reproduced.
  Full/Static are not v3 task conditions and v2/v3 artifacts must not be
  pooled.
- Every Agent request includes the complete machine-readable action contract.
  Accepted positions update authoritative client/dealer accounts, lifecycle
  outcomes enter the reward vector, and all open contracts are liquidated at
  fair value at the episode horizon.
- Issued v2 dynamic positions now retain issue fixing and absolute contract
  levels, process barrier/knock-in/autocall observations, and revalue FV,
  delta, vega and stress on the next market snapshot.

> An earlier NASDAQ-100 "Structurer Playground" prototype (`run.py`,
> `mirage.cli`, `mirage.engine`) has been retired; that entry point no longer
> exists in this repository. `scenarios/structurer_nasdaq/` remains only as
> historical scenario data and is not runnable through any current command.

## What MIRAGE evaluates

- **Synthesizing a contract, not just pricing one.** The tested model
  (`structurer`) can query the client, consult the trading desk/risk
  control/client qualitatively, request deterministic quotes, and submit a
  design -- all under a fixed 3-query / 3-consult / 3-quote per-round budget.
- **Working under partial observability.** In Partial-information rounds the
  client's hard thresholds (capital, loss tolerance, maturity, product
  whitelist, protection requirement) are never disclosed directly; they must
  be drawn out through `query_client`.
- **Carrying risk across decisions.** The v3 environment is always dynamic.
  The v2 compatibility runner still exposes its historical Static ablation.
- **Not gaming a fixed markup.** `dealer_margin` is a funding-aware cost-plus function of
  moneyness, vega exposure, path dependence, Monte Carlo pricing uncertainty,
  static-delta-hedged dealer stress and capacity utilization. Premium-paid
  quotes also use a client-suitability factor. Negative margins are retained,
  and Level-0 par notes still require a future payoff solve-for before
  large-scale training. See the protocol doc for the exact formula.

## Two-layer architecture

```text
StructurerRole (tested LLM)
   | query_client / consult / request_quote / submit_design / skip_round
   v
LongHorizonEnvironment (central router)
   |-- ClientRole LLM      (optional; partial-info disclosure, decision)
   |-- RiskRole LLM        (optional; consult, decision)
   `-- DeskRole LLM        (optional; consult, decision)
   v  read-only, fact-gated calls
DeterministicCore: pricing -> hard constraint checks -> client contract gate
                    -> settlement (never calls an LLM)
   v
Primary settlement (always computed, deterministic):
  hard_execution_rate, hard_execution_rate_given_submission,
  contract_acceptance_rate_given_hard_pass, settlement_acceptance_rate,
  dealer_margin, ...
Secondary WorkflowOutcome (only with --roles-config and a submission):
  workflow_deal, desk/risk/client actions
Offline: judge-runs (blind, two-judge, batched; never feeds back into a run)
```

The deterministic core -- pricing, hard checks, the client contract gate and
settlement -- is the sole authority for the primary leaderboard and is a pure
function of the episode data and the tested model's actions; it never calls an
LLM. The three environment roles (client, risk control, trading desk) are
*optional*: pass `--roles-config config/benchmark_roles.yaml` to back them
with their own models for qualitative dialogue and a secondary
workflow-decision signal. Omitting `--roles-config` leaves every primary
metric byte-for-byte identical to the fully deterministic path.

## Quick start

MIRAGE requires Python 3.10+.

```bash
cd MIRAGE
uv sync --locked --all-extras --dev
```

`uv.lock` is the reproducible development/CI dependency set. A conventional
`python3 -m venv .venv && pip install -e ".[dev]"` install remains supported
for local experimentation but is not the frozen environment.

Configure API keys for real model runs:

```bash
cp .env.example .env
# Edit .env. The default test/env models use DeepSeek, so DEEPSEEK_API_KEY is
# enough to get started; config/models.yaml lists every registered model.
```

The canonical v3 interface is the typed partial-dynamic environment. This
smoke run is deterministic and needs no API key:

```python
import json
from pathlib import Path

from mirage.benchmark import RiskBudget, load_market_snapshots
from mirage.environment import EpisodeTask, MirageStructurerEnv, Skip
from mirage.products import ClientProfile

root = Path.cwd()
snapshots = tuple(
    row for row in load_market_snapshots(
        root / "scenarios/mirage_csi/market_snapshots.example.csv"
    )
    if row.episode_id == "SYNTHETIC_CSI500_DEMO"
)
client = ClientProfile(**json.loads(
    (root / "scenarios/mirage_csi/client.example.json").read_text()
))
budget = RiskBudget(**json.loads(
    (root / "scenarios/mirage_csi/risk_budget.example.json").read_text()
))
env = MirageStructurerEnv(EpisodeTask(snapshots, client, budget, task_seed=7))
observation, info = env.reset()
transition = env.step(Skip("typed v3 smoke test"))
print(observation.available_actions, transition.observation.round_num, info["seed_role"])
```

### v3 contract: input -> interaction -> output -> evaluation

| Stage | Frozen/public artifact | Hidden or evaluator-only truth |
|---|---|---|
| Input | a versioned `TaskSuite`, public market observation, public action schema and fixed public notional grid | client mandate, risk limits, quote policy and complete task manifest |
| Interaction | `ask_client`, `request_quote`, `submit_design`, `submit_product`, `skip` | deterministic pricing, accounts, constraints and lifecycle state |
| Output | one hash-chained trajectory with actions, observations, reward vector, constraint signals and run provenance | the same artifact also carries the sealed task manifest for replay, but it is never sent to the policy |
| Evaluation | replay-verified per-task metrics and task-clustered policy aggregates | exact economic state is reconstructed independently from the sealed inputs |

The last snapshot in every v3 task is terminal valuation truth, not another
decision round. Thus a contract issued on the last decision snapshot must
experience at least one market interval before maturity or horizon
liquidation. Training code can opt into `ScalarizedMirageEnv` with an explicit
versioned `ScalarizationSpec`; the core never silently chooses weights.

Real agents use the v3-native `test-agent` command. A local executable receives
one versioned JSON request on stdin per step and must print one action JSON on
stdout:

```bash
mirage-benchmark test-agent scenarios/mirage_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json scenarios/mirage_csi/risk_budget.example.json \
  --agent-command 'python my_agent.py' \
  --output outputs/v3-cli-agent.trajectory.json
```

An API model can be selected from `config/models.yaml` with
`--model deepseek-v4-flash`, or used directly without editing that file. The
key is read only from the named environment variable and is never put in the
request or trajectory:

```bash
export OPENAI_API_KEY='...'
mirage-benchmark test-agent scenarios/mirage_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json scenarios/mirage_csi/risk_budget.example.json \
  --api-provider openai-compatible \
  --api-base-url https://api.openai.com/v1 \
  --api-model gpt-4o --api-key-env OPENAI_API_KEY \
  --output outputs/v3-api-agent.trajectory.json \
  --summary-output outputs/v3-api-agent.summary.json
```

For a reproducible evaluation rather than a one-off smoke run, freeze a suite,
run a policy over every task, then independently replay and aggregate:

```bash
mirage-benchmark make-v3-suite market_snapshots.csv \
  --episodes CSI500_2024H1 CSI500_2024H2 \
  --client-json client.json --risk-budget-json risk_budget.json \
  --name mirage-dev --version 1 --split dev \
  --output tasks/mirage-dev.v3.json

mirage-benchmark run-v3-suite tasks/mirage-dev.v3.json \
  --agent-command 'python my_agent.py' \
  --replicates 3 \
  --output-dir outputs/mirage-dev \
  --bootstrap-resamples 10000

mirage-benchmark evaluate-trajectory \
  outputs/mirage-dev/0000-r000-TASKHASH.trajectory.json \
  --suite tasks/mirage-dev.v3.json \
  --output outputs/replayed.evaluation.json

mirage-benchmark aggregate-v3 outputs/mirage-dev \
  --output-json outputs/mirage-dev.aggregate.json \
  --output-csv outputs/mirage-dev.aggregate.csv
```

`run-v3-suite` resumes only trajectories whose task, public run seed,
environment configuration and policy configuration still match and whose
economic replay succeeds. Partial or stale runs are rerun; use `--force` to
rerun matching complete artifacts deliberately.

The same boundary is available from Python through `CommandAgentPolicy`,
`LLMAgentPolicy`/`create_api_agent_policy`, and `run_agent_episode`. Malformed
model actions become visible invalid actions. Provider, transport, command and
timeout failures stop as `infrastructure_error` without consuming an
environment step or contaminating invalid-action counts; timed-out command
process groups are cleaned up. Hidden client state is never sent to the
policy.

Saved trajectories include the public action schema, environment configuration,
reset options and public run seed, prompt/model parameters, dependency
versions, implementation hash and Git worktree provenance. Hash verification
alone is not treated as economic verification: `evaluate-trajectory` rebuilds
the task and requires every transition to match a fresh replay exactly. Agent
summaries expose cumulative reward and constraint totals. These additions do
not implement the still-planned TaskGenerator, repair teacher,
counterfactual-pair exporter, payoff DSL/solve-for, or fast/reference pricer.

The `mirage-benchmark` executable also retains the legacy v2 factorial
commands below so frozen v2 runs can still be reproduced
(`python -m mirage.benchmark_cli` works identically):

```bash
# Validate a market snapshot CSV (schema, provenance, contiguous rounds).
mirage-benchmark validate-market scenarios/mirage_csi/market_snapshots.example.csv

# Smoke-test the deterministic desk with the bundled synthetic example.
mirage-benchmark demo scenarios/mirage_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --product-json scenarios/mirage_csi/product.example.json --full

# Calibrate and freeze the risk budget on development episodes only.
mirage-benchmark calibrate-budget scenarios/mirage_csi/market_snapshots.example.csv \
  --episodes SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/mirage_csi/client.example.json \
  --base-budget-json scenarios/mirage_csi/risk_budget.example.json \
  --report-output outputs/budget_report.json \
  --budget-output outputs/risk_budget.frozen.json

# Run one registered model against one frozen episode. Add
# --roles-config config/benchmark_roles.yaml to switch on the LLM-native
# client/risk/desk roles (optional; omit it to stay pure-deterministic).
mirage-benchmark run-episode scenarios/mirage_csi/market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json outputs/risk_budget.frozen.json \
  --model deepseek-v4-flash --strategy ledger_archive \
  --output outputs/demo_run.json

# Freeze the full factorial manifest (Full/Partial x Static/Dynamic, n>=3
# replicates, partial_dynamic n=5), then execute it (reruns resume).
mirage-benchmark make-manifest --episodes SYNTHETIC_CSI500_DEMO \
  --models deepseek-v4-flash --output outputs/experiment_manifest.json
mirage-benchmark run-manifest scenarios/mirage_csi/market_snapshots.example.csv \
  --manifest outputs/experiment_manifest.json \
  --client-json scenarios/mirage_csi/client.example.json \
  --risk-budget-json outputs/risk_budget.frozen.json \
  --outputs-dir outputs/csi_benchmark

# Aggregate: per-(model, condition) cluster-bootstrap CIs, paired
# dynamic/partial-observability degradation contrasts with Holm correction.
mirage-benchmark aggregate outputs/csi_benchmark \
  --output-csv outputs/aggregate.csv --output-md outputs/aggregate.md

# Offline blind judge batch over frozen voluntary submissions.
mirage-benchmark judge-runs outputs/csi_benchmark \
  --judge-models deepseek-v4-pro qwen-max --salt "$(openssl rand -hex 16)"
```

Run tests:

```bash
.venv/bin/python -m pytest
```

## Repository layout

```text
MIRAGE/
|-- pyproject.toml                  # mirage-benchmark console script
|-- config/
|   |-- models.yaml                 # model registry: connection info per model
|   `-- benchmark_roles.yaml        # v2 role behaviour: structurer + 3 env NPCs + judges
|-- mirage/
|   |-- environment/                 # v3 reset/step, CLI/API agent adapters, trajectories
|   |-- benchmark.py                # MarketSnapshot, RiskBudget, ProductDomainSpec,
|   |                               #   TradingDesk, HardConstraintEngine, settlement
|   |-- benchmark_runner.py         # run_episode loop, compute_metrics
|   |-- benchmark_cli.py            # mirage-benchmark CLI (see Quick start)
|   |-- pricing.py                  # Black-Scholes, barriers, Monte Carlo, quote_economics
|   |-- products.py                 # ProductSpec, ClientProfile, MarketState
|   |-- env_agents.py               # FrozenEnvAgent, grounding, response cache
|   |-- role_config.py              # benchmark_roles.yaml loader/validator
|   |-- judge.py                    # offline soft-quality judge + reliability metrics
|   |-- stats.py                    # seeding, bootstrap CI, Wilcoxon, permutation, Holm
|   |-- experiment.py               # factorial manifest + paired condition contrasts
|   |-- llm.py                      # OpenAI-compatible and Anthropic clients, mock client
|   `-- market_builder.py, cffex_data.py, tushare_*.py, formal_*.py, raw_data_audit.py
|                                   # market-data acquisition, provenance audit pipeline
|-- scenarios/
|   |-- mirage_csi/                 # v2 episodes: snapshots, prompts/, benchmark.yaml
|   `-- structurer_nasdaq/          # legacy Playground scenario data (no runnable entry point)
|-- docs/
|   |-- BENCHMARK_PROTOCOL.md       # protocol of record
|   |-- DATA_AUDIT.md               # source audit and go/no-go gates
|   `-- redesign/                   # v2 design record (REDESIGN_PLAN.md and drafts)
|-- data/                           # derived/frozen calibration artifacts (gitignored raw data)
|-- outputs/                        # run artifacts (results JSON, manifests, aggregates)
|-- tests/
|-- README.md                       # English README
`-- README_zh.md                    # Chinese README
```

## Data provenance and license

Production market data must carry explicit source provenance and pass the
gates frozen in [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md); the checked-in
`scenarios/mirage_csi/market_snapshots.example.csv` is synthetic and cannot be
used for reported results. Only abstract workflow, schemas and synthetic
rules may be derived from private institutional material -- no private
document, client, institution, limit or identifying metadata may enter
released data or prompts.

Code is released under the [MIT License](LICENSE).
