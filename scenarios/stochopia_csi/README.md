# Stochopia CSI protocol

This directory freezes the 12-episode research protocol. The checked-in CSV is
synthetic and exists only to document the schema and smoke-test the CLI; it must
not be used as an experimental result.

The production CSV must contain one row per month-end decision and carry a
non-empty `source` field. Preferred sources are CFFEX MO contract-level history
for CSI1000 and SSE 510500 ETF option history for CSI500. When paired calls and
puts exist, infer the forward with put-call parity. The loader then follows the
frozen ATM-IV to realized-vol fallback in `benchmark.yaml`.

Validate a prepared file:

```bash
python -m stochopia.benchmark_cli validate-market path/to/market_snapshots.csv
```

Run the deterministic example quote:

```bash
python -m stochopia.benchmark_cli demo market_snapshots.example.csv \
  --episode SYNTHETIC_CSI500_DEMO --product-json product.example.json
```

Build formal month-end snapshots from a public daily close file containing
`date,underlying,close,source`:

```bash
python -m stochopia.benchmark_cli build-market daily.csv market_snapshots.csv
```

## `prompts/` and `config/benchmark_roles.yaml`

`prompts/` holds the frozen system prompt for each v2 role:

| file | role id (in `config/benchmark_roles.yaml`) |
|------|------|
| `structurer.md` | `structurer` (the tested model) |
| `client_main.md` | `client_main` |
| `risk_control.md` | `risk_control` |
| `trading_desk.md` | `trading_desk` |

`config/benchmark_roles.yaml` is the only place that wires a role id to a
model, temperature and this prompt file (`system_prompt_file`); the repo root
is `stochopia.role_config.load_role_specs`'s default `base_dir`, so the
`system_prompt_file` paths in `benchmark_roles.yaml` are written relative to
the repository root (e.g. `scenarios/stochopia_csi/prompts/client_main.md`), not
relative to this directory. The loader hashes each prompt file's contents
(`system_prompt_sha256`) at load time so a prompt edit is auditable in any
run's recorded `roles_config_sha256`.

Editing a prompt file here changes that role's behaviour the next time
`--roles-config config/benchmark_roles.yaml` is passed to `run-episode` /
`run-manifest`; it has no effect on a run that omits `--roles-config` (the
environment then stays pure-deterministic, per
[docs/BENCHMARK_PROTOCOL.md](../../docs/BENCHMARK_PROTOCOL.md)). The three
env-NPC roles (`client_main`, `risk_control`, `trading_desk`) are pinned to a
frozen model/temperature combination (`main_npc_lineup_id: stochopia.npc-lineup.fixed.v1` in
`benchmark_roles.yaml`) so that only the `structurer` model varies across a
benchmark run -- do not repoint their `model_ref` per tested model. See
"LLM-native roles" in the protocol document for the full behavioural contract
(numeric-authority discipline, grounding validation, response caching,
degraded fallback).

`structurer.md` also documents the `consult` action (qualitative, non-binding
chat with an env role), which is only usable when `--roles-config` is passed;
without it, `consult` returns a fixed degraded-fallback string and the rest of
the protocol -- including the deterministic checks and `dealer_margin` -- is
unaffected.
