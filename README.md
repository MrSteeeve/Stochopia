# Stochopia

**An order-flow market world model revised by an LLM research loop**

[中文 README](README_zh.md)

Stochopia is a market world model. It describes how order placements, cancellations, trades and price changes relate to one another, and it is revised by an LLM research loop that proposes hypotheses, selects evidence and submits revisions. The first research question is the domain of validity of the square-root law of market impact.

## Status

The current repository contains the reusable world-model, market evaluation, and synthetic research infrastructure. The LLM research loop is maintained in [OpenHyra](https://github.com/MrSteeeve/OpenHyra/tree/world-model).

| Path | Contents |
|---|---|
| `stochopia/neural/` | Causal LOB world-model interfaces, training, and reference operators |
| `stochopia/harness/` | Legal event replay and JAX-LOB matching |
| `stochopia/market/` | Market windows, tape metrics, rollout evaluation, and SZSE evaluation |
| `stochopia/measure/` | Data properties, impact curves, and evaluation metrics |
| `stochopia/sim/` | Controlled synthetic propagator and queue-reactive worlds |
| `experiments/synthetic_tau_z/` | Reproducible conditional-operator study and report |

The historical P0 and P1 research reports and code are preserved in the `tier0-p0-final` and `p1-szse-final` tags. Current measurement protocols and P2 reports remain under `docs/research/`.

## Previous version: Benchmark v3

This repository used to be Benchmark v3, a structured-product design benchmark. It is no longer maintained. Its final state is kept under the tag `benchmark-v3-final`:

```bash
git checkout benchmark-v3-final
```

The research notes for that version (in Chinese) are kept under the same tag, in `docs/research/`; the file paths they mention exist only there.

## Development

```bash
uv sync --locked --python 3.12 --all-extras --dev
```

```bash
.venv/bin/python -m pytest
```

The synthetic study requires no private market data or LLM API keys.
