# Stochopia

**An order-flow market world model revised by an LLM research loop**

[中文 README](README_zh.md)

Stochopia is a market world model. It describes how order placements, cancellations, trades and price changes relate to one another, and it is revised by an LLM research loop that proposes hypotheses, selects evidence and submits revisions. The first research question is the domain of validity of the square-root law of market impact.

## Status

The new code is under construction. For now the repository only holds general-purpose modules:

| File | Contents |
|---|---|
| `stochopia/llm.py` | Clients for OpenAI-compatible and Anthropic APIs; models are registered in `config/models.yaml` |
| `stochopia/llm_json.py` | JSON extraction from model replies |
| `stochopia/stats.py` | Seed derivation, bootstrap confidence intervals, paired tests, Holm correction |
| `stochopia/common.py` | Canonical JSON, functional hash IDs, atomic writes |

Research notes are in `docs/research/` (in Chinese).

## Previous version: Benchmark v3

This repository used to be Benchmark v3, a structured-product design benchmark. It is no longer maintained. Its final state is kept under the tag `benchmark-v3-final`:

```bash
git checkout benchmark-v3-final
```

Notes in `docs/research/` dated before 2026-11-11 describe that version; the file paths they mention exist only under that tag.

## Development

```bash
uv sync --all-extras --dev
```

```bash
.venv/bin/python -m pytest
```

API keys are read from environment variables; see `.env.example` for the names.
