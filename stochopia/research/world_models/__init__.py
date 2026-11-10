"""Registry of frozen world models usable as Stochopia's P-side generator.

Each entry is a reviewed, vendored ``forecast(history, query) -> (returns, q)``
implementing ``stochopia.market-world-model-forecast.v1``.  Models produced by
an external search loop are copied here unchanged (apart from a provenance
header comment) so that Stochopia's
simulated world is the loop's current cognition while the realised market
stays the only scorer.

Every entry sits on the generator tier ladder of StructurerDSL SPEC section
11.2: ``draft`` -> ``coherent`` -> ``predictive`` -> ``decision_grade``.
``tier_checks`` records the status of each tier above ``draft``; tiers are
cumulative, so ``tier`` is the highest one whose check and all lower checks
passed.  A generator below ``decision_grade`` is development-only: dominance
results obtained on its paths must not be used as claim evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

Forecast = Callable[[Mapping[str, np.ndarray], Mapping[str, Any]], tuple[np.ndarray, np.ndarray]]

TIERS = ("draft", "coherent", "predictive", "decision_grade")


@dataclass(frozen=True)
class WorldModelEntry:
    name: str
    description: str
    provenance: Mapping[str, Any]
    scores: Mapping[str, Any]
    tier_checks: Mapping[str, str]
    loader: Callable[[], Forecast] = field(repr=False)

    @property
    def tier(self) -> str:
        reached = TIERS[0]
        for name in TIERS[1:]:
            if self.tier_checks.get(name) != "passed":
                break
            reached = name
        return reached

    @property
    def development_only(self) -> bool:
        return self.tier != TIERS[-1]

    def forecast(self) -> Forecast:
        return self.loader()


def _reference() -> Forecast:
    from ..forecast_protocol import reference_forecast

    return reference_forecast


def _r02s01() -> Forecast:
    from .r02s01 import forecast

    return forecast


_CHECKS_NOT_RUN = {name: "not_run" for name in TIERS[1:]}

WORLD_MODELS: dict[str, WorldModelEntry] = {
    "ewma_gbm_reference": WorldModelEntry(
        name="ewma_gbm_reference",
        description="EWMA(0.94) Gaussian GBM with trailing correlation; Q random walk (scoring reference)",
        provenance={"source": "stochopia.research.forecast_protocol.reference_forecast"},
        scores={"dev": 0.0, "test": 0.0, "note": "skill is defined relative to this model"},
        tier_checks=dict(_CHECKS_NOT_RUN),
        loader=_reference,
    ),
    "r02s01": WorldModelEntry(
        name="r02s01",
        description=(
            "Per-path latent stress state (skewed t) driving stress-dependent cross-index correlation; "
            "HAR-style mean-reverting volatility with path-level stochastic volatility and leverage; "
            "log-IV AR(1) with variance-risk-premium drift linked to path realised vol; OU carry linked "
            "to path returns"
        ),
        provenance={
            "source": "OpenHyra artifacts/stochopia-world-model-20261108/deepseek/programs/r02s01.py",
            "loop": "openhyra-stochopia-world-model.v1",
            "generator": "deepseek-v4.1-flash, provider-default reasoning",
            "lineage": ["seed", "r00s01", "r01s00", "r02s01"],
            "date": "2026-11-08",
        },
        scores={
            "protocol": "stochopia.market-world-model-forecast.v1",
            "dev": 0.0478, "dev_lcb_95": 0.0258,
            "test": 0.0783, "test_lcb_95": 0.0280,
            "test_family_skill": {"volatility": 0.241, "q_quote": 0.054, "payoff_probe": 0.018, "return": 0.0},
            "test_caveat": "2023-07..2025-11 was already used by the project and is inside the model's pretraining era",
        },
        tier_checks={
            # The check is not implemented.  Known issues: the model standardises
            # across the sample dimension, so its distribution depends on the
            # requested sample count and the paths of one call are exchangeable
            # but not independent (see the 2026-11-08 research note).
            "coherent": "not_run",
            # CRPS skill vs EWMA-GBM has a positive block-bootstrap LCB on dev
            # and test, which is related to but not the predictive-tier
            # criterion (predictive log score vs single GBM plus stylised-fact
            # checks, and a moving-block-bootstrap / GARCH-t comparison).
            "predictive": "not_run",
            "decision_grade": "not_run",  # no historical-vs-synthetic policy ranking check yet
        },
        loader=_r02s01,
    ),
}


def get_world_model(name: str) -> WorldModelEntry:
    try:
        return WORLD_MODELS[name]
    except KeyError as exc:
        raise KeyError(f"unknown world model {name!r}; available: {sorted(WORLD_MODELS)}") from exc
