"""Training-facing wrappers around the vector-reward Stochopia core."""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Mapping

from .core import StochopiaStructurerEnv
from .types import (
    LEVEL0_UNMEASURED_COMPONENTS,
    EnvironmentAction,
    Observation,
    ScalarizationSpec,
    scalarize_reward,
)


class ScalarizedStochopiaEnv:
    """Expose a standard scalar five-tuple without discarding audit vectors.

    The deterministic core deliberately returns a reward vector.  Training
    code must opt into a versioned :class:`ScalarizationSpec`; the original
    vector and constraint signals remain available in ``info`` on every step.
    """

    def __init__(
        self,
        environment: StochopiaStructurerEnv,
        scalarization: ScalarizationSpec,
    ) -> None:
        if not isinstance(environment, StochopiaStructurerEnv):
            raise TypeError("environment must be a StochopiaStructurerEnv")
        if not isinstance(scalarization, ScalarizationSpec):
            raise TypeError("scalarization must be a ScalarizationSpec")
        # Reject up front: scalarize_reward would raise only after the inner
        # environment had already stepped.
        unmeasured = sorted(
            name
            for name, weight in scalarization.weights.items()
            if weight != 0.0 and name in LEVEL0_UNMEASURED_COMPONENTS
        )
        if unmeasured:
            raise ValueError(f"Level 0 never measures components: {unmeasured}")
        self.environment = environment
        self.scalarization = scalarization

    @property
    def task(self):
        return self.environment.task

    @property
    def done(self) -> bool:
        return self.environment.done

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        observation, info = self.environment.reset(seed=seed, options=options)
        wrapped = dict(info)
        wrapped["scalarization"] = asdict(self.scalarization)
        return observation, wrapped

    def step(
        self,
        action: EnvironmentAction,
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        transition = self.environment.step(action)
        reward = scalarize_reward(
            transition.reward_components,
            self.scalarization,
        )
        info = copy.deepcopy(dict(transition.info))
        info["reward_components"] = asdict(transition.reward_components)
        # Normalisers are hidden (initial client cash, dealer risk capital);
        # value / normalized_value would reveal them to an agent reading info.
        for term in info["reward_components"].values():
            if isinstance(term, dict):
                term.pop("normalized_value", None)
        reward_summary = info.get("episode_summary", {}).get("reward")
        if isinstance(reward_summary, dict):
            reward_summary.pop("normalized_totals", None)
        info["constraint_signals"] = asdict(transition.constraint_signals)
        info["tool_result"] = copy.deepcopy(transition.tool_result)
        info["scalarization"] = asdict(self.scalarization)
        return (
            transition.observation,
            reward,
            transition.terminated,
            transition.truncated,
            info,
        )


__all__ = ["ScalarizedStochopiaEnv"]
