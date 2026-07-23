"""Training-facing wrappers around the vector-reward MIRAGE core."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from typing import Any, Mapping

from .core import MirageStructurerEnv
from .types import (
    EnvironmentAction,
    Observation,
    ScalarizationSpec,
    scalarize_reward,
)


def _spec_hash(spec: ScalarizationSpec) -> str:
    encoded = json.dumps(
        asdict(spec),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ScalarizedMirageEnv:
    """Expose a standard scalar five-tuple without discarding audit vectors.

    The deterministic core deliberately returns a reward vector.  Training
    code must opt into a versioned :class:`ScalarizationSpec`; the original
    vector and constraint signals remain available in ``info`` on every step.
    """

    def __init__(
        self,
        environment: MirageStructurerEnv,
        scalarization: ScalarizationSpec,
    ) -> None:
        if not isinstance(environment, MirageStructurerEnv):
            raise TypeError("environment must be a MirageStructurerEnv")
        if not isinstance(scalarization, ScalarizationSpec):
            raise TypeError("scalarization must be a ScalarizationSpec")
        self.environment = environment
        self.scalarization = scalarization
        self.scalarization_hash = _spec_hash(scalarization)

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
        wrapped["scalarization_hash"] = self.scalarization_hash
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
        info["constraint_signals"] = asdict(transition.constraint_signals)
        info["tool_result"] = copy.deepcopy(transition.tool_result)
        info["state_hash_before"] = transition.state_hash_before
        info["state_hash_after"] = transition.state_hash_after
        info["scalarization"] = asdict(self.scalarization)
        info["scalarization_hash"] = self.scalarization_hash
        return (
            transition.observation,
            reward,
            transition.terminated,
            transition.truncated,
            info,
        )


__all__ = ["ScalarizedMirageEnv"]
