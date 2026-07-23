"""MIRAGE v3-spine training environment (Curriculum Level 0).

The package is intentionally framework-neutral: core has no LLM or Gymnasium
dependency, and exposes only typed tasks, actions, observations, transitions,
and a versioned trajectory recorder.
"""

from .core import ENVIRONMENT_VERSION, PRICING_VERSION, MirageStructurerEnv
from .trajectory import (
    TRAJECTORY_FORMAT,
    TrajectoryEntry,
    TrajectoryMetadata,
    TrajectoryRecorder,
    verify_trajectory,
)
from .types import (
    TASK_SCHEMA,
    TASK_VERSION,
    Action,
    AskClient,
    ConstraintSignals,
    EnvironmentAction,
    EpisodeTask,
    InvalidAction,
    Observation,
    RequestQuote,
    RewardComponents,
    Skip,
    StepTransition,
    SubmitDesign,
    SubmitProduct,
)

__all__ = [
    "ENVIRONMENT_VERSION",
    "PRICING_VERSION",
    "TASK_SCHEMA",
    "TASK_VERSION",
    "TRAJECTORY_FORMAT",
    "Action",
    "AskClient",
    "ConstraintSignals",
    "EnvironmentAction",
    "EpisodeTask",
    "InvalidAction",
    "MirageStructurerEnv",
    "Observation",
    "RequestQuote",
    "RewardComponents",
    "Skip",
    "StepTransition",
    "SubmitDesign",
    "SubmitProduct",
    "TrajectoryEntry",
    "TrajectoryMetadata",
    "TrajectoryRecorder",
    "verify_trajectory",
]
