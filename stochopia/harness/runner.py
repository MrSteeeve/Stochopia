"""Execute P0 scenarios using only the data and fits allowed by the manifest."""

from __future__ import annotations

import json
import os
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from math import isclose, isfinite, sqrt
from numbers import Integral, Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any, Literal
from urllib.parse import quote

import torch
from torch.nn import Module

from experiments.tier0_feasibility import WORLDS, evaluate_model, fit_model, initialize_model, json_safe
from stochopia.common import atomic_write_json, canonical_json
from stochopia.harness.contract import (
    AnalysisSpec, ArmKind, CandidateRef, DataSplit, EvaluationTask, ExperimentManifest, FitRecord,
    FrozenManifest, PairedEvalResult, ReportStatus, ScenarioSpec, UnitObservation,
)
from stochopia.harness.lob_adapter import LobComparison, run_short_lob
from stochopia.harness.paired_stats import evaluate_pairs
from stochopia.neural.seq import ModelSpec, make_model
from stochopia.sim.world import Day, WorldSpec, injection_schedule, simulate_day
from stochopia.stats import derive_seed


Manifest = ExperimentManifest | FrozenManifest
_TRAIN_DEFAULTS = {
    "hidden": 96, "layers": 1, "steps": 1200, "batch": 48, "window": 256,
    "lr": 0.002, "device": "cpu", "own_start": "inside", "label_miss": 0.0, "label_false": 0.0,
}
_ACTION_DEFAULTS = {"context": 256, "children": 50, "spacing": 9, "relax": 500, "windows_per_day": 32}


# --- Phase and simulation inputs ---


def build_scenarios(manifest_view: Manifest, split: Literal["dev", "test"]) -> tuple[ScenarioSpec, ...]:
    """Return registered scenarios for the view's evaluation phase.

    The development view exposes dev only. The frozen view exposes test only.
    Neither selection resolves a data file or simulates a day.
    """
    manifest_view.validate()
    if isinstance(manifest_view, FrozenManifest):
        if split != "test":
            raise ValueError("a frozen runner can only build test scenarios")
        return manifest_view.experiment.scenarios
    if split != "dev":
        raise ValueError("a development runner can only build dev scenarios")
    return manifest_view.scenarios


def _world(environment: Mapping[str, Any]) -> WorldSpec:
    name = environment.get("world", "plain")
    names = {"tier0-plain": "plain", "tier0-informed": "informed", "tier0-propagator": "plain"}
    name = names.get(name, name)
    if name not in WORLDS:
        raise ValueError(f"unknown Tier 0 world: {name}")
    world = WORLDS[name]
    overrides = dict(environment.get("world_overrides", {}))
    if "flow" in overrides:
        overrides["flow"] = replace(world.flow, **overrides["flow"])
    if "kernel" in overrides:
        overrides["kernel"] = replace(world.kernel, **overrides["kernel"])
    return replace(world, **overrides)


def _simulate_units(split: DataSplit, world: WorldSpec, unit_ids: Sequence[str]) -> list[Day]:
    """Resolve unit IDs to simulator indices and verify declared seeds before generation."""
    namespace = split.source.get("namespace")
    seed_world = split.source.get("world", world.name)
    if not isinstance(namespace, str) or not namespace or not isinstance(seed_world, str) or not seed_world:
        raise ValueError("simulated data requires a registered namespace and world name")
    simulated_world = replace(world, name=seed_world)
    registered_indices = split.source.get("day_indices", dict(zip(split.unit_ids, range(len(split.unit_ids)))))
    indices = []
    for unit_id in unit_ids:
        index = registered_indices.get(unit_id)
        if unit_id not in split.unit_ids or not isinstance(index, Integral) or isinstance(index, bool) or index < 0:
            raise ValueError(f"unregistered simulated day index for {unit_id}")
        index = int(index)
        expected = derive_seed("stochopia.sim.world", seed_world, namespace, index)
        if split.unit_seeds.get(unit_id) != expected:
            raise ValueError(f"registered seed does not match simulator identity for {unit_id}")
        indices.append(index)
    return [simulate_day(simulated_world, namespace, index) for index in indices]


def _action_args(scenario: ScenarioSpec) -> dict[str, int]:
    """Resolve trade-clock inputs for the Tier 0 buy/sell antisymmetric protocol."""
    args = {**_ACTION_DEFAULTS}
    aliases = {"n_children": "children", "n_relax": "relax"}
    plan = dict(scenario.action_plan)
    if "side" in plan and plan.pop("side") not in ("both", "antisymmetric"):
        raise ValueError("Tier 0 measures a buy/sell antisymmetric response and requires both sides")
    for alias, name in aliases.items():
        if alias in plan:
            value = plan.pop(alias)
            if name in plan and plan[name] != value:
                raise ValueError(f"conflicting values for {alias} and {name}")
            plan[name] = value
    unknown = plan.keys() - args.keys()
    if unknown:
        raise ValueError(f"unrecognized Tier 0 action plan keys: {sorted(unknown)}")
    args.update(plan)
    for key, value in args.items():
        minimum = 0 if key in ("spacing", "relax") else 1
        if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{key} requires an integer at least {minimum}")
        args[key] = int(value)
    return args


# --- Fitting and immutable model references ---


def _fit_args(plan: Mapping[str, Any], candidate: CandidateRef, price_recipe: str) -> Namespace:
    settings = dict(plan)
    settings.pop("world", None)
    settings.pop("world_overrides", None)
    if "learning_rate" in settings:
        if "lr" in settings:
            raise ValueError("training plan must specify one of lr and learning_rate")
        settings["lr"] = settings.pop("learning_rate")
    unknown = settings.keys() - _TRAIN_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"unrecognized Tier 0 training plan keys: {sorted(unknown)}")
    args = _TRAIN_DEFAULTS | settings
    for key in ("hidden", "layers", "steps", "batch", "window"):
        value = args[key]
        if not isinstance(value, Integral) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{key} requires a positive integer")
        args[key] = int(value)
    for key in ("lr", "label_miss", "label_false"):
        value = args[key]
        if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
            raise ValueError(f"{key} requires a finite real number")
        args[key] = float(value)
    if args["lr"] <= 0:
        raise ValueError("lr must be positive")
    if candidate.backbone not in ("gru", "ssm", "lru"):
        raise ValueError(f"unsupported Tier 0 backbone: {candidate.backbone}")
    return Namespace(**args, cell=candidate.backbone, recipe=price_recipe)


def _save_model(path: Path, model: Module) -> None:
    """Publish a complete tensor checkpoint in the fit's artifact directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = {}
    for key, value in model.state_dict().items():
        weights[key] = value.detach().cpu()
    with NamedTemporaryFile(dir=path.parent, prefix=".weights-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            torch.save({"spec": asdict(model.spec), "weights": weights}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_model(path: str, device: str) -> Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = make_model(ModelSpec(**payload["spec"]))
    model.load_state_dict(payload["weights"])
    return model.to(device).eval()


def _resolve_recipe(recipe: str) -> tuple[str, bool]:
    if recipe in ("seed", "anonymous"):
        return "anonymous", False
    if recipe == "separate_flow":
        return "anonymous", True
    if recipe == "separate_flow_own_price":
        return "self", True
    if recipe in ("self", "self-exogenous"):
        return recipe, False
    raise ValueError(f"unknown Tier 0 recipe: {recipe}")


class Runner:
    """Bind deterministic stages to a development or immutable verification view."""

    def __init__(self, manifest: Manifest) -> None:
        self.phase: Literal["dev", "test"] = "test" if isinstance(manifest, FrozenManifest) else "dev"
        self.scenarios = {item.scenario_id: item for item in build_scenarios(manifest, self.phase)}
        self.frozen = manifest if isinstance(manifest, FrozenManifest) else None
        self.experiment = manifest if self.frozen is None else self.frozen.experiment
        self.candidates = {item.candidate_id: item for item in self.experiment.candidates}
        self.seeds = manifest.seeds
        self._scenes_by_candidate = self._bind_candidate_scenes()

    def _bind_candidate_scenes(self) -> dict[str, list[ScenarioSpec]]:
        """Bind visible candidates and their recipe dependencies to each scene."""
        bindings: dict[str, list[ScenarioSpec]] = {key: [] for key in self.candidates}
        for scene in self.scenarios.values():
            pending = list(scene.candidate_ids)
            visited = set()
            while pending:
                key = pending.pop()
                if key in visited:
                    continue
                visited.add(key)
                bindings[key].append(scene)
                candidate = self.candidates[key]
                if candidate.parent_id is not None:
                    pending.append(candidate.parent_id)
                background = candidate.implementation.get("background_candidate_id")
                if isinstance(background, str) and background in self.candidates:
                    pending.append(background)
        return bindings

    def fit_candidate(self, candidate_id: str, train_repeat_id: str, output_dir: str | Path) -> FitRecord:
        """Fit against train only and atomically publish its resolved weight references.

        Complete records are reused for continuation with identical fitting inputs.
        A frozen Runner cannot invoke fitting and opens only the sealed weight files.
        """
        if self.frozen is not None:
            raise ValueError("frozen runners cannot fit or modify model weights")
        return self._fit_registered(candidate_id, train_repeat_id, Path(output_dir).resolve(), ())

    def _fit_registered(
        self, candidate_id: str, repeat_id: str, output_dir: Path, lineage: tuple[str, ...],
    ) -> FitRecord:
        if candidate_id in lineage:
            raise ValueError("candidate parent references form a cycle")
        if candidate_id not in self.candidates or repeat_id not in self.experiment.train_seeds:
            raise ValueError("fit requires a registered candidate and training repeat")
        candidate = self.candidates[candidate_id]
        seed = self.experiment.train_seeds[repeat_id]
        split = self.experiment.splits["train"]
        related = self._scenes_by_candidate[candidate_id]
        if not related:
            raise ValueError(f"candidate {candidate_id} is not bound to a scenario")
        environment_kind = related[0].environment_kind
        if any(scene.environment_kind != environment_kind for scene in related[1:]):
            raise ValueError(f"candidate {candidate_id} is shared across incompatible environment types")
        is_lob_replay = environment_kind == "lob"
        candidate_spec = candidate.to_dict()
        overrides = candidate_spec["implementation"].get("train_plan_overrides", {})
        plan = {**self.experiment.to_dict()["train_plan"], **overrides}
        world = None
        price_recipe, separated = ("lob_replay", False) if is_lob_replay else _resolve_recipe(candidate.recipe)
        if not is_lob_replay:
            world = _world(plan) if {"world", "world_overrides"} & overrides.keys() else self._training_world(related)
            args = _fit_args(plan, candidate, price_recipe)
            separated = separated or "background_candidate_id" in candidate.implementation
        parent = None
        background_ref = None
        inherited_price_ref = None
        if separated:
            if candidate.parent_id is None:
                raise ValueError("a separated recipe needs a registered parent")
            ancestors = (*lineage, candidate_id)
            background_candidate_id = candidate.implementation.get("background_candidate_id", candidate.parent_id)
            parent = self._fit_registered(candidate.parent_id, repeat_id, output_dir, ancestors)
            background = parent
            if background_candidate_id != candidate.parent_id:
                background = self._fit_registered(background_candidate_id, repeat_id, output_dir, ancestors)
            background_ref = background.background_weights_ref or background.price_weights_ref or background.model_ref
            if candidate.recipe == "separate_flow":
                if candidate.backbone != parent.config["cell"]:
                    raise ValueError("separate_flow must preserve the parent price backbone")
                inherited_price_ref = parent.price_weights_ref or parent.model_ref
        if separated and candidate.channels == ("price",):
            if parent.background_weights_ref is None or parent.background_weights_ref != background_ref:
                raise ValueError("a price-only change requires the same separated background as its parent")

        registered = canonical_json({
            "experiment": self.experiment.exp_id, "fit_version": self.experiment.fit_version,
            "candidate": candidate_spec, "train_split": split.to_dict(), "seed": seed,
            "plan": plan,
            "training_world": None if world is None else world.manifest(),
            "background_ref": background_ref, "inherited_price_ref": inherited_price_ref,
        })
        directory = output_dir / f"candidate-{quote(candidate_id, safe='')}" / f"repeat-{quote(repeat_id, safe='')}"
        record_path = directory / "fit.json"
        if record_path.exists():
            fit = FitRecord.from_dict(json.loads(record_path.read_text(encoding="utf-8")))
            if fit.config.get("registered") != registered:
                raise ValueError("existing fit has a different registered training configuration")
            for ref in dict.fromkeys((fit.model_ref, fit.price_weights_ref, fit.background_weights_ref)):
                if ref is not None and not Path(ref).is_file():
                    raise FileNotFoundError(f"registered fit artifact is missing: {ref}")
            return fit

        fit_id = f"{quote(candidate_id, safe='')}:{quote(repeat_id, safe='')}"
        data_ref = f"{self.experiment.exp_id}/train"
        if is_lob_replay:
            replay_config = {"kind": "lob_replay", "registered": registered, "train_seconds": 0.0}
            artifact = directory / "replay.json"
            atomic_write_json(artifact, replay_config)
            fit = FitRecord(
                fit_id=fit_id, candidate_id=candidate_id, train_repeat_id=repeat_id,
                train_data_ref=data_ref, config=replay_config, seed=seed,
                model_ref=str(artifact), parent_id=candidate.parent_id,
            )
            atomic_write_json(record_path, fit.to_dict())
            return fit

        train_seconds = 0.0
        price_ref = inherited_price_ref
        model_spec = None if parent is None else parent.config.get("model_spec")
        if price_ref is None:
            if world is None or world.n_trades <= args.window + 1:
                raise ValueError("registered training world is too short for a training window")
            started = perf_counter()
            days = _simulate_units(split, world, split.unit_ids)
            model = initialize_model(args, days, seed)
            if not isfinite(model.spec.return_scale) or model.spec.return_scale <= 0:
                raise ValueError("training returns require a finite positive scale")
            fit_model(model, args, days, seed)
            model_spec = asdict(model.spec)
            checkpoint = directory / "weights.pt"
            _save_model(checkpoint, model)
            price_ref = str(checkpoint)
            train_seconds = perf_counter() - started

        config = vars(args) | {
            "registered": registered, "train_seconds": train_seconds, "model_spec": model_spec,
        }
        fit = FitRecord(
            fit_id=fit_id, candidate_id=candidate_id, train_repeat_id=repeat_id, train_data_ref=data_ref,
            config=config, seed=seed, model_ref=price_ref, parent_id=candidate.parent_id,
            price_weights_ref=price_ref, background_weights_ref=background_ref,
        )
        atomic_write_json(record_path, fit.to_dict())
        return fit

    def _training_world(self, scenes: list[ScenarioSpec]) -> WorldSpec:
        plan = self.experiment.train_plan
        if "world" in plan or "world_overrides" in plan:
            return _world(plan)
        world = _world(scenes[0].environment)
        for scene in scenes[1:]:
            if _world(scene.environment) != world:
                raise ValueError("a candidate shared across worlds requires an explicit training world")
        return world


# --- Paired evaluation and development diagnosis ---

    def _data_split(self) -> DataSplit:
        if self.frozen is not None:
            return self.frozen.test_split
        return self.experiment.splits["dev"]

    def _check_registered_fit(self, fit: FitRecord) -> None:
        fit.validate()
        candidate = self.candidates.get(fit.candidate_id)
        if candidate is None or candidate.parent_id != fit.parent_id:
            raise ValueError("fit lineage does not match a registered model")
        if self.experiment.train_seeds.get(fit.train_repeat_id) != fit.seed:
            raise ValueError("fit seed does not match a registered training repeat")

    def _check_pair(self, task: EvaluationTask, parent: FitRecord, candidate: FitRecord) -> ScenarioSpec:
        task.validate()
        self._check_registered_fit(parent)
        self._check_registered_fit(candidate)
        if task.split != self.phase or not set(task.unit_ids) <= set(self._data_split().unit_ids):
            raise ValueError("evaluation task requested units outside the active split")
        if self.frozen is not None:
            if task not in self.frozen.tasks or parent not in self.frozen.fits or candidate not in self.frozen.fits:
                raise ValueError("test evaluation requires the sealed task and fit records")
        scene = self.scenarios.get(task.scenario_id)
        if scene is None or (parent.fit_id, candidate.fit_id) != (task.parent_fit_id, task.candidate_fit_id):
            raise ValueError("evaluation task references the wrong scenario or fits")
        if parent.candidate_id != scene.parent_id or candidate.candidate_id not in scene.candidate_ids:
            raise ValueError("evaluation requires the registered baseline and menu candidate")
        if parent.candidate_id == candidate.candidate_id:
            raise ValueError("paired evaluation requires different candidate and baseline")
        if (parent.train_repeat_id != candidate.train_repeat_id or parent.seed != candidate.seed
                or parent.train_data_ref != candidate.train_data_ref):
            raise ValueError("paired models must share their training data and repeat seed")
        return scene

    def _measure_tier0(
        self, scene: ScenarioSpec, world: WorldSpec, days: list[Day], fit: FitRecord, unit_seeds: tuple[int, ...],
    ) -> dict[str, Any]:
        device = self.experiment.train_plan.get("device", _TRAIN_DEFAULTS["device"])
        args = Namespace(**_action_args(scene), window=int(fit.config["window"]), device=device)
        schedule = injection_schedule(args.children, args.spacing, args.relax)
        if args.context + int((~schedule).sum()) > world.n_trades:
            raise ValueError("context and action schedule exceed the registered simulation day length")
        model = _load_model(fit.price_weights_ref or fit.model_ref, device)
        background = None
        if fit.background_weights_ref is not None:
            if fit.background_weights_ref == (fit.price_weights_ref or fit.model_ref):
                background = model
            else:
                background = _load_model(fit.background_weights_ref, device)
        if background is not None:
            same_scale = isclose(background.spec.return_scale, model.spec.return_scale, rel_tol=1e-9, abs_tol=1e-12)
            if background.spec.own_flag or not same_scale:
                raise ValueError("separated background must be anonymous and use the same training scale")
        return evaluate_model(
            model, args, world, days, rollout_seed=self.seeds["rollout"], seed_ci=self.seeds["delta_ci"],
            background_model=background, return_observations=True, rollout_unit_seeds=unit_seeds,
            include_impact_summary=False,
        )

    def _rollout_seeds(self, scene: ScenarioSpec, reference: str, unit_ids: Sequence[str]) -> tuple[int, ...]:
        seeds = []
        for unit_id in unit_ids:
            seeds.append(derive_seed("stochopia.harness.rollout", self.seeds["rollout"], scene.scenario_id, reference, unit_id))
        return tuple(seeds)

    def run_paired_eval(
        self, task: EvaluationTask, parent_fit: FitRecord, candidate_fit: FitRecord, *, train_seconds: float = 0.0,
    ) -> PairedEvalResult:
        """Measure aligned candidate and parent days using a single registered random plan."""
        scene = self._check_pair(task, parent_fit, candidate_fit)
        if not isfinite(train_seconds) or train_seconds < 0.0:
            raise ValueError("training cost must be finite and nonnegative")
        if self.frozen is not None and train_seconds != 0.0:
            raise ValueError("frozen evaluation only loads models and incurs no training cost")
        analyses = tuple(item for item in self.experiment.analysis if item.scenario_id == scene.scenario_id)
        started = perf_counter()
        if scene.environment_kind == "lob":
            comparison = self.run_short_lob(scene.scenario_id)
            if comparison.status is ReportStatus.UNAVAILABLE:
                return _unavailable_result(task, parent_fit, candidate_fit, comparison.reason, started, train_seconds)
            observations = _lob_observations(task, scene, parent_fit, candidate_fit, analyses, comparison)
        else:
            world = _world(scene.environment)
            days = _simulate_units(self._data_split(), world, task.unit_ids)
            unit_seeds = self._rollout_seeds(scene, task.random_inputs_ref, task.unit_ids)
            parent_raw = self._measure_tier0(scene, world, days, parent_fit, unit_seeds)
            candidate_raw = self._measure_tier0(scene, world, days, candidate_fit, unit_seeds)
            observations = _tier0_observations(task, scene, parent_fit, analyses, parent_raw)
            truth_observations = []
            for observation in observations:
                if observation.truth_path:
                    truth_observations.append(replace(
                        observation, arm=ArmKind.TRUTH, candidate_id=None, response_path=(),
                    ))
            observations.extend(truth_observations)
            observations.extend(_tier0_observations(task, scene, candidate_fit, analyses, candidate_raw))
        if not observations:
            return _unavailable_result(task, parent_fit, candidate_fit, "no registered readout is available", started, train_seconds)
        return evaluate_pairs(
            task, scene, parent_fit, candidate_fit, observations, analyses,
            cost={
                "train_seconds": train_seconds,
                "eval_seconds": perf_counter() - started,
            },
        )

    def run_short_lob(self, scenario_id: str) -> LobComparison:
        """Replay a registered short-book control and check its engine version."""
        scene = self.scenarios.get(scenario_id)
        if scene is None or scene.environment_kind != "lob":
            raise ValueError("a short matching run needs a registered LOB scenario")
        if scene.action_plan != {"short_fixture": True}:
            raise ValueError("the short LOB adapter requires its registered deterministic action fixture")
        replay = run_short_lob(depth=scene.environment.get("depth", 2))
        if replay.matching_version is not None and replay.matching_version != scene.matching_version:
            raise ValueError(f"registered matching engine is {scene.matching_version}, found {replay.matching_version}")
        return replay

    def diagnose(self, scenario_id: str, parent_fit: FitRecord) -> dict[str, Any]:
        """Compare free/clamped and clamped/truth residuals on development days only."""
        if self.frozen is not None:
            raise ValueError("diagnostic evidence can only be generated on development data")
        self._check_registered_fit(parent_fit)
        scene = self.scenarios.get(scenario_id)
        if scene is None or scene.environment_kind != "tier0_sim" or scene.parent_id != parent_fit.candidate_id:
            raise ValueError("diagnosis requires the registered Tier 0 baseline")
        split = self._data_split()
        world = _world(scene.environment)
        days = _simulate_units(split, world, split.unit_ids)
        seeds = self._rollout_seeds(scene, "diagnosis", split.unit_ids)
        measured = self._measure_tier0(scene, world, days, parent_fit, seeds)
        paths = measured["mean_paths"]
        options = scene.environment.get("diagnosis", {})
        conclusion = classify_diagnosis(
            paths["free"], paths["clamped"], paths["truth"],
            min_effect=options.get("min_effect", 0.02), dominance_margin=options.get("dominance_margin", 0.02),
        )
        if len(days) < options.get("min_pairs", 2):
            conclusion.update(status=ReportStatus.INCONCLUSIVE, dominant_channel=None, reason="too few development days")
        return conclusion | {
            "unit_ids": split.unit_ids, "mean_paths": paths,
            "induced_background_sign": json_safe(measured["induced_background_sign"]),
        }


def classify_diagnosis(
    free: Sequence[float], clamped: Sequence[float], truth: Sequence[float],
    *, min_effect: float, dominance_margin: float,
) -> dict[str, ReportStatus | float | str | None]:
    """Label the dominant residual in three aligned, mean response paths.

    The two scores are sqrt(mean((free-clamped)^2)) and sqrt(mean((clamped-truth)^2)).
    All paths must be finite, nonempty and identical in length. A channel is
    identified only when it reaches the effect threshold and beats the other by
    the registered margin. Small signals and ties retain both scores as inconclusive.
    """
    if len(free) == 0 or len(free) != len(clamped) or len(free) != len(truth):
        raise ValueError("diagnostic response paths must be nonempty and aligned")
    if not isfinite(min_effect) or not isfinite(dominance_margin) or min_effect <= 0 or dominance_margin <= 0:
        raise ValueError("diagnostic thresholds must be finite and positive")
    flow_squares = 0.0
    price_squares = 0.0
    for free_point, clamped_point, truth_point in zip(free, clamped, truth):
        if not all(isfinite(value) for value in (free_point, clamped_point, truth_point)):
            raise ValueError("diagnostic paths must contain finite response values")
        flow_squares += (free_point - clamped_point) ** 2
        price_squares += (clamped_point - truth_point) ** 2
    flow = sqrt(flow_squares / len(free))
    price = sqrt(price_squares / len(free))
    channel = None
    if flow >= min_effect and flow - price >= dominance_margin:
        channel = "flow"
    elif price >= min_effect and price - flow >= dominance_margin:
        channel = "price"
    return {
        "status": ReportStatus.SUPPORTED if channel else ReportStatus.INCONCLUSIVE,
        "dominant_channel": channel, "flow_rmse": flow, "price_rmse": price,
        "reason": None if channel else "residuals do not meet the effect and dominance thresholds",
    }


# --- Observations passed to the registered estimators ---


def _window_offsets(window: str, peak: int, length: int) -> tuple[int, ...]:
    if window in ("execution", "during_execution"):
        return tuple(range(peak + 1))
    if window == "peak":
        return (peak,)
    if window in ("recovery", "after_execution", "post_trade"):
        return tuple(range(peak + 1, length))
    if window in ("end", "endpoint"):
        return (length - 1,)
    if window in ("all", "full", "full_path", "horizon", "path"):
        return tuple(range(length))
    return ()


def _tier0_observations(
    task: EvaluationTask, scene: ScenarioSpec, fit: FitRecord, analyses: tuple[AnalysisSpec, ...], raw: dict[str, Any],
) -> list[UnitObservation]:
    rows = []
    seen = set()
    schedule = raw["schedule"]
    peak = max(index for index, child in enumerate(schedule) if child)
    for analysis in analyses:
        readout = analysis.readout
        key = (readout, analysis.window, analysis.unit)
        arm = ArmKind.CLAMPED
        if readout in ("free_rmse", "free_peak_response", "free_end_response", "background_bias", "induced_background_sign"):
            arm = ArmKind.FREE
        elif readout in ("nll", "heldout_loss", "no_injection_drift"):
            arm = ArmKind.NO_INJECTION
        if key in seen or analysis.acceptance.get("arm", arm.value) != arm.value:
            continue
        seen.add(key)
        offsets = _window_offsets(analysis.window, peak, len(schedule))
        for unit_id, measured in zip(task.unit_ids, raw["day_observations"], strict=True):
            observation = _tier0_observation(unit_id, scene, fit, analysis, measured, arm, offsets, peak)
            if observation is not None:
                rows.append(observation)
    return rows


def _tier0_observation(
    unit_id: str, scene: ScenarioSpec, fit: FitRecord, analysis: AnalysisSpec,
    measured: Mapping[str, Any], arm: ArmKind, offsets: tuple[int, ...], peak: int,
) -> UnitObservation | None:
    readout = analysis.readout
    response_path: tuple[float, ...] = ()
    truth_path: tuple[float, ...] = ()
    values: dict[str, Any] = {}
    paths = measured["paths"]
    if readout in ("free_rmse", "clamped_rmse") and offsets:
        response_path = tuple(paths[arm.value][i] for i in offsets)
        truth_path = tuple(paths["truth"][i] for i in offsets)
        values["event_offsets"] = offsets
    elif readout in ("peak_response", "free_peak_response"):
        values["value"] = paths[arm.value][peak]
    elif readout in ("end_response", "free_end_response"):
        values["value"] = paths[arm.value][-1]
    elif readout in ("nll", "heldout_loss"):
        values["value"] = measured["heldout_loss"]
    elif readout == "no_injection_drift":
        values["value"] = measured["no_injection_drift"]
    elif readout in ("background_bias", "induced_background_sign"):
        key = "during_execution" if analysis.window in ("execution", "during_execution", "peak") else "after_execution"
        value = measured["induced_background_sign"][key]
        if isfinite(value):
            values["value"] = value
    if not values and not response_path:
        return None
    return UnitObservation(
        unit_id=unit_id, scenario_id=scene.scenario_id, train_repeat_id=fit.train_repeat_id,
        arm=arm, window=analysis.window, readout=readout, unit=analysis.unit,
        candidate_id=fit.candidate_id, response_path=response_path, truth_path=truth_path, values=values,
    )


def _lob_observations(
    task: EvaluationTask, scene: ScenarioSpec, parent: FitRecord, candidate: FitRecord,
    analyses: tuple[AnalysisSpec, ...], comparison: LobComparison,
) -> list[UnitObservation]:
    if comparison.with_order is None or comparison.without_order is None:
        raise ValueError("successful LOB replay requires both comparison traces")
    if len(task.unit_ids) != 1:
        raise ValueError("the deterministic short LOB replay is one unit and cannot be counted as independent days")
    rows = []
    seen = set()
    controls = (
        (parent, comparison.without_order, ArmKind.NO_INJECTION),
        (candidate, comparison.with_order, ArmKind.VISIBLE),
    )
    for analysis in analyses:
        key = analysis.readout, analysis.window, analysis.unit
        if key in seen:
            continue
        seen.add(key)
        for fit, trace, source_arm in controls:
            if analysis.readout in ("filled_size", "fill_size"):
                value = sum(fill.size for fill in trace.fills)
            elif analysis.readout == "best_ask_depth":
                value = trace.orderbook_rows[-1][1]
                if analysis.window in ("execution", "peak"):
                    value = trace.orderbook_rows[-2][1]
            else:
                continue
            rows.append(UnitObservation(
                unit_id=task.unit_ids[0], scenario_id=scene.scenario_id, train_repeat_id=fit.train_repeat_id,
                arm=ArmKind.VISIBLE, window=analysis.window, readout=analysis.readout, unit=analysis.unit,
                candidate_id=fit.candidate_id, values={"value": value, "source_arm": source_arm.value},
            ))
    return rows


def _unavailable_result(
    task: EvaluationTask, parent: FitRecord, candidate: FitRecord, reason: str | None, started: float,
    train_seconds: float,
) -> PairedEvalResult:
    result = PairedEvalResult(
        result_id=f"result:{task.task_id}", task_id=task.task_id, scenario_id=task.scenario_id,
        split=task.split, parent_id=parent.candidate_id, candidate_id=candidate.candidate_id,
        parent_fit_id=parent.fit_id, candidate_fit_id=candidate.fit_id,
        action_ref=task.action_ref, random_inputs_ref=task.random_inputs_ref,
        arms=(), status="degraded", cost={"train_seconds": train_seconds, "eval_seconds": perf_counter() - started},
        failure_reason=reason or "measurement unavailable",
    )
    result.validate()
    return result
