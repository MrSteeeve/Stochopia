"""No-gradient ablation: memory versus observed previous-event information."""

import torch
from torch import Tensor, nn

from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, FiveTupleLoss, LOBWorldModel, StepObservation, State,
    _probe, world_model_loss,
)
from stochopia.neural.lob_training import evaluate_world_model_nll, world_model_nll_events
from stochopia.neural.primitives import SeedGRULOBModel


class CurrentOnly(LOBWorldModel):
    """Uses the same raw current observation for step and sequence, with no state."""

    def __init__(self) -> None:
        super().__init__()
        self.heads = nn.ModuleList(nn.Linear(2, count) for count in BUCKET_COUNTS)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        return torch.stack(
            (observation.is_own.float(), observation.previous_event[..., 0].float() / 15), dim=-1,
        )

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        return self.sequence_context(observation), None

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        return self.heads[stage](context)


def test_current_only_model_has_zero_gap_across_different_event_counts() -> None:
    model = CurrentOnly()
    batches = [_probe(torch.device("cpu"), 8, 1), _probe(torch.device("cpu"), 14, 5)]
    normal = evaluate_world_model_nll(model, batches, supervised=slice(3, None))
    stateless = evaluate_world_model_nll(model, batches, stateless=True, supervised=slice(3, None))
    assert abs(normal["total"] - stateless["total"]) < 1e-6
    for key in ("dt", "action", "rel_ticks", "volume"):
        assert abs(normal[key] - stateless[key]) < 1e-6
    assert model.training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_stateless_fold_matches_explicit_none_state_for_recurrent_model() -> None:
    torch.manual_seed(198)
    model = SeedGRULOBModel(state_dim=16).eval()
    batch = _probe(torch.device("cpu"), 371, 5)
    has_deep = torch.ones((*batch.targets.shape[:2], 2), dtype=torch.bool)
    batch = batch._replace(observation=batch.observation._replace(has_deep=has_deep))
    actual = world_model_nll_events(model, batch, stateless=True)
    explicit = []
    with torch.no_grad():
        for t in range(batch.targets.size(1)):
            sampler, _ = model.forward_step(batch.observation.at((slice(None), t)), None)
            targets = batch.targets[:, t]
            explicit.append(world_model_loss(sampler.conditioned(targets), targets, reduction="none"))
    expected = FiveTupleLoss(*(torch.stack([row[j] for row in explicit], dim=1) for j in range(5)))
    for actual_term, expected_term in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_term, expected_term, atol=1e-6, rtol=0)
    normal = world_model_nll_events(model, batch)
    assert abs(float(normal.total.mean() - actual.total.mean())) > 1e-5


def test_weighted_head_sums_use_exact_same_event_denominator() -> None:
    torch.manual_seed(97)
    model = SeedGRULOBModel(state_dim=16)
    batches = [_probe(torch.device("cpu"), 11, 1), _probe(torch.device("cpu"), 23, 5)]
    supervised = slice(4, None)
    names = ("dt", "action", "rel_ticks", "volume")
    for stateless in (False, True):
        actual = evaluate_world_model_nll(model, batches, stateless=stateless, supervised=supervised)
        losses = [world_model_nll_events(model, batch, stateless=stateless) for batch in batches]
        denominator = sum(batch.targets[:, supervised].shape[0] * batch.targets[:, supervised].shape[1]
                          for batch in batches)
        for name in names:
            numerator = sum(float(getattr(loss, name)[:, supervised].double().sum()) for loss in losses)
            assert abs(actual[name] - numerator / denominator) < 1e-12
        assert abs(actual["total"] - sum(actual[name] for name in names)) < 1e-12
