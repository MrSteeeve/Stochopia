"""Deterministic checks for the P1 neural contract, independent of market data."""

import pytest
import torch
from torch import Tensor, nn

from stochopia.neural.lob_interface import (
    BUCKET_COUNTS,
    FiveTupleLogits,
    LOBBatch,
    State,
    StepObservation,
    StepSampler,
    decode_events,
    encode_events,
    make_batch,
    valid_action_mask,
    valid_rel_tick_mask,
    verify_world_model_contract,
    world_model_loss,
)
from stochopia.neural.primitives import (
    SeedGRULOBModel,
    affine_scan_step,
    complex_diagonal,
    parallel_affine_scan,
    real_diagonal,
)


def _observation() -> StepObservation:
    bids = torch.tensor([[100, 98, 95, 0, 0, 0, 0, 0, 0, 0], [0] * 10])
    asks = torch.tensor([[103, 105, 106, 107, 108, 109, 110, 111, 112, 113], [103] + [0] * 9])
    books = torch.stack((bids, asks), dim=1)
    depth = 200 * (books > 0).long()
    return StepObservation(books, depth, torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, dtype=torch.bool))


@pytest.mark.parametrize("error", [AssertionError, TypeError, ValueError])
def test_trained_model_contract_failure_is_classified(error: type[Exception], monkeypatch: pytest.MonkeyPatch) -> None:
    from experiments import p1_m2_baseline

    def rejected(model: nn.Module) -> tuple[int, int]:
        raise error("invalid trained state")

    monkeypatch.setattr(p1_m2_baseline, "verify_world_model_contract", rejected)
    with pytest.raises(RuntimeError, match="Trained model failed tensor contract") as caught:
        p1_m2_baseline._verify_trained_model(SeedGRULOBModel())
    assert isinstance(caught.value.__cause__, error)


def test_event_codes_preserve_zero_time_and_price_level_semantics() -> None:
    books = _observation().book_ticks[[0, 0, 0, 0, 0]]
    tokens = encode_events(torch.tensor([0, 1, 20, 500, 11000]),
                           torch.tensor([0, 2, 4, 5, 1]),
                           torch.tensor([-2, 10, 5, 100, 49]),
                           torch.tensor([50, 100, 150, 20100, 800]), books)
    assert tokens[:, 0].tolist() == [0, 1, 2, 9, 15]
    assert tokens[:, 2].tolist() == [3, 10, 2, 10, 25]
    assert tokens[:, 3].tolist() == [0, 1, 2, 11, 6]
    dt, action, rel, shares = decode_events(tokens).unbind(dim=-1)
    assert dt[0] == 0 and (dt[1:] > 0).all() and (dt % 10 == 0).all()
    assert action.tolist() == [0, 2, 4, 5, 1]
    assert rel.tolist() == [-2, 10, 2, 10, 20]
    assert shares.tolist() == [50, 100, 200, 15000, 700]


def test_action_and_level_masks_use_current_book_and_positive_prices() -> None:
    observation = _observation()
    actions = valid_action_mask(observation.book_ticks, observation.book_shares)
    assert actions.tolist() == [[True] * 6, [True, True, True, False, False, True]]
    prices = valid_rel_tick_mask(observation, torch.tensor([0, 0]))
    assert prices[0, :6].tolist() == [False, False, False, True, True, True]
    assert not prices[1, :5].any() and prices[1, 5:].all()
    prices = valid_rel_tick_mask(observation, torch.tensor([4, 5]))
    assert prices[0].nonzero().flatten().tolist() == [0, 1, 2]
    assert prices[1].nonzero().flatten().tolist() == [0]
    prices = valid_rel_tick_mask(observation, torch.tensor([5, 2]))
    assert prices[0].nonzero().flatten().tolist() == list(range(11))
    assert prices[1].nonzero().flatten().tolist() == list(range(11))
    exact = observation._replace(has_deep=torch.zeros(2, 2, dtype=torch.bool))
    assert not valid_rel_tick_mask(exact, torch.tensor([5, 2]))[0, 10]

    tiny = observation.book_ticks.clone()
    tiny[1, 1, 0] = 1
    assert not valid_action_mask(tiny, observation.book_shares)[1, 0]


def test_teacher_forcing_shifts_only_prior_event_and_ownership() -> None:
    raw = _observation().at((slice(0, 1),))
    targets = torch.tensor([[[0, 0, 5, 1], [2, 4, 2, 0], [9, 2, 4, 5]]])
    book = raw.book_ticks.unsqueeze(1).expand(-1, 3, -1, -1)
    depth = raw.book_shares.unsqueeze(1).expand_as(book)
    batch = make_batch(book, depth, targets, torch.tensor([[True, False, True]]))
    assert batch.observation.book_ticks is book
    assert batch.observation.is_own.tolist() == [[False, True, False]]
    assert batch.observation.previous_event.tolist() == [[[0] * 4, [0, 0, 5, 1], [2, 4, 2, 0]]]

    model = SeedGRULOBModel()
    logits = model.forward_sequence(batch)
    assert [x.shape[-1] for x in logits] == list(BUCKET_COUNTS)
    loss = world_model_loss(logits, targets, reduction="none")
    assert all(item.shape == (1, 3) and torch.isfinite(item).all() and (item >= 0).all() for item in loss)
    torch.testing.assert_close(loss.total, sum(loss[1:]))
    torch.testing.assert_close(loss.total.mean(), world_model_loss(logits, targets).total)


def test_sample_and_teacher_forced_heads_agree_on_masked_support() -> None:
    observation = _observation()
    model = SeedGRULOBModel()
    sampler, _ = model.forward_step(observation)
    first = sampler.sample(torch.Generator().manual_seed(123))
    second = sampler.sample(torch.Generator().manual_seed(123))
    assert torch.equal(first, second)
    dt, action, rel, size = first.unbind(dim=-1)
    assert valid_action_mask(observation.book_ticks, observation.book_shares)[torch.arange(2), action].all()
    assert valid_rel_tick_mask(observation, action)[torch.arange(2), rel].all()
    heads = sampler.conditioned(first)
    assert all(torch.isfinite(head[torch.arange(2), idx]).all()
               for head, idx in zip(heads, (dt, action, rel, size), strict=True))


def test_unobserved_own_flag_has_no_random_initial_effect() -> None:
    observation = _observation()
    model = SeedGRULOBModel()
    original, _ = model.forward_step(observation)
    marked = observation._replace(is_own=torch.ones_like(observation.is_own))
    flagged, _ = model.forward_step(marked)
    torch.testing.assert_close(flagged.context, original.context, atol=0, rtol=0)


def _loop_scan(gates: Tensor, inputs: Tensor, state: Tensor) -> Tensor:
    trace: list[Tensor] = []
    for gate, value in zip(gates.unbind(dim=1), inputs.unbind(dim=1), strict=True):
        state = affine_scan_step(gate, value, state)
        trace.append(state)
    return torch.stack(trace, dim=1)


@pytest.mark.parametrize("chunk", [1, 7, 32, 64])
def test_parallel_affine_scan_matches_step_and_gradients(chunk: int) -> None:
    rng = torch.Generator().manual_seed(123)
    logits = (torch.randn(2, 37, 5, dtype=torch.float64, generator=rng) * 11).requires_grad_()
    gates = real_diagonal(logits)
    inputs = torch.randn(2, 37, 5, dtype=torch.float64, generator=rng).requires_grad_()
    h0 = torch.randn(2, 5, dtype=torch.float64, generator=rng).requires_grad_()
    by_block = parallel_affine_scan(gates, inputs, h0, chunk_size=chunk)
    by_step = _loop_scan(gates, inputs, h0)
    torch.testing.assert_close(by_block, by_step, atol=1e-12, rtol=1e-12)
    variables = (logits, inputs, h0)
    grad_block = torch.autograd.grad(by_block.square().sum(), variables, retain_graph=True)
    grad_step = torch.autograd.grad(by_step.square().sum(), variables)
    torch.testing.assert_close(grad_block, grad_step, atol=1e-10, rtol=1e-10)


def test_parallel_affine_scan_accepts_complex_and_zero_time() -> None:
    rng = torch.Generator().manual_seed(342)
    logits = torch.randn(2, 14, 3, dtype=torch.float64, generator=rng)
    phase = torch.randn(2, 14, 3, dtype=torch.float64, generator=rng)
    gates = complex_diagonal(logits, phase)
    inputs = torch.randn(2, 14, 3, dtype=torch.complex128, generator=rng)
    expected = _loop_scan(gates, inputs, inputs.new_zeros(2, 3))
    torch.testing.assert_close(parallel_affine_scan(gates, inputs, chunk_size=3), expected)
    assert parallel_affine_scan(gates[:, :0], inputs[:, :0]).shape == (2, 0, 3)


@pytest.mark.parametrize("complex_gates", [False, True])
def test_one_dimensional_diagonal_broadcasts_for_sequence_and_step(complex_gates: bool) -> None:
    logits = torch.tensor([-1.3, 0.6, 2.0], dtype=torch.float64, requires_grad=True)
    if complex_gates:
        gate = complex_diagonal(logits, torch.tensor([0.3, -0.1, 1.0], dtype=logits.dtype))
    else:
        gate = real_diagonal(logits)
    inputs = torch.arange(1, 31).reshape(2, 5, 3).to(gate.dtype) / 15
    h0 = gate.new_full((2, 3), 0.25)
    scanned = parallel_affine_scan(gate, inputs, h0, chunk_size=2)
    expected = _loop_scan(gate.expand_as(inputs), inputs, h0)
    torch.testing.assert_close(scanned, expected)
    torch.testing.assert_close(affine_scan_step(gate, inputs[:, 0], h0), expected[:, 0])
    derivative = torch.autograd.grad(scanned.abs().square().sum(), logits, retain_graph=True)
    expected_derivative = torch.autograd.grad(expected.abs().square().sum(), logits)
    torch.testing.assert_close(derivative, expected_derivative)


class _FutureBookModel(SeedGRULOBModel):
    def sequence_context(self, observation: StepObservation) -> Tensor:
        context = super().sequence_context(observation)
        return context + observation.book_ticks[:, -1:, 0, 0, None]


class _FutureTargetModel(SeedGRULOBModel):
    def forward_sequence(self, batch: LOBBatch) -> FiveTupleLogits:
        logits = super().forward_sequence(batch)
        return logits._replace(volume=logits.volume + batch.targets[:, -1:, 3, None])


class _WrongStepSampler(StepSampler):
    def conditioned(self, targets: Tensor) -> FiveTupleLogits:
        wrong_relative = targets.clone()
        wrong_relative[..., 2] = (wrong_relative[..., 2] + 1) % BUCKET_COUNTS[2]
        return super().conditioned(wrong_relative)


class _WrongStepModel(SeedGRULOBModel):
    def forward_step(self, step_obs: StepObservation, hidden_state: State = None) -> tuple[StepSampler, State]:
        sampler, state = super().forward_step(step_obs, hidden_state)
        return _WrongStepSampler(*sampler), state


class _LargeStateModel(SeedGRULOBModel):
    def forward_step(self, step_obs: StepObservation, hidden_state: State = None) -> tuple[StepSampler, State]:
        state = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
        sampler, state = super().forward_step(step_obs, state)
        return sampler, (state, torch.zeros(1, 41, dtype=torch.complex64, device=sampler.context.device))


@pytest.mark.parametrize("model_type", [_FutureBookModel, _FutureTargetModel, _WrongStepModel])
def test_contract_rejects_future_or_inconsistent_conditional_heads(model_type: type[SeedGRULOBModel]) -> None:
    with pytest.raises(AssertionError):
        verify_world_model_contract(model_type())


def test_contract_checks_learnable_parameters_and_actual_state() -> None:
    model = SeedGRULOBModel()
    parameters, state_dim = verify_world_model_contract(model)
    assert state_dim == 48
    assert model.training
    model.extra = nn.Parameter(torch.zeros(28_001 - parameters))
    with pytest.raises(ValueError, match="max_params"):
        verify_world_model_contract(model)
    with pytest.raises(ValueError, match="max_state_dim"):
        verify_world_model_contract(_LargeStateModel())
