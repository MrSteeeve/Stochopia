"""Composite recurrent geometry and new causal model primitives."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from stochopia.neural.lob_interface import (
    BUCKET_COUNTS, LOBWorldModel, State, StepObservation, _probe,
    concat_states, detach_or_clone_state, index_select_state,
    state_num_elements_per_sample, verify_world_model_contract,
)
from stochopia.neural.primitives import (
    apply_rotary_2d, causal_depthwise_conv1d, flip_lob_observation, parallel_affine_scan,
)


class CompositeModel(LOBWorldModel):
    """A minimal stable level-state model with one of three state geometries."""

    def __init__(self, geometry: str) -> None:
        super().__init__()
        self.geometry = geometry
        self.heads = nn.ModuleList(nn.Linear(16, count) for count in BUCKET_COUNTS)
        self.seen_batches: set[int] = set()

    @staticmethod
    def _drive(obs: StepObservation) -> Tensor:
        x = obs.book_shares[..., 0, :].float() / 10_000
        return x[..., None].expand(*x.shape, 16)

    @staticmethod
    def _context(book: Tensor) -> Tensor:
        return book.mean(dim=-2)

    def sequence_context(self, observation: StepObservation) -> Tensor:
        drive = self._drive(observation)
        return self._context(parallel_affine_scan(drive.new_full((10, 16), 0.2), drive))

    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        drive = self._drive(observation)
        self.seen_batches.add(drive.size(0))
        if state is None:
            book = drive
        else:
            if isinstance(state, tuple):
                state = state[0]
            elif isinstance(state, dict):
                state = state["book"]
            book = state * 0.2 + drive
        if self.geometry == "matrix":
            updated = book
        elif self.geometry == "tuple":
            updated = (book, book[:, 0])
        else:
            updated = {"book": book, "extra": (book[:, 0], None)}
        return self._context(book), updated

    def head_logits(self, context: Tensor, stage: int, prior: tuple[Tensor, ...]) -> Tensor:
        return self.heads[stage](context)


@pytest.mark.parametrize(("kind", "scalars"), [("matrix", 160), ("tuple", 176), ("dict", 176)])
def test_composite_contract_dynamic_batch_and_per_sample_cap(kind: str, scalars: int) -> None:
    torch.manual_seed(5)
    model = CompositeModel(kind)
    _, actual = verify_world_model_contract(model, max_state_dim=512)
    assert actual == scalars
    assert model.seen_batches == {1, 5}
    assert model.training
    with pytest.raises(ValueError, match="max_state_dim"):
        verify_world_model_contract(model, max_state_dim=scalars - 1)


class WrongBatch(CompositeModel):
    def step_context(self, observation: StepObservation, state: State) -> tuple[Tensor, State]:
        context, state = super().step_context(observation, state)
        return context, state[:1]


def test_state_contract_rejects_hardcoded_single_batch() -> None:
    with pytest.raises(ValueError, match="batch size"):
        verify_world_model_contract(WrongBatch("matrix"), max_state_dim=512)


def test_tree_count_index_concat_and_detach_preserve_geometry() -> None:
    a = torch.arange(2 * 10 * 16, dtype=torch.float32).reshape(2, 10, 16).requires_grad_()
    b = torch.ones(2, 3, dtype=torch.complex64)
    original = {"book": a, "slow": (b, None)}
    assert state_num_elements_per_sample(original) == 160 + 2 * 3
    assert state_num_elements_per_sample(None) == 0
    with pytest.raises(ValueError, match="batch size"):
        state_num_elements_per_sample((a, b[:1]))

    first = index_select_state(original, torch.tensor([1]))
    second = index_select_state(original, torch.tensor([0, 1]))
    merged = concat_states([first, second])
    torch.testing.assert_close(merged["book"], a[[1, 0, 1]])
    torch.testing.assert_close(merged["slow"][0], b[[1, 0, 1]])
    assert merged["slow"][1] is None
    assert state_num_elements_per_sample(merged) == 166
    cloned = detach_or_clone_state(original)
    assert not cloned["book"].requires_grad
    assert cloned["book"].data_ptr() != a.data_ptr()
    assert concat_states([None, None]) is None
    assert detach_or_clone_state(None) is None
    assert index_select_state(None, torch.tensor([0])) is None
    with pytest.raises(ValueError, match="matching structure"):
        concat_states([None, a])
    with pytest.raises(ValueError, match="matching structure"):
        concat_states([{"x": a}, {"y": a}])
    with pytest.raises(ValueError, match="empty sequence"):
        concat_states([])


def _loop(gates: Tensor, values: Tensor, init_state: Tensor) -> Tensor:
    states = []
    state = init_state
    for t in range(values.size(1)):
        state = gates[:, t] * state + values[:, t]
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize("tail", [(7,), (10, 16)])
@pytest.mark.parametrize("gate_mode", ["static", "singleton", "batch", "batch_no_time", "full"])
def test_fp32_broadcasted_scan_output_and_gradients(tail: tuple[int, ...], gate_mode: str) -> None:
    rng = torch.Generator().manual_seed(493)
    shape = (2, 13, *tail)
    gate_shape = {
        "static": tail, "singleton": (1, 1, *tail), "batch": (shape[0], 1, *tail),
        "batch_no_time": (shape[0], *tail), "full": shape,
    }[gate_mode]
    gates = (0.1 + 0.2 * torch.rand(gate_shape, generator=rng)).requires_grad_()
    values = (0.2 * torch.randn(shape, generator=rng)).requires_grad_()
    initial = torch.randn((shape[0], *tail), generator=rng).requires_grad_()
    actual = parallel_affine_scan(gates, values, init_state=initial, chunk_size=5)
    expanded = gates.unsqueeze(1) if gate_mode == "batch_no_time" else gates
    expected = _loop(expanded.expand_as(values), values, initial)
    assert float((actual - expected).detach().abs().max()) < 1e-5
    variables = (gates, values, initial)
    grad_scan = torch.autograd.grad(actual.square().sum(), variables)
    grad_loop = torch.autograd.grad(expected.square().sum(), variables)
    for actual_grad, expected_grad in zip(grad_scan, grad_loop, strict=True):
        assert float((actual_grad - expected_grad).abs().max()) < 1e-5


@pytest.mark.parametrize("shape", [(2, 11, 4), (2, 11, 10, 4)])
def test_depthwise_convolution_causal_and_independent(shape: tuple[int, ...]) -> None:
    rng = torch.Generator().manual_seed(643)
    x = torch.randn(shape, generator=rng)
    weight = torch.randn(4, 3, generator=rng)
    bias = torch.randn(4, generator=rng)
    actual = causal_depthwise_conv1d(x, weight, bias)
    padded = torch.cat((torch.zeros_like(x[:, :2]), x), dim=1)
    expected = torch.stack(
        [sum(padded[:, t + k] * weight[:, k] for k in range(3)) + bias for t in range(x.size(1))], dim=1,
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    changed = x.clone()
    changed[:, 6:] += 123
    assert torch.equal(actual[:, :6], causal_depthwise_conv1d(changed, weight, bias)[:, :6])
    changed = x.clone()
    changed[..., 0] += 7
    assert torch.equal(actual[..., 1:], causal_depthwise_conv1d(changed, weight, bias)[..., 1:])


def test_rotary_conserves_norm_and_is_invertible() -> None:
    rng = torch.Generator().manual_seed(85)
    x = torch.randn(2, 8, 10, 16, generator=rng)
    angles = torch.randn(8, 1, 8, generator=rng)
    rotated = apply_rotary_2d(x, angles)
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=0)
    torch.testing.assert_close(apply_rotary_2d(rotated, -angles), x, atol=1e-6, rtol=0)


def test_book_reflection_is_an_involution_and_flips_action_side() -> None:
    obs = torch.arange(47, dtype=torch.float32).expand(6, -1).clone()
    obs[:, 44] = torch.arange(6)
    reflected = flip_lob_observation(obs)
    assert torch.equal(reflected[:, :10], -obs[:, 10:20])
    assert torch.equal(reflected[:, 10:20], -obs[:, :10])
    assert torch.equal(reflected[:, 20:30], obs[:, 30:40])
    assert torch.equal(reflected[:, 30:40], obs[:, 20:30])
    assert torch.equal(reflected[:, 40], obs[:, 40])
    assert torch.equal(reflected[:, 41], -obs[:, 41])
    assert torch.equal(reflected[:, 42:44], obs[:, 42:44])
    assert reflected[:, 44].tolist() == [1, 0, 3, 2, 5, 4]
    assert torch.equal(reflected[:, 45:], obs[:, 45:])
    assert torch.equal(flip_lob_observation(reflected), obs)
    book = obs[:, :43]
    assert torch.equal(flip_lob_observation(flip_lob_observation(book)), book)
