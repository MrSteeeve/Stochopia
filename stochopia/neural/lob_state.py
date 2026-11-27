"""Batch operations for the order-book model's recurrent state.

The tree is either absent, a tensor whose leading axis is the sample axis, or
a tuple/dictionary of subtrees. A complex element occupies two real scalars.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import TypeAlias

import torch
from torch import Tensor

State: TypeAlias = Tensor | tuple["State", ...] | dict[str, "State"] | None


def _leaves(state: State) -> list[Tensor]:
    if state is None:
        return []
    if isinstance(state, Tensor):
        return [state]
    if isinstance(state, tuple):
        values = state
    elif isinstance(state, dict) and all(isinstance(key, str) for key in state):
        values = tuple(state.values())
    else:
        raise TypeError("state must be a Tensor, tuple, string-keyed dict, or None")
    return [leaf for value in values for leaf in _leaves(value)]


def state_num_elements_per_sample(state: State, *, batch_size: int | None = None) -> int:
    """Sum real scalar counts over leaves of shape [B, *state_shape].

    The batch axis is never charged to the state budget. Tensor leaves must
    share a batch size; an explicit batch_size also checks the caller's input
    batch against every returned leaf. Empty states count as zero.
    """
    leaves = _leaves(state)
    if any(leaf.ndim == 0 for leaf in leaves):
        raise ValueError("state tensor leaves require a leading batch axis")
    if batch_size is None and leaves:
        batch_size = leaves[0].size(0)
    if any(leaf.size(0) != batch_size for leaf in leaves):
        raise ValueError("state tensor leaves must share the input batch size")
    return sum(prod(leaf.shape[1:]) * (2 if leaf.is_complex() else 1) for leaf in leaves)


def detach_or_clone_state(state: State) -> State:
    """Detach and clone leaves, preserving tree structure and batch shapes."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.detach().clone()
    if isinstance(state, tuple):
        return tuple(detach_or_clone_state(item) for item in state)
    return {key: detach_or_clone_state(item) for key, item in state.items()}


def concat_states(states: Sequence[State], dim: int = 0) -> State:
    """Concatenate corresponding leaves of like-structured state trees."""
    if len(states) == 0:
        raise ValueError("cannot concatenate an empty sequence of states")
    first = states[0]
    if first is None and all(state is None for state in states):
        return None
    if isinstance(first, Tensor) and all(isinstance(state, Tensor) for state in states):
        return torch.cat(tuple(states), dim=dim)
    if isinstance(first, tuple) and all(
        isinstance(state, tuple) and len(state) == len(first) for state in states
    ):
        return tuple(concat_states([state[index] for state in states], dim) for index in range(len(first)))
    if isinstance(first, dict) and all(
        isinstance(state, dict) and state.keys() == first.keys() for state in states
    ):
        return {key: concat_states([state[key] for state in states], dim) for key in first}
    raise ValueError("state trees must have matching structure")


def index_select_state(state: State, indices: Tensor) -> State:
    """Select the same sample indices from every leaf (leading batch axis)."""
    if state is None:
        return None
    if isinstance(state, Tensor):
        return state.index_select(0, indices.to(device=state.device, dtype=torch.long))
    if isinstance(state, tuple):
        return tuple(index_select_state(item, indices) for item in state)
    return {key: index_select_state(item, indices) for key, item in state.items()}
