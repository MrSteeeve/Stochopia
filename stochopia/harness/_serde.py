"""Explicit contract field rules and detached, read-only JSON snapshots."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import fields
from enum import Enum
from math import isfinite
from numbers import Integral, Real
from typing import Any, ClassVar, TypeVar


_RecordT = TypeVar("_RecordT", bound="_Record")
_EnumT = TypeVar("_EnumT", bound=Enum)
Rule = Callable[[Any, str, bool], Any]


# --- Snapshots and wire values ---


def _scalar(value: Any) -> Any:
    """Normalize concrete numeric scalars without importing an array library."""
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        try:
            return float(value)
        except OverflowError:
            return value
    module = type(value).__module__.partition(".")[0]
    if module in ("numpy", "jax", "jaxlib") and getattr(value, "ndim", None) == 0:
        try:
            item = value.item()
        except (TypeError, ValueError, OverflowError):
            return value
        if type(item) in (bool, int, float):
            return item
    return value


class _ReadOnlyMapping(Mapping[str, Any]):
    """Read-only view of owned data, wrapping nested mappings when accessed."""

    __slots__ = ("__data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self.__data = data

    def __getitem__(self, key: str) -> Any:
        return _view(self.__data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return repr(self.__data)


def _snapshot(value: Any) -> Any:
    if isinstance(value, _ReadOnlyMapping):
        return value
    if isinstance(value, Mapping):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(item) for item in value)
    return _scalar(value)


def _view(value: Any) -> Any:
    if isinstance(value, dict):
        return _ReadOnlyMapping(value)
    if isinstance(value, tuple):
        return tuple(_view(item) for item in value)
    return value


def _encode(value: Any) -> Any:
    if isinstance(value, _Record):
        return {item.name: _encode(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    return value


# --- Explicit field rules ---


def text(value: Any, path: str, wire: bool) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path}: expected str")
    return value


def integer(value: Any, path: str, wire: bool) -> int:
    number = _scalar(value)
    if not isinstance(number, Integral) or isinstance(number, bool):
        raise ValueError(f"{path}: expected int")
    return number


def real(value: Any, path: str, wire: bool) -> int | float:
    number = _scalar(value)
    if isinstance(number, Real) and not isinstance(number, bool):
        try:
            if isfinite(number):
                return number
        except OverflowError:
            pass
    raise ValueError(f"{path}: expected finite float")


def boolean(value: Any, path: str, wire: bool) -> bool:
    value = _scalar(value)
    if not isinstance(value, bool):
        raise ValueError(f"{path}: expected bool")
    return value


def literal(*choices: str) -> Rule:
    def parse(value: Any, path: str, wire: bool) -> str:
        if value not in choices:
            raise ValueError(f"{path}: expected one of {choices!r}")
        return value

    return parse


def enum(enum_type: type[_EnumT]) -> Rule:
    def parse(value: Any, path: str, wire: bool) -> _EnumT:
        if isinstance(value, enum_type):
            return value
        if wire and isinstance(value, str):
            try:
                return enum_type(value)
            except ValueError:
                pass
        raise ValueError(f"{path}: expected {enum_type.__name__}")

    return parse


def optional(rule: Rule) -> Rule:
    def parse(value: Any, path: str, wire: bool) -> Any:
        return None if value is None else rule(value, path, wire)

    return parse


def sequence(rule: Rule, *, length: int | None = None) -> Rule:
    def parse(value: Any, path: str, wire: bool) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple) if wire else tuple):
            raise ValueError(f"{path}: expected an array")
        if length is not None and len(value) != length:
            raise ValueError(f"{path}: expected {length} items")
        if wire:
            return tuple(rule(item, f"{path}[{index}]", True) for index, item in enumerate(value))
        for index, item in enumerate(value):
            rule(item, f"{path}[{index}]", False)
        return value

    return parse


def mapping(rule: Rule) -> Rule:
    def parse(value: Any, path: str, wire: bool) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}: expected an object")
        if wire:
            decoded = {}
            for key, item in value.items():
                text(key, f"{path}.key", True)
                decoded[key] = rule(item, f"{path}.{key}", True)
            return _ReadOnlyMapping(decoded)
        for key, item in value.items():
            text(key, f"{path}.key", False)
            rule(item, f"{path}.{key}", False)
        return value

    return parse


def record(record_type: type[_RecordT]) -> Rule:
    def parse(value: Any, path: str, wire: bool) -> _RecordT:
        if wire:
            return record_type.from_dict(value)
        if not isinstance(value, record_type):
            raise ValueError(f"{path}: expected {record_type.__name__}")
        value.validate()
        return value

    return parse


def json_value(value: Any, path: str, wire: bool) -> Any:
    """Check an open JSON field and normalize its payload while decoding."""
    if isinstance(value, Mapping):
        decoded = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{path}: JSON object keys must be strings")
            normalized = json_value(item, f"{path}.{key}", wire)
            if wire:
                decoded[key] = normalized
        return decoded if wire else value
    if isinstance(value, (list, tuple)):
        if wire:
            return tuple(json_value(item, f"{path}[{index}]", True) for index, item in enumerate(value))
        for index, item in enumerate(value):
            json_value(item, f"{path}[{index}]", False)
        return value
    normalized = _scalar(value)
    if normalized is None or type(normalized) in (str, bool, int):
        return normalized
    if type(normalized) is float and isfinite(normalized):
        return normalized
    raise ValueError(f"{path}: expected a finite JSON value")


# --- Record boundaries ---


class _Record:
    _field_rules: ClassVar[Mapping[str, Rule]] = {}

    def __post_init__(self) -> None:
        """Detach inputs without changing when semantic errors are reported."""
        for item in fields(self):
            value = getattr(self, item.name)
            detached = _view(_snapshot(value))
            if detached is not value:
                object.__setattr__(self, item.name, detached)

    def validate(self) -> None:
        for item in fields(self):
            rule = self._field_rules.get(item.name, text)
            rule(getattr(self, item.name), item.name, False)
        self._check()

    def _check(self) -> None:
        """Check a field-validated record's domain invariants."""

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return _encode(self)

    @classmethod
    def from_dict(cls: type[_RecordT], raw: Mapping[str, Any]) -> _RecordT:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{cls.__name__}: expected an object")
        unknown = set(raw) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"{cls.__name__}: unknown fields {sorted(map(str, unknown))}")
        decoded = {}
        for name, value in raw.items():
            rule = cls._field_rules.get(name, text)
            decoded[name] = rule(value, name, True)
        try:
            result = cls(**decoded)
        except TypeError as exc:
            raise ValueError(f"{cls.__name__}: {exc}") from exc
        # Nested records are already validated by their own wire rules.
        result._check()
        return result
