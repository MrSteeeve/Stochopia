"""Extract JSON objects from free-form LLM output.

Fenced ```json blocks are tried first, then every balanced ``{...}`` span in
order of appearance.  Shared by the environment-agent loader and the episode
runner so both sides parse model replies identically.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def find_balanced_end(text: str, start: int) -> int | None:
    """Return the index of the ``}`` matching the ``{`` at ``start``.

    String literals and escapes are respected; ``None`` when unbalanced.
    """

    depth = 0
    in_str = False
    escaped = False
    for k in range(start, len(text)):
        c = text[k]
        if in_str:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return k
    return None


def iter_json_candidates(text: str) -> Iterator[Any]:
    """Yield every parseable JSON value in ``text``: fenced blocks first, then
    balanced-brace spans in order of appearance."""

    for block in _FENCE_RE.findall(text):
        try:
            yield json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    index = 0
    while index < len(text):
        if text[index] == "{":
            end = find_balanced_end(text, index)
            if end is not None:
                try:
                    yield json.loads(text[index : end + 1])
                    index = end + 1
                    continue
                except json.JSONDecodeError:
                    pass
        index += 1


def extract_first_json(raw: str) -> Any:
    """Return the first parseable JSON value in ``raw``; ``ValueError`` if none."""

    for candidate in iter_json_candidates(raw):
        return candidate
    raise ValueError("未找到可解析的 JSON 对象")
