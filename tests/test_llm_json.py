import json

import pytest

from stochopia.llm_json import extract_first_json, find_balanced_end, iter_json_candidates


def test_find_balanced_end_respects_strings_and_escapes():
    text = '{"a": "}\\"}", "b": {"c": 1}} tail'
    end = find_balanced_end(text, 0)
    assert json.loads(text[: end + 1]) == {"a": '}"}', "b": {"c": 1}}
    assert find_balanced_end("{ unbalanced", 0) is None


def test_fenced_block_wins_over_earlier_inline_object():
    raw = '{"x": 1} then ```json\n{"action": "query"}\n```'
    assert extract_first_json(raw) == {"action": "query"}


def test_inline_objects_in_order_and_broken_blocks_skipped():
    raw = "prose {not json} more {\"k\": [1, 2]} and {\"z\": null}"
    assert list(iter_json_candidates(raw)) == [{"k": [1, 2]}, {"z": None}]
    assert extract_first_json(raw) == {"k": [1, 2]}


def test_no_json_raises():
    with pytest.raises(ValueError):
        extract_first_json("只有文字，没有对象")
