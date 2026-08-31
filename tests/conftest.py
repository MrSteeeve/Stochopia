"""测试共享夹具：仓库路径、场景夹具与可记录消息的 Mock 客户端。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stochopia.llm import MockLLMClient  # noqa: E402

SCENARIOS_ROOT = ROOT / "scenarios"


class RecordingMockClient(MockLLMClient):
    """在 Mock 基础上记录每次 chat 收到的 messages 与 max_tokens，便于断言调用行为。"""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__(responses)
        self.calls: list[list[dict]] = []
        self.max_tokens_calls: list[int | None] = []

    async def chat(self, messages, temperature=None, max_tokens=None) -> str:
        self.calls.append([dict(m) for m in messages])
        self.max_tokens_calls.append(max_tokens)
        return await super().chat(messages, temperature, max_tokens)


def query_json(target: str, message: str) -> str:
    """构造一条合法的 query 动作 JSON 文本。"""
    return json.dumps(
        {"action": "query", "target": target, "message": message}, ensure_ascii=False
    )


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件为字典列表。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def scenarios_root() -> Path:
    return SCENARIOS_ROOT
