"""智能体测试：动作解析、环境智能体、结构化产品设计师智能体。"""

from __future__ import annotations

import json

import pytest

from conftest import RecordingMockClient, query_json

from stochopia.agents import (
    ENV_AGENT_PREAMBLE,
    STRUCTURER_PREAMBLE,
    ActionParseError,
    EnvironmentAgent,
    StructurerAgent,
    parse_action,
)
from stochopia.products import ClientProfile, Constraint, MarketState
from stochopia.scenario import AgentSpec

VALID_IDS = ["client_a", "client_b", "client_c", "risk_control", "trading_desk"]


def _submit_product_json(**overrides) -> str:
    product = {
        "product_type": "vanilla_call",
        "notional": 5_000_000,
        "maturity_months": 6,
        "strike_pct": 1.0,
        "barrier_pct": None,
        "barrier_type": None,
        "coupon_rate": None,
        "participation_rate": 1.0,
        "principal_protected": False,
        "target_client": "client_b",
        "pitch": "推销话术",
        "hedging_plan": "对冲方案",
    }
    product.update(overrides)
    return json.dumps({"action": "submit_product", "product": product}, ensure_ascii=False)


SKIP_ROUND_JSON = json.dumps({"action": "skip_round"})


class StubScenario:
    """测试用的最小场景桩：只暴露 StructurerAgent/EnvironmentAgent 依赖的鸭子类型接口。"""

    def __init__(self, background: str, agents: list[AgentSpec]) -> None:
        self.background = background
        self.agents = agents

    def agent_ids(self) -> list[str]:
        return [a.id for a in self.agents]

    def get_agent(self, agent_id: str) -> AgentSpec:
        for a in self.agents:
            if a.id == agent_id:
                return a
        raise KeyError(f"unknown agent id: {agent_id}")


def _agent_spec(agent_id: str = "client_a", display_name: str = "客户经理老王") -> AgentSpec:
    return AgentSpec(
        id=agent_id,
        display_name=display_name,
        public_desc="负责对接客户",
        prompt=f"你是{display_name}，一名{agent_id}角色。",
    )


def _stub_scenario() -> StubScenario:
    return StubScenario(
        background="这是一个结构化产品设计场景的背景说明。",
        agents=[
            _agent_spec("client_a", "客户经理老王"),
            _agent_spec("risk_control", "风控张总"),
        ],
    )


# ---------------------------------------------------------------------------
# parse_action
# ---------------------------------------------------------------------------


def test_parse_action_query_ok():
    action = parse_action(query_json("client_a", "你好，介绍一下客户需求"), VALID_IDS)
    assert action.type == "query"
    assert action.target == "client_a"
    assert action.message == "你好，介绍一下客户需求"
    assert action.raw is not None


def test_parse_action_submit_product_ok():
    raw = _submit_product_json(notional=8_000_000)
    action = parse_action(raw, VALID_IDS)
    assert action.type == "submit_product"
    assert isinstance(action.product_data, dict)
    assert action.product_data["notional"] == 8_000_000
    assert action.product_data["product_type"] == "vanilla_call"
    assert action.raw == raw


def test_parse_action_skip_round_ok():
    action = parse_action(SKIP_ROUND_JSON, VALID_IDS)
    assert action.type == "skip_round"
    assert action.raw == SKIP_ROUND_JSON


def test_parse_action_submit_product_without_product_dict_raises():
    raw = json.dumps({"action": "submit_product"})
    with pytest.raises(ActionParseError):
        parse_action(raw, VALID_IDS)

    raw2 = json.dumps({"action": "submit_product", "product": "not-a-dict"})
    with pytest.raises(ActionParseError):
        parse_action(raw2, VALID_IDS)


def test_parse_action_unknown_action_lists_valid_ones():
    raw = json.dumps({"action": "submit_report"})
    with pytest.raises(ActionParseError) as excinfo:
        parse_action(raw, VALID_IDS)
    msg = str(excinfo.value)
    assert "query" in msg
    assert "submit_product" in msg
    assert "skip_round" in msg


def test_parse_action_fenced_json_ok():
    raw = "这是我的思考过程。\n```json\n" + query_json("risk_control", "有什么合规限制？") + "\n```\n"
    action = parse_action(raw, VALID_IDS)
    assert action.type == "query"
    assert action.target == "risk_control"


# ---------------------------------------------------------------------------
# EnvironmentAgent
# ---------------------------------------------------------------------------


async def test_environment_agent_system_prompt():
    spec = _agent_spec()
    client = RecordingMockClient(["回复内容"])
    agent = EnvironmentAgent(spec, client)
    await agent.respond("你好")
    system_msg = client.calls[0][0]
    assert system_msg["role"] == "system"
    assert system_msg["content"] == ENV_AGENT_PREAMBLE + "\n\n" + spec.prompt


async def test_environment_agent_announcement_injected_once():
    spec = _agent_spec()
    client = RecordingMockClient(["回复一", "回复二"])
    agent = EnvironmentAgent(spec, client)
    agent.inject_info("市场发生重大变化")

    await agent.respond("第一条消息")
    first_user_msg = client.calls[0][-1]
    assert "【市场公告】市场发生重大变化" in first_user_msg["content"]
    assert "第一条消息" in first_user_msg["content"]

    await agent.respond("第二条消息")
    second_user_msg = client.calls[1][-1]
    assert "市场发生重大变化" not in second_user_msg["content"]
    assert "第二条消息" in second_user_msg["content"]


# ---------------------------------------------------------------------------
# StructurerAgent
# ---------------------------------------------------------------------------


def test_structurer_agent_system_message():
    scenario = _stub_scenario()
    client = RecordingMockClient()
    agent = StructurerAgent(client, scenario)
    assert agent.messages[0]["role"] == "system"
    assert agent.messages[0]["content"] == STRUCTURER_PREAMBLE + "\n\n" + scenario.background


def _market() -> MarketState:
    return MarketState(
        round_num=1,
        index_name="沪深300",
        spot=3800.0,
        volatility=0.22,
        risk_free_rate=0.025,
        dividend_yield=0.018,
        recent_trend="震荡上行",
        vix_level="中等",
    )


def _constraints() -> list[Constraint]:
    return [
        Constraint(
            id="c1",
            activate_round=1,
            source="risk_control",
            description="单笔名义本金不得超过 1000 万元",
            check_field="notional",
            check_type="max",
            check_value=10_000_000,
        )
    ]


def _clients() -> list[ClientProfile]:
    return [
        ClientProfile(
            id="client_b",
            name="李女士",
            capital=2_000_000.0,
            max_loss_pct=0.1,
            min_return_pct=0.03,
            risk_appetite="balanced",
        )
    ]


def test_send_round_instruction_renders_all_sections():
    scenario = _stub_scenario()
    client = RecordingMockClient()
    agent = StructurerAgent(client, scenario)
    agent.send_round_instruction(
        round_num=2,
        total_rounds=8,
        market=_market(),
        constraints=_constraints(),
        clients=_clients(),
        reputation=7.5,
        cumulative_commission=120_000.0,
        interactions=5,
    )
    content = agent.messages[-1]["content"]
    assert "第 2/8 轮" in content
    assert "沪深300" in content and "3800" in content
    assert "22.0%" in content  # volatility
    assert "2.5%" in content  # risk_free_rate
    assert "1.8%" in content  # dividend_yield
    assert "震荡上行" in content
    assert "中等" in content
    assert "单笔名义本金不得超过 1000 万元" in content
    assert "client_b：李女士，balanced" in content
    assert "可投" not in content  # 资本额已从客户列表移除
    assert "只能向本轮沟通过的客户提交产品" in content
    assert "累计佣金：120,000 元" in content
    assert "当前声誉：7.5/10" in content
    assert "5 次交互机会" in content


def test_send_round_instruction_no_constraints_shows_placeholder():
    scenario = _stub_scenario()
    client = RecordingMockClient()
    agent = StructurerAgent(client, scenario)
    agent.send_round_instruction(
        round_num=1,
        total_rounds=8,
        market=_market(),
        constraints=[],
        clients=_clients(),
        reputation=5.0,
        cumulative_commission=0.0,
        interactions=5,
    )
    content = agent.messages[-1]["content"]
    assert "无额外约束" in content


def test_inject_round_memory_appears_once_then_cleared():
    scenario = _stub_scenario()
    client = RecordingMockClient()
    agent = StructurerAgent(client, scenario)
    agent.inject_round_memory("上一轮客户拒绝了雪球产品")

    agent.send_round_instruction(
        round_num=2,
        total_rounds=8,
        market=_market(),
        constraints=[],
        clients=_clients(),
        reputation=5.0,
        cumulative_commission=0.0,
        interactions=5,
    )
    first_content = agent.messages[-1]["content"]
    assert "上一轮客户拒绝了雪球产品" in first_content

    agent.send_round_instruction(
        round_num=3,
        total_rounds=8,
        market=_market(),
        constraints=[],
        clients=_clients(),
        reputation=5.0,
        cumulative_commission=0.0,
        interactions=5,
    )
    second_content = agent.messages[-1]["content"]
    assert "上一轮客户拒绝了雪球产品" not in second_content


async def test_decide_action_happy_path():
    scenario = _stub_scenario()
    client = RecordingMockClient([query_json("client_a", "客户想要什么产品？")])
    agent = StructurerAgent(client, scenario)
    action = await agent.decide_action(remaining=5)
    assert action.type == "query"
    assert action.target == "client_a"
    assert len(client.calls) == 1


async def test_decide_action_retry_then_success():
    scenario = _stub_scenario()
    client = RecordingMockClient(["这不是合法 JSON", SKIP_ROUND_JSON])
    agent = StructurerAgent(client, scenario)
    action = await agent.decide_action(remaining=3)
    assert action.type == "skip_round"
    assert len(client.calls) == 2


async def test_decide_action_exhausted_raises():
    scenario = _stub_scenario()
    client = RecordingMockClient(["坏输出一", "坏输出二", "坏输出三"])
    agent = StructurerAgent(client, scenario)
    with pytest.raises(ActionParseError):
        await agent.decide_action(remaining=1)
    assert len(client.calls) == 3


async def test_force_submit_accepts_submit_product():
    scenario = _stub_scenario()
    client = RecordingMockClient([_submit_product_json()])
    agent = StructurerAgent(client, scenario)
    action = await agent.force_submit()
    assert action.type == "submit_product"
    assert len(client.calls) == 1


async def test_force_submit_rejects_query_then_accepts_skip_round():
    scenario = _stub_scenario()
    client = RecordingMockClient(
        [query_json("client_a", "还想再问一句"), SKIP_ROUND_JSON]
    )
    agent = StructurerAgent(client, scenario)
    action = await agent.force_submit()
    assert action.type == "skip_round"
    assert len(client.calls) == 2


async def test_force_submit_raises_after_three_queries():
    scenario = _stub_scenario()
    client = RecordingMockClient(
        [
            query_json("client_a", "问题一"),
            query_json("client_a", "问题二"),
            query_json("client_a", "问题三"),
        ]
    )
    agent = StructurerAgent(client, scenario)
    with pytest.raises(ActionParseError):
        await agent.force_submit()
    assert len(client.calls) == 3


def test_receive_feedback_appends_user_message():
    scenario = _stub_scenario()
    client = RecordingMockClient()
    agent = StructurerAgent(client, scenario)
    n_before = len(agent.messages)
    agent.receive_feedback("产品方案不合法：notional 必须是大于 0 的数字。请修正后重新提交。")
    assert len(agent.messages) == n_before + 1
    last = agent.messages[-1]
    assert last["role"] == "user"
    assert "产品方案不合法" in last["content"]


async def test_force_submit_with_custom_prompt_used_as_user_message():
    scenario = _stub_scenario()
    client = RecordingMockClient([_submit_product_json()])
    agent = StructurerAgent(client, scenario)
    custom_prompt = "产品违反约束：单笔名义本金不得超过 1000 万元。请提交修改后的产品或放弃本轮。"
    action = await agent.force_submit(prompt=custom_prompt)
    assert action.type == "submit_product"
    # 自定义 prompt 应作为 user 消息出现在发给模型的消息中
    sent_messages = client.calls[0]
    assert any(
        m["role"] == "user" and m["content"] == custom_prompt for m in sent_messages
    )


async def test_force_submit_default_prompt_unchanged_when_no_prompt_given():
    scenario = _stub_scenario()
    client = RecordingMockClient([SKIP_ROUND_JSON])
    agent = StructurerAgent(client, scenario)
    await agent.force_submit()
    sent_messages = client.calls[0]
    assert any(
        m["role"] == "user" and "交互机会已用完" in m["content"] for m in sent_messages
    )
