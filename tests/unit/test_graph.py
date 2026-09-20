"""图与两个节点的行为测试：循环上限、工具轮数、护栏换算。

这里用的是**假模型**（不联网、不需要密钥）：它按预置脚本返回"要不要调用工具"。
除了测上限，它顺便证明了图能离线驱动——这是阶段 3 回放模型的地基。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent.core.bus import EventBus
from agent.core.reliability import EventEmitter
from agent.graph.bridge import recursion_limit_for, stream_turn
from agent.graph.builder import build_graph
from agent.graph.nodes import build_model_node, build_tool_node
from agent.tools.base import ToolContext
from agent.tools.policy import Policy
from agent.tools.registry import default_registry


class ScriptedModel:
    """假模型：按脚本返回回复，脚本用完就一直重复最后一条。

    注意每次都要返回**新的消息对象**：`add_messages` 归约器会按 id 去重，
    同一个对象重复返回等于没追加（id 都是 None 时它按"是否已在列表里"判断），
    于是最后一条消息会停留在 ToolMessage，条件边直接收工。
    真实模型每次返回的都是新对象，这里用 `model_copy` 模拟这一点。
    """

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = replies
        self.calls = 0

    def bind_tools(self, tools: Any) -> ScriptedModel:
        return self  # 图在构造节点时会调这一个方法

    async def ainvoke(self, messages: Any) -> AIMessage:
        reply = self._replies[min(self.calls, len(self._replies) - 1)].model_copy(deep=True)
        self.calls += 1
        return reply


def make_emitter() -> EventEmitter:
    return EventEmitter("sess_test", EventBus())


def tool_call_reply() -> AIMessage:
    """一条"我要调 fs_list"的回复。"""
    call = {"name": "fs_list", "args": {"path": "."}, "id": "call_1", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def make_ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(workspace=tmp_path)


async def test_model_node_stops_at_the_round_limit() -> None:
    model = ScriptedModel([AIMessage(content="这条不该被用到")])
    node = build_model_node(model, default_registry(), max_tool_rounds=2)

    update = await node({"messages": [], "tool_rounds": 2})

    assert model.calls == 0, "轮数用尽后不该再调模型"
    message = update["messages"][0]
    assert not message.tool_calls, "收尾消息不能带 tool_calls，否则条件边还会转回工具"
    assert "上限" in message.content


async def test_model_node_calls_model_below_the_limit() -> None:
    model = ScriptedModel([AIMessage(content="答案")])
    node = build_model_node(model, default_registry(), max_tool_rounds=3)

    update = await node({"messages": [HumanMessage(content="hi")], "tool_rounds": 1})

    assert model.calls == 1
    assert update["rounds"] == 1
    assert update["messages"][0].content == "答案"


async def test_tool_node_counts_tool_rounds(tmp_path: Path) -> None:
    node = build_tool_node(
        registry=default_registry(),
        policy=Policy(tmp_path),
        ctx=make_ctx(tmp_path),
        emitter=make_emitter(),
    )

    update = await node({"messages": [tool_call_reply()], "tool_rounds": 0})

    assert update["tool_rounds"] == 1
    assert update["messages"][0].type == "tool"


def test_recursion_limit_leaves_room_for_every_round() -> None:
    # 一次工具往返 = agent + tools 两个 superstep，再加进入与收尾的余量
    assert recursion_limit_for(12) == 28
    assert recursion_limit_for(1) > 2


async def test_graph_stops_after_max_tool_rounds(tmp_path: Path) -> None:
    """模型一直要工具时，图必须在配置的轮数上停住，而不是撞 recursion_limit 崩掉。"""
    model = ScriptedModel([tool_call_reply()])  # 永远要工具
    emitter = make_emitter()
    events = []
    emitter.on_event = events.append
    graph = build_graph(
        model=model,
        registry=default_registry(),
        policy=Policy(tmp_path),
        ctx=make_ctx(tmp_path),
        emitter=emitter,
        max_tool_rounds=2,
    )

    await stream_turn(
        graph=graph,
        prompt="随便问问",
        emitter=emitter,
        session_id="sess_test",
        turn_id="turn_1",
        recursion_limit=recursion_limit_for(2),
    )

    types = [event.type.value for event in events]
    assert types.count("tool.call") == 2, "只允许两轮工具往返"
    assert model.calls == 2, "用尽轮数后不再调模型"
    done = next(event for event in events if event.type.value == "text.done")
    assert "上限" in done.data["text"], "收尾消息要能出现在最终文本里（非流式时的兜底）"
