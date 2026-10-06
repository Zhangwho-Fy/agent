"""审批流测试：挂起 → 恢复 → 执行或拒绝。

钉住两条不变量：

1. 挂起时工具**不能**被执行（还没批准就先动了，那审批就是摆设）；
2. 没通道、被拒、超时，**一律不放行**——审批的默认值必须是拒绝。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from agent.core.bus import EventBus
from agent.core.events import Event
from agent.core.reliability import EventEmitter
from agent.graph.bridge import stream_turn
from agent.graph.builder import build_graph
from agent.graph.checkpointer import open_checkpointer
from agent.tools.base import ToolContext
from agent.tools.policy import Policy
from agent.tools.registry import default_registry


class ScriptedModel:
    """按脚本回复的假模型：不联网、不需要 key。"""

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = replies
        self.calls = 0

    def bind_tools(self, tools: Any) -> ScriptedModel:
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        reply = self._replies[min(self.calls, len(self._replies) - 1)].model_copy(deep=True)
        self.calls += 1
        return reply


def write_reply(filename: str = "out.txt") -> AIMessage:
    """让模型请求一条有副作用的命令：重定向写文件 → 分级为 write → 需要审批。"""
    call = {
        "name": "shell_exec",
        "args": {"command": f"echo hi > {filename}"},
        "id": "call_w",
        "type": "tool_call",
    }
    return AIMessage(content="", tool_calls=[call])


def build(tmp_path: Path, model: Any) -> tuple[Any, EventEmitter, list[Event]]:
    emitter = EventEmitter("sess_t", EventBus())
    events: list[Event] = []
    emitter.on_event = events.append
    graph = build_graph(
        model=model,
        registry=default_registry(),
        policy=Policy(tmp_path),
        ctx=ToolContext(workspace=tmp_path),
        emitter=emitter,
        checkpointer=open_checkpointer(tmp_path / "ckpt.db"),
    )
    return graph, emitter, events


async def test_granted_approval_executes_the_tool(tmp_path: Path) -> None:
    model = ScriptedModel([write_reply(), AIMessage(content="写好了")])
    graph, emitter, events = build(tmp_path, model)
    asked: list[dict[str, Any]] = []

    async def approver(requests: list[dict[str, Any]]) -> dict[str, bool]:
        asked.extend(requests)
        return {request["call_id"]: True for request in requests}

    result = await stream_turn(
        graph=graph,
        prompt="写个文件",
        emitter=emitter,
        session_id="sess_t",
        turn_id="turn_1",
        approver=approver,
        approval_timeout_s=5,
    )

    assert [request["name"] for request in asked] == ["shell_exec"]
    assert (tmp_path / "out.txt").exists(), "批准之后应当真的执行"
    assert result.text == "写好了"
    assert result.status == "done"
    types = [event.type.value for event in events]
    assert types.index("approval.required") < types.index("tool.result")


async def test_denied_approval_leaves_the_workspace_untouched(tmp_path: Path) -> None:
    model = ScriptedModel([write_reply(), AIMessage(content="那我换个做法")])
    graph, emitter, events = build(tmp_path, model)

    async def approver(requests: list[dict[str, Any]]) -> dict[str, bool]:
        return {request["call_id"]: False for request in requests}

    await stream_turn(
        graph=graph,
        prompt="写个文件",
        emitter=emitter,
        session_id="sess_t",
        turn_id="turn_1",
        approver=approver,
        approval_timeout_s=5,
    )

    assert not (tmp_path / "out.txt").exists(), "没批准就不能动工作区"
    denial = next(event for event in events if event.type.value == "tool.result")
    assert "未执行" in denial.data["preview"], "要把拒绝原因回填给模型"


async def test_denied_approval_stops_the_turn(tmp_path: Path) -> None:
    """拒绝之后**这一轮直接停**：不该再叫模型"换个命令再问一次"。

    用户的原话：拒绝了它还在思考、一直问我。所以这里不靠提示词劝模型住手——
    路由直接走到 END，模型根本不会被叫第二次。
    """
    model = ScriptedModel([write_reply(), AIMessage(content="我再试一次")])
    graph, emitter, events = build(tmp_path, model)

    async def approver(requests: list[dict[str, Any]]) -> dict[str, bool]:
        return {request["call_id"]: False for request in requests}

    result = await stream_turn(
        graph=graph,
        prompt="写个文件",
        emitter=emitter,
        session_id="sess_t",
        turn_id="turn_1",
        approver=approver,
        approval_timeout_s=5,
    )

    assert model.calls == 1, "拒绝之后模型不该被再叫一次"
    assert result.status == "stopped"
    assert "已停下" in result.text
    assert any(
        event.type.value == "text.delta" and "已停下" in event.data["text"] for event in events
    ), "这句话要能被界面画出来（text.done 界面上不重画）"
    done = next(event for event in events if event.type.value == "turn.done")
    assert done.data["status"] == "stopped"


async def test_timeout_counts_as_denied(tmp_path: Path) -> None:
    model = ScriptedModel([write_reply(), AIMessage(content="好")])
    graph, emitter, _ = build(tmp_path, model)

    async def slow_approver(requests: list[dict[str, Any]]) -> dict[str, bool]:
        await asyncio.sleep(10)
        return {request["call_id"]: True for request in requests}

    await stream_turn(
        graph=graph,
        prompt="写个文件",
        emitter=emitter,
        session_id="sess_t",
        turn_id="turn_1",
        approver=slow_approver,
        approval_timeout_s=0.2,
    )

    assert not (tmp_path / "out.txt").exists(), "超时按拒绝"


async def test_missing_approver_counts_as_denied(tmp_path: Path) -> None:
    """无人值守又没开 --yes：不能悄悄放行。"""
    model = ScriptedModel([write_reply(), AIMessage(content="好")])
    graph, emitter, _ = build(tmp_path, model)

    await stream_turn(
        graph=graph,
        prompt="写个文件",
        emitter=emitter,
        session_id="sess_t",
        turn_id="turn_1",
        approver=None,
    )

    assert not (tmp_path / "out.txt").exists()


async def test_read_only_calls_do_not_ask_for_approval(tmp_path: Path) -> None:
    call = {"name": "fs_list", "args": {"path": "."}, "id": "call_r", "type": "tool_call"}
    model = ScriptedModel([AIMessage(content="", tool_calls=[call]), AIMessage(content="看完了")])
    graph, emitter, events = build(tmp_path, model)
    asked: list[dict[str, Any]] = []

    async def approver(requests: list[dict[str, Any]]) -> dict[str, bool]:
        asked.extend(requests)
        return {}

    await stream_turn(
        graph=graph,
        prompt="看看目录",
        emitter=emitter,
        session_id="sess_t",
        turn_id="turn_1",
        approver=approver,
    )

    assert asked == [], "只读调用不该打扰人"
    assert "approval.required" not in [event.type.value for event in events]
