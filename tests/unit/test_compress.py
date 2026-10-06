"""压缩：阶梯选择、指针化、摘要、找回通道。

对应 docs/design.md 第 7.5 节的验收表。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.core.bus import EventBus
from agent.core.compress import (
    MIN_COMPRESS_CHARS,
    POINTER_HEAD_CHARS,
    plan,
    pointerize,
    split_for_summary,
    summary_message,
    summary_request,
)
from agent.core.reliability import EventEmitter
from agent.graph.compress import build_compress_node
from agent.store import repo
from agent.store.db import Database
from agent.tools.base import ToolContext
from agent.tools.recall import RECALL_TOOL


def tool_message(call_id: str, body: str = "x" * (MIN_COMPRESS_CHARS + 50)) -> ToolMessage:
    return ToolMessage(
        content=f'<untrusted source="fs_read">\n{body}\n</untrusted>', tool_call_id=call_id
    )


def history(count: int) -> list[Any]:
    """构造 count 组 (用户 → 模型要工具 → 工具结果)，再补一条用户消息。"""
    messages: list[Any] = [HumanMessage(content="开始")]
    for index in range(count):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fs_read", "args": {}, "id": f"c{index}", "type": "tool_call"}
                ],
            )
        )
        messages.append(
            tool_message(f"c{index}", body=f"第{index}份输出" + "长" * MIN_COMPRESS_CHARS)
        )
    return messages


def test_plan_has_two_thresholds() -> None:
    assert not plan(0.3).active
    assert plan(0.65).pointerize and not plan(0.65).summarize
    assert plan(0.85).pointerize and plan(0.85).summarize


def test_pointerize_keeps_the_recent_tail_intact() -> None:
    messages = history(3)

    pointers, compressed = pointerize(messages, keep_recent=1)

    assert compressed == ["c0", "c1"]
    assert len(pointers) == len(messages), "只换内容，不删消息——配对关系必须保住"
    assert [str(m.tool_call_id) for m in pointers if isinstance(m, ToolMessage)] == [
        "c0",
        "c1",
        "c2",
    ]
    assert pointers[-1].content == messages[-1].content, "最近一条原样保留"


def test_pointer_carries_the_call_id_and_a_way_back() -> None:
    pointers, _ = pointerize(history(2), keep_recent=0)
    first = pointers[2]
    assert isinstance(first, ToolMessage)

    assert first.content.startswith('<compressed call_id="c0">')
    assert 'recall(call_id="c0")' in first.content
    assert '<untrusted source="fs_read">' in first.content, "来源标记要留在指针里"
    assert len(first.content) < POINTER_HEAD_CHARS * 2


def test_pointerize_is_idempotent_and_skips_small_results() -> None:
    messages = [
        HumanMessage(content="开始"),
        tool_message("c0"),
        ToolMessage(content="很短", tool_call_id="c1"),
    ]

    once, _ = pointerize(messages, keep_recent=0)
    twice, again = pointerize(once, keep_recent=0)

    assert again == [], "已经是指针的不再压"
    assert [m.content for m in twice] == [m.content for m in once]
    assert twice[2].content == "很短", "短结果不值得压"


def test_split_for_summary_never_cuts_a_tool_pair() -> None:
    messages = history(3)

    prefix, keep = split_for_summary(messages, keep_recent=2)

    assert prefix and keep
    assert prefix + keep == messages
    # 真正的约束在保留段这一侧：它不能从一个"配不上的工具结果"开始
    assert not isinstance(keep[0], ToolMessage)
    assert not getattr(prefix[-1], "tool_calls", None), "摘要段不能以'要工具'的回复结尾"


def test_split_grows_the_tail_to_reach_a_boundary() -> None:
    """宁可比要求的多留一条，也不能把 tool_call / tool_result 切开。"""
    messages = history(3)

    prefix, keep = split_for_summary(messages, keep_recent=1)

    assert prefix + keep == messages
    assert not isinstance(keep[0], ToolMessage)


def test_summary_request_states_the_protected_facts() -> None:
    system, human = summary_request(history(2))

    assert isinstance(system, SystemMessage)
    assert "文件路径" in system.content and "错误信息原文" in system.content
    assert "## 未完成" in system.content
    assert isinstance(human, HumanMessage)
    assert "第0份输出" in human.content


def test_summary_message_is_wrapped_in_a_session_summary_tag() -> None:
    """D36：压缩产物的标签是 `<session_summary>`，不是 `<memory>`。"""
    text = summary_message("## 任务\n修 bug").content
    assert text.startswith("<session_summary ")
    assert text.rstrip().endswith("</session_summary>")


class FakeModel:
    """只回一句话，用来验证"该调才调"。"""

    def __init__(self, reply: str = "## 任务\n（无）") -> None:
        self.reply = reply
        self.calls = 0
        self.fail = False

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        if self.fail:
            raise RuntimeError("压缩器炸了")
        return AIMessage(content=self.reply)


def make_emitter() -> tuple[EventEmitter, list[Any]]:
    emitter = EventEmitter("sess_test", EventBus())
    events: list[Any] = []
    emitter.on_event = events.append
    return emitter, events


async def test_node_does_nothing_below_the_threshold() -> None:
    emitter, events = make_emitter()
    node = build_compress_node(emitter=emitter, model=FakeModel(), context_limit=1000)

    update = await node({"messages": history(4), "context_tokens": 300, "turn_id": "t1"})

    assert update == {}
    assert events == []


async def test_node_pointerizes_without_calling_the_model() -> None:
    model = FakeModel()
    emitter, events = make_emitter()
    node = build_compress_node(
        emitter=emitter, model=model, context_limit=1000, keep_recent_tool_results=1
    )
    messages = history(3)

    update = await node({"messages": messages, "context_tokens": 650, "turn_id": "t1"})

    assert model.calls == 0, "60% 档是零成本的，不该调模型"
    assert len(update["messages"]) == len(messages) + 1, "清空标记 + 原序重建"
    event = events[0]
    assert event.type.value == "context.compressed"
    assert event.data["pointerized"] == ["c0", "c1"]
    assert event.data["summarized"] == 0


async def test_node_summarizes_at_the_high_threshold() -> None:
    model = FakeModel("## 任务\n修 bug\n## 未完成\na.py")
    emitter, events = make_emitter()
    node = build_compress_node(emitter=emitter, model=model, context_limit=1000)

    update = await node({"messages": history(4), "context_tokens": 850, "turn_id": "t1"})

    assert model.calls == 1
    first = update["messages"][1]
    assert isinstance(first, SystemMessage) and "<session_summary" in first.content
    assert "a.py" in first.content
    assert events[0].data["summarized"] > 0


async def test_node_fuses_after_repeated_failures() -> None:
    model = FakeModel()
    model.fail = True
    emitter, _ = make_emitter()
    node = build_compress_node(emitter=emitter, model=model, context_limit=1000, max_failures=2)
    state = {"messages": history(4), "context_tokens": 850, "turn_id": "t1"}

    await node(state)
    await node(state)
    await node(state)

    assert model.calls == 2, "连续失败到阈值就熔断，别一直烧钱"


async def test_compression_survives_a_checkpointer(tmp_path: Path) -> None:
    """整表替换 + 检查点：第二轮读回来的是压缩后的列表，且配对仍然成立。"""
    from langgraph.graph import END, START, StateGraph

    from agent.graph.checkpointer import open_checkpointer
    from agent.graph.state import AgentState

    emitter, _ = make_emitter()
    builder = StateGraph(AgentState)
    builder.add_node(
        "compress",
        build_compress_node(
            emitter=emitter,
            model=FakeModel("## 任务\n（无）"),
            context_limit=1000,
            keep_recent_tool_results=1,
        ),
    )
    builder.add_edge(START, "compress")
    builder.add_edge("compress", END)
    graph = builder.compile(checkpointer=open_checkpointer(tmp_path / "ckpt.db"))
    config = {"configurable": {"thread_id": "sess_1"}}
    messages = history(4)

    await graph.ainvoke(
        {"messages": messages, "context_tokens": 900, "turn_id": "turn_1"}, config=config
    )
    state = graph.get_state(config)
    stored = state.values["messages"]

    assert len(stored) < len(messages), "前段被换成了一条摘要"
    assert isinstance(stored[0], SystemMessage) and "<session_summary" in stored[0].content
    assert any(
        isinstance(message, ToolMessage) and message.content.startswith("<compressed ")
        for message in stored
    ), "旧工具结果已经是指针"
    assert stored[-1].content.startswith("<untrusted "), "最近一条工具结果原样保留"


@pytest.fixture()
async def store(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "agent.db")
    database.connect()
    yield database
    await database.close()


async def test_recall_returns_the_original_output(store: Database, tmp_path: Path) -> None:
    await repo.create_session(store, session_id="sess_1", profile="code", workspace=str(tmp_path))
    await repo.start_turn(store, turn_id="turn_1", session_id="sess_1")
    await repo.start_tool_call(
        store,
        call_id="call_1",
        session_id="sess_1",
        turn_id="turn_1",
        name="fs_read",
        args={"path": "a.py"},
        tier="read",
    )
    await repo.finish_tool_call(store, "call_1", status="ok", decision="auto", result="原始全文")
    # 工具不认识数据库：装配层把"按 call_id 取回原文"包成回调交给它
    ctx = ToolContext(
        workspace=tmp_path,
        fetch_tool_call=lambda call_id: repo.get_tool_call(store, call_id),
    )

    result = await RECALL_TOOL.run({"call_id": "call_1"}, ctx)
    missing = await RECALL_TOOL.run({"call_id": "nope"}, ctx)
    no_db = await RECALL_TOOL.run({"call_id": "call_1"}, ToolContext(workspace=tmp_path))

    assert result.ok and "原始全文" in result.content and "fs_read" in result.content
    assert not missing.ok and "没有这条工具调用记录" in missing.content
    assert not no_db.ok and "没有接会话库" in no_db.content
