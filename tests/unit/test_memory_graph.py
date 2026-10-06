"""记忆与检索接进图之后的行为：免审批、结果进上下文、摘要进尾部。

单测（test_memory / test_search_code）证明工具本身对；这里证明**整条链路**对：
`agent → approve → tools → agent` 走通、`memory_write` 不触发审批、检索结果带来源标记、
记忆摘要只拼在请求尾部而不进 state。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.config import Settings
from agent.core.bus import EventBus
from agent.core.reliability import EventEmitter
from agent.graph.bridge import recursion_limit_for, stream_turn
from agent.graph.nodes import build_model_node
from agent.graph.wiring import build_session_graph, get_memory_store, reset_runtime_caches
from agent.memory import GLOBAL_SCOPE, MemoryKind, MemoryStore
from agent.memory.render import render_memories
from agent.retrieval.embeddings import HashingEmbedder
from agent.store import repo
from agent.store.db import Database
from agent.tools.registry import default_registry


class ScriptedModel:
    """假模型：按脚本返回回复，并记下每次收到的完整消息列表（不联网）。"""

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = replies
        self.calls = 0
        self.seen: list[Any] = []

    def bind_tools(self, tools: Any) -> ScriptedModel:
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.seen.append(messages)
        reply = self._replies[min(self.calls, len(self._replies) - 1)].model_copy(deep=True)
        self.calls += 1
        return reply


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    call = {"name": name, "args": args, "id": call_id, "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "api_key": "",
        "provider": "replay",
        "workspace": tmp_path,
        "db_path": tmp_path / "agent.db",
        "memory_db_path": tmp_path / "memory.db",
        "index_dir": tmp_path / "index",
        "compress_enabled": False,  # 评测与集成测试都要确定性（D32）
    }
    base.update(overrides)
    return Settings(**base)


@pytest.fixture(autouse=True)
def _reset_caches() -> Iterator[None]:
    """进程级缓存会跨用例复用 sqlite 连接，测完必须清掉。"""
    reset_runtime_caches()
    yield
    reset_runtime_caches()


async def test_memory_write_runs_through_the_graph_without_approval(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    session_db = Database(tmp_path / "agent.db")
    session_db.connect()
    await repo.create_session(
        session_db, session_id="sess_mem", profile="code", workspace=str(tmp_path)
    )
    await repo.start_turn(session_db, turn_id="turn_mem", session_id="sess_mem")
    model = ScriptedModel(
        [
            tool_call(
                "memory_write",
                {"content": "用户偏好中文回答", "kind": "preference", "scope": "global"},
            ),
            AIMessage(content="记住了"),
        ]
    )
    emitter = EventEmitter("sess_mem", EventBus())
    wiring = build_session_graph(
        settings, workspace=tmp_path, model=model, emitter=emitter, db=session_db
    )
    asked: list[Any] = []

    async def approver(requests: list[dict[str, Any]]) -> dict[str, bool]:
        asked.append(requests)
        return {request["call_id"]: False for request in requests}

    try:
        result = await stream_turn(
            graph=wiring.graph,
            prompt="记住：以后都用中文回答",
            emitter=emitter,
            session_id="sess_mem",
            turn_id="turn_mem",
            recursion_limit=recursion_limit_for(settings.max_tool_rounds),
            approver=approver,
            approval_timeout_s=1,
        )
        # 审计免费复用现有流水线（D47）：记忆写入也躺在 tool_calls 表里
        calls = await repo.list_tool_calls(session_db, "turn_mem")
    finally:
        await session_db.close()

    assert result.status == "done", result.text
    assert asked == [], "D42：记忆写入免审批，approver 不该被叫到"
    assert [str(call["name"]) for call in calls] == ["memory_write"]
    assert calls[0]["decision"] == "auto"
    cards = await get_memory_store(settings).list_cards(scopes=[GLOBAL_SCOPE])
    assert [card.content for card in cards] == ["用户偏好中文回答"]


async def test_search_code_runs_through_the_graph_and_marks_the_source(tmp_path: Path) -> None:
    (tmp_path / "alpha.py").write_text("def needle_function():\n    return 1\n", encoding="utf-8")
    settings = make_settings(tmp_path)
    model = ScriptedModel(
        [
            tool_call("search_code", {"query": "needle_function"}),
            AIMessage(content="在 alpha.py"),
        ]
    )
    emitter = EventEmitter("sess_rag", EventBus())
    wiring = build_session_graph(settings, workspace=tmp_path, model=model, emitter=emitter)

    result = await stream_turn(
        graph=wiring.graph,
        prompt="needle_function 在哪定义",
        emitter=emitter,
        session_id="sess_rag",
        turn_id="turn_rag",
        recursion_limit=recursion_limit_for(settings.max_tool_rounds),
    )

    assert result.status == "done"
    assert model.calls == 2, "模型应该被叫第二次：看到命中后收尾"
    second_request = model.seen[1]
    tool_messages = [m for m in second_request if getattr(m, "type", "") == "tool"]
    assert tool_messages, "检索结果必须以工具消息回填"
    assert "alpha.py" in tool_messages[0].content
    assert tool_messages[0].content.startswith("<untrusted"), "仓库内容进上下文要带来源标记"


async def test_memory_digest_is_appended_to_the_request_tail(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "m.db", embedder=HashingEmbedder())
    try:
        await store.write(
            scope=GLOBAL_SCOPE,
            kind=MemoryKind.PREFERENCE,
            content="用户偏好中文回答",
            importance=0.9,
        )
        model = ScriptedModel([AIMessage(content="ok")])

        async def digest(_state: Any) -> str:
            cards = await store.digest(scopes=[GLOBAL_SCOPE], limit=5)
            return render_memories(cards)

        node = build_model_node(model, default_registry(), memory_digest=digest)
        update = await node({"messages": [HumanMessage(content="你好")]})
    finally:
        await store.close()

    request = model.seen[0]
    assert "<memory_context" in request[-2].content, "记忆摘要拼在尾部（D44）"
    assert "<agent_state" in request[-1].content, "状态块仍在最后"
    # 不进 state：写进历史会留过期副本，被后面几轮当同样可信的事实
    assert all(
        "memory_context" not in getattr(message, "content", "") for message in update["messages"]
    )


def test_settings_derive_memory_and_index_paths(tmp_path: Path) -> None:
    """默认落在会话库旁边：一个数据目录，三样东西（会话 / 记忆 / 索引）。"""
    settings = Settings(_env_file=None, api_key="", db_path=tmp_path / "data" / "agent.db")

    assert settings.resolved_memory_db_path == tmp_path / "data" / "memory.db"
    assert settings.resolved_index_dir == tmp_path / "data" / "index"


async def test_broken_memory_db_does_not_break_the_turn(tmp_path: Path) -> None:
    """记忆是增强：库打不开只跳过这一轮，不能让主链路失败。"""
    blocked = tmp_path / "blocked"
    blocked.write_text("这是一个文件，不是一个目录", encoding="utf-8")
    settings = make_settings(tmp_path, memory_db_path=blocked / "memory.db")
    model = ScriptedModel([AIMessage(content="照常回答")])
    emitter = EventEmitter("sess_broken", EventBus())
    wiring = build_session_graph(settings, workspace=tmp_path, model=model, emitter=emitter)

    result = await stream_turn(
        graph=wiring.graph,
        prompt="你好",
        emitter=emitter,
        session_id="sess_broken",
        turn_id="turn_broken",
        recursion_limit=recursion_limit_for(settings.max_tool_rounds),
    )

    assert result.status == "done"
    assert result.text == "照常回答"
