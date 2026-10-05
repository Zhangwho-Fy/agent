"""检查点接线测试：状态要跨轮、跨"进程"都还在。

阶段 2 的验收之一是"重启后能续跑"。这里用**新开一个 checkpointer 实例**来模拟重启：
同一个 SQLite 文件，新的图、新的模型，第二轮应当能看见第一轮说过的话。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from agent.core.bus import EventBus
from agent.core.reliability import EventEmitter
from agent.graph.bridge import stream_turn
from agent.graph.builder import build_graph
from agent.graph.checkpointer import open_checkpointer
from agent.tools.base import ToolContext
from agent.tools.policy import Policy
from agent.tools.registry import default_registry


class RecordingModel:
    """记录每次被喂进去的消息，用来断言"历史还在"。"""

    def __init__(self, replies: list[AIMessage]) -> None:
        self._replies = replies
        self.calls = 0
        self.seen: list[list[Any]] = []

    def bind_tools(self, tools: Any) -> RecordingModel:
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.seen.append(list(messages))
        reply = self._replies[min(self.calls, len(self._replies) - 1)].model_copy(deep=True)
        self.calls += 1
        return reply


def build(tmp_path: Path, model: Any, db_path: Path) -> Any:
    return build_graph(
        model=model,
        registry=default_registry(),
        policy=Policy(tmp_path),
        ctx=ToolContext(workspace=tmp_path),
        emitter=EventEmitter("sess_1", EventBus()),
        checkpointer=open_checkpointer(db_path),
    )


async def test_history_survives_a_fresh_checkpointer(tmp_path: Path) -> None:
    db_path = tmp_path / "ckpt.db"

    first_model = RecordingModel([AIMessage(content="答案是 42")])
    await stream_turn(
        graph=build(tmp_path, first_model, db_path),
        prompt="我喜欢的数字是几？",
        emitter=EventEmitter("sess_1", EventBus()),
        session_id="sess_1",
        turn_id="turn_1",
    )

    # 模拟进程重启：新图、新模型、新 saver，只有 SQLite 文件是同一个
    second_model = RecordingModel([AIMessage(content="42")])
    await stream_turn(
        graph=build(tmp_path, second_model, db_path),
        prompt="刚才我说的是哪个数字？",
        emitter=EventEmitter("sess_1", EventBus()),
        session_id="sess_1",
        turn_id="turn_2",
    )

    contents = [getattr(message, "content", "") for message in second_model.seen[0]]
    assert "我喜欢的数字是几？" in contents, "第一轮的提问要还在"
    assert "答案是 42" in contents, "第一轮的回答也要还在"
    assert "刚才我说的是哪个数字？" in contents, "本轮提问当然要在"


async def test_different_sessions_do_not_share_state(tmp_path: Path) -> None:
    """thread_id 就是会话边界：另一个会话不该看到这边的历史。"""
    db_path = tmp_path / "ckpt.db"
    model_a = RecordingModel([AIMessage(content="A")])
    await stream_turn(
        graph=build(tmp_path, model_a, db_path),
        prompt="甲的问题",
        emitter=EventEmitter("sess_a", EventBus()),
        session_id="sess_a",
        turn_id="turn_a",
    )

    model_b = RecordingModel([AIMessage(content="B")])
    await stream_turn(
        graph=build(tmp_path, model_b, db_path),
        prompt="乙的问题",
        emitter=EventEmitter("sess_b", EventBus()),
        session_id="sess_b",
        turn_id="turn_b",
    )

    contents = [getattr(message, "content", "") for message in model_b.seen[0]]
    assert "甲的问题" not in contents
    assert "乙的问题" in contents
