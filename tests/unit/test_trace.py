"""录放测试。

核心断言只有一条：**同一份图、同一个提示词，录一遍再放一遍，事件序列必须一致**。
如果哪天这条挂了，说明图的行为变了——这正是录放存在的意义。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent.core.bus import EventBus
from agent.core.events import Event
from agent.core.reliability import EventEmitter
from agent.graph.bridge import stream_turn
from agent.graph.builder import build_graph
from agent.models.trace import (
    RecordingChatModel,
    ReplayChatModel,
    TraceExhausted,
    TracePlayer,
    TraceRecorder,
)
from agent.tools.base import ToolContext
from agent.tools.policy import Policy
from agent.tools.registry import default_registry


class FakeModel(BaseChatModel):
    """按脚本回复的假模型：代替真实 API，测试里不联网。"""

    replies: list[AIMessage] = Field(default_factory=list)
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeModel:
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        reply = self.replies[min(self.calls, len(self.replies) - 1)].model_copy(deep=True)
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ):
        return self._generate(messages)


def list_reply() -> AIMessage:
    call = {"name": "fs_list", "args": {"path": "."}, "id": "call_1", "type": "tool_call"}
    return AIMessage(content="", tool_calls=[call])


async def run_turn(model: Any, tmp_path: Path, prompt: str = "看看目录") -> tuple[list[Event], Any]:
    emitter = EventEmitter("sess_t", EventBus())
    events: list[Event] = []
    emitter.on_event = events.append
    graph = build_graph(
        model=model,
        registry=default_registry(),
        policy=Policy(tmp_path),
        ctx=ToolContext(workspace=tmp_path),
        emitter=emitter,
    )
    result = await stream_turn(
        graph=graph,
        prompt=prompt,
        emitter=emitter,
        session_id="sess_t",
        turn_id="turn_1",
    )
    return events, result


def stable(events: list[Event]) -> list[dict[str, Any]]:
    """去掉不参与"行为"的字段再比较：耗时每次都不一样，不是行为差异。"""
    return [{k: v for k, v in event.data.items() if k != "duration_ms"} for event in events]


async def test_record_then_replay_reproduces_the_same_events(tmp_path: Path) -> None:
    # 录制文件放在工作区**外面**：否则第二轮 `fs_list` 会看到它，工具输出就不一样了
    workspace = tmp_path / "ws"
    workspace.mkdir()
    trace = tmp_path / "record.jsonl"
    replies = [list_reply(), AIMessage(content="看完了")]

    recorded_model = RecordingChatModel(
        inner=FakeModel(replies=replies),
        recorder=TraceRecorder(path=trace),
        model_name="fake",
    )
    events_a, result_a = await run_turn(recorded_model, workspace)
    assert trace.exists()

    replayed_model = ReplayChatModel(player=TracePlayer.from_file(trace))
    events_b, result_b = await run_turn(replayed_model, workspace)

    assert result_b.text == result_a.text == "看完了"
    assert [event.type.value for event in events_b] == [event.type.value for event in events_a]
    assert stable(events_b) == stable(events_a)


async def test_trace_file_is_line_delimited_json(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    model = RecordingChatModel(
        inner=FakeModel(replies=[AIMessage(content="一次")]),
        recorder=TraceRecorder(path=trace),
        model_name="fake",
    )
    await run_turn(model, tmp_path)

    lines = trace.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["model"] == "fake"
    assert record["response"]["type"] == "ai"
    assert record["response"]["data"]["content"] == "一次"
    assert record["request"], "请求也要录下来，便于排查"


async def test_tool_calls_and_usage_survive_the_round_trip(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    reply = AIMessage(
        content="",
        tool_calls=[
            {"name": "fs_list", "args": {"path": "src"}, "id": "call_9", "type": "tool_call"}
        ],
        usage_metadata={"input_tokens": 11, "output_tokens": 22, "total_tokens": 33},
    )
    model = RecordingChatModel(
        inner=FakeModel(replies=[reply]),
        recorder=TraceRecorder(path=trace),
        model_name="fake",
    )
    await run_turn(model, tmp_path)

    replayed = ReplayChatModel(player=TracePlayer.from_file(trace)).player.next()
    assert replayed.tool_calls[0]["name"] == "fs_list"
    assert replayed.tool_calls[0]["args"] == {"path": "src"}
    assert replayed.usage_metadata == {
        "input_tokens": 11,
        "output_tokens": 22,
        "total_tokens": 33,
    }


def test_running_out_of_records_says_what_to_do(tmp_path: Path) -> None:
    """图多调了一次模型时，报错要说人话——这是录放最有价值的告警。"""
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"model": "fake", "response": {"type": "ai", "data": {"content": "唯一一条"}}})
        + "\n",
        encoding="utf-8",
    )
    player = TracePlayer.from_file(trace)

    assert player.next().content == "唯一一条"
    with pytest.raises(TraceExhausted) as excinfo:
        player.next()
    assert "只有 1 次模型调用" in str(excinfo.value)
    assert "重新用 AGENT_TRACE_PATH 录一份" in str(excinfo.value)


def test_missing_trace_file_is_reported_early(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="回放文件不存在"):
        TracePlayer.from_file(tmp_path / "nope.jsonl")
