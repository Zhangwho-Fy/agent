"""回放 Provider 测试：它是后面所有"不联网也能跑"的测试的地基。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.core.chunks import ProviderChunk, TextDelta, ToolCallDelta
from agent.core.messages import Message
from agent.core.tool_spec import ToolSpec
from agent.providers.replay import ReplayExhausted, ReplayProvider


async def collect(
    provider: ReplayProvider,
    messages: list[Message] | None = None,
    tools: list[ToolSpec] | None = None,
) -> list[ProviderChunk]:
    chunks: list[ProviderChunk] = []
    async for chunk in provider.stream(
        messages=messages or [Message.user("hi")], tools=tools or []
    ):
        chunks.append(chunk)
    return chunks


async def test_replays_chunks_turn_by_turn() -> None:
    provider = ReplayProvider(
        [
            [TextDelta(text="第一"), TextDelta(text="轮")],
            [TextDelta(text="第二轮")],
        ]
    )

    first = await collect(provider)
    second = await collect(provider)

    assert [chunk.text for chunk in first if isinstance(chunk, TextDelta)] == ["第一", "轮"]
    assert [chunk.text for chunk in second if isinstance(chunk, TextDelta)] == ["第二轮"]
    assert provider.remaining == 0


async def test_running_out_of_recording_raises() -> None:
    provider = ReplayProvider([[TextDelta(text="只有一轮")]])
    await collect(provider)

    with pytest.raises(ReplayExhausted) as excinfo:
        await collect(provider)

    assert "主循环多跑了一轮" in str(excinfo.value)


async def test_records_requests_for_assertions() -> None:
    provider = ReplayProvider([[TextDelta(text="ok")]])
    tools = [ToolSpec(name="fs.read", description="读文件")]

    await collect(provider, [Message.user("看看 src/app.py")], tools)

    assert provider.requests[0]["messages"][0]["content"] == "看看 src/app.py"
    assert provider.requests[0]["messages"][0]["role"] == "user"
    assert provider.requests[0]["tools"][0]["name"] == "fs.read"


async def test_loads_from_file_with_wrapped_turns(tmp_path: Path) -> None:
    fixture = tmp_path / "turn.json"
    fixture.write_text(
        json.dumps(
            {
                "turns": [
                    {"chunks": [{"type": "text_delta", "text": "hi"}]},
                    [{"type": "tool_call_delta", "index": 0, "id": "c1", "name": "fs.read"}],
                ]
            }
        ),
        encoding="utf-8",
    )

    provider = ReplayProvider.from_file(fixture)

    first = await collect(provider)
    second = await collect(provider)

    assert isinstance(first[0], TextDelta)
    assert isinstance(second[0], ToolCallDelta)
    assert second[0].name == "fs.read"
