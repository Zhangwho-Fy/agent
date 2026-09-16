"""事件模型测试：这些字段是客户端协议的基础，不能随意改。"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agent.core.events import Event, EventType


def test_create_fills_envelope() -> None:
    event = Event.create(
        session_id="sess_1",
        seq=7,
        type=EventType.TOOL_CALL,
        data={"name": "fs.read"},
        turn_id="turn_1",
    )

    assert event.id.startswith("evt_")
    assert event.seq == 7
    assert event.type is EventType.TOOL_CALL
    assert event.turn_id == "turn_1"
    assert event.data == {"name": "fs.read"}
    assert event.ts.tzinfo is not None


def test_payload_is_json_serializable() -> None:
    event = Event.create(
        session_id="sess_1", seq=1, type=EventType.TEXT_DELTA, data={"text": "你好"}
    )

    encoded = json.dumps(event.to_payload(), ensure_ascii=False)

    assert '"text": "你好"' in encoded
    assert '"type": "text.delta"' in encoded


def test_sse_frame_carries_seq_as_event_id() -> None:
    event = Event.create(session_id="sess_1", seq=42, type=EventType.TEXT_DONE, data={"text": "hi"})

    frame = event.to_sse()

    assert frame.startswith("id: 42\nevent: text.done\ndata: ")
    assert frame.endswith("\n\n")
    payload = json.loads(frame.split("data: ", 1)[1])
    assert payload["seq"] == 42


def test_event_is_immutable() -> None:
    event = Event.create(session_id="sess_1", seq=1, type=EventType.TURN_STARTED)

    with pytest.raises(ValidationError):
        event.seq = 2  # type: ignore[misc]


def test_unknown_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Event(
            id="evt_1",
            seq=1,
            session_id="sess_1",
            type=EventType.TURN_STARTED,
            surprise="x",  # type: ignore[call-arg]
        )
