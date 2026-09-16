"""事件模型：整个系统对外可见的状态变化。

事件是**只追加的事实源**（append-only log）：先落库、再推送给客户端。
顺序反了就会出现"客户端看到过、事后查不到"的诡异问题。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .ids import new_id


class EventType(StrEnum):
    TURN_STARTED = "turn.started"
    TEXT_DELTA = "text.delta"
    TEXT_DONE = "text.done"
    TOOL_CALL = "tool.call"
    APPROVAL_REQUIRED = "approval.required"
    TOOL_RESULT = "tool.result"
    TURN_DONE = "turn.done"
    ERROR = "error"


class Event(BaseModel):
    """一条事件。frozen=True：事件是既成事实，不允许事后修改。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    seq: int = Field(description="会话内单调递增，客户端据此判断丢失与补齐")
    session_id: str
    turn_id: str | None = None
    type: EventType
    data: dict[str, Any] = Field(default_factory=dict)
    ts: datetime

    @classmethod
    def create(
        cls,
        *,
        session_id: str,
        seq: int,
        type: EventType,
        data: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> Event:
        return cls(
            id=new_id("evt"),
            seq=seq,
            session_id=session_id,
            turn_id=turn_id,
            type=type,
            data=data or {},
            ts=datetime.now(UTC),
        )

    def to_payload(self) -> dict[str, Any]:
        """转成 JSON 可序列化的字典（时间转 ISO 字符串）。"""
        return self.model_dump(mode="json")

    def to_sse(self) -> str:
        """编码成一个 SSE 帧。

        `id:` 用 seq，客户端重连时把它放进 Last-Event-ID 就能续传。
        """
        payload = json.dumps(self.to_payload(), ensure_ascii=False)
        return f"id: {self.seq}\nevent: {self.type.value}\ndata: {payload}\n\n"
