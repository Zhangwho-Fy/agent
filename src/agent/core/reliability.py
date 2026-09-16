"""可靠性语义：事件编号、对外事件流、幂等键。

这些是框架**不提供**的部分，也是本项目区别于"照教程搭的 demo"的地方：
框架管"怎么跑"，这里管"跑过之后有什么凭据、重发会怎样、断线后怎么补齐"。

阶段 1 先在内存里实现，阶段 2 把落库接进来——接口不变，换的是实现。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from .bus import EventBus
from .events import Event, EventType
from .ids import new_id

logger = logging.getLogger(__name__)


class EventEmitter:
    """给事件分配会话内单调递增的 seq，推送到总线。

    阶段 2 会在这里插入"先落库、再推送"，调用方无需改动。
    """

    def __init__(self, session_id: str, bus: EventBus, *, start_seq: int = 0) -> None:
        self.session_id = session_id
        self._bus = bus
        self._seq = start_seq
        #: 进程内消费者（例如 CLI 渲染、结构化日志），SSE 消费者走 bus
        self.on_event: Callable[[Event], None] | None = None

    @property
    def last_seq(self) -> int:
        return self._seq

    def emit(
        self,
        type: EventType,
        data: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
    ) -> Event:
        self._seq += 1
        event = Event.create(
            session_id=self.session_id,
            seq=self._seq,
            type=type,
            data=data,
            turn_id=turn_id,
        )
        self._bus.publish(event)
        if self.on_event is not None:
            self.on_event(event)
        return event


class IdempotencyStore:
    """幂等键 → 已有 turn 的映射。

    为什么需要：Agent 的一轮可能跑几十秒，客户端超时重发是常态。
    没有幂等键，同一条消息会被执行两次——工具可能真的改了文件。
    """

    def __init__(self) -> None:
        self._turns: dict[str, str] = {}

    def lookup(self, key: str) -> str | None:
        """返回已存在的 turn_id；没有则返回 None。"""
        return self._turns.get(key)

    def remember(self, key: str, turn_id: str) -> None:
        self._turns[key] = turn_id

    def new_turn_id(self) -> str:
        return new_id("turn")
