"""可靠性语义：事件编号与对外事件流。

这些是框架**不提供**的部分，也是本项目区别于"照教程搭的 demo"的地方：
框架管"怎么跑"，这里管"跑过之后有什么凭据、重发会怎样、断线后怎么补齐"。

幂等键的**事实源在 SQLite**（`store/schema.sql` 的 `idempotency` 表 +
`store/repo.py` 的读写），不在这里——进程内的 map 顶不住重启，
阶段 2 接落库时就把它换掉了。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .bus import EventBus
from .events import Event, EventType

logger = logging.getLogger(__name__)


class EventEmitter:
    """给事件分配会话内单调递增的 seq，推送到总线。

    **顺序不能反**：先推后存的话，会出现"客户端看到过、事后去库里查不到"。
    落库通过 `sink` 注入（阶段 2 起由 `store.repo.append_event` 提供），
    这样 core 层不用知道数据库的存在，测试里也能换成内存实现。

    事件是 IO（要落库），所以 `emit` 是 async 的。
    """

    def __init__(
        self,
        session_id: str,
        bus: EventBus,
        *,
        start_seq: int = 0,
        sink: Callable[[Event], Awaitable[None]] | None = None,
    ) -> None:
        self.session_id = session_id
        self._bus = bus
        self._seq = start_seq
        self._sink = sink
        #: 进程内消费者（例如 CLI 渲染、结构化日志），SSE 消费者走 bus
        self.on_event: Callable[[Event], None] | None = None

    @property
    def last_seq(self) -> int:
        return self._seq

    async def emit(
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
        if self._sink is not None:
            # 先落库：它是事实源，落不下去就别对外宣称发生过
            await self._sink(event)
        self._bus.publish(event)
        if self.on_event is not None:
            self.on_event(event)
        return event
