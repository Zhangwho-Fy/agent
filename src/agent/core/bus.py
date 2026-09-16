"""会话内事件总线（进程内）。

推送是**尽力而为**的：订阅队列有界，慢了就丢最老的事件。
敢丢的前提是——事件在推送前已经落库，客户端积压或断线后按 seq 从库里补齐，
所以推送只承担"实时性"，不承担"正确性"。
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable

from .events import Event

logger = logging.getLogger(__name__)

#: 队列里的哨兵，用来让 `async for` 正常结束
_CLOSED = object()


class Subscription:
    """一个订阅者的事件队列。"""

    def __init__(
        self,
        session_id: str,
        *,
        queue_size: int = 256,
        on_close: Callable[[str, Subscription], None] | None = None,
    ) -> None:
        self.session_id = session_id
        self.dropped = 0
        self._queue: asyncio.Queue[Event | object] = asyncio.Queue(maxsize=queue_size)
        self._closed = False
        self._on_close = on_close

    @property
    def closed(self) -> bool:
        return self._closed

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Event]:
        while True:
            # 已关闭就不再补发积压：先判状态，再等待，避免"哨兵被前一个消费者吃掉后
            # 后续消费者在空队列上永久阻塞"。
            if self._closed:
                return
            item = await self._queue.get()
            if item is _CLOSED:
                return
            yield item

    async def get(self) -> Event | None:
        """取一条事件；订阅已关闭时返回 None。"""
        if self._closed:
            return None
        item = await self._queue.get()
        if item is _CLOSED:
            return None
        return item

    def _deliver(self, event: Event) -> bool:
        """投递一条事件。返回 True 表示因队列满而丢了最老的一条。"""
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(event)
            self.dropped += 1
            return True
        return False

    def close(self) -> None:
        """关闭订阅：清空积压并放入哨兵，让正在迭代的消费者退出。"""
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._queue.put_nowait(_CLOSED)
        if self._on_close is not None:
            self._on_close(self.session_id, self)


class EventBus:
    """按会话分发事件。"""

    def __init__(self, *, queue_size: int = 256) -> None:
        self._subscribers: dict[str, list[Subscription]] = defaultdict(list)
        self._queue_size = queue_size

    def subscribe(self, session_id: str) -> Subscription:
        subscription = Subscription(session_id, queue_size=self._queue_size, on_close=self._remove)
        self._subscribers[session_id].append(subscription)
        return subscription

    def publish(self, event: Event) -> int:
        """推给该会话的所有订阅者，返回发生丢弃的次数。"""
        dropped = 0
        for subscription in list(self._subscribers.get(event.session_id, ())):
            if subscription._deliver(event):
                dropped += 1
        if dropped:
            logger.warning(
                "事件推送积压，已丢弃最老的事件",
                extra={"session_id": event.session_id, "dropped": dropped, "seq": event.seq},
            )
        return dropped

    def subscriber_count(self, session_id: str) -> int:
        return len(self._subscribers.get(session_id, ()))

    def _remove(self, session_id: str, subscription: Subscription) -> None:
        subscribers = self._subscribers.get(session_id)
        if not subscribers:
            return
        if subscription in subscribers:
            subscribers.remove(subscription)
        if not subscribers:
            del self._subscribers[session_id]
