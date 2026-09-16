"""事件总线测试：重点是"慢消费者不会拖垮生产者"。"""

from __future__ import annotations

import asyncio

from agent.core.bus import EventBus
from agent.core.events import Event, EventType


def make_event(seq: int, session_id: str = "sess_1") -> Event:
    return Event.create(
        session_id=session_id, seq=seq, type=EventType.TEXT_DELTA, data={"text": str(seq)}
    )


async def test_subscriber_receives_published_events() -> None:
    bus = EventBus()
    subscription = bus.subscribe("sess_1")

    bus.publish(make_event(1))
    bus.publish(make_event(2))

    first = await subscription.get()
    second = await subscription.get()
    assert first is not None and first.seq == 1
    assert second is not None and second.seq == 2


async def test_events_are_fanned_out_to_every_subscriber() -> None:
    bus = EventBus()
    first = bus.subscribe("sess_1")
    second = bus.subscribe("sess_1")

    bus.publish(make_event(1))

    assert (await first.get()).seq == 1  # type: ignore[union-attr]
    assert (await second.get()).seq == 1  # type: ignore[union-attr]


async def test_other_sessions_do_not_receive() -> None:
    bus = EventBus()
    subscription = bus.subscribe("sess_1")

    dropped = bus.publish(make_event(1, session_id="sess_2"))

    assert dropped == 0
    assert subscription.dropped == 0


async def test_slow_subscriber_drops_oldest_and_counts() -> None:
    bus = EventBus(queue_size=3)
    subscription = bus.subscribe("sess_1")

    for seq in range(1, 6):  # 推 5 条，队列只能装 3 条
        bus.publish(make_event(seq))

    assert subscription.dropped == 2
    assert (await subscription.get()).seq == 3  # type: ignore[union-attr]


async def test_close_ends_iteration_and_unregisters() -> None:
    bus = EventBus()
    subscription = bus.subscribe("sess_1")
    bus.publish(make_event(1))

    subscription.close()

    remaining = [event async for event in subscription]
    assert remaining == []
    assert subscription.closed is True
    assert bus.subscriber_count("sess_1") == 0
    assert await subscription.get() is None


async def test_close_wakes_up_a_blocked_consumer() -> None:
    """消费者正卡在 get() 上时关闭订阅，必须被唤醒而不是永久阻塞。"""
    bus = EventBus()
    subscription = bus.subscribe("sess_1")

    waiter = asyncio.create_task(subscription.get())
    await asyncio.sleep(0)  # 让 get() 先真正进入等待

    subscription.close()

    assert await asyncio.wait_for(waiter, timeout=1.0) is None


async def test_subscribe_increments_count() -> None:
    bus = EventBus()
    bus.subscribe("sess_1")
    bus.subscribe("sess_1")

    assert bus.subscriber_count("sess_1") == 2
    assert bus.subscriber_count("sess_missing") == 0
