"""可靠性语义测试：事件必须**先落库、再推送**。

顺序反了会出现最难受的一类 bug——客户端看到过某条事件，拿它去查却查不到。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent.core.bus import EventBus
from agent.core.events import Event, EventType
from agent.core.reliability import EventEmitter
from agent.store import repo
from agent.store.db import Database


@pytest.fixture()
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "agent.db")
    database.connect()
    yield database
    await database.close()


async def test_sink_runs_before_subscribers_are_notified() -> None:
    order: list[str] = []

    async def sink(event: Event) -> None:
        order.append("sink")

    emitter = EventEmitter("sess_1", EventBus(), sink=sink)
    emitter.on_event = lambda event: order.append("push")

    await emitter.emit(EventType.TEXT_DELTA, {"text": "x"})

    assert order == ["sink", "push"]


async def test_failed_persist_means_the_event_never_happened() -> None:
    """落库失败就不该往外推：宁可这一轮报错，也不要留下"看到过但查不到"。"""
    pushed: list[Event] = []

    async def broken_sink(event: Event) -> None:
        msg = "库挂了"
        raise RuntimeError(msg)

    emitter = EventEmitter("sess_1", EventBus(), sink=broken_sink)
    emitter.on_event = pushed.append

    with pytest.raises(RuntimeError):
        await emitter.emit(EventType.TEXT_DELTA, {"text": "x"})

    assert pushed == []


async def test_seq_continues_from_stored_events(db: Database) -> None:
    """重启后接着已有事件往下编号，不能重号。"""
    await repo.create_session(db, session_id="sess_1", profile="code")
    first = EventEmitter(
        "sess_1", EventBus(), start_seq=0, sink=lambda event: repo.append_event(db, event)
    )
    await first.emit(EventType.TURN_STARTED, {"prompt": "第一轮"}, turn_id="turn_1")

    # 模拟进程重启：新的 emitter，起始号从库里读
    second = EventEmitter(
        "sess_1",
        EventBus(),
        start_seq=await repo.last_event_seq(db, "sess_1"),
        sink=lambda event: repo.append_event(db, event),
    )
    await second.emit(EventType.TURN_STARTED, {"prompt": "第二轮"}, turn_id="turn_2")

    rows = await repo.list_events(db, "sess_1")
    assert [row["seq"] for row in rows] == [1, 2]
    assert await repo.last_event_seq(db, "sess_1") == 2
