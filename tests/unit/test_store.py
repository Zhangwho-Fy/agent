"""持久化层测试：临时库、跑完即删、不联网、不需要 key。

这里盯的是阶段 2 的地基——事件能落库且顺序不重、seq 从库里现算、
崩溃后能看出哪些 turn 没跑完、幂等键能挡重复执行。
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent.core.events import Event, EventType
from agent.store import repo
from agent.store.db import Database

TABLES = frozenset({"sessions", "messages", "events", "turns", "tool_calls", "idempotency"})


@pytest.fixture()
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(tmp_path / "agent.db")
    database.connect()
    yield database
    await database.close()


async def _new_session(database: Database, session_id: str = "sess_1") -> str:
    await repo.create_session(database, session_id=session_id, profile="code", workspace="/tmp/ws")
    return session_id


async def test_session_summaries_carry_the_last_user_message(db: Database) -> None:
    """列表要认得出人：标题只记第一句（常常是"你好"），最后一句才是这次在聊什么。"""
    session_id = await _new_session(db, "sess_a")
    await repo.create_session(db, session_id="sess_b", profile="code", title="空会话")
    await repo.append_message(db, session_id=session_id, seq=1, role="user", content="你好")
    await repo.append_message(db, session_id=session_id, seq=2, role="assistant", content="在的")
    await repo.append_message(
        db, session_id=session_id, seq=3, role="user", content="把 README 前 5 行读出来"
    )

    by_id = {row["id"]: row for row in await repo.session_summaries(db, limit=10)}

    assert by_id[session_id]["last_user"] == "把 README 前 5 行读出来", "取最后一条**用户**消息"
    assert by_id[session_id]["turns"] == 0
    assert by_id["sess_b"]["last_user"] is None, "一句都没说过的会话不能炸"


async def test_delete_session_clears_every_trace(db: Database) -> None:
    """删会话要删全：只删 sessions 一行的话，事件/消息/turn 会变成查不到主人的孤儿。"""
    session_id = await _new_session(db, "sess_gone")
    await repo.append_event(db, _event(1, session_id=session_id))
    await repo.start_turn(db, turn_id="turn_1", session_id=session_id)
    await repo.append_message(db, session_id=session_id, seq=1, role="user", content="删我")
    # 工具调用引用 turn：删表顺序错了会撞外键（tool_calls 得在 turns 之前删）
    await repo.start_tool_call(
        db,
        call_id="call_1",
        session_id=session_id,
        turn_id="turn_1",
        name="fs_read",
        args={"path": "a.py"},
        tier="read",
    )
    await repo.remember_idempotency(db, key="k1", session_id=session_id, turn_id="turn_1")
    other = await _new_session(db, "sess_keep")
    await repo.append_event(db, _event(1, session_id=other))

    await repo.delete_session(db, session_id)

    assert await repo.get_session(db, session_id) is None
    assert await repo.list_events(db, session_id) == []
    assert await repo.list_messages(db, session_id) == []
    assert await repo.list_turns(db, session_id) == []
    assert await repo.lookup_idempotency(db, "k1") is None
    assert await repo.get_session(db, other) is not None, "别误删别的会话"
    assert len(await repo.list_events(db, other)) == 1


def _event(seq: int, *, session_id: str = "sess_1", text: str = "hi") -> Event:
    return Event.create(
        session_id=session_id,
        seq=seq,
        type=EventType.TEXT_DELTA,
        data={"text": text},
        turn_id="turn_1",
    )


# ---------------------------------------------------------------- schema


def test_connect_creates_all_tables(tmp_path: Path) -> None:
    database = Database(tmp_path / "agent.db")
    database.connect()

    names = {
        row[0]
        for row in database._connection()
        .execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        .fetchall()
    }
    assert TABLES <= names


def test_connect_is_idempotent(tmp_path: Path) -> None:
    """重复打开同一个文件不应该报错，也不应该丢数据。"""
    path = tmp_path / "agent.db"
    first = Database(path)
    first.connect()
    first._connection().execute(
        "INSERT INTO sessions (id, title, profile, created_at, updated_at)"
        " VALUES ('sess_keep', '', 'chat', '2026-01-01T00:00:00+00:00',"
        " '2026-01-01T00:00:00+00:00')"
    )
    first._connection().commit()
    first._connection().close()

    second = Database(path)
    second.connect()
    assert second._connection().execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


async def test_foreign_keys_are_enforced(db: Database) -> None:
    """`PRAGMA foreign_keys=ON` 没打开的话，这条会静默写进去。"""
    with pytest.raises(sqlite3.IntegrityError):
        await repo.append_message(db, session_id="sess_missing", seq=1, role="user", content="hi")


# ---------------------------------------------------------------- events


async def test_events_are_append_only_and_ordered(db: Database) -> None:
    session_id = await _new_session(db)
    for seq in (1, 2, 3):
        await repo.append_event(db, _event(seq, session_id=session_id, text=f"t{seq}"))

    rows = await repo.list_events(db, session_id)
    assert [row["seq"] for row in rows] == [1, 2, 3]
    assert [row["data"] for row in rows] == ['{"text": "t1"}', '{"text": "t2"}', '{"text": "t3"}']


async def test_list_events_after_seq_is_the_resume_semantic(db: Database) -> None:
    """断线续传：客户端带着 Last-Event-ID 回来，只补缺口。"""
    session_id = await _new_session(db)
    for seq in (1, 2, 3):
        await repo.append_event(db, _event(seq, session_id=session_id))

    rows = await repo.list_events(db, session_id, after_seq=1)
    assert [row["seq"] for row in rows] == [2, 3]
    assert await repo.last_event_seq(db, session_id) == 3


async def test_duplicate_seq_is_rejected(db: Database) -> None:
    """顺序错乱必须炸出来，不能静默接受。"""
    session_id = await _new_session(db)
    await repo.append_event(db, _event(1, session_id=session_id))

    with pytest.raises(sqlite3.IntegrityError):
        await repo.append_event(db, _event(1, session_id=session_id))


async def test_next_event_seq_is_computed_from_storage(db: Database) -> None:
    """从库里现算，所以"进程重启"（换个 Database 连同一个文件）也不会重号。"""
    session_id = await _new_session(db)
    assert await repo.next_event_seq(db, session_id) == 1

    await repo.append_event(db, _event(1, session_id=session_id))
    assert await repo.next_event_seq(db, session_id) == 2


async def test_row_to_event_roundtrip(db: Database) -> None:
    session_id = await _new_session(db)
    original = _event(1, session_id=session_id, text="往返")
    await repo.append_event(db, original)

    row = (await repo.list_events(db, session_id))[0]
    restored = repo.row_to_event(row)

    assert restored.id == original.id
    assert restored.type is EventType.TEXT_DELTA
    assert restored.data == {"text": "往返"}
    assert restored.turn_id == "turn_1"


# ---------------------------------------------------------------- turns


async def test_turn_lifecycle_records_tokens(db: Database) -> None:
    session_id = await _new_session(db)
    await repo.start_turn(db, turn_id="turn_1", session_id=session_id)

    running = await repo.get_turn(db, "turn_1")
    assert running is not None
    assert running["status"] == "running"
    assert running["ended_at"] is None

    await repo.finish_turn(db, "turn_1", status="done", input_tokens=120, output_tokens=45)
    finished = await repo.get_turn(db, "turn_1")
    assert finished is not None
    assert finished["status"] == "done"
    assert finished["input_tokens"] == 120
    assert finished["output_tokens"] == 45
    assert finished["ended_at"] is not None


async def test_interrupt_running_turns_only_touches_running(db: Database) -> None:
    """崩溃恢复：只有没跑完的会被标记，已经结束的不动。"""
    session_id = await _new_session(db)
    await repo.start_turn(db, turn_id="turn_done", session_id=session_id)
    await repo.start_turn(db, turn_id="turn_stuck", session_id=session_id)
    await repo.finish_turn(db, "turn_done", status="done")

    marked = await repo.interrupt_running_turns(db)

    assert marked == 1
    done_row = await repo.get_turn(db, "turn_done")
    stuck_row = await repo.get_turn(db, "turn_stuck")
    assert done_row is not None
    assert stuck_row is not None
    assert done_row["status"] == "done"
    assert stuck_row["status"] == "interrupted"
    assert stuck_row["ended_at"] is not None


# ---------------------------------------------------------------- messages


async def test_messages_keep_order_and_link_tool_calls(db: Database) -> None:
    session_id = await _new_session(db)
    await repo.append_message(db, session_id=session_id, seq=1, role="user", content="读文件")
    await repo.append_message(
        db,
        session_id=session_id,
        seq=2,
        role="tool",
        content="文件内容",
        tool_call_id="call_1",
    )

    rows = await repo.list_messages(db, session_id)
    assert [row["role"] for row in rows] == ["user", "tool"]
    assert rows[1]["tool_call_id"] == "call_1"
    assert await repo.next_message_seq(db, session_id) == 3


# ---------------------------------------------------------------- tool_calls


async def test_tool_call_records_full_output(db: Database) -> None:
    session_id = await _new_session(db)
    await repo.start_turn(db, turn_id="turn_1", session_id=session_id)
    await repo.start_tool_call(
        db,
        call_id="call_1",
        session_id=session_id,
        turn_id="turn_1",
        name="shell_exec",
        args={"command": "ls"},
        tier="read",
    )

    await repo.finish_tool_call(
        db,
        "call_1",
        status="ok",
        decision="auto",
        result="x" * 10_000,
        exit_code=0,
    )

    row = (await repo.list_tool_calls(db, "turn_1"))[0]
    assert row["status"] == "ok"
    assert row["decision"] == "auto"
    assert row["exit_code"] == 0
    assert len(row["result"]) == 10_000


# ---------------------------------------------------------------- idempotency


async def test_idempotency_key_maps_to_existing_turn(db: Database) -> None:
    session_id = await _new_session(db)
    assert await repo.lookup_idempotency(db, "k1") is None

    await repo.remember_idempotency(db, key="k1", session_id=session_id, turn_id="turn_1")

    hit = await repo.lookup_idempotency(db, "k1")
    assert hit is not None
    assert hit["turn_id"] == "turn_1"


async def test_duplicate_idempotency_key_is_rejected(db: Database) -> None:
    session_id = await _new_session(db)
    await repo.remember_idempotency(db, key="k1", session_id=session_id, turn_id="turn_1")

    with pytest.raises(sqlite3.IntegrityError):
        await repo.remember_idempotency(db, key="k1", session_id=session_id, turn_id="turn_2")
