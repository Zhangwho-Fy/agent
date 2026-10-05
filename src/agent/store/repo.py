"""各表的读写函数。只放 SQL 和字段映射，不放业务判断。

业务规则（什么时候该写事件、失败怎么处理）留在调用方，
这样表结构变更时，改动范围有明确边界。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any

from ..core.events import Event, EventType
from ..core.ids import new_id
from .db import Database, now_iso

# ---------------------------------------------------------------- sessions


async def create_session(
    db: Database,
    *,
    session_id: str,
    profile: str,
    workspace: str | None = None,
    title: str = "",
) -> None:
    now = now_iso()
    await db.execute(
        """
        INSERT INTO sessions (id, title, profile, workspace, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, title, profile, workspace, now, now),
    )


async def get_session(db: Database, session_id: str) -> sqlite3.Row | None:
    return await db.one("SELECT * FROM sessions WHERE id = ?", (session_id,))


async def list_sessions(db: Database, *, limit: int = 50) -> list[sqlite3.Row]:
    return await db.all("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,))


async def session_summaries(db: Database, *, limit: int = 20) -> list[dict[str, Any]]:
    """会话列表 + 每个会话**最后一条用户消息**（认人用）与 turn 数。

    标题只记第一句（经常是"你好"），光看标题认不出哪次是哪次；最后一句通常最能
    说明这次在聊什么。**一条 SQL 拿完**，不做 N+1 查询——列表是给人扫的，快才有人用。
    """
    rows = await db.all(
        """
        SELECT s.id, s.title, s.profile, s.workspace, s.created_at, s.updated_at,
               (SELECT m.content FROM messages m
                 WHERE m.session_id = s.id AND m.role = 'user'
                 ORDER BY m.seq DESC LIMIT 1) AS last_user,
               (SELECT COUNT(*) FROM turns t WHERE t.session_id = s.id) AS turns
        FROM sessions s
        ORDER BY s.updated_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [dict(row) for row in rows]


#: 删会话时要一起清掉的业务表，**顺序有讲究**：SQLite 开了外键约束，
#: 引用别人的表要先删（tool_calls 引用 turns，所以它在 turns 前面）。
_SESSION_TABLES = ("tool_calls", "events", "messages", "idempotency", "turns")


async def delete_session(db: Database, session_id: str) -> None:
    """删掉一个会话，连同它的全部痕迹。

    **要删全**：只删 `sessions` 一行，事件/消息/turn 就成了查不到主人的孤儿数据，
    `agent sessions` 上看不见、库却一直涨。LangGraph 的 `checkpoints` / `writes`
    用 `thread_id = session_id`，也一并清掉——不然删掉再建同名会话会继承旧的图状态。
    """
    for table in _SESSION_TABLES:
        await db.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
    for table in ("checkpoints", "writes"):
        try:
            await db.execute(f"DELETE FROM {table} WHERE thread_id = ?", (session_id,))
        except sqlite3.OperationalError:
            pass  # 这份库还没跑过图，langgraph 的表还没建出来
    await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


async def touch_session(db: Database, session_id: str) -> None:
    """会话有活动时更新 `updated_at`，用于"最近会话"排序。"""
    await db.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now_iso(), session_id))


async def title_session_if_empty(db: Database, session_id: str, title: str) -> None:
    """没标题的会话，拿第一句用户输入当标题——`agent sessions` / resume 列表靠它认人。

    只在标题为空（或还是建会话时的占位符 `chat`）时写：用户后来改过的不覆盖。
    """
    clean = " ".join(title.split())[:60]
    if not clean:
        return
    await db.execute(
        "UPDATE sessions SET title = ? WHERE id = ? AND TRIM(title) IN ('', 'chat')",
        (clean, session_id),
    )


# ---------------------------------------------------------------- events


async def next_event_seq(db: Database, session_id: str) -> int:
    """下一个事件序号。

    从库里现算而不是靠内存计数器：进程重启后不会重号，
    也不依赖调用方记得把起始值传对。
    """
    current = await db.scalar(
        "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?", (session_id,)
    )
    return int(current or 0) + 1


async def last_event_seq(db: Database, session_id: str) -> int:
    current = await db.scalar(
        "SELECT COALESCE(MAX(seq), 0) FROM events WHERE session_id = ?", (session_id,)
    )
    return int(current or 0)


async def append_event(db: Database, event: Event) -> None:
    """写入一条事件。

    `seq` 上有唯一约束，重复写入会抛 `IntegrityError`——这正是想要的行为：
    宁可报错，也不要静默地把事件顺序搞乱。
    """
    await db.execute(
        """
        INSERT INTO events (id, session_id, turn_id, seq, type, data, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.id,
            event.session_id,
            event.turn_id,
            event.seq,
            event.type.value,
            json.dumps(event.data, ensure_ascii=False),
            event.ts.isoformat(timespec="seconds"),
        ),
    )


async def list_events(
    db: Database,
    session_id: str,
    *,
    after_seq: int = 0,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """按 seq 升序取事件。`after_seq` 就是断线续传的语义。"""
    sql = "SELECT * FROM events WHERE session_id = ? AND seq > ? ORDER BY seq"
    params: list[Any] = [session_id, after_seq]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return await db.all(sql, params)


def row_to_event(row: sqlite3.Row) -> Event:
    """把库里的行还原成 `Event`，重放时用。"""
    return Event(
        id=row["id"],
        session_id=row["session_id"],
        turn_id=row["turn_id"],
        seq=row["seq"],
        type=EventType(row["type"]),
        data=json.loads(row["data"]),
        ts=datetime.fromisoformat(row["created_at"]),
    )


# ---------------------------------------------------------------- messages


async def next_message_seq(db: Database, session_id: str) -> int:
    current = await db.scalar(
        "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?", (session_id,)
    )
    return int(current or 0) + 1


async def append_message(
    db: Database,
    *,
    session_id: str,
    seq: int,
    role: str,
    content: str,
    tool_call_id: str | None = None,
) -> str:
    """写一条消息，返回新生成的 id。"""
    message_id = new_id("msg")
    await db.execute(
        """
        INSERT INTO messages (id, session_id, seq, role, content, tool_call_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (message_id, session_id, seq, role, content, tool_call_id, now_iso()),
    )
    return message_id


async def list_messages(
    db: Database,
    session_id: str,
    *,
    after_seq: int = 0,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    sql = "SELECT * FROM messages WHERE session_id = ? AND seq > ? ORDER BY seq"
    params: list[Any] = [session_id, after_seq]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return await db.all(sql, params)


# ---------------------------------------------------------------- turns


async def start_turn(db: Database, *, turn_id: str, session_id: str) -> None:
    await db.execute(
        """
        INSERT INTO turns (id, session_id, status, started_at)
        VALUES (?, ?, 'running', ?)
        """,
        (turn_id, session_id, now_iso()),
    )


async def finish_turn(
    db: Database,
    turn_id: str,
    *,
    status: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    await db.execute(
        """
        UPDATE turns
           SET status = ?, input_tokens = ?, output_tokens = ?, ended_at = ?
         WHERE id = ?
        """,
        (status, input_tokens, output_tokens, now_iso(), turn_id),
    )


async def get_turn(db: Database, turn_id: str) -> sqlite3.Row | None:
    return await db.one("SELECT * FROM turns WHERE id = ?", (turn_id,))


async def list_turns(db: Database, session_id: str) -> list[sqlite3.Row]:
    return await db.all(
        "SELECT * FROM turns WHERE session_id = ? ORDER BY started_at", (session_id,)
    )


async def interrupt_running_turns(db: Database) -> int:
    """崩溃恢复：把还挂在 `running` 的 turn 标成 `interrupted`。

    进程都重启了，这些 turn 不可能还在跑。不清理的话，界面会永远显示"进行中"，
    而且"同会话串行"的语义也会被占用。返回被标记的行数，启动日志里能看到恢复了几条。
    """
    return await db.execute(
        "UPDATE turns SET status = 'interrupted', ended_at = ? WHERE status = 'running'",
        (now_iso(),),
    )


# ---------------------------------------------------------------- tool_calls


async def start_tool_call(
    db: Database,
    *,
    call_id: str,
    session_id: str,
    turn_id: str,
    name: str,
    args: dict[str, Any],
    tier: str,
    status: str = "running",
) -> None:
    await db.execute(
        """
        INSERT INTO tool_calls
            (id, session_id, turn_id, name, args, tier, status, started_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            call_id,
            session_id,
            turn_id,
            name,
            json.dumps(args, ensure_ascii=False),
            tier,
            status,
            now_iso(),
        ),
    )


async def finish_tool_call(
    db: Database,
    call_id: str,
    *,
    status: str,
    decision: str | None = None,
    result: str | None = None,
    exit_code: int | None = None,
) -> None:
    """收尾一次工具调用。

    `result` 存全量输出——模型只看得到截断版，人要排查时得能看到全文。
    """
    await db.execute(
        """
        UPDATE tool_calls
           SET status = ?, decision = COALESCE(?, decision),
               result = ?, exit_code = ?, ended_at = ?
         WHERE id = ?
        """,
        (status, decision, result, exit_code, now_iso(), call_id),
    )


async def list_tool_calls(db: Database, turn_id: str) -> list[sqlite3.Row]:
    return await db.all(
        "SELECT * FROM tool_calls WHERE turn_id = ? ORDER BY started_at", (turn_id,)
    )


# ---------------------------------------------------------------- idempotency


async def lookup_idempotency(db: Database, key: str) -> sqlite3.Row | None:
    """命中说明这条消息已经处理过，直接返回原来的 turn_id，不重复执行。"""
    return await db.one("SELECT * FROM idempotency WHERE key = ?", (key,))


async def remember_idempotency(
    db: Database,
    *,
    key: str,
    session_id: str,
    turn_id: str,
) -> None:
    await db.execute(
        """
        INSERT INTO idempotency (key, session_id, turn_id, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (key, session_id, turn_id, now_iso()),
    )
