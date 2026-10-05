"""CLI 辅助函数的测试：会话解析、doctor 探针、replay 跟随。

CLI 的编排层（发消息、收事件）已经在 test_server / test_store 里覆盖；这里只钉
那些**可判定**的小零件——它们才是容易写错、又不容易发现的地方。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.client.main import (
    _db_probe,
    _follow_events,
    _looks_like_placeholder,
    _resolve_session,
)
from agent.core.events import Event, EventType
from agent.store import repo
from agent.store.db import Database


def _event(seq: int, text: str = "hi") -> Event:
    return Event.create(
        session_id="sess_1", seq=seq, type=EventType.TEXT_DELTA, data={"text": text}
    )


def test_placeholder_key_is_flagged() -> None:
    """`.env.example` 里那种占位符是"已设置但根本调不通"，doctor 要拦下来。"""
    assert _looks_like_placeholder("sk-your-key-here")
    assert _looks_like_placeholder("sk-xxxxxxxxxxxxxxxxxxxx")
    assert _looks_like_placeholder("短key")
    assert not _looks_like_placeholder("sk-test-fixture-not-a-real-key")
    assert not _looks_like_placeholder(""), "空值是另一条路（未设置），别报两次"


async def test_resolve_session_understands_last(tmp_path: Path) -> None:
    """`-s last` / `--last` 解析成最近一个会话；空库时返回 None（新建），不报错。"""
    db = Database(tmp_path / "agent.db")
    db.connect()
    try:
        assert await _resolve_session(db, None, False) is None
        assert await _resolve_session(db, "sess_x", False) == "sess_x"
        assert await _resolve_session(db, None, True) is None, "空库不该报错"

        await repo.create_session(
            db, session_id="sess_old", profile="code", workspace=str(tmp_path)
        )
        await repo.create_session(
            db, session_id="sess_new", profile="code", workspace=str(tmp_path)
        )
        # 时间戳是秒级，同一秒建两个会话会并列——显式拉开，测试才确定
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            ("2030-01-01T00:00:00+00:00", "sess_new"),
        )

        assert await _resolve_session(db, "last", False) == "sess_new"
        assert await _resolve_session(db, None, True) == "sess_new"
    finally:
        await db.close()


def test_db_probe_reports_counts_and_failures(tmp_path: Path) -> None:
    """doctor 的库探针：没见过 / 空库 / 坏文件，三种都要说人话。"""
    path = tmp_path / "agent.db"
    assert "还没有" in _db_probe(path)

    db = Database(path)
    db.connect()  # 建表（同步方法）
    assert "0 会话 / 0 事件" in _db_probe(path)

    broken = tmp_path / "broken.db"
    broken.write_text("这不是数据库", encoding="utf-8")
    assert "读失败" in _db_probe(broken)


async def test_follow_prints_new_events(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`replay --follow`：库里有新事件就打印（纯读库，不依赖服务端）。"""
    db = Database(tmp_path / "agent.db")
    db.connect()
    await repo.create_session(db, session_id="sess_1", profile="code", workspace=str(tmp_path))
    await repo.append_event(db, _event(1))

    task = asyncio.create_task(
        _follow_events(db, "sess_1", after_seq=0, raw=True, poll_seconds=0.01)
    )
    await asyncio.sleep(0.05)  # 先空转一会儿：没有新事件时不该刷屏
    await repo.append_event(db, _event(2, "第二条"))
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    out = capsys.readouterr().out
    assert "第二条" in out, "跟到的新事件要打出来"
    assert "hi" in out, "起点的历史也一并渲染（after_seq=0）"
    await db.close()
