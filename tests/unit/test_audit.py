"""会话审计（L2）：从真实会话库里找异常模式。

设计见 `docs/design.md` 第 10.2 节。这里钉两件事：异常找得到、干净的库不误报；
外加一条纪律——**扫描是只读的**。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.audit import scan
from agent.store import repo
from agent.store.db import Database


async def _session(path: Path) -> Database:
    db = Database(path)
    db.connect()
    await repo.create_session(db, session_id="s1", profile="code", workspace=str(path.parent))
    return db


async def _call(db: Database, turn_id: str, index: int, *, name: str, status: str) -> None:
    call_id = f"{turn_id}-c{index}"
    await repo.start_tool_call(
        db,
        call_id=call_id,
        session_id="s1",
        turn_id=turn_id,
        name=name,
        args={},
        tier="read",
    )
    await repo.finish_tool_call(db, call_id, status=status, decision="auto")


async def test_scan_reports_a_clean_database_as_clean(tmp_path: Path) -> None:
    db = await _session(tmp_path / "agent.db")
    await repo.start_turn(db, turn_id="t_ok", session_id="s1")
    await _call(db, "t_ok", 0, name="fs_read", status="ok")
    await repo.finish_turn(db, "t_ok", status="done", input_tokens=100, output_tokens=20)
    await db.close()

    report = scan(tmp_path / "agent.db")

    assert report.sessions == 1
    assert report.turns == 1
    assert report.input_tokens == 100
    assert report.output_tokens == 20
    assert report.findings == [], "正常的一轮不该被当成异常"


async def test_scan_finds_failed_turns_busy_turns_and_stuck_tools(tmp_path: Path) -> None:
    db = await _session(tmp_path / "agent.db")

    # 失败的一轮：同一个工具连续失败三次
    await repo.start_turn(db, turn_id="t_bad", session_id="s1")
    for index in range(3):
        await _call(db, "t_bad", index, name="fs_read", status="error")
    await repo.finish_turn(db, "t_bad", status="failed")

    # 很忙的一轮：12 次工具调用（超过 BUSY_ROUNDS = 8）
    await repo.start_turn(db, turn_id="t_busy", session_id="s1")
    for index in range(12):
        await _call(db, "t_busy", index, name="shell_exec", status="ok")
    await repo.finish_turn(db, "t_busy", status="done")

    # 被拦下的一轮
    await repo.start_turn(db, turn_id="t_denied", session_id="s1")
    await repo.start_tool_call(
        db,
        call_id="t_denied-c0",
        session_id="s1",
        turn_id="t_denied",
        name="shell_exec",
        args={},
        tier="write",
    )
    await repo.finish_tool_call(db, "t_denied-c0", status="denied", decision="deny")
    await repo.finish_turn(db, "t_denied", status="done")
    await db.close()

    report = scan(tmp_path / "agent.db")
    kinds = {finding.kind for finding in report.findings}

    assert "abnormal-turn" in kinds, "failed 的轮次要被点出来"
    assert "busy-turn" in kinds, "12 次工具调用明显不正常"
    assert "stuck-tool" in kinds, "同一轮里同一个工具失败 3 次"
    assert "denied" in kinds, "被拦下是'拦住过危险动作'的证据"


def test_scan_missing_database_is_a_friendly_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        scan(tmp_path / "nope.db")


async def test_scan_is_read_only(tmp_path: Path) -> None:
    """诊断不该改变被诊断的对象（D65）：扫完库一个字节都不该变。"""
    path = tmp_path / "agent.db"
    db = await _session(path)
    await repo.start_turn(db, turn_id="t_ok", session_id="s1")
    await repo.finish_turn(db, "t_ok", status="done", input_tokens=7, output_tokens=3)
    await db.close()

    before = path.stat().st_mtime_ns
    size_before = path.stat().st_size
    scan(path)

    assert path.stat().st_size == size_before
    assert path.stat().st_mtime_ns == before
