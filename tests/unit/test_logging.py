"""日志的契约：一行一条 JSON、能落到文件、**带得上会话/轮次字段**。

分工见 `agent/logging.py`：正常路径写事件，异常与降级路径写日志。
这里不测"打了哪些日志"，只测"格式与字段能不能用来排查"。
"""

from __future__ import annotations

import io
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest

from agent.core.bus import EventBus
from agent.core.reliability import EventEmitter
from agent.graph.bridge import _resolve_approvals
from agent.logging import configure_logging, log_extra


def test_log_line_is_single_json_object() -> None:
    buffer = io.StringIO()
    configure_logging("info", stream=buffer)

    logging.getLogger("agent.test").info("工具执行完成", extra={"tool": "fs_read", "ms": 12})

    lines = buffer.getvalue().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["msg"] == "工具执行完成"
    assert record["level"] == "info"
    assert record["tool"] == "fs_read"
    assert record["ms"] == 12
    assert record["ts"].endswith("+00:00")


def test_noisy_third_party_loggers_are_silenced() -> None:
    """httpx 在 info 级会把每个请求都打出来，CLI 输出会被冲散。"""
    buffer = io.StringIO()
    configure_logging("info", stream=buffer)

    logging.getLogger("httpx2").info("HTTP Request: POST ...")
    logging.getLogger("agent.tools").info("工具调用完成")

    lines = buffer.getvalue().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["logger"] == "agent.tools"


def test_log_extra_drops_empty_values() -> None:
    """日志里不该出现一堆 null：空字段直接不写。"""
    assert log_extra(session_id="s1", turn_id=None, call_ids=[], note="") == {"session_id": "s1"}


def test_logs_can_land_on_disk_and_rotate(tmp_path: Path) -> None:
    """长驻进程的日志不能只留在终端里：给 path 就同时写文件，并按大小轮转。"""
    buffer = io.StringIO()
    log_file = tmp_path / "logs" / "agent.log"

    configure_logging("info", stream=buffer, path=log_file, max_bytes=1000, backup_count=2)
    logging.getLogger("agent.test").warning("出事了", extra=log_extra(session_id="s1"))

    handler = logging.getLogger().handlers[-1]
    assert isinstance(handler, RotatingFileHandler)
    assert (handler.maxBytes, handler.backupCount) == (1000, 2)
    record = json.loads(log_file.read_text(encoding="utf-8").strip())
    assert record["session_id"] == "s1" and record["level"] == "warning"
    assert "出事了" in buffer.getvalue(), "终端里也要有一份，别把眼前的问题藏进文件"


class _Suspension:
    """顶替 LangGraph 的 `Interrupt`：`_resolve_approvals` 只读它的 `value`。"""

    def __init__(self, requests: list[dict[str, object]]) -> None:
        self.value = {"requests": requests}


async def test_approval_warning_carries_session_and_turn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """按会话排查时不该去 grep 消息字符串——字段必须真的在。"""
    emitter = EventEmitter("sess_log", EventBus())
    suspended = [
        _Suspension([{"call_id": "c1", "name": "shell_exec", "args": {}, "reason": "写操作"}])
    ]

    with caplog.at_level(logging.WARNING):
        denied, note = await _resolve_approvals(
            suspended,
            emitter=emitter,
            turn_id="turn_1",
            approver=None,
            timeout_s=1,
        )

    assert denied == {"c1": False} and note == "当时没有审批通道"
    record = next(r for r in caplog.records if r.name == "agent.graph.bridge")
    assert record.session_id == "sess_log"
    assert record.turn_id == "turn_1"
