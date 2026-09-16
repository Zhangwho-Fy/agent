"""日志格式测试：结构化日志的核心契约是"一行一条 JSON"。"""

from __future__ import annotations

import io
import json
import logging

from agent.logging import configure_logging


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
