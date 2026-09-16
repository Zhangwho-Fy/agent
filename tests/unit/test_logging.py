"""日志格式测试：结构化日志的核心契约是"一行一条 JSON"。"""

from __future__ import annotations

import io
import json
import logging

from agent.logging import configure_logging


def test_log_line_is_single_json_object() -> None:
    buffer = io.StringIO()
    configure_logging("info", stream=buffer)

    logging.getLogger("agent.test").info("工具执行完成", extra={"tool": "fs.read", "ms": 12})

    lines = buffer.getvalue().strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["msg"] == "工具执行完成"
    assert record["level"] == "info"
    assert record["tool"] == "fs.read"
    assert record["ms"] == 12
    assert record["ts"].endswith("+00:00")
