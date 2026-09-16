"""结构化日志：每个事件一行 JSON。

为什么不用默认格式：Agent 靠日志复盘线上行为（哪一步慢、哪次重试、哪个工具超时），
一行一条 JSON 才能被 grep / jq / 日志系统直接消费。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import IO

# LogRecord 自带的标准属性，序列化时要排除，只留我们自己通过 extra= 传进去的字段
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None))
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成一行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "info", *, stream: IO[str] | None = None) -> None:
    """配置根 logger。重复调用是幂等的（先清空旧 handler）。"""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
