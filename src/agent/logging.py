"""结构化日志：一行一条 JSON，写给**排查问题的人**看。

### 和事件的分工（别混在一起）

| | 事件（`events` 表） | 日志（这里） |
| --- | --- | --- |
| 是什么 | 业务事实、**对外契约** | 进程诊断 |
| 存哪 | SQLite，按会话 + seq，永久 | stderr（可选落盘 + 轮转） |
| 给谁看 | 客户端、`agent replay`、审计、评测 | 排查问题的人 |
| 写什么 | 每一次工具调用、每一段文本 | **异常与降级路径** |

所以**正常路径不写日志**：正常路径的事实都在事件里，重复写一遍等于同一件事存两份，
还让"日志里出现一条 warning"失去告警价值。反过来，日志里的东西也不能当业务事实用——
它不按会话有序、可以丢、进程结束就没了。

### 三条惯例

1. 一行一条 JSON（`JsonFormatter`），能直接 `jq` / grep。
2. 有会话 / 轮次上下文的调用点，**一律带** `extra=log_extra(session_id=…, turn_id=…)`，
   按会话过滤时才不用去 grep 消息字符串。
3. 要落盘就设 `AGENT_LOG_PATH`（按 2MB × 3 份轮转）；`agent serve` 没配时默认写到
   会话库旁边——长驻进程的日志不该只留在终端里。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO

# LogRecord 自带的标准属性，序列化时要排除，只留我们自己通过 extra= 传进去的字段
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord(name="", level=0, pathname="", lineno=0, msg="", args=(), exc_info=None))
) | {"message", "asctime"}

# 这些库在 info 级会把每个 HTTP 请求都打出来，CLI 输出会被冲散。
# 我们自己的关键信息走 "agent.*" 命名空间，不受影响。
_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "openai", "langsmith")

#: 单个日志文件的大小上限与保留份数：本地单机够用，也不会把磁盘吃光。
LOG_MAX_BYTES = 2_000_000
LOG_BACKUP_COUNT = 3


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


def log_extra(**fields: object) -> dict[str, object]:
    """构造 `extra=`，顺手丢掉空值——日志里不该出现一堆 `null`。"""
    return {key: value for key, value in fields.items() if value not in (None, "", [], {})}


def configure_logging(
    level: str = "info",
    *,
    stream: IO[str] | None = None,
    path: str | Path | None = None,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
) -> None:
    """配置根 logger。重复调用是幂等的（先清空旧 handler）。

    `path` 给了就**同时**写 stderr 和文件：终端里立刻能看到，文件里留得住、能轮转。
    """
    handler_list: list[logging.Handler] = [logging.StreamHandler(stream or sys.stderr)]
    if path:
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        handler_list.append(
            RotatingFileHandler(
                target, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
        )

    formatter = JsonFormatter()
    for handler in handler_list:
        handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    for handler in handler_list:
        root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
