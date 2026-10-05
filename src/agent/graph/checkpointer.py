"""LangGraph 的检查点存储。

**为什么不用官方的 `AsyncSqliteSaver`**：它基于 `aiosqlite`，而 aiosqlite 把 sqlite
放进后台线程、再用事件循环回调把结果传回来。受限容器里这一步会失败，进程静默卡死
——和 `store/db.py` 记录的是同一个根因（见 AGENTS.md 已知坑表）。

官方的同步 `SqliteSaver` 反倒是好的：它的逻辑本来就在当前线程里跑，只是异步方法
直接 `raise NotImplementedError`。所以这里只补一层薄适配：异步方法转手调用同步实现。
序列化、版本管理、SQL 全部复用官方实现，我们一行都不重写。

**为什么和业务库共用一个文件**：checkpointer 存的是图的运行态（可丢、可重建），
业务库存的是对外事实源（不可丢）。两者的生命周期不同，但放在同一个 SQLite 文件里
用不同的表互相不干扰，还省掉一个文件路径配置。真要分开，改 `open_checkpointer` 一行即可。
"""

from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import (
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    RunnableConfig,
)
from langgraph.checkpoint.sqlite import SqliteSaver


class SqliteCheckpointer(SqliteSaver):
    """把官方 `SqliteSaver` 的同步实现直接暴露成异步接口。"""

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return self.get_tuple(config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return self.put(config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        self.put_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        self.delete_thread(thread_id)


def open_checkpointer(db_path: str | Path) -> SqliteCheckpointer:
    """打开（必要时创建）检查点存储，并建好它自己的表。"""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    # 业务库和检查点库共用一个文件，两个连接可能同时想写：等一会儿而不是直接报错
    conn.execute("PRAGMA busy_timeout=5000")
    saver = SqliteCheckpointer(conn)
    saver.setup()  # 建 checkpoints / writes 表，幂等
    saver.is_setup = True
    return saver
