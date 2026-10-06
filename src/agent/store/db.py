"""SQLite 连接与访问。

这个模块只管"怎么把语句发到数据库"，不关心表里存什么——
表结构在 `schema.sql`，读写函数在 `repo.py`。

**为什么不开线程**（这是对详细设计 2.3 的一处有意偏离，理由写在这里）：

详细设计写的是"`sqlite3` 同步驱动 + `asyncio.to_thread`"。实测发现，
在受限容器里 `to_thread` 不可靠：把阻塞调用丢进线程池后，工作线程的完成
需要反向唤醒事件循环，而这个唤醒在沙箱里会失败，表现为整个进程静默卡死。
判断依据：`to_thread` 里跑 `time.sleep` / 写文件 / 任何 sqlite 语句都挂住，
而同样的代码用手写的 `threading.Thread` 执行则正常——差别就在"要不要唤醒事件循环"。
`AGENTS.md` 的已知坑表里记过同一类问题（框架把同步可调用对象丢线程池导致卡死），
这是第二次踩，所以在源头避开。

不开线程的代价是：每次数据库调用会短暂占用事件循环。可以接受，因为
本机 SQLite 的单条语句是微秒级，且 WAL + `synchronous=NORMAL` 不会有每次提交的 fsync。
真换成 Postgres 那天，再把这一层换成异步驱动即可——接口已经全是 `async def`，
调用方不用改。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def now_iso() -> str:
    """统一的时间戳：UTC、秒级精度、带时区，可以直接字符串比较。"""
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    """一个 SQLite 文件，加一套异步签名（实现是同步的，见模块说明）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

    # ---- 生命周期 ----

    def connect(self, *, apply_schema: bool = True) -> None:
        """打开连接、建目录、执行建表语句。

        幂等：`schema.sql` 全是 `IF NOT EXISTS`，重复调用没有副作用。
        写成同步方法是因为它只在启动时跑一次，而且失败要立刻暴露。

        `apply_schema=False` 给**旁路库**用（记忆库、检索索引库）：它们各有自己的建表
        语句，不该被塞进会话库的六张表——那样删会话时 `_SESSION_TABLES` 会多删一堆
        与它无关的表。
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        # WAL：写入不阻塞读，外部工具（sqlite3 命令行）也能边跑边查
        conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL：WAL 下只有 checkpoint 才 fsync。进程崩溃不丢数据，
        # 只有断电可能丢最后几条提交——本地工具这个取舍是划算的。
        conn.execute("PRAGMA synchronous=NORMAL")
        # 外键默认是关的，必须显式打开，否则 REFERENCES 只是注释
        conn.execute("PRAGMA foreign_keys=ON")
        if apply_schema:
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
        self._conn = conn

    async def close(self) -> None:
        """关闭连接。异步签名只是为了和其余方法一致。"""
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    # ---- 读写 ----

    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """执行写语句并提交，返回受影响的行数。"""
        conn = self._connection()
        try:
            cursor = conn.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()  # 不回滚的话，下一条语句会接着半截事务继续跑
            raise
        return cursor.rowcount

    async def one(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
        """取一行，没有则返回 None。"""
        return self._connection().execute(sql, params).fetchone()

    async def all(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        """取多行。"""
        return self._connection().execute(sql, params).fetchall()

    async def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        """取第一行第一列，常用于 COUNT / MAX。"""
        row = await self.one(sql, params)
        return None if row is None else row[0]

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            msg = "数据库尚未打开，请先调用 connect()"
            raise RuntimeError(msg)
        return self._conn
