"""检索索引：词法层 + 向量层，都放在 SQLite 里。

- **词法层**用 FTS5 的 `trigram` 分词器：按 3 字符切，中文注释也能检索
  （默认的 `unicode61` 会把整句中文当成一个词，基本没法用）。
- **向量层**把向量按 JSON 存在同一张表里。样例规模（几千块）够用，
  真上规模该换成向量库——`search.py` 只依赖"给我 (id, 向量)"，换实现不动调用方。

支持增量：按文件 mtime 判断哪些变过，只重切那些文件；删掉的文件也会清出索引。
"""

from __future__ import annotations

import json
from pathlib import Path

from ..store.db import Database, now_iso
from .chunker import chunk_file, iter_source_files
from .embeddings import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  path        TEXT NOT NULL,
  start_line  INTEGER NOT NULL,
  end_line    INTEGER NOT NULL,
  content     TEXT NOT NULL,
  embedding   TEXT NOT NULL,
  indexed_at  TEXT NOT NULL,
  UNIQUE (path, start_line)
);

-- 独立的 FTS 表，rowid 跟 chunks.id 对齐，方便两边一起删
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(content, path, tokenize='trigram');

-- 文件级水位：增量重建时只看 mtime 变过的文件
CREATE TABLE IF NOT EXISTS chunk_files (
  path    TEXT PRIMARY KEY,
  mtime   REAL NOT NULL
);
"""


class RetrievalIndex:
    """某个工作区的检索索引。"""

    def __init__(self, db: Database, embedder: Embedder, root: Path) -> None:
        self.db = db
        self.embedder = embedder
        self.root = root.expanduser().resolve()

    async def ensure_schema(self) -> None:
        for statement in SCHEMA.split(";"):
            if statement.strip():
                await self.db.execute(statement)

    async def rebuild(self, *, incremental: bool = False) -> int:
        """重建索引，返回写入的块数。

        `incremental=True` 时只处理 mtime 变过的新文件，并清掉已删除文件。
        """
        await self.ensure_schema()
        files = list(iter_source_files(self.root))
        current = {path.relative_to(self.root).as_posix(): path for path in files}
        known = {
            row["path"]: row["mtime"]
            for row in await self.db.all("SELECT path, mtime FROM chunk_files")
        }

        if not incremental:
            await self._clear()
            known = {}

        stale = [path for path in known if path not in current]
        changed = [
            path for path, file in current.items() if known.get(path) != file.stat().st_mtime
        ]
        for path in stale:
            await self._drop_path(path)

        written = 0
        for path in changed:
            file = current[path]
            await self._drop_path(path)
            chunks = chunk_file(file, self.root)
            if chunks:
                vectors = self.embedder.embed([chunk.content for chunk in chunks])
                for chunk, vector in zip(chunks, vectors, strict=True):
                    await self.db.execute(
                        """
                        INSERT INTO chunks
                            (path, start_line, end_line, content, embedding, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chunk.path,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.content,
                            json.dumps(vector),
                            now_iso(),
                        ),
                    )
                    chunk_id = await self.db.scalar("SELECT last_insert_rowid()")
                    await self.db.execute(
                        "INSERT INTO chunks_fts (rowid, content, path) VALUES (?, ?, ?)",
                        (chunk_id, chunk.content, chunk.path),
                    )
                    written += 1
            await self.db.execute(
                "INSERT OR REPLACE INTO chunk_files (path, mtime) VALUES (?, ?)",
                (path, file.stat().st_mtime),
            )
        return written

    async def stats(self) -> dict[str, int]:
        await self.ensure_schema()
        return {
            "chunks": int(await self.db.scalar("SELECT COUNT(*) FROM chunks") or 0),
            "files": int(await self.db.scalar("SELECT COUNT(*) FROM chunk_files") or 0),
            "dim": self.embedder.dim,
        }

    async def _clear(self) -> None:
        await self.db.execute("DELETE FROM chunks")
        await self.db.execute("DELETE FROM chunks_fts")
        await self.db.execute("DELETE FROM chunk_files")

    async def _drop_path(self, path: str) -> None:
        ids = [
            row["id"] for row in await self.db.all("SELECT id FROM chunks WHERE path = ?", (path,))
        ]
        for chunk_id in ids:
            await self.db.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
        await self.db.execute("DELETE FROM chunks WHERE path = ?", (path,))
