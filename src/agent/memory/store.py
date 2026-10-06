"""长期记忆库：独立 SQLite 文件 + 卡片读写 + 混合检索。

设计见 `docs/design.md` 第 8.3 / 8.4 / 8.6 节。四个要点：

1. **独立文件**（D37）：记忆的寿命和会话不一致，删会话不该删记忆；对会话只存
   `source_session` 这类**软引用**，不建外键。
2. **写入先过三连判据 + 脱敏**（8.4）：内容为空、命中密钥形态、超出五类，都拒绝。
3. **同 `key` 的活卡被新卡替换时走 supersede**（D39）：旧卡只改状态字段，正文一个字不动，
   变更同时写 `memory_ops`——"当时为什么变成这样"永远查得到。
4. **检索复用代码域那套**（D53）：FTS5 trigram 词法 + 向量语义 + RRF 融合，只是索引分开。

数据库连接与建表都是**懒的**：构造这个对象不碰磁盘，第一次真正读写才开库。
测试里不传路径就不会在开发机上留下垃圾文件。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from ..core.ids import new_id
from ..retrieval.embeddings import Embedder, cosine
from ..retrieval.search import build_fts_query
from ..store.db import Database, now_iso
from .cards import (
    GLOBAL_SCOPE,
    MemoryCard,
    MemoryKind,
    MemoryStatus,
    MemoryTrust,
    find_secrets,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  row_id           INTEGER PRIMARY KEY AUTOINCREMENT,
  id               TEXT NOT NULL UNIQUE,
  owner            TEXT NOT NULL,
  scope            TEXT NOT NULL,
  kind             TEXT NOT NULL,
  "key"            TEXT,
  content          TEXT NOT NULL,
  attributes       TEXT NOT NULL DEFAULT '{}',
  embedding        TEXT NOT NULL DEFAULT '[]',
  trust            TEXT NOT NULL,
  status           TEXT NOT NULL,
  importance       REAL NOT NULL DEFAULT 0.5,
  valid_from       TEXT,
  valid_to         TEXT,
  supersedes       TEXT,
  superseded_by    TEXT,
  source_session   TEXT,
  source_turn      TEXT,
  source_event_seq INTEGER,
  source_quote     TEXT,
  created_at       TEXT NOT NULL,
  updated_at       TEXT NOT NULL,
  last_used_at     TEXT,
  use_count        INTEGER NOT NULL DEFAULT 0
);

-- trigram 分词：中文才检索得到（同代码索引，见 retrieval/index.py）
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
  USING fts5(content, tokenize='trigram');

-- 只追加的变更日志：write / supersede / archive / consolidate / forget
CREATE TABLE IF NOT EXISTS memory_ops (
  id         TEXT PRIMARY KEY,
  memory_id  TEXT NOT NULL,
  op         TEXT NOT NULL,
  actor      TEXT NOT NULL DEFAULT 'model',
  detail     TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_scope_status ON memories(scope, status);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories(scope, "key");
"""

#: RRF 的经验值，和 retrieval/search.py 保持一致
RRF_K = 60


class MemoryRejected(ValueError):
    """这条记忆不能写入（空内容 / 命中密钥 / 非法类型）。"""


class MemoryStore:
    """某个 `memory.db` 上的读写。所有方法都是 async，第一次调用才连库。"""

    def __init__(
        self,
        path: str | Path,
        *,
        embedder: Embedder | None = None,
        owner: str = "local",
    ) -> None:
        self.path = Path(path)
        self.owner = owner
        self._embedder = embedder
        self._db = Database(self.path)
        self._connected = False
        self._schema_ready = False

    # ---- 生命周期 ----

    async def _ensure(self) -> None:
        if not self._connected:
            # 旁路库：不要塞进会话库那六张表（见 Database.connect 的说明）
            self._db.connect(apply_schema=False)
            self._connected = True
        if not self._schema_ready:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    await self._db.execute(statement)
            self._schema_ready = True

    async def close(self) -> None:
        await self._db.close()

    # ---- 写 ----

    async def write(
        self,
        *,
        scope: str,
        kind: MemoryKind | str,
        content: str,
        key: str | None = None,
        attributes: dict[str, object] | None = None,
        trust: MemoryTrust | str = MemoryTrust.USER_STATED,
        importance: float = 0.5,
        source_session: str | None = None,
        source_turn: str | None = None,
        source_event_seq: int | None = None,
        source_quote: str | None = None,
    ) -> MemoryCard:
        """写入一条记忆；与已有活卡冲突时走 supersede，重复内容直接去重。"""
        await self._ensure()
        text = content.strip()
        if not text:
            raise MemoryRejected("content 不能为空")
        hits = find_secrets(text)
        if source_quote:
            hits += find_secrets(source_quote)
        if hits:
            raise MemoryRejected(f"内容里检测到密钥形态，拒绝写入（命中 {len(hits)} 条规则）")

        kind = MemoryKind(kind)
        trust = MemoryTrust(trust)
        scope = scope or GLOBAL_SCOPE
        now = now_iso()

        # 完全相同的活卡：去重，只把"用过"的计数加一
        duplicate = await self._db.one(
            "SELECT id FROM memories WHERE scope = ? AND content = ? AND status = 'active'"
            " ORDER BY row_id DESC LIMIT 1",
            (scope, text),
        )
        if duplicate is not None:
            await self._touch([str(duplicate["id"])], now)
            card = await self.get(str(duplicate["id"]))
            assert card is not None  # 刚查到，不可能没有
            return card

        supersedes: str | None = None
        if key:
            previous = await self._db.one(
                "SELECT id FROM memories WHERE scope = ? AND \"key\" = ? AND status = 'active'"
                " ORDER BY row_id DESC LIMIT 1",
                (scope, key),
            )
            if previous is not None:
                supersedes = str(previous["id"])

        card = MemoryCard(
            id=new_id("mem"),
            owner=self.owner,
            scope=scope,
            kind=kind,
            key=key,
            content=text,
            attributes=dict(attributes or {}),
            trust=trust,
            status=MemoryStatus.ACTIVE,
            importance=importance,
            valid_from=now,
            supersedes=supersedes,
            source_session=source_session,
            source_turn=source_turn,
            source_event_seq=source_event_seq,
            source_quote=source_quote,
            created_at=now,
            updated_at=now,
        )
        await self._db.execute(
            """
            INSERT INTO memories
                (id, owner, scope, kind, "key", content, attributes, embedding, trust, status,
                 importance, valid_from, supersedes, source_session, source_turn,
                 source_event_seq, source_quote, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                card.id,
                card.owner,
                card.scope,
                card.kind.value,
                card.key,
                card.content,
                json.dumps(card.attributes, ensure_ascii=False),
                json.dumps(self._embed(text)),
                card.trust.value,
                card.status.value,
                card.importance,
                card.valid_from,
                card.supersedes,
                card.source_session,
                card.source_turn,
                card.source_event_seq,
                card.source_quote,
                card.created_at,
                card.updated_at,
            ),
        )
        row_id = await self._db.scalar("SELECT last_insert_rowid()")
        await self._db.execute(
            "INSERT INTO memories_fts (rowid, content) VALUES (?, ?)",
            (row_id, text),
        )
        await self._log_op(card.id, "write", detail={"kind": kind.value, "scope": scope})

        if supersedes:
            # 只动状态字段：正文一个字不改（D39），变更同时进 memory_ops
            await self._db.execute(
                "UPDATE memories SET status = 'superseded', superseded_by = ?,"
                " valid_to = ?, updated_at = ? WHERE id = ?",
                (card.id, now, now, supersedes),
            )
            await self._log_op(supersedes, "supersede", detail={"by": card.id})
        return card

    # ---- 读 ----

    async def get(self, memory_id: str) -> MemoryCard | None:
        await self._ensure()
        row = await self._db.one("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return None if row is None else _row_to_card(row)

    async def list_cards(
        self,
        *,
        scopes: Sequence[str],
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        limit: int = 100,
    ) -> list[MemoryCard]:
        await self._ensure()
        clause, params = _scope_clause(scopes)
        sql = f"SELECT * FROM memories WHERE {clause}"
        if status is not None:
            sql += " AND status = ?"
            params.append(status.value)
        sql += " ORDER BY importance DESC, row_id DESC LIMIT ?"
        params.append(limit)
        rows = await self._db.all(sql, params)
        return [_row_to_card(row) for row in rows]

    async def digest(self, *, scopes: Sequence[str], limit: int = 5) -> list[MemoryCard]:
        """每轮摘要的来源：活跃卡片按 importance 排序取前几条（D44 / 8.5）。"""
        return await self.list_cards(scopes=scopes, status=MemoryStatus.ACTIVE, limit=limit)

    async def search(
        self, query: str, *, scopes: Sequence[str], limit: int = 5, candidates: int = 20
    ) -> list[MemoryCard]:
        """FTS5 词法 + 向量语义 + RRF 融合（D53）。"""
        await self._ensure()
        if not query.strip():
            return []
        scopes = list(scopes) or [GLOBAL_SCOPE]

        lexical = await self._lexical(query, scopes, candidates)
        semantic = await self._semantic(query, scopes, candidates)
        fused: dict[str, float] = {}
        for rank, (card_id, _) in enumerate(lexical, start=1):
            fused[card_id] = fused.get(card_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (card_id, _) in enumerate(semantic, start=1):
            fused[card_id] = fused.get(card_id, 0.0) + 1.0 / (RRF_K + rank)
        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
        if not ordered:
            return []

        ids = [card_id for card_id, _ in ordered]
        placeholders = ",".join("?" for _ in ids)
        rows = await self._db.all(f"SELECT * FROM memories WHERE id IN ({placeholders})", ids)
        by_id = {str(row["id"]): _row_to_card(row) for row in rows}
        cards = [by_id[card_id] for card_id, _ in ordered if card_id in by_id]
        await self._touch([card.id for card in cards], now_iso())
        return cards

    async def archive(self, memory_id: str, *, actor: str = "user") -> bool:
        """软删除：只标 archived，正文与 supersede 链都留着（D48）。"""
        await self._ensure()
        changed = await self._db.execute(
            "UPDATE memories SET status = 'archived', updated_at = ? WHERE id = ?"
            " AND status != 'archived'",
            (now_iso(), memory_id),
        )
        if changed:
            await self._log_op(memory_id, "archive", actor=actor)
        return bool(changed)

    # ---- 内部 ----

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            return []
        return [float(value) for value in self._embedder.embed([text])[0]]

    async def _touch(self, ids: Sequence[str], now: str) -> None:
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        await self._db.execute(
            f"UPDATE memories SET last_used_at = ?, use_count = use_count + 1"
            f" WHERE id IN ({placeholders})",
            (now, *ids),
        )

    async def _log_op(
        self,
        memory_id: str,
        op: str,
        *,
        actor: str = "model",
        detail: dict[str, object] | None = None,
    ) -> None:
        await self._db.execute(
            "INSERT INTO memory_ops (id, memory_id, op, actor, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                new_id("mop"),
                memory_id,
                op,
                actor,
                json.dumps(detail or {}, ensure_ascii=False),
                now_iso(),
            ),
        )

    async def _lexical(
        self, query: str, scopes: Sequence[str], limit: int
    ) -> list[tuple[str, float]]:
        expression = build_fts_query(query)
        if not expression:
            return []
        placeholders = ",".join("?" for _ in scopes)
        rows = await self._db.all(
            f"""
            SELECT m.id AS id, bm25(memories_fts) AS score
              FROM memories_fts
              JOIN memories m ON m.row_id = memories_fts.rowid
             WHERE memories_fts MATCH ?
               AND m.status = 'active'
               AND m.scope IN ({placeholders})
             ORDER BY score
             LIMIT ?
            """,
            [expression, *scopes, limit],
        )
        return [(str(row["id"]), float(row["score"])) for row in rows]

    async def _semantic(
        self, query: str, scopes: Sequence[str], limit: int
    ) -> list[tuple[str, float]]:
        if self._embedder is None:
            return []
        placeholders = ",".join("?" for _ in scopes)
        rows = await self._db.all(
            f"SELECT id, embedding FROM memories WHERE status = 'active'"
            f" AND scope IN ({placeholders})",
            list(scopes),
        )
        if not rows:
            return []
        probe = self._embedder.embed([query])[0]
        scored = [
            (str(row["id"]), cosine(probe, json.loads(row["embedding"])))
            for row in rows
            if row["embedding"]
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]


def _scope_clause(scopes: Sequence[str]) -> tuple[str, list[object]]:
    values = list(scopes) or [GLOBAL_SCOPE]
    placeholders = ",".join("?" for _ in values)
    return f"scope IN ({placeholders})", list(values)


def _row_to_card(row: sqlite3.Row) -> MemoryCard:
    """sqlite3.Row → MemoryCard。列名和 cards.MemoryCard 的字段一一对应。"""
    return MemoryCard(
        id=str(row["id"]),
        owner=str(row["owner"]),
        scope=str(row["scope"]),
        kind=MemoryKind(str(row["kind"])),
        key=row["key"],
        content=str(row["content"]),
        attributes=json.loads(row["attributes"] or "{}"),
        trust=MemoryTrust(str(row["trust"])),
        status=MemoryStatus(str(row["status"])),
        importance=float(row["importance"]),
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        supersedes=row["supersedes"],
        superseded_by=row["superseded_by"],
        source_session=row["source_session"],
        source_turn=row["source_turn"],
        source_event_seq=row["source_event_seq"],
        source_quote=row["source_quote"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_used_at=row["last_used_at"],
        use_count=int(row["use_count"]),
    )
