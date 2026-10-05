"""混合检索：词法 + 语义，用 RRF 融合。

**为什么要混合**：词法（FTS5）对标识符和精确词最灵——搜 `resolve_within` 它一击即中；
语义（向量）能处理换句话说的问法——"路径越界怎么拦"能命中 `resolve_within`。
两者错的不是同一类样本，合起来比任何一方都稳。

**为什么用 RRF 而不是加权求和**：两种分数的量纲根本不可比（BM25 是负数、余弦是 -1~1），
归一化又引入新参数。RRF 只看**名次**，不需要调权重，是省事且结实的默认选择：

    score = Σ 1 / (k + rank)，k 取 60（原论文的经验值）
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from ..store.db import Database
from .embeddings import Embedder, cosine

#: 查询里的词：字母数字下划线，或连续的 CJK
_TERM = re.compile(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{3,}")

#: 一条查询最多拆出多少个检索项，防止表达式失控
MAX_TERMS = 32


@dataclass(frozen=True, slots=True)
class SearchHit:
    path: str
    start_line: int
    end_line: int
    content: str
    score: float
    sources: tuple[str, ...]  # 命中它的检索路径：lexical / semantic


def build_fts_query(query: str) -> str:
    """把自然语言查询变成安全的 FTS5 表达式。

    - 每个词/三元组用双引号包起来（内部引号翻倍），避免 `.` `-` 之类被当成语法；
    - 用 OR 连接：宁可多召回几条，排序交给 RRF 融合；
    - 只保留长度 ≥3 的项——trigram 分词器索引的就是 3 字符序列，更短的匹配不上。
    """
    terms: list[str] = []
    for token in _TERM.findall(query.lower()):
        if token.isascii():
            terms.append(token)
        else:
            terms.extend(token[i : i + 3] for i in range(len(token) - 2))
    unique = list(dict.fromkeys(terms))[:MAX_TERMS]
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in unique)


async def lexical_search(db: Database, query: str, limit: int) -> list[tuple[int, float]]:
    """FTS5 检索，返回 `(chunk_id, bm25 分数)`，按相关性排序。"""
    expression = build_fts_query(query)
    if not expression:
        return []
    rows = await db.all(
        """
        SELECT rowid AS id, bm25(chunks_fts) AS score
          FROM chunks_fts
         WHERE chunks_fts MATCH ?
         ORDER BY score
         LIMIT ?
        """,
        (expression, limit),
    )
    return [(int(row["id"]), float(row["score"])) for row in rows]


async def semantic_search(
    db: Database, embedder: Embedder, query: str, limit: int
) -> list[tuple[int, float]]:
    """向量检索。

    样例规模下把向量全读出来算余弦；真上规模这里要换成向量库的 ANN 查询。
    """
    rows = await db.all("SELECT id, embedding FROM chunks")
    if not rows:
        return []
    probe = embedder.embed([query])[0]
    scored = [(int(row["id"]), cosine(probe, json.loads(row["embedding"]))) for row in rows]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


async def hybrid_search(
    db: Database,
    embedder: Embedder,
    query: str,
    *,
    limit: int = 5,
    candidates: int = 20,
    rrf_k: int = 60,
) -> list[SearchHit]:
    """两路各取前 `candidates` 条，按 RRF 融合，返回前 `limit` 条。"""
    lexical = await lexical_search(db, query, candidates)
    semantic = await semantic_search(db, embedder, query, candidates)

    fused: dict[int, float] = {}
    sources: dict[int, list[str]] = {}
    for rank, (chunk_id, _) in enumerate(lexical, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        sources.setdefault(chunk_id, []).append("lexical")
    for rank, (chunk_id, _) in enumerate(semantic, start=1):
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        sources.setdefault(chunk_id, []).append("semantic")

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:limit]
    if not ordered:
        return []
    return await _load(db, ordered, sources)


async def _load(
    db: Database,
    ordered: Sequence[tuple[int, float]],
    sources: dict[int, list[str]],
) -> list[SearchHit]:
    ids = [chunk_id for chunk_id, _ in ordered]
    placeholders = ",".join("?" for _ in ids)
    rows = await db.all(f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids)
    by_id = {int(row["id"]): row for row in rows}
    hits: list[SearchHit] = []
    for chunk_id, score in ordered:
        row = by_id.get(chunk_id)
        if row is None:  # 索引在查询过程中被重建过，跳过
            continue
        hits.append(
            SearchHit(
                path=str(row["path"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                content=str(row["content"]),
                score=score,
                sources=tuple(sources.get(chunk_id, [])),
            )
        )
    return hits
