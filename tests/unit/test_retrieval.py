"""检索层单测：切块、嵌入、FTS 表达式、融合与增量重建。

全部离线——默认嵌入后端是确定性的哈希实现，不下载任何权重。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent.retrieval import HashingEmbedder, RetrievalIndex, build_embedder, chunk_file
from agent.retrieval.chunker import iter_source_files
from agent.retrieval.embeddings import cosine
from agent.retrieval.search import (
    build_fts_query,
    hybrid_search,
    lexical_search,
    semantic_search,
)
from agent.store.db import Database


@pytest.fixture()
def corpus(tmp_path: Path) -> Path:
    """一个小代码库：两个主题不同的文件，用来验排序。"""
    (tmp_path / "alpha.py").write_text(
        "def resolve_within(workspace, raw):\n"
        "    '''把路径解析到工作区内，越界就抛错。'''\n"
        "    return workspace / raw\n",
        encoding="utf-8",
    )
    (tmp_path / "beta.py").write_text(
        "def truncate(text, limit):\n"
        "    '''输出太长时保留头尾，免得冲爆上下文。'''\n"
        "    return text[:limit]\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("# 说明\n\n这是文档，不是代码。\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "junk.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
async def index(corpus: Path, tmp_path: Path) -> AsyncIterator[tuple[Database, RetrievalIndex]]:
    db = Database(tmp_path / "idx.db")
    db.connect()
    idx = RetrievalIndex(db, build_embedder("offline"), corpus)
    await idx.rebuild()
    yield db, idx
    await db.close()


def test_iter_source_files_skips_noise(corpus: Path) -> None:
    assert {path.name for path in iter_source_files(corpus)} == {
        "alpha.py",
        "beta.py",
        "notes.md",
    }, "__pycache__ 不该进来"


def test_chunker_keeps_line_numbers_and_overlap(tmp_path: Path) -> None:
    file = tmp_path / "long.py"
    file.write_text("\n".join(f"line {i}" for i in range(1, 26)), encoding="utf-8")

    chunks = chunk_file(file, tmp_path, max_lines=10, overlap=2)

    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 10
    assert chunks[1].start_line == 9, "重叠两行，边界内容两块都有"
    assert chunks[-1].end_line == 25


def test_chunker_skips_blank_windows(tmp_path: Path) -> None:
    file = tmp_path / "blank.py"
    file.write_text("\n\n\n\n", encoding="utf-8")
    assert chunk_file(file, tmp_path, max_lines=2, overlap=0) == []


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dim=64)
    first = embedder.embed(["审批为什么要单独一个节点"])[0]
    assert first == embedder.embed(["审批为什么要单独一个节点"])[0], "同样输入必须同样输出"
    assert len(first) == 64
    # 分量按 6 位小数存（为了库里别太大），所以自相似是 1 而不是 1.000001
    assert abs(cosine(first, first) - 1.0) < 1e-5


def test_fts_query_quotes_terms_and_drops_short_ones() -> None:
    expression = build_fts_query("resolve_within 越界 a")
    assert '"resolve_within"' in expression
    assert "越界" not in expression, "短于 3 字符的项 trigram 匹配不上"


def test_fts_query_is_immune_to_punctuation() -> None:
    """查询里的标点不能把 FTS5 表达式拼坏——每一项都必须是带引号的短语。"""
    expression = build_fts_query('foo.bar "baz" (qux) -not')
    assert expression
    assert all(part.strip().startswith('"') for part in expression.split(" OR "))


async def test_lexical_search_finds_identifiers(
    index: tuple[Database, RetrievalIndex],
) -> None:
    db, _ = index
    hits = await lexical_search(db, "resolve_within", 5)
    assert hits, "标识符应当被词法层命中"
    top = await db.one("SELECT path FROM chunks WHERE id = ?", (hits[0][0],))
    assert top is not None and top["path"] == "alpha.py"


async def test_semantic_search_finds_related_words(
    index: tuple[Database, RetrievalIndex],
) -> None:
    db, idx = index
    hits = await semantic_search(db, idx.embedder, "输出太长怎么截断", 5)
    top = await db.one("SELECT path FROM chunks WHERE id = ?", (hits[0][0],))
    assert top is not None and top["path"] == "beta.py"


async def test_hybrid_search_reports_which_route_hit(
    index: tuple[Database, RetrievalIndex],
) -> None:
    db, idx = index
    hits = await hybrid_search(db, idx.embedder, "resolve_within 越界", limit=3)
    assert hits[0].path == "alpha.py"
    assert hits[0].sources, "要能说清是哪一路找到的"


async def test_rebuild_is_incremental_by_mtime(corpus: Path, tmp_path: Path) -> None:
    db = Database(tmp_path / "inc.db")
    db.connect()
    idx = RetrievalIndex(db, build_embedder("offline"), corpus)
    assert await idx.rebuild() > 0
    assert await idx.rebuild(incremental=True) == 0, "没改过就不该重切"

    (corpus / "beta.py").write_text("def changed():\n    return 1\n", encoding="utf-8")
    await asyncio.sleep(0.01)
    assert await idx.rebuild(incremental=True) > 0, "改过的文件要重切"

    (corpus / "notes.md").unlink()
    await idx.rebuild(incremental=True)
    remaining = {row["path"] for row in await db.all("SELECT DISTINCT path FROM chunks")}
    assert "notes.md" not in remaining, "删掉的文件要清出索引"
    await db.close()
