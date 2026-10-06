"""`search_code`：检索的接入层（R0）。

设计见 `docs/design.md` 第 9.3 / 9.4 节。这里钉住的是：能搜到、能按路径收窄、
增量能看到新文件、输出是 L0 形态、没接索引时给一句人话而不是崩。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.retrieval.embeddings import HashingEmbedder
from agent.retrieval.service import SearchService
from agent.tools.base import ToolContext
from agent.tools.search_code import SEARCH_CODE_TOOL


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "alpha.py").write_text(
        "def resolve_within(path):\n    return path\n", encoding="utf-8"
    )
    (root / "src" / "beta.py").write_text("# 完全无关的模块\nVALUE = 1\n", encoding="utf-8")
    return root


def make_service(
    tmp_path: Path, workspace: Path, *, refresh_interval_s: float = 5.0
) -> SearchService:
    return SearchService(
        index_path=tmp_path / "index.db",
        workspace=workspace,
        embedder=HashingEmbedder(),
        refresh_interval_s=refresh_interval_s,
    )


async def test_search_finds_the_defining_file(tmp_path: Path, workspace: Path) -> None:
    service = make_service(tmp_path, workspace)
    try:
        hits = await service.search("resolve_within", limit=5)
    finally:
        await service.close()

    assert hits
    assert hits[0].path == "src/alpha.py"
    assert hits[0].start_line >= 1


async def test_path_prefix_narrows_the_search(tmp_path: Path, workspace: Path) -> None:
    service = make_service(tmp_path, workspace)
    try:
        inside = await service.search("VALUE", limit=5, path_prefix="src")
        outside = await service.search("VALUE", limit=5, path_prefix="nope")
    finally:
        await service.close()

    # 离线嵌入是字面级兜底，语义那一路会把两个文件都召回；这里验的是前缀过滤本身
    assert inside and all(hit.path.startswith("src/") for hit in inside)
    assert "src/beta.py" in [hit.path for hit in inside]
    assert outside == []


async def test_incremental_index_picks_up_new_files(tmp_path: Path, workspace: Path) -> None:
    service = make_service(tmp_path, workspace, refresh_interval_s=0.0)
    try:
        await service.search("resolve_within", limit=5)
        (workspace / "src" / "gamma.py").write_text("GAMMA_MARKER = 42\n", encoding="utf-8")
        hits = await service.search("GAMMA_MARKER", limit=5)
    finally:
        await service.close()

    assert any(hit.path == "src/gamma.py" for hit in hits), "mtime 增量应该看到新文件"


async def test_tool_formats_l0_hits(tmp_path: Path, workspace: Path) -> None:
    service = make_service(tmp_path, workspace)
    ctx = ToolContext(
        workspace=workspace,
        fetch_search=lambda query, limit, path_prefix: service.search(
            query, limit=limit, path_prefix=path_prefix
        ),
    )
    try:
        result = await SEARCH_CODE_TOOL.run({"query": "resolve_within"}, ctx)
    finally:
        await service.close()

    assert result.ok, result.content
    assert "src/alpha.py:" in result.content
    assert "fs_read" in result.content, "L0 要告诉模型怎么拿全文"


async def test_tool_without_an_index_degrades_gracefully(tmp_path: Path) -> None:
    result = await SEARCH_CODE_TOOL.run({"query": "x"}, ToolContext(workspace=tmp_path))

    assert result.ok is False
    assert "没有接检索索引" in result.content


async def test_empty_query_returns_nothing(tmp_path: Path, workspace: Path) -> None:
    service = make_service(tmp_path, workspace)
    try:
        assert await service.search("   ", limit=5) == []
    finally:
        await service.close()
