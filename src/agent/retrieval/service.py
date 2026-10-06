"""检索服务：把索引的生命周期包起来，给工具一个"给我命中"的入口。

设计见 `docs/design.md` 第 9.3 节。工具不认识数据库，也不认识索引——它只拿到一个
异步回调（`ToolContext.fetch_search`）；这个模块就是装配层塞进去的那个实现。

两条调度约定：

1. **懒建**：第一次检索才建索引，会话启动不为几千个文件付代价。
2. **增量 + 节流**：之后每次检索按 mtime 增量，但最多每 `refresh_interval_s` 秒重扫一次——
   每次检索都遍历工作区，在挂载盘上等于把检索的收益吃光。
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

from ..store.db import Database
from .embeddings import Embedder
from .index import RetrievalIndex
from .search import SearchHit, hybrid_search

#: 两次增量重扫之间的最小间隔（秒）。索引更新及时性 vs 遍历开销的折中。
REFRESH_INTERVAL_S = 5.0


def index_path_for(index_dir: Path, workspace: Path) -> Path:
    """一个工作区一个索引文件，名字用路径的 sha1 前 12 位——不同工作区不互相污染。"""
    digest = hashlib.sha1(str(workspace.expanduser().resolve()).encode("utf-8")).hexdigest()
    return index_dir.expanduser() / f"{digest[:12]}.db"


class SearchService:
    """某个工作区上的代码检索。构造不碰磁盘，第一次 `search()` 才连库建索引。"""

    def __init__(
        self,
        *,
        index_path: Path,
        workspace: Path,
        embedder: Embedder,
        refresh_interval_s: float = REFRESH_INTERVAL_S,
    ) -> None:
        self.index_path = index_path
        self.workspace = workspace.expanduser().resolve()
        self.refresh_interval_s = refresh_interval_s
        self._embedder = embedder
        self._db = Database(index_path)
        self._index = RetrievalIndex(self._db, embedder, self.workspace)
        self._connected = False
        self._last_check = 0.0

    async def _refresh(self) -> None:
        now = time.monotonic()
        if self._connected and now - self._last_check < self.refresh_interval_s:
            return
        if not self._connected:
            # 旁路库：索引有自己的表（chunks / chunks_fts / chunk_files）
            self._db.connect(apply_schema=False)
            self._connected = True
        await self._index.rebuild(incremental=True)
        self._last_check = now

    async def search(
        self, query: str, *, limit: int = 5, path_prefix: str | None = None
    ) -> list[SearchHit]:
        """混合检索，返回命中片段（L0 形态由工具负责渲染，见 9.4）。"""
        if not query.strip():
            return []
        await self._refresh()
        # 先多取一些，再按 path_prefix 过滤，最后才截断——否则过滤后可能一条都不剩
        hits = await hybrid_search(
            self._db,
            self._embedder,
            query,
            limit=max(limit * 3, limit),
            candidates=20,
        )
        if path_prefix:
            prefix = path_prefix.strip().strip("/")
            hits = [hit for hit in hits if hit.path == prefix or hit.path.startswith(f"{prefix}/")]
        return hits[:limit]

    async def close(self) -> None:
        await self._db.close()


#: 进程级缓存：同一进程里同一工作区只开一次索引、只连一次库。
_SERVICES: dict[tuple[str, str], SearchService] = {}


def get_search_service(
    *,
    index_dir: Path,
    workspace: Path,
    embedder: Embedder,
    refresh_interval_s: float = REFRESH_INTERVAL_S,
) -> SearchService:
    """按 (索引文件, 工作区) 取一个常驻的 `SearchService`。"""
    index_path = index_path_for(index_dir, workspace)
    key = (str(index_path), str(workspace.expanduser().resolve()))
    service = _SERVICES.get(key)
    if service is None:
        service = SearchService(
            index_path=index_path,
            workspace=workspace,
            embedder=embedder,
            refresh_interval_s=refresh_interval_s,
        )
        _SERVICES[key] = service
    return service


def clear_search_services() -> None:
    """测试用：丢掉进程级缓存（避免测试之间共享同一个 sqlite 连接）。"""
    _SERVICES.clear()
