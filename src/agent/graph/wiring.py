"""会话装配：把「配置 + 工作区 + 模型」装成一张能跑的图。

为什么单独一个模块：客户端（`agent run`）、服务端（每个会话）、评测（每条用例）
要装的是**同一套东西**——之前是三份复制粘贴。加一个开关就得改三处，漏一处就会变成
"评测跟线上行为不一致"这种最难查的 bug（压缩那次就差点漏掉：评测必须关掉它）。

所以规则是：**装配只有一处，开关只有一份**。谁要跑图，就来这里拿。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from ..config import Settings
from ..core.reliability import EventEmitter
from ..memory import GLOBAL_SCOPE, MemoryStore, scope_for
from ..memory.render import render_memories
from ..retrieval import build_embedder, clear_search_services, get_search_service
from ..store import repo
from ..store.db import Database
from ..tools.base import ToolContext
from ..tools.policy import Policy
from ..tools.registry import ToolRegistry, default_registry
from .builder import build_graph
from .state import AgentState

logger = logging.getLogger(__name__)

#: 进程级缓存。装配函数可能被每个会话各调一次（服务端就是这样），但嵌入模型、
#: 记忆库、索引都不该跟着复制——fastembed 加载一次权重是秒级的。
_EMBEDDERS: dict[tuple[str, str], Any] = {}
_MEMORY_STORES: dict[str, MemoryStore] = {}


def _embedder(settings: Settings) -> Any:
    key = (settings.embed_backend, settings.embed_model)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = build_embedder(settings.embed_backend, settings.embed_model)
    return _EMBEDDERS[key]


def get_memory_store(settings: Settings) -> MemoryStore:
    """按路径拿一个常驻的 `MemoryStore`（构造不碰磁盘，第一次读写才连库）。"""
    path = settings.resolved_memory_db_path
    store = _MEMORY_STORES.get(str(path))
    if store is None:
        store = MemoryStore(path, embedder=_embedder(settings))
        _MEMORY_STORES[str(path)] = store
    return store


def reset_runtime_caches() -> None:
    """测试用：丢掉进程级的嵌入器 / 记忆库 / 索引缓存。"""
    _EMBEDDERS.clear()
    _MEMORY_STORES.clear()
    clear_search_services()


@dataclass(slots=True)
class SessionGraph:
    """一套运行时装配：工具上下文、注册表、策略、编译好的图。

    调用方通常只要 `graph`；`ctx` / `registry` 留着是为了让 CLI 打印工作区和工具清单。
    """

    ctx: ToolContext
    registry: ToolRegistry
    policy: Policy
    graph: Any


def build_session_graph(
    settings: Settings,
    *,
    workspace: Path,
    emitter: EventEmitter,
    model: BaseChatModel,
    db: Database | None = None,
    checkpointer: Any | None = None,
    compress_enabled: bool | None = None,
) -> SessionGraph:
    """按配置装配一套运行时。

    三个口子单独说明：

    - `db`：有库才接得上 `recall`（取回被压缩掉的工具原文）。评测不传。
    - `compress_enabled`：评测要**关掉压缩**——它会改写消息序列，把按序号回放的
      夹具打乱（D32）。其余情况跟随配置。
    - 检索与记忆：索引、记忆库都由装配层注入 `ctx`（工具不认识它们），
      没有网络/磁盘副作用——真正的建索引与建表都推到第一次使用（见 9.3 / 8.3）。
    """
    embedder = _embedder(settings)
    memory = get_memory_store(settings)
    search = get_search_service(
        index_dir=settings.resolved_index_dir,
        workspace=workspace,
        embedder=embedder,
    )
    memory_scope = scope_for(workspace)
    ctx = ToolContext(
        workspace=workspace,
        timeout_s=settings.tool_timeout_s,
        output_limit_bytes=settings.output_limit_bytes,
        # 工具不认识数据库：装配层把"按 call_id 取回原文"这件事包成一个回调交给它
        fetch_tool_call=(
            (lambda call_id: repo.get_tool_call(db, call_id)) if db is not None else None
        ),
        # 检索与记忆同理：工具只知道"有个检索入口 / 有个记忆库"
        fetch_search=lambda query, limit, path_prefix: search.search(
            query, limit=limit, path_prefix=path_prefix
        ),
        memory=memory,
        memory_scope=memory_scope,
        session_id=emitter.session_id,
    )

    async def memory_digest(state: AgentState) -> str:
        """每轮现拼的记忆摘要（8.5 / D44）：不进 state，只拼进这一次请求的尾部。"""
        limit = settings.memory_digest_limit
        if limit <= 0:
            return ""
        try:
            cards = await memory.digest(scopes=[memory_scope, GLOBAL_SCOPE], limit=limit)
        except Exception:
            # 记忆是增强，不是主链路：库打不开、磁盘满了都只跳过这一轮，
            # 不能让整个会话跟着失败。真正的异常留给日志，模型看不到。
            logger.warning("记忆摘要失败，这一轮跳过", exc_info=True)
            return ""
        return render_memories(cards)

    registry = default_registry()
    policy = Policy(ctx.workspace)
    graph = build_graph(
        model=model,
        registry=registry,
        policy=policy,
        ctx=ctx,
        emitter=emitter,
        memory_digest=memory_digest,
        max_tool_rounds=settings.max_tool_rounds,
        context_limit=settings.context_limit,
        compress_enabled=(
            settings.compress_enabled if compress_enabled is None else compress_enabled
        ),
        compress_lossless_ratio=settings.compress_lossless_ratio,
        compress_summary_ratio=settings.compress_summary_ratio,
        compress_keep_recent=settings.compress_keep_recent_tool_results,
        checkpointer=checkpointer,
        db=db,
    )
    return SessionGraph(ctx=ctx, registry=registry, policy=policy, graph=graph)
