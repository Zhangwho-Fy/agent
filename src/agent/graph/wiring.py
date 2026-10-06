"""会话装配：把「配置 + 工作区 + 模型」装成一张能跑的图。

为什么单独一个模块：客户端（`agent run`）、服务端（每个会话）、评测（每条用例）
要装的是**同一套东西**——之前是三份复制粘贴。加一个开关就得改三处，漏一处就会变成
"评测跟线上行为不一致"这种最难查的 bug（压缩那次就差点漏掉：评测必须关掉它）。

所以规则是：**装配只有一处，开关只有一份**。谁要跑图，就来这里拿。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from ..config import Settings
from ..core.reliability import EventEmitter
from ..store import repo
from ..store.db import Database
from ..tools.base import ToolContext
from ..tools.policy import Policy
from ..tools.registry import ToolRegistry, default_registry
from .builder import build_graph


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

    两个口子单独说明：

    - `db`：有库才接得上 `recall`（取回被压缩掉的工具原文）。评测不传。
    - `compress_enabled`：评测要**关掉压缩**——它会改写消息序列，把按序号回放的
      夹具打乱（D32）。其余情况跟随配置。
    """
    ctx = ToolContext(
        workspace=workspace,
        timeout_s=settings.tool_timeout_s,
        output_limit_bytes=settings.output_limit_bytes,
        # 工具不认识数据库：装配层把"按 call_id 取回原文"这件事包成一个回调交给它
        fetch_tool_call=(
            (lambda call_id: repo.get_tool_call(db, call_id)) if db is not None else None
        ),
    )
    registry = default_registry()
    policy = Policy(ctx.workspace)
    graph = build_graph(
        model=model,
        registry=registry,
        policy=policy,
        ctx=ctx,
        emitter=emitter,
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
