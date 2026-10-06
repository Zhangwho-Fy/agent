"""组装 StateGraph。

形状：

    START → compress → agent ──(有工具调用)──→ approve ──→ tools ──→ agent
                          └────(没有工具调用)────────────→ END

条件边 `_route` 就是"循环要不要继续"的判据，等价于手写版里的
`if not tool_calls: return`。

`approve` 只做分级判断、可能在这里 `interrupt()` 挂起；真正的执行在 `tools`。
分开是因为**挂起恢复时节点会重跑**，而重跑工具执行是灾难。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from ..core.prompt import render_system_prompt
from ..core.reliability import EventEmitter
from ..skills.loader import catalog_text
from ..store.db import Database
from ..tools.base import ToolContext
from ..tools.policy import Policy
from ..tools.registry import ToolRegistry
from .compress import build_compress_node
from .nodes import build_approval_node, build_model_node, build_tool_node
from .state import AgentState


async def _route(state: AgentState) -> str:
    """条件边：最后一条消息带工单就去工具节点，否则收工。

    写成 `async` 不是为了"异步做事"（它一行同步逻辑都没有），而是**避免下一次线程跳**：
    框架遇到同步函数会把它丢进线程池执行，而这个项目其余部分全是 async。
    少一次线程交接，在受限容器（线程池交接可能被限制）里也不会卡住。
    """
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


async def _route_after_tools(state: AgentState) -> str:
    """工具跑完：除非这一轮被人工拒绝、要求停下，否则回模型继续。"""
    return "end" if state.get("halt") else "agent"


def build_graph(
    *,
    model: BaseChatModel,
    registry: ToolRegistry,
    policy: Policy,
    ctx: ToolContext,
    emitter: EventEmitter,
    max_tool_rounds: int = 12,
    context_limit: int = 0,
    compress_enabled: bool = True,
    compress_lossless_ratio: float = 0.6,
    compress_summary_ratio: float = 0.8,
    compress_keep_recent: int = 4,
    checkpointer: Any | None = None,
    db: Database | None = None,
) -> Any:
    """编译图。

    `checkpointer` 传 None 时图不能被中断（`interrupt()` 依赖检查点），
    所以要用审批流就必须接上它。`db` 用来记工具调用审计，可以不传。
    """
    builder = StateGraph(AgentState)
    builder.add_node(
        "compress",
        build_compress_node(
            emitter=emitter,
            model=model,
            context_limit=context_limit,
            lossless_ratio=compress_lossless_ratio,
            summary_ratio=compress_summary_ratio,
            keep_recent_tool_results=compress_keep_recent,
            enabled=compress_enabled,
        ),
    )

    def _system_prompt(state: AgentState) -> str:
        """每次模型调用现拼（L1 + L2）。

        传函数而不是字符串：技能目录按轮冻结（D11），"当期"只能从 state 的 `turn_id` 看出来。
        """
        return render_system_prompt(
            workspace=ctx.workspace,
            tool_names=registry.names,
            skills_catalog=catalog_text(ctx.workspace, str(state.get("turn_id") or "")),
        )

    builder.add_node(
        "agent",
        build_model_node(
            model,
            registry,
            system_prompt=_system_prompt,
            max_tool_rounds=max_tool_rounds,
            context_limit=context_limit,
        ),
    )
    builder.add_node("approve", build_approval_node(registry, policy))
    builder.add_node(
        "tools",
        build_tool_node(registry=registry, policy=policy, ctx=ctx, emitter=emitter, db=db),
    )
    builder.add_edge(START, "compress")
    builder.add_edge("compress", "agent")
    builder.add_conditional_edges("agent", _route, {"tools": "approve", "end": END})
    builder.add_edge("approve", "tools")
    builder.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", "end": END})
    return builder.compile(checkpointer=checkpointer)
