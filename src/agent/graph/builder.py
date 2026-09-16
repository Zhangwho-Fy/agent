"""组装 StateGraph。

形状很简单：

    START → agent ──(有工具调用)──→ tools ──→ agent
              └────(没有工具调用)────→ END

条件边 `_route` 就是"循环要不要继续"的判据，等价于手写版里的
`if not tool_calls: return`。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from ..core.reliability import EventEmitter
from ..tools.base import ToolContext
from ..tools.policy import Policy
from ..tools.registry import ToolRegistry
from .nodes import build_model_node, build_tool_node
from .state import AgentState


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_graph(
    *,
    model: BaseChatModel,
    registry: ToolRegistry,
    policy: Policy,
    ctx: ToolContext,
    emitter: EventEmitter,
    checkpointer: Any | None = None,
) -> Any:
    """编译图。

    `checkpointer` 阶段 1 传 None（无持久化），阶段 2 接 SQLite——
    接线方式不变，这正是用框架的收益。
    """
    builder = StateGraph(AgentState)
    builder.add_node("agent", build_model_node(model, registry))
    builder.add_node(
        "tools", build_tool_node(registry=registry, policy=policy, ctx=ctx, emitter=emitter)
    )
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
