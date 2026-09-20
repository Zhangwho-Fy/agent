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


async def _route(state: AgentState) -> str:
    """条件边：最后一条消息带工单就去工具节点，否则收工。

    写成 `async` 不是为了"异步做事"（它一行同步逻辑都没有），而是**避免下一次线程跳**：
    框架遇到同步函数会把它丢进线程池执行，而这个项目其余部分全是 async。
    少一次线程交接，在受限容器（线程池交接可能被限制）里也不会卡住。
    """
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "end"


def build_graph(
    *,
    model: BaseChatModel,
    registry: ToolRegistry,
    policy: Policy,
    ctx: ToolContext,
    emitter: EventEmitter,
    max_tool_rounds: int = 12,
    checkpointer: Any | None = None,
) -> Any:
    """编译图。

    `checkpointer` 阶段 1 传 None（无持久化），阶段 2 接 SQLite——
    接线方式不变，这正是用框架的收益。
    """
    builder = StateGraph(AgentState)
    builder.add_node("agent", build_model_node(model, registry, max_tool_rounds=max_tool_rounds))
    builder.add_node(
        "tools", build_tool_node(registry=registry, policy=policy, ctx=ctx, emitter=emitter)
    )
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", _route, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")
    return builder.compile(checkpointer=checkpointer)
