"""图的节点：模型节点与工具节点。

为什么不用 LangGraph 预置的 `ToolNode`：预置节点直接执行工具，中间没有位置插
"分级 → 拦截 → 记事件"。而这几步正是本项目要展示的部分，所以工具节点自己写。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage

from ..core.events import EventType
from ..core.prompt import CODE_SYSTEM_PROMPT
from ..core.reliability import EventEmitter
from ..tools.base import ToolContext
from ..tools.policy import Decision, Policy
from ..tools.registry import ToolRegistry
from .state import AgentState


def build_model_node(model: BaseChatModel, registry: ToolRegistry) -> Any:
    """模型节点：把工具清单绑给模型，收到回复就追加进状态。"""
    bound_model = model.bind_tools(registry.to_openai_tools())

    async def call_model(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=CODE_SYSTEM_PROMPT), *state["messages"]]
        response = await bound_model.ainvoke(messages)

        usage = getattr(response, "usage_metadata", None) or {}
        previous = state.get("usage") or {}
        return {
            "messages": [response],
            "rounds": (state.get("rounds") or 0) + 1,
            "usage": {
                "input_tokens": previous.get("input_tokens", 0) + int(usage.get("input_tokens", 0)),
                "output_tokens": previous.get("output_tokens", 0)
                + int(usage.get("output_tokens", 0)),
            },
        }

    return call_model


def build_tool_node(
    *,
    registry: ToolRegistry,
    policy: Policy,
    ctx: ToolContext,
    emitter: EventEmitter,
) -> Any:
    """工具节点：分级 → 执行或拒绝 → 发事件 → 把结果回填给模型。"""

    async def call_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        results: list[AnyMessage] = []

        for call in getattr(last, "tool_calls", None) or []:
            name = str(call.get("name", ""))
            args = dict(call.get("args") or {})
            call_id = str(call.get("id", ""))
            emitter.emit(EventType.TOOL_CALL, {"call_id": call_id, "name": name, "args": args})

            tool = registry.get(name)
            if tool is None:
                content = f"未知工具：{name}"
                results.append(ToolMessage(content=content, tool_call_id=call_id))
                emitter.emit(
                    EventType.TOOL_RESULT,
                    {"call_id": call_id, "name": name, "status": "unknown_tool"},
                )
                continue

            decision = policy.classify(tool, args)
            if decision.decision is not Decision.AUTO:
                content = f"未执行（{decision.decision.value}）：{decision.reason}"
                results.append(ToolMessage(content=content, tool_call_id=call_id))
                emitter.emit(
                    EventType.TOOL_RESULT,
                    {
                        "call_id": call_id,
                        "name": name,
                        "status": decision.decision.value,
                        "preview": content,
                    },
                )
                continue

            result = await tool.run(args, ctx)
            results.append(ToolMessage(content=result.content, tool_call_id=call_id))
            emitter.emit(
                EventType.TOOL_RESULT,
                {
                    "call_id": call_id,
                    "name": name,
                    "status": "ok" if result.ok else "error",
                    "exit_code": result.exit_code,
                    "truncated": result.truncated,
                    "duration_ms": result.duration_ms,
                    "preview": result.content[:400],
                },
            )

        return {"messages": results}

    return call_tools
