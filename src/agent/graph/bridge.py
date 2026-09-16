"""事件桥：把框架的流映射成我们对外的事件。

这是全项目最关键的适配层。**框架负责"怎么跑"，这里负责"对外怎么说"**：

- 框架的 `stream_mode="messages"` → 我们的 `text.delta`
- 工具节点自己发 `tool.call` / `tool.result`（因为分级与拦截都在那一层）
- 一轮结束 → `text.done` / `turn.done`

把映射集中在一个文件里，是为了让框架升级或换实现时，改动有明确边界。
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage

from ..core.events import EventType
from ..core.reliability import EventEmitter


async def stream_turn(
    *,
    graph: Any,
    prompt: str,
    emitter: EventEmitter,
    session_id: str,
    turn_id: str,
    recursion_limit: int = 40,
) -> str:
    """驱动一轮对话，返回模型最终文本。"""
    emitter.emit(EventType.TURN_STARTED, {"prompt": prompt}, turn_id=turn_id)

    inputs: dict[str, Any] = {"messages": [HumanMessage(content=prompt)], "rounds": 0}
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},  # 与 checkpointer 关联的线程 ID
        "recursion_limit": recursion_limit,  # 循环次数硬上限，防止停不下来
    }

    text_parts: list[str] = []
    status = "done"
    try:
        async for mode, payload in graph.astream(
            inputs, config=config, stream_mode=["messages", "updates"]
        ):
            if mode != "messages":
                continue  # updates 里的事件由工具节点自己发，避免重复
            chunk, metadata = payload
            if metadata.get("langgraph_node") != "agent":
                continue
            piece = getattr(chunk, "content", "")
            if isinstance(piece, str) and piece:
                text_parts.append(piece)
                emitter.emit(EventType.TEXT_DELTA, {"text": piece}, turn_id=turn_id)
    except Exception as exc:  # 基础设施级故障：模型调用失败、图出错
        status = "failed"
        emitter.emit(
            EventType.ERROR,
            {"kind": type(exc).__name__, "message": str(exc), "retryable": False},
            turn_id=turn_id,
        )

    final_text = "".join(text_parts)
    emitter.emit(EventType.TEXT_DONE, {"text": final_text}, turn_id=turn_id)
    emitter.emit(
        EventType.TURN_DONE,
        {"status": status, "chars": len(final_text)},
        turn_id=turn_id,
    )
    return final_text
