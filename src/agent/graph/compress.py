"""压缩节点：在**轮边界**把该做的压缩做掉。

为什么挂在图的最前面（`START → compress → agent`）：一轮开始时是天然的轮边界，
而且只有在这里能拿到完整状态（消息、用量、turn_id）——bridge 在轮末尾只看得到增量。
被 `interrupt()` 挂起后的恢复**不会**回到这个节点（LangGraph 从挂起点继续），
所以半轮中间不会被压缩，正合 D32。

**整表替换**用 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 清空再按原序追加：这样
`tool_call` 与 `tool_result` 的配对不会被打乱。不能用"删一条、补一条"——补的那条会被
追加到列表末尾，模型看到的是"一堆结果堆在最后"，协议层面就不合法了。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from ..core.compress import (
    plan,
    pointerize,
    split_for_summary,
    summary_message,
    summary_request,
)
from ..core.events import EventType
from ..core.reliability import EventEmitter
from ..logging import log_extra
from .state import AgentState

logger = logging.getLogger(__name__)


def build_compress_node(
    *,
    emitter: EventEmitter,
    model: BaseChatModel | None = None,
    context_limit: int = 0,
    lossless_ratio: float = 0.6,
    summary_ratio: float = 0.8,
    keep_recent_tool_results: int = 4,
    keep_recent_messages: int = 6,
    max_failures: int = 2,
    enabled: bool = True,
) -> Any:
    """构造压缩节点。

    `max_failures` 是熔断器：生产数据里大量会话会困在"反复压缩失败"的循环里烧钱，
    连续失败到阈值就这一轮不再试（计数器在图的闭包里，不跨会话）。
    """
    failures = 0

    async def compress(state: AgentState) -> dict[str, Any]:
        nonlocal failures
        if not enabled or not context_limit:
            return {}

        used = int(state.get("context_tokens") or 0)
        if not used:
            return {}
        ratio = used / context_limit
        decision = plan(ratio, lossless_ratio=lossless_ratio, summary_ratio=summary_ratio)
        if not decision.active:
            return {}

        messages = list(state["messages"])
        remaining, pointerized = pointerize(messages, keep_recent=keep_recent_tool_results)
        summarized = 0
        if decision.summarize and model is not None and failures < max_failures:
            prefix, keep = split_for_summary(remaining, keep_recent=keep_recent_messages)
            if prefix:
                try:
                    response = await model.ainvoke(summary_request(prefix))
                    text = response.content if isinstance(response.content, str) else ""
                    if text.strip():
                        remaining = [summary_message(text), *keep]
                        summarized = len(prefix)
                        failures = 0
                    else:
                        failures += 1
                except Exception:
                    failures += 1
                    logger.exception(
                        "压缩失败",
                        extra=log_extra(
                            session_id=emitter.session_id,
                            turn_id=state.get("turn_id"),
                            failures=failures,
                            max_failures=max_failures,
                        ),
                    )

        if not pointerized and not summarized:
            return {}

        await emitter.emit(
            EventType.CONTEXT_COMPRESSED,
            {
                "ratio": round(ratio, 3),
                "pointerized": pointerized,
                "summarized": summarized,
                "failures": failures,
            },
            turn_id=state.get("turn_id") or None,
        )
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *remaining]}

    return compress
