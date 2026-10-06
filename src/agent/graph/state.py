"""图状态：节点之间传递的数据。

`messages` 上挂的 `add_messages` 是**归约器**：节点返回的消息会被追加进列表，
而不是覆盖整个列表。这是 LangGraph 的核心机制之一——节点只需要返回增量。
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    rounds: int  # 模型被调用的次数
    tool_rounds: int  # 工具往返次数：每进一次 tools 节点 +1
    usage: dict[str, int]
    #: 本轮 turn 的 id，工具节点记审计时要用（checkpointer 恢复后仍然拿得到）
    turn_id: str
    #: 审批结果：call_id → 是否批准。审批节点写入，工具节点读取
    approvals: dict[str, bool]
    #: 本轮每个工具调用了多少次、失败多少次。工具节点累加，状态块只读（第 4 节）
    tool_stats: dict[str, dict[str, int]]
    #: 上一次模型调用实际塞进上下文的 token 数——上下文占用估算的锚点
    context_tokens: int
