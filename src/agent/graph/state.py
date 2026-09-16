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
    rounds: int
    usage: dict[str, int]
