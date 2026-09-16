"""对话消息模型。

对照 C++：相当于一个带校验的 struct；`Message.user("你好")` 这类工厂方法
等价于带默认值的构造函数，避免到处手写 role 字符串。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """模型请求调用某个工具。"""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    parse_error: str | None = Field(
        default=None,
        description="模型给出的参数不是合法 JSON 时的原因，用于回填给模型让它重来",
    )


class Message(BaseModel):
    role: Role
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, *, tool_call_id: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)
