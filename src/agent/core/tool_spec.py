"""工具契约：模型需要知道的工具描述。

放在 core 而不是 tools/ 下的原因：事件、审批与工具实现都要用到它，
放在共同的下游就不会出现互相 import。
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

#: 工具名必须满足的字符集。
#:
#: 这不是我们的规定，是 OpenAI 兼容协议（含 DeepSeek）强制的：函数名只允许
#: `[A-Za-z0-9_-]`。踩过一次——用 `fs.read` 这种带点号的命名，请求被直接 400 拒绝，
#: 报错是 `Invalid 'tools[0].function.name': string does not match pattern`。
TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class Tier(StrEnum):
    """工具的风险等级，决定要不要人工审批。"""

    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


class ToolSpec(BaseModel):
    """一个工具对模型可见的部分：名字、用途、参数 JSON Schema。"""

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema，模型据此生成参数",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not TOOL_NAME_PATTERN.match(value):
            raise ValueError(
                f"工具名 {value!r} 不符合协议要求：只允许字母、数字、下划线、连字符（1~64 位）"
            )
        return value

    def to_openai(self) -> dict[str, Any]:
        """转成 chat.completions 的 tools 参数格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
