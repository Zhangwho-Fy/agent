"""工具契约：模型需要知道的工具描述。

放在 core 而不是 tools/ 下的原因：providers 要把它发给模型，tools 要实现它，
两边都依赖它，放在共同的下游就不会出现互相 import。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


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
