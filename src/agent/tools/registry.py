"""工具注册表：把工具集合暴露给模型，并按名字查回来执行。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.tool_spec import ToolSpec
from .base import Tool
from .fs import LIST_TOOL, READ_TOOL
from .shell import EXEC_TOOL
from .skill import SKILL_CREATE_TOOL, SKILL_LOAD_TOOL


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: dict[str, Tool] = {tool.name: tool for tool in tools}

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=tool.name, description=tool.description, parameters=tool.parameters)
            for tool in self._tools.values()
        ]

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """转成模型接口要的 tools 参数。"""
        return [spec.to_openai() for spec in self.specs()]


def default_registry() -> ToolRegistry:
    """默认工具集：文件、命令、技能。

    检索类工具还没接（`search_code` 是 AGENTS.md 7.4 记的那笔小账）——
    在它落地之前，读代码靠 `fs_list` + `fs_read`。
    """
    return ToolRegistry([READ_TOOL, LIST_TOOL, EXEC_TOOL, SKILL_LOAD_TOOL, SKILL_CREATE_TOOL])
