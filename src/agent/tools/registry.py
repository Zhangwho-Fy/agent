"""工具注册表：把工具集合暴露给模型，并按名字查回来执行。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..core.tool_spec import ToolSpec
from .base import Tool
from .fs import LIST_TOOL, READ_TOOL
from .memory import SEARCH_TOOL as MEMORY_SEARCH_TOOL
from .memory import WRITE_TOOL as MEMORY_WRITE_TOOL
from .recall import RECALL_TOOL
from .search_code import SEARCH_CODE_TOOL
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
    """默认工具集：文件、命令、技能、检索、记忆。

    `search_code` / `memory_search` / `memory_write` 都**常驻注册表**：它们的运行时
    依赖（索引、记忆库）由装配层通过 `ToolContext` 注入，没接的时候工具返回一句人话，
    和 `recall` 在没接会话库时的做法一样。这样"工具清单"不随运行方式变化，
    L2 里的工具列表与提示缓存前缀也就稳定。
    """
    return ToolRegistry(
        [
            READ_TOOL,
            LIST_TOOL,
            EXEC_TOOL,
            RECALL_TOOL,
            SKILL_LOAD_TOOL,
            SKILL_CREATE_TOOL,
            SEARCH_CODE_TOOL,
            MEMORY_SEARCH_TOOL,
            MEMORY_WRITE_TOOL,
        ]
    )
