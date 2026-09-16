"""Provider 抽象。

注意签名写法：`def stream(...) -> AsyncIterator[...]` 而不是
`async def stream(...) -> ...`。因为实现是**异步生成器**（函数体里有 yield），
调用它拿到的是"异步迭代器"而非"协程"；写成协程会让 `async for` 直接报错，
这是接第一个 provider 时最常见的坑。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Protocol, runtime_checkable

from ..core.chunks import ProviderChunk
from ..core.messages import Message
from ..core.tool_spec import ToolSpec


@runtime_checkable
class Provider(Protocol):
    """模型供应商的统一接口。"""

    def stream(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> AsyncIterator[ProviderChunk]:
        """流式产出模型输出。实现必须是异步生成器。"""
        ...
