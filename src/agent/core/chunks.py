"""流式输出的数据块，以及把增量拼成完整工具调用的装配器。

为什么要有这一层：模型是**逐块**吐字的，工具调用更麻烦——id、名字、参数 JSON
可能被切在好几个块里。provider 负责把各家的协议统一成这里的块类型，
主循环只认这一套块，换模型供应商就不用改主循环。
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

from .messages import ToolCall


class TextDelta(BaseModel):
    """一段文本增量。"""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallDelta(BaseModel):
    """工具调用的增量片段。"""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    index: int = Field(description="并行调用时的序号，同一个 index 的片段属于同一个调用")
    id: str | None = None
    name: str | None = None
    arguments: str = Field(default="", description="参数 JSON 的片段，不保证每块都是完整 JSON")


class Usage(BaseModel):
    """token 用量，用于记账与评测。"""

    type: Literal["usage"] = "usage"
    input_tokens: int = 0
    output_tokens: int = 0


class Finished(BaseModel):
    """一段流结束。reason 例如 stop / tool_calls / length。"""

    type: Literal["finished"] = "finished"
    reason: str | None = None


ProviderChunk = Annotated[
    TextDelta | ToolCallDelta | Usage | Finished,
    Field(discriminator="type"),
]

#: 判别式解析器：从 JSON 读回块对象时用（录制回放）
CHUNK_ADAPTER: TypeAdapter[ProviderChunk] = TypeAdapter(ProviderChunk)


class _PartialCall:
    """一个正在拼装的工具调用。"""

    __slots__ = ("arguments", "id", "name")

    def __init__(self) -> None:
        self.id: str | None = None
        self.name: str | None = None
        self.arguments: list[str] = []


class ToolCallBuilder:
    """把流式增量拼成完整的 ToolCall。

    参数的 JSON 必须等整段流结束才能解析——中途的片段几乎必然不是合法 JSON。
    解析失败不抛异常，而是把原因塞进 `parse_error` 回填给模型，让它自己改正。
    """

    def __init__(self) -> None:
        self._partials: dict[int, _PartialCall] = {}

    def add(self, delta: ToolCallDelta) -> None:
        partial = self._partials.setdefault(delta.index, _PartialCall())
        if delta.id:
            partial.id = delta.id
        if delta.name:
            partial.name = delta.name
        if delta.arguments:
            partial.arguments.append(delta.arguments)

    @property
    def has_calls(self) -> bool:
        return bool(self._partials)

    def build(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for index in sorted(self._partials):
            partial = self._partials[index]
            raw = "".join(partial.arguments).strip()
            arguments: dict[str, Any] = {}
            parse_error: str | None = None

            if raw:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as exc:
                    parse_error = f"参数不是合法 JSON：{exc.msg}"
                else:
                    if isinstance(parsed, dict):
                        arguments = parsed
                    else:
                        parse_error = "参数不是 JSON 对象"

            if partial.name is None:
                parse_error = parse_error or "模型没有给出工具名"

            calls.append(
                ToolCall(
                    id=partial.id or f"call_{index}",
                    name=partial.name or "",
                    arguments=arguments,
                    parse_error=parse_error,
                )
            )
        return calls
