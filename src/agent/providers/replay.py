"""回放 Provider：把录制好的模型输出重放一遍。

它撑起两件事：

1. **可评测**：golden 任务集用固定录制跑，结果可比、CI 不花钱、不联网；
2. **可复现 bug**：线上出问题，把那段流录下来，本地回放即可稳定复现。

用完即抛 `ReplayExhausted`：测试里少录一段会立刻炸掉，
而不是静默返回空流让人查半天。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core.chunks import CHUNK_ADAPTER, ProviderChunk
from ..core.messages import Message
from ..core.tool_spec import ToolSpec


class ReplayExhausted(RuntimeError):
    """录制内容用完了。"""


class ReplayProvider:
    """按轮次回放的 Provider。"""

    def __init__(self, turns: Iterable[Iterable[ProviderChunk]]) -> None:
        self._turns: list[list[ProviderChunk]] = [list(turn) for turn in turns]
        self._cursor = 0
        #: 记录每次 stream 收到的入参，测试里用来断言"喂进去的上下文对不对"
        self.requests: list[dict[str, Any]] = []

    # ---- 构造 ----
    @classmethod
    def from_file(cls, path: str | Path) -> ReplayProvider:
        """从 JSON 文件加载。

        两种结构都支持，后者更啰嗦但便于夹带请求信息：

            {"turns": [[{"type": "text_delta", "text": "hi"}]]}
            {"turns": [{"chunks": [...]}]}
        """
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_turns: Sequence[Any] = payload["turns"] if isinstance(payload, Mapping) else payload
        turns: list[list[ProviderChunk]] = []
        for raw_turn in raw_turns:
            chunks = raw_turn["chunks"] if isinstance(raw_turn, Mapping) else raw_turn
            turns.append([CHUNK_ADAPTER.validate_python(chunk) for chunk in chunks])
        return cls(turns)

    # ---- 查询 ----
    @property
    def remaining(self) -> int:
        """还没被消费的轮次数。"""
        return max(len(self._turns) - self._cursor, 0)

    # ---- Provider 接口 ----
    async def stream(
        self,
        *,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
    ) -> AsyncIterator[ProviderChunk]:
        if self._cursor >= len(self._turns):
            raise ReplayExhausted(
                f"录制内容已用完（共 {len(self._turns)} 轮），"
                "说明实际发生的模型调用比录制时多——通常是主循环多跑了一轮"
            )

        self.requests.append(
            {
                "messages": [message.model_dump(mode="json") for message in messages],
                "tools": [tool.model_dump(mode="json") for tool in tools],
            }
        )
        turn = self._turns[self._cursor]
        self._cursor += 1
        for chunk in turn:
            yield chunk
