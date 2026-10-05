"""录放：把模型调用变成可以提交进仓库的夹具。

**要解决的问题**：真实调用的测试有三个致命伤——慢（一次几十秒）、花钱、
结果不确定（同一个输入两次回答可能不同），而且 CI 机器上根本没有 key。

**做法**：包一层自定义 `BaseChatModel`。

- 录制：真实调用照常走，把 `(请求消息, 响应消息)` 追加到 JSONL；
- 回放：完全不联网，按顺序返回录制好的响应。

**为什么包模型、而不是 monkeypatch 或者拦 HTTP**：图、工具节点、事件桥
**完全不知道**自己在回放——它们看到的是同一个模型接口。所以除了"网络那一跳"，
所有真实代码路径都被测到了。打补丁的方案等于在跑另一套代码。

**一条硬规则：回放严格按顺序匹配。** 图的行为一变（比如多调了一次模型、
少调一次），回放就会在越界时报错——这不是缺陷，而是它最有用的地方：
提示词、图结构、工具清单的改动，都会在这里露出马脚。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, ConfigDict, Field

from ..store.db import now_iso


class TraceExhausted(RuntimeError):
    """回放文件里的调用用完了，但图还想再调一次。"""


class TraceRecorder(BaseModel):
    """把每次模型调用追加到 JSONL 文件。"""

    path: Path

    def append(
        self,
        *,
        model: str,
        request: Sequence[BaseMessage],
        response: BaseMessage,
    ) -> None:
        record: dict[str, Any] = {
            "model": model,
            "ts": now_iso(),
            "request": [message_to_dict(message) for message in request],
            "response": message_to_dict(response),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class TracePlayer(BaseModel):
    """按顺序回放录制好的响应。"""

    path: Path
    records: list[dict[str, Any]] = Field(default_factory=list)
    index: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> TracePlayer:
        trace = Path(path)
        if not trace.exists():
            msg = f"回放文件不存在：{trace}（先用 AGENT_TRACE_PATH 录一次）"
            raise FileNotFoundError(msg)
        records = [
            json.loads(line)
            for line in trace.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            msg = f"回放文件是空的：{trace}"
            raise ValueError(msg)
        return cls(path=trace, records=records)

    @property
    def total(self) -> int:
        return len(self.records)

    def next(self) -> AIMessage:
        if self.index >= len(self.records):
            msg = (
                f"回放文件里只有 {self.total} 次模型调用，但这次运行要第 "
                f"{self.index + 1} 次：图或提示词跟录制时不一样了。"
                f"（文件：{self.path}）重新用 AGENT_TRACE_PATH 录一份，或者检查改动。"
            )
            raise TraceExhausted(msg)
        record = self.records[self.index]
        self.index += 1
        message = messages_from_dict([record["response"]])[0]
        if not isinstance(message, AIMessage):  # pragma: no cover - 录制文件被手改过
            msg = f"回放文件里的第 {self.index} 条不是 AI 消息：{type(message).__name__}"
            raise TraceExhausted(msg)
        return message


class RecordingChatModel(BaseChatModel):
    """真实模型外面套一层录制。调用完就把 `(请求, 响应)` 记下来。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: Any
    recorder: TraceRecorder
    model_name: str = "unknown"

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(self, tools: Any, **kwargs: Any) -> RecordingChatModel:
        """把工具清单转交给真实模型，录制包装本身不需要知道工具是什么。"""
        return self.model_copy(update={"inner": self.inner.bind_tools(tools, **kwargs)})

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        response = self.inner.invoke(messages, stop=stop, **kwargs)
        self.recorder.append(model=self.model_name, request=messages, response=response)
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ):
        response = await self.inner.ainvoke(messages, stop=stop, **kwargs)
        self.recorder.append(model=self.model_name, request=messages, response=response)
        return ChatResult(generations=[ChatGeneration(message=response)])


class ReplayChatModel(BaseChatModel):
    """回放模型：不联网、不要密钥，按顺序吐出录制好的响应。"""

    player: TracePlayer
    model_name: str = "replay"

    @property
    def _llm_type(self) -> str:
        return "replay"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ReplayChatModel:
        """工具清单在回放时没有意义：响应里已经带着录好的 `tool_calls` 了。"""
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        return ChatResult(generations=[ChatGeneration(message=self.player.next())])

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
