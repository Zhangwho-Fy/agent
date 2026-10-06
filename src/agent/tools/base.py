"""工具的公共契约：上下文、结果、路径边界与输出截断。

对照 C++：`ToolContext` 相当于执行器构造时注入的运行时环境（工作区、超时、限额），
工具实现不自己去读全局配置——这样测试里塞一个临时目录就能跑。
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ..core.errors import PathEscapeError
from ..core.tool_spec import Tier


class ToolContext(BaseModel):
    """一次执行的全部环境信息。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Path
    timeout_s: float = 60.0
    output_limit_bytes: int = 8192
    #: 按 call_id 取回工具调用记录的回调（只有 `recall` 用得上，可以不传）。
    #: 由装配层注入 store 的函数——**工具不认识数据库**，它们只知道"有个取回原文的入口"。
    fetch_tool_call: Callable[[str], Awaitable[Any]] | None = None
    #: 代码检索回调（只有 `search_code` 用得上）。签名：
    #: `(query, limit, path_prefix) -> list[SearchHit]`。同样由装配层注入，
    #: 工具不认识索引、更不认识 sqlite（见 9.3）。
    fetch_search: Callable[..., Awaitable[Any]] | None = None
    #: 长期记忆库（只有 memory_write / memory_search 用得上）。鸭子类型：
    #: 工具只调 `write()` / `search()`，不 import `agent.memory`（见 8.7）。
    memory: Any | None = None
    #: 本会话工作区对应的记忆 scope（`workspace:<realpath>`）
    memory_scope: str = ""
    #: 会话与轮次，写入记忆时当出处（8.4 的"有明确来源"）
    session_id: str = ""
    turn_id: str = ""


class ToolResult(BaseModel):
    """工具执行结果。`content` 是要回填给模型的文本（已截断）。"""

    ok: bool
    content: str
    truncated: bool = False
    exit_code: int | None = None
    duration_ms: int = 0
    #: 进上下文时的包装方式（见 core/guard.py）：
    #: - `untrusted`：外部内容，包 `<untrusted source=…>`（默认）
    #: - `none`：工具自己已经包好了。技能正文走这条——它是**操作说明**，
    #:   不是"仅供参考的资料"，所以用 `<skill>` 块（D12）
    wrap: Literal["untrusted", "none"] = "untrusted"


def resolve_within(workspace: Path, raw: str) -> Path:
    """把用户/模型给的路径解析成工作区内的绝对路径。

    关键点：**先 resolve 再判边界**。只做字符串前缀判断会被 `../` 和软链接绕过；
    解析之后软链接已经展开，再比较父目录才可靠。
    """
    root = workspace.expanduser().resolve()
    candidate = (root / raw).expanduser()
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise PathEscapeError(f"路径落在工作区之外：{raw} → {resolved}")
    return resolved


def truncate(text: str, limit_bytes: int) -> tuple[str, bool]:
    """超过限额时保留头尾各一半，中间标注省略了多少字节。

    为什么保留尾部：报错信息、测试结论通常都在末尾，砍掉尾巴等于把最有用的部分丢了。
    """
    data = text.encode("utf-8")
    if len(data) <= limit_bytes:
        return text, False

    half = max(limit_bytes // 2, 1)
    head = data[:half].decode("utf-8", errors="ignore")
    tail = data[-half:].decode("utf-8", errors="ignore")
    omitted = len(data) - len(head.encode()) - len(tail.encode())
    return f"{head}\n... [省略 {omitted} 字节] ...\n{tail}", True


class Tool:
    """一个工具：对模型可见的描述 + 真实的异步实现。"""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        tier: Tier,
        args_model: type[BaseModel],
        run: Callable[[Any, ToolContext], Awaitable[ToolResult]],
    ) -> None:
        self.name = name
        self.description = description
        self.tier = tier
        self.args_model = args_model
        self._run = run

    @property
    def parameters(self) -> dict[str, Any]:
        """给模型看的 JSON Schema，直接从 pydantic 模型导出，两边不会不一致。"""
        return self.args_model.model_json_schema()

    async def run(self, raw_args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """校验参数后执行。参数不合法属于可预期失败，编码成结果回填给模型。"""
        started = time.perf_counter()
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult(ok=False, content=f"参数不合法：{exc.errors(include_url=False)}")

        result = await self._run(args, ctx)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result.model_copy(update={"duration_ms": elapsed_ms})
