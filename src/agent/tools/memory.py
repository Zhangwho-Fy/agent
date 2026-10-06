"""`memory_write` / `memory_search`：跨会话记忆的两个入口。

设计见 `docs/design.md` 第 8.4 / 8.5 节。

- `memory_write` 语义上是**写操作**，但策略默认放行（见 `policy.py` 与 D42）：
  它可撤销、没有工作区外副作用；每次记一条偏好都弹审批会把功能废掉。
- 两个工具都是**普通工具**，所以审计、事件、幂等、录放全部免费复用现有流水线（D47）。
- 工具不认识 `memory.db`：库通过 `ctx.memory` 注入，和 `recall` 拿到的是同一种鸭子类型。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..core.tool_spec import Tier
from ..memory.cards import GLOBAL_SCOPE, MemoryKind, MemoryTrust
from ..memory.render import render_memories
from ..memory.store import MemoryRejected
from .base import Tool, ToolContext, ToolResult


class MemoryWriteArgs(BaseModel):
    content: str = Field(description="要记住的内容。写成一条自洽的陈述——脱离当前对话也能独立读懂。")
    kind: MemoryKind = Field(
        description=(
            "记忆类型：fact=项目事实 / preference=用户偏好 / decision=决策+理由 / "
            "thread=未完成事项 / episode=带因果的经历"
        )
    )
    key: str | None = Field(
        default=None,
        description=(
            "结构化定位（可选），如 deps.manager、decision.mcp。"
            "带同一个 key 的新记忆会让旧记忆失效（supersede），不要用它当分类标签。"
        ),
    )
    scope: Literal["workspace", "global"] = Field(
        default="workspace",
        description="workspace=只在这个项目里有用；global=与代码无关的用户偏好",
    )
    trust: MemoryTrust = Field(
        default=MemoryTrust.USER_STATED,
        description=(
            "信息来源：user_stated=用户明说 / inferred=你推断的 / "
            "from_workspace=从仓库文件读到的 / from_web=外部内容"
        ),
    )
    importance: float = Field(default=0.5, ge=0.0, le=1.0, description="重要性，检索时排序用")
    source_quote: str | None = Field(
        default=None, description="写下这条记忆所依据的原文（可选，用于事后核查）"
    )


async def _write(args: MemoryWriteArgs, ctx: ToolContext) -> ToolResult:
    if ctx.memory is None:
        return ToolResult(ok=False, content="本次运行没有接记忆库，这条没有写入")
    scope = GLOBAL_SCOPE if args.scope == "global" else (ctx.memory_scope or GLOBAL_SCOPE)
    try:
        card = await ctx.memory.write(
            scope=scope,
            kind=args.kind,
            content=args.content,
            key=args.key,
            trust=args.trust,
            importance=args.importance,
            source_session=ctx.session_id or None,
            source_turn=ctx.turn_id or None,
            source_quote=args.source_quote,
        )
    except MemoryRejected as exc:
        return ToolResult(ok=False, content=f"这条记忆没有写入：{exc}")
    suffix = f"；它让上一条同 key 的记忆（{card.supersedes}）失效了" if card.supersedes else ""
    return ToolResult(
        ok=True, content=f"已记住（{card.kind.value} / {card.scope}）：{card.content}{suffix}"
    )


class MemorySearchArgs(BaseModel):
    query: str = Field(description="要回忆什么，用自然语言写")
    limit: int = Field(default=5, ge=1, le=20, description="最多返回几条")
    scope: Literal["all", "workspace", "global"] = Field(
        default="all", description="记忆范围：all=本项目 + 全局偏好"
    )


async def _search(args: MemorySearchArgs, ctx: ToolContext) -> ToolResult:
    if ctx.memory is None:
        return ToolResult(ok=False, content="本次运行没有接记忆库")
    scopes = {
        "all": [ctx.memory_scope, GLOBAL_SCOPE],
        "workspace": [ctx.memory_scope],
        "global": [GLOBAL_SCOPE],
    }[args.scope]
    scopes = [scope for scope in dict.fromkeys(scopes) if scope]
    cards = await ctx.memory.search(args.query, scopes=scopes, limit=args.limit)
    if not cards:
        return ToolResult(ok=True, content=f"没有找到相关记忆（查询：{args.query}）")
    return ToolResult(ok=True, content=render_memories(cards), wrap="none")


WRITE_TOOL = Tool(
    name="memory_write",
    description=(
        "把值得跨会话保留的东西记下来：用户偏好、项目事实、决策与理由、未完成事项、"
        "带因果的经历。只在满足三条时写——跨会话有用、足够稳定、有明确来源；"
        "本轮任务的临时细节不要写。"
    ),
    tier=Tier.WRITE,
    args_model=MemoryWriteArgs,
    run=_write,
)

SEARCH_TOOL = Tool(
    name="memory_search",
    description=(
        "回忆跨会话保存的记忆：用户偏好、项目事实、之前的决策与理由、未完成事项。"
        "当用户提到'之前''上次''我们说过'，或你要做与既有决策相关的改动时，先搜一下。"
    ),
    tier=Tier.READ,
    args_model=MemorySearchArgs,
    run=_search,
)
