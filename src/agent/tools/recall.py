"""`recall`：把被压缩掉的工具原文取回来。

D30 的前提：压缩是**可逆**的——上下文里只留指针，原文一直在 `tool_calls` 表里。
没有这条通道，指针化就只是"有损但假装无损"。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.tool_spec import Tier
from ..store import repo
from .base import Tool, ToolContext, ToolResult, truncate


class RecallArgs(BaseModel):
    call_id: str = Field(description="要取回的工具调用 id，见 <compressed call_id=…>")


async def recall(args: RecallArgs, ctx: ToolContext) -> ToolResult:
    if ctx.db is None:
        return ToolResult(ok=False, content="这个运行没有接会话库，取不回原文")
    row = await repo.get_tool_call(ctx.db, args.call_id)
    if row is None:
        return ToolResult(ok=False, content=f"没有这条工具调用记录：{args.call_id}")
    original = row["result"]
    if not original:
        return ToolResult(
            ok=False,
            content=f"这条调用没有留下输出（状态：{row['status']}，工具：{row['name']}）",
        )
    content, truncated = truncate(str(original), ctx.output_limit_bytes)
    return ToolResult(
        ok=True,
        content=f"call_id={args.call_id}（{row['name']}）的原始输出：\n{content}",
        truncated=truncated,
    )


RECALL_TOOL = Tool(
    name="recall",
    description="取回某次工具调用的原始输出全文（上下文里被压缩成指针的那些）",
    tier=Tier.READ,
    args_model=RecallArgs,
    run=recall,
)
