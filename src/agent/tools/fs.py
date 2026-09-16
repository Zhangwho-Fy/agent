"""文件系统工具：读文件、列目录。两个都是只读，自动执行。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.errors import PathEscapeError
from ..core.tool_spec import Tier
from .base import Tool, ToolContext, ToolResult, resolve_within, truncate


class ReadArgs(BaseModel):
    path: str = Field(description="相对工作区的文件路径")
    start_line: int = Field(default=1, ge=1, description="起始行号，从 1 开始")
    max_lines: int = Field(default=200, ge=1, le=2000, description="最多读取多少行")


class ListArgs(BaseModel):
    path: str = Field(default=".", description="相对工作区的目录路径")
    max_entries: int = Field(default=200, ge=1, le=2000, description="最多列出多少项")


async def read_file(args: ReadArgs, ctx: ToolContext) -> ToolResult:
    try:
        target = resolve_within(ctx.workspace, args.path)
    except PathEscapeError as exc:
        return ToolResult(ok=False, content=str(exc))

    if not target.exists():
        return ToolResult(ok=False, content=f"文件不存在：{args.path}")
    if target.is_dir():
        return ToolResult(ok=False, content=f"{args.path} 是目录，请用 fs_list")

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult(ok=False, content=f"读取失败：{exc}")

    lines = text.splitlines()
    start = args.start_line - 1
    window = lines[start : start + args.max_lines]
    body = "\n".join(f"{start + offset + 1:>5} | {line}" for offset, line in enumerate(window))
    if len(lines) > start + len(window):
        body += f"\n... 文件共 {len(lines)} 行，还有 {len(lines) - start - len(window)} 行未显示"

    content, truncated = truncate(body, ctx.output_limit_bytes)
    return ToolResult(ok=True, content=f"{args.path}\n{content}", truncated=truncated)


async def list_dir(args: ListArgs, ctx: ToolContext) -> ToolResult:
    try:
        target = resolve_within(ctx.workspace, args.path)
    except PathEscapeError as exc:
        return ToolResult(ok=False, content=str(exc))
    if not target.exists():
        return ToolResult(ok=False, content=f"目录不存在：{args.path}")
    if not target.is_dir():
        return ToolResult(ok=False, content=f"{args.path} 不是目录")

    entries = sorted(target.iterdir(), key=lambda entry: (not entry.is_dir(), entry.name))
    shown = entries[: args.max_entries]
    lines: list[str] = []
    for entry in shown:
        if entry.is_dir():
            lines.append(f"{entry.name}/")
        else:
            try:
                size = entry.stat().st_size
            except OSError:
                size = -1
            lines.append(f"{entry.name}  ({size} B)")
    if len(entries) > len(shown):
        lines.append(f"... 还有 {len(entries) - len(shown)} 项未列出")

    body = "\n".join(lines) or "(空目录)"
    content, truncated = truncate(body, ctx.output_limit_bytes)
    return ToolResult(ok=True, content=f"{args.path} 下的内容：\n{content}", truncated=truncated)


READ_TOOL = Tool(
    name="fs_read",
    description="读取工作区内某个文本文件（带行号），支持指定起止范围",
    tier=Tier.READ,
    args_model=ReadArgs,
    run=read_file,
)

LIST_TOOL = Tool(
    name="fs_list",
    description="列出工作区内某个目录的直接子项",
    tier=Tier.READ,
    args_model=ListArgs,
    run=list_dir,
)
