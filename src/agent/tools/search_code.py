"""`search_code`：把代码检索接成工具（R0，见 `docs/design.md` 第 9.3 节）。

它比 `fs_list` + `fs_read` 省上下文的地方在于：**索引在上下文外读几千个文件，
只把命中的片段送进来**。所以它也是子 agent 那种"隔离"模式的最便宜替代品（D33）。

输出是 L0 形态（9.4）：文件、行号、命中来源，加片段的前几行。要看全文，模型自己
再 `fs_read` 那个行区间——把"要不要深入"的决定权留给它。

工具不认识索引与数据库：装配层把 `SearchService.search` 包成 `ctx.fetch_search`
注入进来（和 `recall` 的 `fetch_tool_call` 是同一个模式）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.tool_spec import Tier
from .base import Tool, ToolContext, ToolResult, truncate

#: 每条命中默认给几行。给多了就把"分层"的意义抵消了。
MAX_SNIPPET_LINES = 6


class SearchCodeArgs(BaseModel):
    query: str = Field(
        description=(
            "要搜什么。自然语言和标识符都行：「路径越界是怎么拦的」和「resolve_within」都能命中。"
        )
    )
    limit: int = Field(default=5, ge=1, le=20, description="最多几条命中")
    path_prefix: str | None = Field(
        default=None, description="只在某个相对路径下搜，例如 src/agent/tools"
    )


def _format_hits(hits: list[Any]) -> str:
    blocks: list[str] = [f"命中 {len(hits)} 条（词法 + 向量 + RRF 融合）："]
    for index, hit in enumerate(hits, start=1):
        sources = ",".join(getattr(hit, "sources", ()) or ()) or "?"
        blocks.append(f"{index}. {hit.path}:{hit.start_line}-{hit.end_line}  [{sources}]")
        for line in str(hit.content).splitlines()[:MAX_SNIPPET_LINES]:
            blocks.append(f"   {line}")
        blocks.append("   …（要看全文用 fs_read 读这个行区间）")
    return "\n".join(blocks)


async def _search_code(args: SearchCodeArgs, ctx: ToolContext) -> ToolResult:
    if ctx.fetch_search is None:
        return ToolResult(
            ok=False,
            content="本次运行没有接检索索引；改用 fs_list + fs_read 找代码。",
        )
    hits = await ctx.fetch_search(args.query, args.limit, args.path_prefix)
    if not hits:
        return ToolResult(ok=True, content=f"没有命中（查询：{args.query}）")
    content, truncated = truncate(_format_hits(list(hits)), ctx.output_limit_bytes)
    return ToolResult(ok=True, content=content, truncated=truncated)


SEARCH_CODE_TOOL = Tool(
    name="search_code",
    description=(
        "在代码库里做混合检索（词法 + 语义），返回文件、行号和命中片段。"
        "找函数定义、看某件事在哪儿实现、按自然语言描述找代码时，先用它，"
        "比 fs_list 逐层翻目录省得多；拿到行号后再用 fs_read 读全文。"
    ),
    tier=Tier.READ,
    args_model=SearchCodeArgs,
    run=_search_code,
)
