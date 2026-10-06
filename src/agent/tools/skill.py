"""技能工具：按需加载正文、把新技能写进项目技能目录。

为什么正文走 tool 结果而不是塞进系统提示词：见 `docs/design.md` 7.2（D12）
——系统提示每轮现拼，正文进去就是每轮重发；tool 通道天然带审计、事件与回放。

为什么是 `<skill>` 而不是 `<untrusted>`：技能正文是**操作说明**。所以这两个工具返回
`wrap="none"`，由自己包容器（`<skill name=… scope=…>`），L1 里写了这条例外。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.errors import PathEscapeError
from ..core.tool_spec import Tier
from ..skills.loader import NAME_PATTERN, WORKSPACE_SUBDIR, find_skill, one_line, scan_skills
from .base import Tool, ToolContext, ToolResult, resolve_within, truncate

#: 正文单次加载的上限（字符）。超过就不是"一个技能"，该拆子文档了。
MAX_BODY_CHARS = 20_000


class SkillLoadArgs(BaseModel):
    name: str = Field(description="技能名，见系统提示里的技能目录")
    file: str | None = Field(
        default=None, description="技能目录里的子文档（如 reference.md），第三层细则走这个"
    )


def _skill_block(*, name: str, scope: str, body: str) -> str:
    """包成 `<skill>` 块。name 已过 NAME_PATTERN（不含引号和尖括号），scope 是我们定的。"""
    return f'<skill name="{name}" scope="{scope}">\n{body}\n</skill>'


async def load_skill(args: SkillLoadArgs, ctx: ToolContext) -> ToolResult:
    skill = find_skill(ctx.workspace, args.name)
    if skill is None:
        available = ", ".join(item.name for item in scan_skills(ctx.workspace)) or "（无）"
        return ToolResult(ok=False, content=f"没有技能 {args.name}。可用：{available}")

    target = skill.path
    if args.file:
        try:
            # 作用域根从工作区换成这个技能自己的目录：内置技能在工作区外，fs_read 够不着。
            target = resolve_within(skill.path.parent, args.file)
        except PathEscapeError as exc:
            return ToolResult(ok=False, content=str(exc))
        if not target.is_file():
            return ToolResult(ok=False, content=f"技能 {args.name} 里没有这个文件：{args.file}")

    try:
        body = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult(ok=False, content=f"读取技能失败：{exc}")

    truncated = len(body) > MAX_BODY_CHARS
    if truncated:
        body = body[:MAX_BODY_CHARS] + "\n... [技能正文过长，已截断] ..."
    body, truncated_by_bytes = truncate(body, ctx.output_limit_bytes)
    return ToolResult(
        ok=True,
        content=_skill_block(name=skill.name, scope=skill.scope, body=body),
        truncated=truncated or truncated_by_bytes,
        wrap="none",
    )


class SkillCreateArgs(BaseModel):
    name: str = Field(description="技能名：字母、数字、下划线、连字符或中文，1~64 位")
    description: str = Field(description="一句话说清'什么时候该用我'，不超过 200 字")
    body: str = Field(description="技能正文（Markdown）：角色、核心原则、禁止清单")


async def create_skill(args: SkillCreateArgs, ctx: ToolContext) -> ToolResult:
    """把新技能写进 `<workspace>/.agent/skills/<name>/SKILL.md`。

    tier=WRITE，所以一定会走人工审批——这是**有意的**：写入工作区是用户的动作。
    **不在这里刷新目录缓存**：按 D11，新技能要到下一个轮边界才进目录；但
    `skill_load` 走实时扫描，所以本轮照样能按名字加载它。
    """
    name = args.name.strip()
    if not NAME_PATTERN.match(name):
        return ToolResult(
            ok=False, content="技能名只能用字母、数字、下划线、连字符或中文，长度 1~64"
        )
    description = one_line(args.description)
    if not description:
        return ToolResult(ok=False, content="description 不能为空——它决定什么时候该用这个技能")
    body = args.body.strip()
    if not body:
        return ToolResult(ok=False, content="正文不能为空")

    relative = f"{WORKSPACE_SUBDIR}/{name}/SKILL.md"
    try:
        path = resolve_within(ctx.workspace, relative)
    except PathEscapeError as exc:  # pragma: no cover - NAME_PATTERN 已经挡住穿越
        return ToolResult(ok=False, content=str(exc))
    if path.exists():
        return ToolResult(ok=False, content=f"技能 {name} 已存在：{relative}，先想个别的名字")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return ToolResult(
        ok=True,
        content=f"已写入 {relative}。它会在下一轮出现在技能目录里，本轮可以直接 skill_load。",
    )


SKILL_LOAD_TOOL = Tool(
    name="skill_load",
    description="读取一个技能的完整正文（或它目录里的子文档）。技能目录在系统提示里",
    tier=Tier.READ,
    args_model=SkillLoadArgs,
    run=load_skill,
)

SKILL_CREATE_TOOL = Tool(
    name="skill_create",
    description="把一套可复用的做法写成新技能，存进项目的 .agent/skills/（需要人工批准）",
    tier=Tier.WRITE,
    args_model=SkillCreateArgs,
    run=create_skill,
)
