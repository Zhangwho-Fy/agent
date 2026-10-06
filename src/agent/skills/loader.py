"""技能目录：扫描、渲染、加载。

设计见 `docs/design.md` 第 7.2 节。三条要点：

1. **目录进 L2，正文走 tool result**（`<skill>` 块）——正文永远不会进系统提示词（D9），
   否则每轮重发，正好把渐进式披露省下来的钱花回去。
2. **信任按来源定**（D13）：内置技能随包发布，可信；工作区里的
   `<workspace>/.agent/skills/` 可能来自第三方仓库，只当参考资料。
3. **目录渲染是安全边界**（D14）：工作区里的文本要进系统提示词，必须 XML 转义、
   压成单行、截断、固定排序。不转义的话，一个 `</skill><skill …>` 就逃出了格子，
   等于把任意文本塞进最高信任通道。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..logging import log_extra

logger = logging.getLogger(__name__)

#: 内置技能目录：随包发布，**在工作区之外**，所以 `fs_read` 够不着，只能走 skill_load。
BUILTIN_DIR = Path(__file__).resolve().parent

#: 项目级技能目录（相对工作区）。
WORKSPACE_SUBDIR = ".agent/skills"

#: 目录条目上限（D15）。超出的只列名字——不静默截断。
MAX_SKILLS = 30

#: 单条 description 的字符上限（D14）。description 是路由条件，不是文档。
DESCRIPTION_LIMIT = 200

#: 超出上限时最多列几个名字，避免 `<more>` 自己长成新的膨胀源。
MAX_OVERFLOW_NAMES = 20

#: 技能名：路径安全 + 可读。中文允许（面向中文用户），但**永远按目录扫描结果查找**，
#: 不拿它拼路径。
NAME_PATTERN = re.compile(r"^[\w\-\u4e00-\u9fff]{1,64}$")

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass(frozen=True)
class Skill:
    """一个技能的元数据 + 正文位置。"""

    name: str
    description: str
    scope: Literal["builtin", "workspace"]
    path: Path
    chars: int

    @property
    def token_hint(self) -> int:
        """正文的粗略 token 数（中文约 1.6 字符/token），给模型一个"值不值得加载"的提示。"""
        return max(100, round(self.chars / 1.6 / 100) * 100)


def _resolve(workspace: Path) -> Path:
    return workspace.expanduser().resolve()


def _escape(text: str) -> str:
    """XML 转义。目录里的每个来自磁盘的字符串都要过这一步。"""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def one_line(text: str) -> str:
    """压成单行、剥控制字符、截断。

    description 会被拼进系统提示词，任何换行都等于"能自己开新的一行写指令"。
    """
    flat = _CONTROL.sub(" ", text.replace("\n", " ").replace("\r", " ").replace("\t", " "))
    flat = re.sub(r"\s+", " ", flat).strip()
    if len(flat) > DESCRIPTION_LIMIT:
        flat = flat[: DESCRIPTION_LIMIT - 1] + "…"
    return flat


def _parse(path: Path) -> tuple[str, str, str] | None:
    """解析 `SKILL.md` 的 frontmatter，返回 (name, description, body)。

    只认单行 `key: value`：frontmatter 就这两个字段，为了它引一个 YAML 依赖不划算。
    解析不出 name/description 就整个跳过——一个坏技能不该把目录搞坏。
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - 权限之类
        logger.warning("技能读取失败", extra=log_extra(skill_file=str(path), error=str(exc)))
        return None
    if not text.startswith("---"):
        logger.warning("技能缺少 frontmatter，跳过", extra=log_extra(skill_file=str(path)))
        return None

    end = text.find("\n---", 3)
    if end < 0:
        logger.warning("技能 frontmatter 没有结束标记，跳过", extra=log_extra(skill_file=str(path)))
        return None

    fields: dict[str, str] = {}
    for line in text[3:end].splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()

    name = fields.get("name", "").strip()
    description = one_line(fields.get("description", ""))
    if not NAME_PATTERN.match(name) or not description:
        logger.warning("技能名不合规或缺 description，跳过", extra=log_extra(skill_file=str(path)))
        return None
    return name, description, text[end + 4 :].lstrip("\n")


def scan_skills(workspace: Path) -> list[Skill]:
    """扫描内置 + 项目级技能。**实时读盘**，不走缓存（`skill_load` 用这个）。"""
    roots: list[tuple[Literal["builtin", "workspace"], Path]] = [
        ("builtin", BUILTIN_DIR),
        ("workspace", _resolve(workspace) / WORKSPACE_SUBDIR),
    ]
    found: dict[str, Skill] = {}
    for scope, root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not child.is_dir():
                continue
            path = child / "SKILL.md"
            if not path.is_file():
                continue
            parsed = _parse(path)
            if parsed is None:
                continue
            name, description, body = parsed
            found[name] = Skill(
                name=name,
                description=description,
                scope=scope,
                path=path,
                chars=len(body),
            )
    # 固定排序：scope → name。**不能按 mtime 或相关度**，否则前缀和回放夹具都不稳（D14）。
    return sorted(found.values(), key=lambda skill: (skill.scope != "builtin", skill.name))


def render_catalog(skills: list[Skill]) -> str:
    """渲染成 `<skills>` 块。超出上限的只留名字，并且不静默（D15）。"""
    shown = skills[:MAX_SKILLS]
    overflow = skills[MAX_SKILLS:]
    lines = ['<skills note="下面是可用技能目录。需要时用 skill_load 读取完整内容">']
    for skill in shown:
        lines.append(
            f'  <skill name="{_escape(skill.name)}" scope="{skill.scope}"'
            f' size="~{skill.token_hint}t">{_escape(skill.description)}</skill>'
        )
    if overflow:
        names = " ".join(_escape(skill.name) for skill in overflow[:MAX_OVERFLOW_NAMES])
        lines.append(f'  <more count="{len(overflow)}">{names}</more>')
        logger.warning(
            "技能目录超上限，超出的只列名字",
            extra=log_extra(limit=MAX_SKILLS, overflow=len(overflow)),
        )
    lines.append("</skills>")
    return "\n".join(lines)


#: 会话视图快照：workspace → (turn_id, 目录文本)。
#:
#: **按轮冻结**（D11）：同一个 `turn_id` 内返回同一份，换一轮才重扫——所以一轮之内
#: 新建的技能不会中途改变系统提示，而下一轮立刻能看到它。`skill_load` 走
#: `scan_skills()` 实时读盘，不受这里影响，刚建的技能本轮照样能按名字加载。
_CATALOG_CACHE: dict[Path, tuple[str, str]] = {}


def catalog_text(workspace: Path, turn_id: str = "") -> str:
    """当前这一轮的技能目录文本。"""
    key = _resolve(workspace)
    cached = _CATALOG_CACHE.get(key)
    if cached is None or cached[0] != turn_id:
        cached = (turn_id, render_catalog(scan_skills(key)))
        _CATALOG_CACHE[key] = cached
    return cached[1]


def find_skill(workspace: Path, name: str) -> Skill | None:
    """按名字找技能（实时扫描）。找到的不是路径拼接结果，所以不存在穿越问题。"""
    return next((skill for skill in scan_skills(workspace) if skill.name == name), None)
