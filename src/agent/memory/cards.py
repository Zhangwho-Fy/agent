"""记忆卡片：类型、信任、状态与模型。

设计见 `docs/design.md` 第 8.3 节。三条不能丢的约定：

1. **卡片 = 自洽段落 + JSON 键值 + 结构化 `key`**（D38）。三种都留着不是冗余：
   段落保证脱离原会话也能读懂，键值支持部分更新，`key` 用来找"同一条事实"。
2. **信任按来源定**（D40）：`user_stated` 才当参考，其余按数据、检索时用 `<untrusted>` 包裹。
3. **更新靠 supersede 链**（D39），不 UPDATE、不硬删。
"""

from __future__ import annotations

import re
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: 与代码无关的用户偏好落在 global；其余按工作区隔离（D49）
GLOBAL_SCOPE = "global"


class MemoryKind(StrEnum):
    """只写这五类，超出的一律留在会话里（D43）。"""

    FACT = "fact"  # 项目事实
    PREFERENCE = "preference"  # 用户偏好
    DECISION = "decision"  # 决策 + 理由
    THREAD = "thread"  # 未完成事项
    EPISODE = "episode"  # 带因果的经历


class MemoryTrust(StrEnum):
    USER_STATED = "user_stated"
    INFERRED = "inferred"
    FROM_WORKSPACE = "from_workspace"
    FROM_WEB = "from_web"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


def scope_for(workspace: Path) -> str:
    """工作区 scope：用 realpath，避免软链接绕出隔离（同 6.1 的边界做法）。"""
    return f"workspace:{workspace.expanduser().resolve()}"


#: 密钥形态。命中的内容**拒绝入库**——记忆会长期留在库里，写进去就等于泄漏。
SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{8,}"),
    re.compile(r"\bgho_[A-Za-z0-9]{8,}"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{8,}"),
)


def find_secrets(text: str) -> list[str]:
    """返回命中的密钥形态（用于日志，不回显原文）。"""
    return [pattern.pattern for pattern in SECRET_PATTERNS if pattern.search(text)]


class MemoryCard(BaseModel):
    """一条长期记忆。不可变：改写走 supersede，不原地修改（D39）。"""

    model_config = ConfigDict(frozen=True)

    id: str
    owner: str = "local"
    scope: str
    kind: MemoryKind
    key: str | None = None
    content: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    trust: MemoryTrust = MemoryTrust.USER_STATED
    status: MemoryStatus = MemoryStatus.ACTIVE
    importance: float = 0.5
    valid_from: str | None = None
    valid_to: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None
    source_session: str | None = None
    source_turn: str | None = None
    source_event_seq: int | None = None
    source_quote: str | None = None
    created_at: str = ""
    updated_at: str = ""
    last_used_at: str | None = None
    use_count: int = 0

    @field_validator("content")
    @classmethod
    def _content_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content 不能为空")
        return value

    @field_validator("importance")
    @classmethod
    def _importance_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("importance 必须在 0~1 之间")
        return value
