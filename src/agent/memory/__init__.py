"""用户记忆：跨会话的卡片库（写 / 读 / 整理）。

设计见 `docs/design.md` 第 8 节。对外只有两样东西：卡片模型与存储。
工具在 `agent.tools.memory`，装配在 `agent.graph.wiring`。
"""

from .cards import (
    GLOBAL_SCOPE,
    MemoryCard,
    MemoryKind,
    MemoryStatus,
    MemoryTrust,
    find_secrets,
    scope_for,
)
from .store import MemoryRejected, MemoryStore

__all__ = [
    "GLOBAL_SCOPE",
    "MemoryCard",
    "MemoryKind",
    "MemoryRejected",
    "MemoryStatus",
    "MemoryStore",
    "MemoryTrust",
    "find_secrets",
    "scope_for",
]
