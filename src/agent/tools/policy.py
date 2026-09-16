"""工具的风险分级与执行决策。

分三级（设计文档里的约定）：

- `read`：只读且无副作用 → 自动执行
- `write`：会改动工作区（写文件、提交、跑测试脚手架）→ 需要人工审批
- `dangerous`：删除、提权、推送、访问工作区外 → 直接拒绝

阶段 1 只落地"自动执行 / 拒绝"两条路径，审批流在阶段 2 接 LangGraph 的 interrupt。
"""

from __future__ import annotations

import re
import shlex
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..core.tool_spec import Tier
from .base import Tool, resolve_within


class Decision(StrEnum):
    AUTO = "auto"
    APPROVAL = "approval"
    DENY = "deny"


class PolicyDecision(BaseModel):
    decision: Decision
    reason: str = ""


#: 危险命令：命中即拒绝。写成"模式 + 原因"，拒绝时把原因回填给模型，它才知道换什么做法。
DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[A-Za-z]*[rf][A-Za-z]*\s+)+", "递归/强制删除"),
    (r"\bsudo\b|\bsu\s+-", "提权"),
    (r"\bmkfs|\bdd\s+if=|>\s*/dev/(sd|nvme)", "写裸设备"),
    (r"\bgit\s+push\b", "推送远程仓库"),
    (r"\bgit\s+reset\s+--hard\b", "丢弃工作区改动"),
    (r"\bgit\s+clean\s+-[A-Za-z]*f", "强制清理未跟踪文件"),
    (r"\bchmod\s+(-R\s+)?777\b", "放开权限"),
    (r"\b(shutdown|reboot|systemctl)\b", "影响系统状态"),
    (r"\|\s*(sh|bash|zsh)\b", "把下载内容直接管道给 shell 执行"),
    (r"\b(curl|wget)\b.*\|\s*(sh|bash|zsh)\b", "下载即执行"),
    (r"(^|\s)>\s*/(?!tmp/|dev/null)", "写工作区外的绝对路径"),
)

#: 只读命令白名单：整条命令里每一段的第一 token 都在这里，才允许自动执行
READ_ONLY_COMMANDS: frozenset[str] = frozenset(
    {
        "ls",
        "cat",
        "head",
        "tail",
        "wc",
        "grep",
        "rg",
        "find",
        "pwd",
        "echo",
        "tree",
        "file",
        "stat",
        "diff",
        "sort",
        "uniq",
        "cut",
        "awk",
    }
)

#: 这些命令只有在子命令是只读形态时才算只读
READ_ONLY_SUBCOMMANDS: dict[str, frozenset[str]] = {
    "git": frozenset({"status", "log", "diff", "show", "branch", "remote", "describe", "blame"}),
}

_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\|")


class Policy:
    """判断一次工具调用该自动执行、等审批，还是直接拒绝。"""

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace.expanduser().resolve()

    @property
    def workspace(self) -> Path:
        return self._workspace

    def resolve_path(self, raw: str) -> Path:
        """工作区内路径解析，越界抛 `PathEscapeError`。"""
        return resolve_within(self._workspace, raw)

    def classify(self, tool: Tool, args: dict[str, Any]) -> PolicyDecision:
        if tool.tier is Tier.DANGEROUS:
            return PolicyDecision(decision=Decision.DENY, reason=f"{tool.name} 属于危险工具")

        if tool.name == "shell_exec":
            return self._classify_command(str(args.get("command", "")))

        if tool.tier is Tier.WRITE:
            return PolicyDecision(decision=Decision.APPROVAL, reason=f"{tool.name} 会修改工作区")

        return PolicyDecision(decision=Decision.AUTO)

    def _classify_command(self, command: str) -> PolicyDecision:
        stripped = command.strip()
        if not stripped:
            return PolicyDecision(decision=Decision.DENY, reason="空命令")

        for pattern, reason in DANGEROUS_PATTERNS:
            if re.search(pattern, stripped):
                return PolicyDecision(decision=Decision.DENY, reason=f"命中危险模式：{reason}")

        segments = [seg.strip() for seg in _SEGMENT_SPLIT.split(stripped) if seg.strip()]
        if segments and all(self._is_read_only_segment(seg) for seg in segments):
            return PolicyDecision(decision=Decision.AUTO)

        return PolicyDecision(
            decision=Decision.APPROVAL,
            reason="命令有副作用，需要人工确认",
        )

    @staticmethod
    def _is_read_only_segment(segment: str) -> bool:
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        # 去掉环境变量赋值前缀，例如 `FOO=1 grep ...`
        while tokens and "=" in tokens[0] and not tokens[0].startswith("-"):
            tokens = tokens[1:]
        if not tokens:
            return False

        head = Path(tokens[0]).name
        if head in READ_ONLY_COMMANDS:
            return True
        allowed_subcommands = READ_ONLY_SUBCOMMANDS.get(head)
        if allowed_subcommands is not None:
            return len(tokens) > 1 and tokens[1] in allowed_subcommands
        return False
