"""shell_exec 的三条约束：正常执行、超时、输出截断、环境变量白名单。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent.tools.base import ToolContext
from agent.tools.shell import EXEC_TOOL


def make_ctx(workspace: Path, *, timeout_s: float = 10.0, limit: int = 8192) -> ToolContext:
    return ToolContext(workspace=workspace, timeout_s=timeout_s, output_limit_bytes=limit)


async def test_runs_command_and_reports_exit_code(tmp_path: Path) -> None:
    result = await EXEC_TOOL.run({"command": "echo hello"}, make_ctx(tmp_path))

    assert result.ok is True
    assert result.exit_code == 0
    assert "hello" in result.content
    assert result.duration_ms >= 0


async def test_non_zero_exit_is_not_an_exception(tmp_path: Path) -> None:
    result = await EXEC_TOOL.run({"command": "exit 3"}, make_ctx(tmp_path))

    assert result.ok is False
    assert result.exit_code == 3
    assert "exit=3" in result.content


async def test_timeout_kills_the_process_group(tmp_path: Path) -> None:
    started = time.perf_counter()

    result = await EXEC_TOOL.run({"command": "sleep 30"}, make_ctx(tmp_path, timeout_s=0.5))

    elapsed = time.perf_counter() - started
    assert result.ok is False
    assert "超时" in result.content
    assert elapsed < 5, "超时后必须立刻返回，不能等命令自己结束"


async def test_large_output_is_truncated(tmp_path: Path) -> None:
    result = await EXEC_TOOL.run({"command": "seq 1 5000"}, make_ctx(tmp_path, limit=512))

    assert result.truncated is True
    assert "省略" in result.content
    assert len(result.content.encode()) < 1024


async def test_secrets_are_not_passed_to_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_API_KEY", "sk-should-not-leak")

    result = await EXEC_TOOL.run({"command": "env"}, make_ctx(tmp_path))

    assert result.ok is True
    assert "sk-should-not-leak" not in result.content


async def test_cwd_outside_workspace_is_rejected(tmp_path: Path) -> None:
    result = await EXEC_TOOL.run({"command": "pwd", "cwd": "../.."}, make_ctx(tmp_path))

    assert result.ok is False
    assert "工作区之外" in result.content
