"""`shell_exec`：在工作区里执行命令。

三条约束缺一不可：

1. **超时杀整个进程组**——只杀父进程会把 `sleep 100 &` 这类子进程留成孤儿；
2. **输出截断后再回填**——一条 `ls -R` 就能把上下文冲爆，既烧钱又挤掉有用信息；
3. **环境变量白名单**——不把 `AGENT_API_KEY` 之类的密钥泄漏给子进程。
"""

from __future__ import annotations

import asyncio
import os
import signal

from pydantic import BaseModel, Field

from ..core.errors import PathEscapeError
from ..core.tool_spec import Tier
from .base import Tool, ToolContext, ToolResult, resolve_within, truncate

#: 允许传给子进程的环境变量：够命令正常运行，又不含任何密钥
ENV_WHITELIST: tuple[str, ...] = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "USER",
    "SHELL",
)


class ExecArgs(BaseModel):
    command: str = Field(description="要执行的 shell 命令")
    cwd: str | None = Field(default=None, description="相对工作区的子目录，默认工作区根")
    timeout_s: float | None = Field(
        default=None, gt=0, le=600, description="超时秒数，默认取全局配置"
    )


def safe_env() -> dict[str, str]:
    """只透传白名单里的环境变量。"""
    return {name: os.environ[name] for name in ENV_WHITELIST if name in os.environ}


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # pragma: no cover - 进程已退出
        proc.kill()


async def run_command(args: ExecArgs, ctx: ToolContext) -> ToolResult:
    try:
        cwd = (
            resolve_within(ctx.workspace, args.cwd)
            if args.cwd
            else ctx.workspace.expanduser().resolve()
        )
    except PathEscapeError as exc:
        return ToolResult(ok=False, content=str(exc))

    if not cwd.is_dir():
        return ToolResult(ok=False, content=f"工作目录不存在：{args.cwd}")

    timeout_s = args.timeout_s or ctx.timeout_s
    proc = await asyncio.create_subprocess_shell(
        args.command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # 独立进程组，超时才能整组杀掉
        env=safe_env(),
    )

    try:
        async with asyncio.timeout(timeout_s):
            stdout, _ = await proc.communicate()
    except TimeoutError:
        _kill_process_group(proc)
        await proc.wait()
        return ToolResult(
            ok=False,
            content=f"命令超时（{timeout_s:g}s），已终止整个进程组：{args.command}",
        )

    output = stdout.decode("utf-8", errors="replace")
    body, truncated = truncate(output, ctx.output_limit_bytes)
    header = f"$ {args.command}\n(exit={proc.returncode})"
    return ToolResult(
        ok=proc.returncode == 0,
        content=f"{header}\n{body}",
        truncated=truncated,
        exit_code=proc.returncode,
    )


EXEC_TOOL = Tool(
    name="shell_exec",
    description=(
        "在工作区里执行 shell 命令并返回输出（含退出码）。"
        "只读命令（ls/cat/grep/git status 等）会自动执行，有副作用的命令需要人工确认，"
        "危险命令会被直接拒绝。"
    ),
    tier=Tier.WRITE,  # 兜底等级：无法识别时按"有副作用"处理
    args_model=ExecArgs,
    run=run_command,
)
