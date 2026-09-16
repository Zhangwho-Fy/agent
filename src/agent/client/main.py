"""CLI 入口（Typer）。

当前只有 `version` / `doctor` / `config` 三个命令，够验证环境与配置。
`run` / `serve` / `sessions` / `replay` / `eval` 在后续阶段补齐。

对照 C++：Typer 相当于给你一个自动生成 `--help` 的参数解析器，
而参数定义就是函数签名本身——类型标注既是文档也是校验。
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .. import __version__
from ..config import Settings
from ..logging import configure_logging

app = typer.Typer(
    help="本地代码库助手 Agent",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

#: doctor 需要确认能导入的第三方依赖
REQUIRED_MODULES = ("pydantic", "pydantic_settings", "openai", "fastapi", "typer", "rich")


@app.callback()
def main(
    log_level: str | None = typer.Option(None, "--log-level", help="日志级别，默认取配置里的值"),
) -> None:
    """所有子命令共用的入口：先把日志配好。"""
    settings = Settings()
    configure_logging(log_level or settings.log_level)


@app.command()
def version() -> None:
    """显示版本号。"""
    try:
        installed = importlib.metadata.version("agent")
    except importlib.metadata.PackageNotFoundError:
        installed = __version__
    console.print(f"agent {installed}")


@app.command()
def doctor() -> None:
    """体检：Python 版本、依赖、配置、密钥。"""
    problems: list[str] = []
    table = Table(title="环境体检")
    table.add_column("检查项")
    table.add_column("结果")

    current = sys.version_info
    py_ok = (current.major, current.minor) >= (3, 12)
    table.add_row(
        "Python",
        f"{current.major}.{current.minor}.{current.micro}" + ("" if py_ok else "  需要 >= 3.12"),
    )
    if not py_ok:
        problems.append("Python 版本过低")

    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
        except ImportError:
            table.add_row(f"依赖 {module}", "缺失")
            problems.append(f"缺少依赖 {module}")
        else:
            table.add_row(f"依赖 {module}", "OK")

    settings: Settings | None = None
    try:
        settings = Settings()
    except Exception as exc:  # pydantic 的校验错误信息已经很详细，直接展示
        table.add_row("配置加载", f"失败：{exc}")
        problems.append("配置加载失败")
    else:
        table.add_row("配置加载", "OK")
        table.add_row("模型", f"{settings.provider} / {settings.model}")
        table.add_row("工作区", str(settings.resolved_workspace))
        if settings.provider == "replay":
            table.add_row("密钥", "不需要（回放模式）")
        elif settings.api_key:
            table.add_row("密钥", "已设置")
        else:
            table.add_row("密钥", "未设置  需要在 .env 里填 AGENT_API_KEY")
            problems.append("未设置 AGENT_API_KEY")

    console.print(table)
    if problems:
        console.print("[red]体检未通过：[/red]" + "；".join(problems))
        raise typer.Exit(code=1)
    console.print("[green]体检通过[/green]")


@app.command("config")
def show_config() -> None:
    """显示解析后的配置（敏感字段只显示长度）。"""
    settings = Settings()
    table = Table(title="配置")
    table.add_column("字段")
    table.add_column("值")
    for key, value in settings.describe().items():
        table.add_row(key, value)
    console.print(table)


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="交给 agent 的任务")],
    workspace: Annotated[
        Path | None,
        typer.Option("--workspace", "-w", help="工作区目录，默认取配置里的 AGENT_WORKSPACE"),
    ] = None,
) -> None:
    """跑一次任务：驱动 LangGraph 图并在终端流式显示。"""
    settings = Settings()
    if workspace is not None:
        settings.workspace = workspace
    raise typer.Exit(code=asyncio.run(_run_once(prompt, settings)))


async def _run_once(prompt: str, settings: Settings) -> int:
    """构建图并跑一轮。

    重依赖（langgraph / langchain）在这里才导入，这样 `agent doctor`、`agent config`
    在没有装齐依赖时也能用——诊断工具本身不该依赖被诊断的东西。
    """
    from ..core.bus import EventBus
    from ..core.events import Event, EventType
    from ..core.ids import new_id
    from ..core.reliability import EventEmitter
    from ..graph.bridge import stream_turn
    from ..graph.builder import build_graph
    from ..models.factory import build_chat_model
    from ..tools.base import ToolContext
    from ..tools.policy import Policy
    from ..tools.registry import default_registry

    ctx = ToolContext(
        workspace=settings.resolved_workspace,
        timeout_s=settings.tool_timeout_s,
        output_limit_bytes=settings.output_limit_bytes,
    )
    if not ctx.workspace.is_dir():
        console.print(f"[red]工作区不存在：{ctx.workspace}[/red]")
        return 2

    registry = default_registry()
    policy = Policy(ctx.workspace)
    try:
        model = build_chat_model(settings)
    except Exception as exc:
        console.print(f"[red]模型初始化失败：{exc}[/red]")
        return 1

    session_id = new_id("sess")
    turn_id = new_id("turn")
    emitter = EventEmitter(session_id, EventBus())

    def render(event: Event) -> None:
        data = event.data
        if event.type is EventType.TEXT_DELTA:
            console.print(str(data.get("text", "")), end="", markup=False, highlight=False)
        elif event.type is EventType.TOOL_CALL:
            args = json.dumps(data.get("args", {}), ensure_ascii=False)
            console.print(f"\n[dim]→ {data.get('name')} {args}[/dim]")
        elif event.type is EventType.TOOL_RESULT:
            status = data.get("status")
            extra = f" {data['duration_ms']}ms" if data.get("duration_ms") else ""
            style = "dim" if status == "ok" else "yellow"
            console.print(f"[{style}]  ← {status}{extra}[/{style}]")
        elif event.type is EventType.ERROR:
            console.print(f"\n[red]错误：{data.get('message')}[/red]")

    emitter.on_event = render

    graph = build_graph(model=model, registry=registry, policy=policy, ctx=ctx, emitter=emitter)
    tool_names = ", ".join(registry.names)
    console.print(f"[dim]工作区 {ctx.workspace}｜模型 {settings.model}｜工具 {tool_names}[/dim]")
    console.print(f"[bold]你[/bold] {prompt}")
    console.print("[bold]agent[/bold] ", end="")

    await stream_turn(
        graph=graph,
        prompt=prompt,
        emitter=emitter,
        session_id=session_id,
        turn_id=turn_id,
    )
    console.print()
    return 0


if __name__ == "__main__":
    app()
