"""CLI 入口（Typer）。

命令分两类：

- **验环境**：`version` / `doctor` / `config`——不依赖 langgraph，缺依赖时也能跑
  （诊断工具本身不该依赖被诊断的东西）。
- **干活**：`run` 跑一轮任务、`sessions` 列会话、`replay` 按原顺序重放事件流（不调模型）。

`serve` / `eval` 在后续阶段（服务端在阶段 4）。

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
from typing import Annotated, Any

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
        table.add_row("会话库", str(settings.resolved_db_path))
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
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="接着已有会话跑（默认每次新建）"),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="自动批准所有写操作（无人值守时用）"),
    ] = False,
) -> None:
    """跑一次任务：驱动 LangGraph 图并在终端流式显示。"""
    settings = Settings()
    if workspace is not None:
        settings.workspace = workspace
    code = asyncio.run(_run_once(prompt, settings, session_id=session, auto_approve=yes))
    raise typer.Exit(code=code)


async def _run_once(
    prompt: str,
    settings: Settings,
    *,
    session_id: str | None = None,
    auto_approve: bool = False,
) -> int:
    """构建图并跑一轮：落库 → 驱动图（可能等审批）→ 收尾落库。

    重依赖（langgraph / langchain）在这里才导入，这样 `agent doctor`、`agent config`
    在没有装齐依赖时也能用。
    """
    from ..core.bus import EventBus
    from ..core.events import Event, EventType
    from ..core.ids import new_id
    from ..core.reliability import EventEmitter
    from ..graph.bridge import Approver, recursion_limit_for, stream_turn
    from ..graph.builder import build_graph
    from ..graph.checkpointer import open_checkpointer
    from ..models.factory import build_chat_model
    from ..store import repo
    from ..store.db import Database
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

    # ---- 打开会话库，并做一次崩溃恢复 ----
    db = Database(settings.resolved_db_path)
    db.connect()
    recovered = await repo.interrupt_running_turns(db)
    if recovered:
        console.print(f"[yellow]恢复：{recovered} 个没跑完的 turn 已标记为 interrupted[/yellow]")

    if session_id is None:
        session_id = new_id("sess")
        await repo.create_session(
            db,
            session_id=session_id,
            profile="code",
            workspace=str(ctx.workspace),
            title=prompt[:60],
        )
    elif await repo.get_session(db, session_id) is None:
        console.print(f"[red]会话不存在：{session_id}[/red]")
        return 2

    turn_id = new_id("turn")
    await repo.start_turn(db, turn_id=turn_id, session_id=session_id)
    await repo.append_message(
        db,
        session_id=session_id,
        seq=await repo.next_message_seq(db, session_id),
        role="user",
        content=prompt,
    )

    # 事件总线之后的每一条事件都先进库、再推送；seq 接着库里已有的往下排
    emitter = EventEmitter(
        session_id,
        EventBus(),
        start_seq=await repo.last_event_seq(db, session_id),
        sink=lambda event: repo.append_event(db, event),
    )

    streamed = False

    def render(event: Event) -> None:
        nonlocal streamed
        data = event.data
        if event.type is EventType.TEXT_DELTA:
            streamed = True
            console.print(str(data.get("text", "")), end="", markup=False, highlight=False)
        elif event.type is EventType.TEXT_DONE:
            # 逐字流已经打过了就不再重复；没有流式分片时（例如轮数用尽的收尾消息）
            # 在这里补打一次，否则终端上会是空白。
            if not streamed and data.get("text"):
                console.print(str(data["text"]), markup=False, highlight=False)
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

    graph = build_graph(
        model=model,
        registry=registry,
        policy=policy,
        ctx=ctx,
        emitter=emitter,
        max_tool_rounds=settings.max_tool_rounds,
        checkpointer=open_checkpointer(settings.resolved_db_path),
        db=db,
    )
    tool_names = ", ".join(registry.names)
    console.print(f"[dim]工作区 {ctx.workspace}｜模型 {settings.model}｜工具 {tool_names}[/dim]")
    console.print(f"[dim]会话 {session_id}[/dim]")
    console.print(f"[bold]你[/bold] {prompt}")
    console.print("[bold]agent[/bold] ", end="")

    approver: Approver = _make_approver(auto_approve)
    result = await stream_turn(
        graph=graph,
        prompt=prompt,
        emitter=emitter,
        session_id=session_id,
        turn_id=turn_id,
        recursion_limit=recursion_limit_for(settings.max_tool_rounds),
        approver=approver,
        approval_timeout_s=settings.approval_timeout_s,
    )
    console.print()

    await repo.append_message(
        db,
        session_id=session_id,
        seq=await repo.next_message_seq(db, session_id),
        role="assistant",
        content=result.text,
    )
    await repo.finish_turn(
        db,
        turn_id,
        status="done" if result.status == "done" else "failed",
        input_tokens=int(result.usage.get("input_tokens", 0)),
        output_tokens=int(result.usage.get("output_tokens", 0)),
    )
    await repo.touch_session(db, session_id)
    await db.close()

    console.print(f"[dim]用时 {result.duration_ms}ms｜token {result.usage or '无'}[/dim]")
    return 0 if result.status == "done" else 1


def _make_approver(auto_approve: bool) -> Any:
    """审批回调：交互式问答，`--yes` 时整批放行。

    注：这里的 `input()` 是阻塞调用，会挡住事件循环——对 CLI 无所谓（本来就在等用户），
    所以 `approval_timeout_s` 在人机交互场景下不会真正触发；超时是给服务端用的。
    """

    async def approve(requests: list[dict[str, Any]]) -> dict[str, bool]:
        decisions: dict[str, bool] = {}
        for request in requests:
            if auto_approve:
                console.print(f"[yellow]自动批准：{request['name']}[/yellow]")
                decisions[request["call_id"]] = True
                continue
            args = json.dumps(request.get("args", {}), ensure_ascii=False)
            console.print(f"\n[yellow]需要确认[/yellow] {request['name']} {args}")
            console.print(f"[dim]原因：{request.get('reason', '')}[/dim]")
            # 故意阻塞：CLI 本来就在等用户，而且这里换成线程池会踩 AGENTS.md 记的那个坑
            answer = input("  放行吗？[y/N] ").strip().lower()  # noqa: ASYNC250
            decisions[request["call_id"]] = answer in {"y", "yes", "是"}
        return decisions

    return approve


@app.command()
def sessions(
    limit: Annotated[int, typer.Option("--limit", "-n", help="最多列出多少条")] = 20,
) -> None:
    """列出最近的会话。"""
    raise typer.Exit(code=asyncio.run(_list_sessions(Settings(), limit)))


async def _list_sessions(settings: Settings, limit: int) -> int:
    from ..store import repo
    from ..store.db import Database

    db = Database(settings.resolved_db_path)
    if not db.path.exists():
        console.print("[yellow]还没有会话库，先跑一次 `agent run`[/yellow]")
        return 0
    db.connect()
    rows = await repo.list_sessions(db, limit=limit)
    if not rows:
        console.print("[yellow]还没有会话[/yellow]")
        return 0

    table = Table(title="会话")
    table.add_column("session_id")
    table.add_column("标题")
    table.add_column("最后活动")
    table.add_column("turn 数", justify="right")
    for row in rows:
        turns = await repo.list_turns(db, row["id"])
        table.add_row(row["id"], row["title"] or "-", row["updated_at"], str(len(turns)))
    console.print(table)
    await db.close()
    return 0


def format_event(event: Any) -> str:
    """把一条事件渲染成一行文本。纯函数，重放和测试都用它。"""
    data = event.data or {}
    kind = event.type.value
    if kind == "turn.started":
        return f"[{event.seq}] 你：{data.get('prompt', '')}"
    if kind == "text.delta":
        return f"[{event.seq}] 文本：{data.get('text', '')}"
    if kind == "text.done":
        return f"[{event.seq}] 回答完成：{data.get('text', '')}"
    if kind == "tool.call":
        args = json.dumps(data.get("args", {}), ensure_ascii=False)
        return f"[{event.seq}] → {data.get('name')} {args}"
    if kind == "approval.required":
        return f"[{event.seq}] ? 需审批 {data.get('name')}：{data.get('reason', '')}"
    if kind == "tool.result":
        return f"[{event.seq}] ← {data.get('status')}"
    if kind == "turn.done":
        return f"[{event.seq}] 结束 status={data.get('status')} usage={data.get('usage', {})}"
    if kind == "error":
        return f"[{event.seq}] 错误：{data.get('message')}"
    return f"[{event.seq}] {kind} {json.dumps(data, ensure_ascii=False)}"


@app.command()
def replay(
    session_id: Annotated[str, typer.Argument(help="会话 id（用 agent sessions 查）")],
    from_seq: Annotated[
        int, typer.Option("--from-seq", help="从第几条事件之后开始，断线续传的语义")
    ] = 0,
    raw: Annotated[bool, typer.Option("--raw", help="逐条显示，不合并连续的 text.delta")] = False,
) -> None:
    """按原顺序重放会话的事件流。**不调用模型**，纯读库。"""
    raise typer.Exit(code=asyncio.run(_replay(Settings(), session_id, from_seq, raw)))


async def _replay(settings: Settings, session_id: str, from_seq: int, raw: bool = False) -> int:
    from ..store import repo
    from ..store.db import Database

    db = Database(settings.resolved_db_path)
    if not db.path.exists():
        console.print(f"[red]没有会话库：{db.path}[/red]")
        return 1
    db.connect()
    session = await repo.get_session(db, session_id)
    if session is None:
        console.print(f"[red]会话不存在：{session_id}[/red]")
        return 1

    rows = await repo.list_events(db, session_id, after_seq=from_seq)
    console.print(
        f"[dim]会话 {session_id}｜profile {session['profile']}｜"
        f"{len(rows)} 条事件（after seq {from_seq}）[/dim]"
    )
    events = [repo.row_to_event(row) for row in rows]
    for line in render_events(events, raw=raw):
        console.print(line, markup=False, highlight=False)

    turns = await repo.list_turns(db, session_id)
    console.print(
        "[dim]turn 状态：" + "，".join(f"{row['id']}={row['status']}" for row in turns) + "[/dim]"
    )
    await db.close()
    return 0


def render_events(events: list[Any], *, raw: bool = False) -> list[str]:
    """事件流 → 可读的多行文本。

    默认把连续的 `text.delta` 合并成一行：一次回答会产生几十上百条分片，
    逐条打印会把真正重要的工具调用和审批淹没掉。`raw=True` 时保持原样，
    用于核对"事件序列本身是否完整、有序"。
    """
    lines: list[str] = []
    buffer: list[Any] = []

    def flush() -> None:
        if not buffer:
            return
        text = "".join(str(event.data.get("text", "")) for event in buffer)
        span = str(buffer[0].seq) if len(buffer) == 1 else f"{buffer[0].seq}-{buffer[-1].seq}"
        lines.append(f"[{span}] 文本：{text}")
        buffer.clear()

    for event in events:
        if not raw and event.type.value == "text.delta":
            buffer.append(event)
            continue
        flush()
        lines.append(format_event(event))
    flush()
    return lines


if __name__ == "__main__":
    app()
