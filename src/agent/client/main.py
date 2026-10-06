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
import secrets
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table

from .. import __version__
from ..config import Settings
from ..logging import configure_logging
from .tui import (
    _loop_tui,
    _pick_session,
)
from .ui import (
    ChatStatus,
    EventRenderer,
    _fit,
    console,
    tui_problem,
)

app = typer.Typer(
    help="本地代码库助手 Agent",
    no_args_is_help=True,
    add_completion=False,
)

#: 顶层 callback 解析出来的日志级别。子命令要**再配一次**日志（比如 serve 要加文件
#: handler）时得沿用同一个级别，不能自己拿 settings 覆盖掉用户传的 `--log-level`。
_active_log_level = "info"

#: doctor 需要确认能导入的第三方依赖
REQUIRED_MODULES = ("pydantic", "pydantic_settings", "openai", "fastapi", "typer", "rich")


@app.callback()
def main(
    log_level: str | None = typer.Option(None, "--log-level", help="日志级别，默认取配置里的值"),
) -> None:
    """所有子命令共用的入口：先把日志配好。"""
    global _active_log_level
    settings = Settings()
    _active_log_level = log_level or settings.log_level
    configure_logging(_active_log_level, path=settings.log_path or None)


@app.command()
def version() -> None:
    """显示版本号。"""
    try:
        installed = importlib.metadata.version("agent")
    except importlib.metadata.PackageNotFoundError:
        installed = __version__
    console.print(f"agent {installed}")


def _looks_like_placeholder(key: str) -> bool:
    """密钥是不是 `.env.example` 里那种占位符（"已设置"但根本调不通，最坑）。"""
    if not key:
        return False
    lowered = key.lower()
    if len(key) < 20:
        return True
    return any(marker in lowered for marker in ("xxx", "your", "changeme", "placeholder", "填入"))


def _db_probe(path: Path) -> str:
    """会话库体检：打不开（权限/被锁）或表结构不对，都要在 doctor 里现形。"""
    import sqlite3

    if not path.exists():
        return f"{path}（还没有，跑一次 `agent run` 会建）"
    size_mb = path.stat().st_size / 1024 / 1024
    try:
        conn = sqlite3.connect(path, timeout=1.0)
        try:
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return f"{path}（{size_mb:.1f}MB，读失败：{type(exc).__name__}: {exc}）"
    return f"{path}（{size_mb:.1f}MB，{sessions} 会话 / {events} 事件）"


def _port_probe(host: str, port: int) -> str:
    """默认端口通不通——"服务端到底起没起"这个问题，别再靠猜。"""
    import socket

    try:
        with socket.create_connection((host, port), timeout=0.3):
            return f"{host}:{port} 已被占用（服务端大概率在跑）"
    except OSError:
        return f"{host}:{port} 空闲"


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
        table.add_row("会话库", _db_probe(settings.resolved_db_path))
        if settings.provider == "replay":
            table.add_row("密钥", "不需要（回放模式）")
        elif _looks_like_placeholder(settings.api_key):
            table.add_row("密钥", "看起来还是占位符  去 .env 填真 key")
            problems.append("AGENT_API_KEY 像占位符")
        elif settings.api_key:
            table.add_row("密钥", f"已设置（{len(settings.api_key)} 字符）")
        else:
            table.add_row("密钥", "未设置  需要在 .env 里填 AGENT_API_KEY")
            problems.append("未设置 AGENT_API_KEY")
        table.add_row("端口", _port_probe(settings.host, settings.port))

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
        typer.Option(
            "--session", "-s", help="接着已有会话跑（默认每次新建）；`-s last` 接最近一个"
        ),
    ] = None,
    last: Annotated[
        bool, typer.Option("--last", help="接着最近一个会话跑（等价于 -s last）")
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="自动批准所有写操作（无人值守时用）"),
    ] = False,
) -> None:
    """跑一次任务：驱动 LangGraph 图并在终端流式显示。"""
    settings = Settings()
    if workspace is not None:
        settings.workspace = workspace
    code = asyncio.run(
        _run_once(prompt, settings, session_id=session, use_last=last, auto_approve=yes)
    )
    raise typer.Exit(code=code)


async def _resolve_session(db: Any, session_id: str | None, use_last: bool) -> str | None:
    """把 `-s last` / `--last` 解析成真实的会话 id。

    空库时返回 None（调用方新建一个）——"接着上次"在还没跑过任何任务时不该直接报错。
    """
    if session_id != "last" and not use_last:
        return session_id
    from ..store import repo

    rows = await repo.session_summaries(db, limit=1)
    if not rows:
        console.print("[yellow]还没有历史会话，这次新建一个[/yellow]")
        return None
    latest = rows[0]
    title = " ".join(str(latest.get("title") or "").split())[:24]
    console.print(f"[dim]接着最近一个会话 …{str(latest['id'])[-6:]}（{title or '未命名'}）[/dim]")
    return str(latest["id"])


async def _run_once(
    prompt: str,
    settings: Settings,
    *,
    session_id: str | None = None,
    use_last: bool = False,
    auto_approve: bool = False,
) -> int:
    """构建图并跑一轮：落库 → 驱动图（可能等审批）→ 收尾落库。

    重依赖（langgraph / langchain）在这里才导入，这样 `agent doctor`、`agent config`
    在没有装齐依赖时也能用。
    """
    from ..core.bus import EventBus
    from ..core.ids import new_id
    from ..core.reliability import EventEmitter
    from ..graph.bridge import Approver, recursion_limit_for, stream_turn
    from ..graph.checkpointer import open_checkpointer
    from ..graph.wiring import build_session_graph
    from ..models.factory import build_chat_model
    from ..store import repo
    from ..store.db import Database

    if not settings.resolved_workspace.is_dir():
        console.print(f"[red]工作区不存在：{settings.resolved_workspace}[/red]")
        return 2

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

    session_id = await _resolve_session(db, session_id, use_last)
    if session_id is None:
        session_id = new_id("sess")
        await repo.create_session(
            db,
            session_id=session_id,
            profile="code",
            workspace=str(settings.resolved_workspace),
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

    emitter.on_event = EventRenderer()

    wiring = build_session_graph(
        settings,
        workspace=settings.resolved_workspace,
        model=model,
        emitter=emitter,
        checkpointer=open_checkpointer(settings.resolved_db_path),
        db=db,
    )
    graph = wiring.graph
    tool_names = ", ".join(wiring.registry.names)
    console.print(
        f"[dim]工作区 {wiring.ctx.workspace}｜模型 {settings.model}｜工具 {tool_names}[/dim]"
    )
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
    as_json: Annotated[bool, typer.Option("--json", help="按 JSON 输出，给脚本用")] = False,
) -> None:
    """列出最近的会话：会话 id、标题、**最后一句用户消息**、最后活动、turn 数。"""
    raise typer.Exit(code=asyncio.run(_list_sessions(Settings(), limit, as_json)))


async def _list_sessions(settings: Settings, limit: int, as_json: bool = False) -> int:
    from rich.markup import escape

    from ..store import repo
    from ..store.db import Database

    db = Database(settings.resolved_db_path)
    if not db.path.exists():
        console.print("[yellow]还没有会话库，先跑一次 `agent run`[/yellow]")
        return 0
    db.connect()
    rows = await repo.session_summaries(db, limit=limit)
    if not rows:
        console.print("[yellow]还没有会话[/yellow]")
        await db.close()
        return 0

    if as_json:
        console.print_json(json.dumps(rows, ensure_ascii=False, default=str))
        await db.close()
        return 0

    table = Table(title="会话")
    table.add_column("会话 id")
    table.add_column("标题")
    table.add_column("最后一句")
    table.add_column("最后活动")
    table.add_column("turn 数", justify="right")
    for row in rows:
        last = " ".join(str(row.get("last_user") or "").split())
        table.add_row(
            str(row["id"]),
            escape(str(row["title"] or "-")),  # 用户内容里可能有 `[`，别被 rich 当标记
            escape(_fit(last, 40).rstrip() or "-"),
            str(row["updated_at"]),
            str(row["turns"]),
        )
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
    follow: Annotated[
        bool,
        typer.Option("--follow", "-f", help="放完已有的继续跟新事件（Ctrl-C 退出）"),
    ] = False,
) -> None:
    """按原顺序重放会话的事件流。**不调用模型**，纯读库。

    `--from-seq` 从某条之后开始（断线续传语义）、`--raw` 逐条看分片、
    `--follow` 放完已有的继续跟新事件（另一个终端在跑任务时，这里能实时看）。
    """
    raise typer.Exit(code=asyncio.run(_replay(Settings(), session_id, from_seq, raw, follow)))


async def _replay(
    settings: Settings,
    session_id: str,
    from_seq: int,
    raw: bool = False,
    follow: bool = False,
    *,
    poll_seconds: float = 0.2,
) -> int:
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

    if follow:
        start = events[-1].seq if events else from_seq
        await _follow_events(db, session_id, start, raw, poll_seconds)

    turns = await repo.list_turns(db, session_id)
    console.print(
        "[dim]turn 状态：" + "，".join(f"{row['id']}={row['status']}" for row in turns) + "[/dim]"
    )
    await db.close()
    return 0


async def _follow_events(
    db: Any, session_id: str, after_seq: int, raw: bool, poll_seconds: float
) -> None:
    """接着跟新事件：**轮询库里的 events**（纯读库，不依赖服务端）。

    为什么不用总线订阅：`replay` 是离线工具，可能跟正在跑任务的进程不在同一个
    进程里——跨进程可见的事实源只有库。轮询间隔默认 0.2s，本地读一条 SQL 的开销
    可以忽略。
    """
    from ..store import repo

    console.print("[dim]--follow：有新事件就打印，Ctrl-C 退出[/dim]")
    try:
        while True:
            await asyncio.sleep(poll_seconds)
            rows = await repo.list_events(db, session_id, after_seq=after_seq)
            if not rows:
                continue
            events = [repo.row_to_event(row) for row in rows]
            after_seq = events[-1].seq
            for line in render_events(events, raw=raw):
                console.print(line, markup=False, highlight=False)
    except KeyboardInterrupt:
        console.print()
        console.print("[dim]停止跟随[/dim]")


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


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host", help="监听地址，默认取配置")] = None,
    port: Annotated[int | None, typer.Option("--port", help="端口，默认取配置")] = None,
    token: Annotated[
        str | None, typer.Option("--token", help="访问令牌，不给就随机生成一个")
    ] = None,
) -> None:
    """启动 HTTP + SSE 服务端：任务在这里执行，客户端只是订阅者。"""
    settings = Settings()
    if host is not None:
        settings.host = host
    if port is not None:
        settings.port = port
    if token is not None:
        settings.auth_token = token
    if not settings.auth_token:
        # 默认不是"没有校验"：没配就现生成一个，本地开发也走同一套路径
        settings.auth_token = secrets.token_urlsafe(16)

    # 长驻进程的日志不该只留在终端里：没配 AGENT_LOG_PATH 就写到会话库旁边
    configure_logging(
        _active_log_level,
        path=settings.log_path or (settings.resolved_db_path.parent / "agent.log"),
    )

    import uvicorn

    from ..server.app import create_app

    console.print(f"[dim]会话库 {settings.resolved_db_path}[/dim]")
    console.print(f"[bold]服务端[/bold] http://{settings.host}:{settings.port}")
    console.print(f"[bold]令牌[/bold] {settings.auth_token}")
    console.print(
        "[dim]另开一个终端：agent chat --token "
        f"{settings.auth_token}（或把它写进 AGENT_AUTH_TOKEN）[/dim]"
    )
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level="warning",
    )


@app.command()
def chat(
    server: Annotated[str, typer.Option("--server", help="服务端地址")] = "http://127.0.0.1:8765",
    token: Annotated[
        str | None, typer.Option("--token", help="访问令牌，默认取 AGENT_AUTH_TOKEN")
    ] = None,
    session: Annotated[str | None, typer.Option("--session", "-s", help="接着已有会话聊")] = None,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", "-w", help="新建会话时的工作区")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="自动批准所有写操作")] = False,
    resume: Annotated[
        bool, typer.Option("--resume", "-r", help="先列出历史会话，挑一个继续")
    ] = False,
    last: Annotated[
        bool, typer.Option("--last", help="配合 --resume：直接接最近一个，不弹选择")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="列出多少条历史会话")] = 20,
) -> None:
    """交互式对话：一次启动、多轮问答（任务在服务端跑）。"""
    settings = Settings()
    code = asyncio.run(
        _chat(
            server,
            token or settings.auth_token,
            session,
            workspace,
            yes,
            resume=resume,
            last=last,
            limit=limit,
        )
    )
    raise typer.Exit(code=code)


@app.command()
def resume(
    server: Annotated[str, typer.Option("--server", help="服务端地址")] = "http://127.0.0.1:8765",
    token: Annotated[
        str | None, typer.Option("--token", help="访问令牌，默认取 AGENT_AUTH_TOKEN")
    ] = None,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", "-w", help="新建会话时的工作区")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="自动批准所有写操作")] = False,
    last: Annotated[bool, typer.Option("--last", help="直接接最近一个会话，不弹选择")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="列出多少条历史会话")] = 20,
) -> None:
    """挑一个历史会话继续聊：等价于 `agent chat --resume`。"""
    settings = Settings()
    code = asyncio.run(
        _chat(
            server,
            token or settings.auth_token,
            None,
            workspace,
            yes,
            resume=True,
            last=last,
            limit=limit,
        )
    )
    raise typer.Exit(code=code)


async def _chat(
    server: str,
    token: str,
    session_id: str | None,
    workspace: Path | None,
    auto_approve: bool,
    *,
    resume: bool = False,
    last: bool = False,
    limit: int = 20,
) -> int:
    from ..client.api import AgentClient

    problem = tui_problem()
    if problem:
        console.print(f"[red]agent chat 需要真实终端：{problem}[/red]")
        console.print('[dim]一次性的行式任务用：agent run "你的问题"（-s 接着已有会话）[/dim]')
        return 1

    settings = Settings()
    async with AgentClient(server, token) as client:
        try:
            health = await client.health()
        except Exception as exc:
            console.print(f"[red]连不上服务端 {server}：{exc}[/red]")
            console.print("[dim]先在另一个终端跑：agent serve[/dim]")
            return 1
        console.print(f"[dim]已连接 {server}（agent {health.get('version')}）[/dim]")

        if resume and session_id is None:
            sessions = await client.list_sessions(limit=limit)
            if not sessions:
                console.print("[yellow]还没有历史会话，直接开一个新的吧[/yellow]")
            elif last:
                session_id = str(sessions[0]["id"])
            else:
                picked = await _pick_session(sessions, deleter=client.delete_session)
                if picked is None:
                    console.print("[dim]已取消[/dim]")
                    return 0
                session_id = picked

        if session_id is None:
            created = await client.create_session(
                workspace=str(workspace) if workspace else None, profile="code"
            )
            session_id = str(created["session_id"])
            console.print(f"[dim]新会话 {session_id}｜工作区 {created['workspace']}[/dim]")
        else:
            console.print(f"[dim]继续会话 {session_id}（历史会先画出来）[/dim]")
        console.print("[dim]输入内容回车发送；exit / quit 退出[/dim]")

        status = ChatStatus(
            model=settings.model,
            workspace=str(workspace) if workspace else "",
            session_id=session_id,
            context_limit=settings.context_limit,
            tool_limit=settings.max_tool_rounds,
        )
        await _loop_tui(client, session_id, status, auto_approve)
        console.print("\n[dim]再见[/dim]")
    return 0


if __name__ == "__main__":
    app()
