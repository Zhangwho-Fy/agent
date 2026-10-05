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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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


class EventRenderer:
    """把事件渲染到终端。`run`（本地直连）和 `chat`（走服务端）共用这一份。"""

    def __init__(self) -> None:
        self.streamed = False

    def reset(self) -> None:
        """新的一轮开始前清掉状态，否则上一轮流过的文本会把这一轮吞掉。"""
        self.streamed = False

    def __call__(self, event: Any) -> None:
        data = event.data
        kind = event.type.value
        if kind == "text.delta":
            self.streamed = True
            console.print(str(data.get("text", "")), end="", markup=False, highlight=False)
        elif kind == "text.done":
            # 逐字流已经打过了就不重复；没有流式分片时（例如轮数用尽的收尾消息）
            # 在这里补打一次，否则终端上是空白
            if not self.streamed and data.get("text"):
                console.print(str(data["text"]), markup=False, highlight=False)
        elif kind == "tool.call":
            args = json.dumps(data.get("args", {}), ensure_ascii=False)
            console.print(f"\n[dim]→ {data.get('name')} {args}[/dim]")
        elif kind == "tool.result":
            status = data.get("status")
            extra = f" {data['duration_ms']}ms" if data.get("duration_ms") else ""
            style = "dim" if status == "ok" else "yellow"
            console.print(f"[{style}]  ← {status}{extra}[/{style}]")
        elif kind == "approval.required":
            args = json.dumps(data.get("args", {}), ensure_ascii=False)
            console.print(f"\n[yellow]需要确认[/yellow] {data.get('name')} {args}")
        elif kind == "error":
            console.print(f"\n[red]错误：{data.get('message')}[/red]")


@dataclass
class ChatStatus:
    """交互式会话的状态，给底部状态栏用。

    数据全部来自 `turn.done` 事件里的 `usage`——本来就是模型真实报出来的 token 数，
    不需要额外估算。`last_input_tokens` 是最近一次调用实际塞进上下文的量，
    所以它就是"当前上下文占用"最直接的度量。
    """

    model: str = ""
    workspace: str = ""
    session_id: str = ""
    turns: int = 0
    last_input: int = 0
    last_output: int = 0
    total: int = 0
    context_limit: int = 0

    def track(self, usage: dict[str, Any]) -> None:
        self.last_input = int(usage.get("input_tokens", 0))
        self.last_output = int(usage.get("output_tokens", 0))
        self.total += self.last_input + self.last_output
        self.turns += 1

    def text(self) -> str:
        short = self.session_id[-6:] if self.session_id else "-"
        used = f"上下文 {self.last_input}"
        if self.context_limit:
            used += f"/{self.context_limit}（{self.last_input / self.context_limit:.0%}）"
        return (
            f" {self.model} │ 会话 …{short} │ {self.workspace} │ "
            f"{used} │ 本轮 ↑{self.last_input} ↓{self.last_output} │ "
            f"累计 {self.total} │ 第 {self.turns} 轮"
        )


def make_plain_asker() -> Callable[[], Awaitable[str]]:
    """最朴素的输入：`input()`。也是底栏出问题时的退路。"""

    async def plain() -> str:
        # 故意阻塞：CLI 本来就在等用户；换成线程池会踩 AGENTS.md 记的那个坑
        return input("\n你 > ")  # noqa: ASYNC250

    return plain


def make_asker(status: ChatStatus) -> Callable[[], Awaitable[str]]:
    """返回"读一行输入"的函数。

    有 `prompt_toolkit` 且确实在终端里跑，就用带底栏的输入框（输入固定在底部，
    状态栏挂在它下面）；否则退回 `input()`——重定向、CI、管道里没有 tty，
    底栏那一套根本渲染不出来，硬上只会输出一堆控制字符。

    **必须是 async**：`chat` 跑在事件循环里，而 prompt_toolkit 的同步 `prompt()`
    会自己 `asyncio.run()`，从运行中的循环里调它会直接抛
    "asyncio.run() cannot be called from a running event loop"。异步版
    `prompt_async()` 复用的是当前循环，没有这个问题。

    加依赖：`uv add prompt_toolkit`（没装也能跑，只是没有底栏）。
    """
    try:
        from prompt_toolkit import PromptSession

        if not sys.stdout.isatty():
            raise RuntimeError("不在终端里")
    except Exception:
        return make_plain_asker()

    session = PromptSession()

    async def with_toolbar() -> str:
        return str(await session.prompt_async("你 > ", bottom_toolbar=status.text))

    return with_toolbar


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

    emitter.on_event = EventRenderer()

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
) -> None:
    """交互式对话：一次启动、多轮问答（任务在服务端跑）。"""
    settings = Settings()
    code = asyncio.run(_chat(server, token or settings.auth_token, session, workspace, yes))
    raise typer.Exit(code=code)


async def _chat(
    server: str,
    token: str,
    session_id: str | None,
    workspace: Path | None,
    auto_approve: bool,
) -> int:
    import httpx

    from ..client.api import AgentClient

    settings = Settings()
    renderer = EventRenderer()
    async with AgentClient(server, token) as client:
        try:
            health = await client.health()
        except Exception as exc:
            console.print(f"[red]连不上服务端 {server}：{exc}[/red]")
            console.print("[dim]先在另一个终端跑：agent serve[/dim]")
            return 1
        console.print(f"[dim]已连接 {server}（agent {health.get('version')}）[/dim]")

        if session_id is None:
            created = await client.create_session(
                workspace=str(workspace) if workspace else None, profile="code", title="chat"
            )
            session_id = str(created["session_id"])
            console.print(f"[dim]新会话 {session_id}｜工作区 {created['workspace']}[/dim]")
        else:
            console.print(f"[dim]继续会话 {session_id}[/dim]")
        console.print("[dim]输入内容回车发送；exit / quit 退出[/dim]")

        status = ChatStatus(
            model=settings.model,
            workspace=str(workspace) if workspace else "",
            session_id=session_id,
            context_limit=settings.context_limit,
        )
        ask = make_asker(status)
        last_seq = 0
        while True:
            try:
                # 故意阻塞：CLI 本来就在等用户输入（换成线程池会踩 AGENTS.md 记的坑）
                line = (await ask()).strip()
            except (EOFError, KeyboardInterrupt):
                break
            except Exception as exc:
                # 底栏是"锦上添花"：终端不配合（老终端、奇怪的 tty、库版本差异）
                # 就退回朴素输入，不能让一次渲染失败把整个会话带走
                console.print(
                    f"[yellow]底栏输入不可用（{type(exc).__name__}），已退回普通输入[/yellow]"
                )
                ask = make_plain_asker()
                continue
            if not line:
                continue
            if line.lower() in {"exit", "quit", ":q"}:
                break

            renderer.reset()
            console.print("[bold]agent[/bold] ", end="")
            try:
                sent = await client.send_message(session_id, line)
            except httpx.HTTPStatusError as exc:
                console.print(
                    f"[red]发送失败：{exc.response.status_code} {exc.response.text}[/red]"
                )
                continue
            if sent.get("duplicate"):
                console.print("[yellow]（幂等命中：这条消息已经发过了）[/yellow]")
            turn_id = str(sent["turn_id"])

            async for event in client.stream_events(session_id, after_seq=last_seq):
                last_seq = event.seq
                if event.type.value == "approval.required":
                    granted = auto_approve or _ask_approval(event.data)
                    await client.approve(
                        session_id, str(event.data.get("call_id", "")), granted=granted
                    )
                    continue
                renderer(event)
                if event.type.value == "turn.done" and event.turn_id == turn_id:
                    status.track(event.data.get("usage") or {})
                    break
        console.print("\n[dim]再见[/dim]")
    return 0


def _ask_approval(data: Any) -> bool:
    """终端里问一句。默认拒绝：直接回车不放行。"""
    args = json.dumps(data.get("args", {}), ensure_ascii=False)
    console.print(f"[dim]原因：{data.get('reason', '')}[/dim]")
    answer = input(f"  允许执行 {args} 吗？[y/N] ").strip().lower()
    return answer in {"y", "yes", "是"}


if __name__ == "__main__":
    app()
