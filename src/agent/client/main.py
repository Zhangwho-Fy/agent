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
import contextlib
import importlib
import importlib.metadata
import json
import logging
import os
import secrets
import sys
import time
import traceback
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
        self.thinking = False

    def reset(self) -> None:
        """新的一轮开始前清掉状态，否则上一轮流过的文本会把这一轮吞掉。"""
        self.streamed = False
        self.thinking = False

    def __call__(self, event: Any) -> None:
        data = event.data
        kind = event.type.value
        if kind == "text.delta":
            self.streamed = True
            console.print(str(data.get("text", "")), end="", markup=False, highlight=False)
        elif kind == "reasoning.delta":
            # 行式模式显示不了"折叠"，但至少要让人知道模型在推理；
            # 完整的思考过程留给全屏界面（Ctrl-O 展开）。
            if not self.thinking:
                self.thinking = True
                console.print("[dim]· 思考中…（全屏界面里可展开）[/dim]")
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

    `phase` / `frame` 是给常驻底栏用的"这一刻在干什么"：空闲时为空串，
    执行时是"思考中 / 执行 fs_read / 输出中"，`frame` 由 `_spin()` 逐帧推进，
    底栏据此转圈。**它不进 `text()`**——行式底栏（PromptSession 那条路）不需要它。
    """

    #: 点状 spinner 的帧序，纯文本、无依赖
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    model: str = ""
    workspace: str = ""
    session_id: str = ""
    turns: int = 0
    last_input: int = 0
    last_output: int = 0
    total: int = 0
    context_limit: int = 0
    phase: str = ""
    frame: int = 0
    #: 本轮工具调用次数与上限，以及"同一个工具调了几次"的告警（第 4 节）。
    #: 全部由事件本地计数——和给模型的状态块同源，但**渲染是两份**（4.6）。
    tool_calls: int = 0
    tool_limit: int = 0
    repeat: str = ""
    _tool_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def track(self, usage: dict[str, Any]) -> None:
        self.last_input = int(usage.get("input_tokens", 0))
        self.last_output = int(usage.get("output_tokens", 0))
        self.total += self.last_input + self.last_output
        self.turns += 1

    def begin_turn(self) -> None:
        """新一轮开始：工具计数归零——它数的是"这一轮"，不是整个会话。"""
        self.tool_calls = 0
        self.repeat = ""
        self._tool_counts = {}

    def count_tool(self, name: str) -> None:
        """记一次工具调用。调用次数 ≥2 时底栏亮出重复告警（人也要能看见）。"""
        self.tool_calls += 1
        self._tool_counts[name] = self._tool_counts.get(name, 0) + 1
        worst = max(self._tool_counts.items(), key=lambda item: item[1])
        self.repeat = f"{worst[0]}×{worst[1]}" if worst[1] >= 2 else ""

    def segments(self, *, aligned: bool = False) -> list[tuple[str, str]]:
        """状态栏的分段内容（样式类, 文本）。

        这里只负责"哪一段是什么"，颜色在 `TUI_STYLE` 里定；纯文本就是把这些
        片段拼起来（测试里就是这么断言每一段内容和顺序的）。

        `aligned=True` 给全屏界面用：会变宽的字段（模型名、数字）补到固定宽度，
        这样阶段名从"空闲"变成"执行 fs_read"、token 数进一位，整条也不会左右抖。
        """
        short = self.session_id[-6:] if self.session_id else "-"
        ratio = self.last_input / self.context_limit if self.context_limit else 0.0
        usage = (
            "class:bar.usage.hot"
            if ratio >= 0.8
            else "class:bar.usage.warn"
            if ratio >= 0.5
            else "class:bar.usage.ok"
        )

        digits = len(str(self.context_limit)) if self.context_limit else 4
        if aligned:
            # 数字右对齐到固定列：不进位就不挪窝
            history = f"本轮 ↑{self.last_input:>{digits}} ↓{self.last_output:>{digits}}"
            total = f"累计 {self.total:>{digits + 2}}"
            turns = f"第 {self.turns:>3} 轮"
            model = _fit(self.model, 18)
            session = _fit(f"…{short}", 8)
            used = f"上下文 {self.last_input:>{digits}}"
        else:
            history = f"本轮 ↑{self.last_input} ↓{self.last_output}"
            total = f"累计 {self.total}"
            turns = f"第 {self.turns} 轮"
            model = self.model
            session = f"…{short}"
            used = f"上下文 {self.last_input}"

        if self.context_limit:
            percent = f"{ratio:>4.0%}" if aligned else f"{ratio:.0%}"
            used += f"/{self.context_limit}（{percent}）"

        tools = (
            f"工具 {self.tool_calls:>2}/{self.tool_limit}"
            if self.tool_limit
            else f"工具 {self.tool_calls:>2}"
        )

        segments = [
            ("class:bar.model", f" {model}"),
            ("class:bar.dim", " │ 会话 "),
            ("class:bar.session", session),
            ("class:bar.dim", f" │ {self.workspace} │ "),
            (usage, used),
            ("class:bar.dim", f" │ {history} │ {total} │ {turns}"),
            ("class:bar.dim", f" │ {tools}"),
        ]
        if self.repeat:
            segments.append(("class:bar.warn", f" │ ⚠ {self.repeat}"))
        return segments

    def set_phase(self, phase: str) -> None:
        """进入某个执行阶段，底栏开始转圈。"""
        self.phase = phase

    def clear_phase(self) -> None:
        """回到空闲——但底栏本身不消失，只是不再转圈。"""
        self.phase = ""

    def tick(self) -> None:
        self.frame += 1

    def spinner(self) -> str:
        return self.SPINNER[self.frame % len(self.SPINNER)]


def tui_problem() -> str | None:
    """能不能开全屏界面？不行就返回一句人话原因。

    `agent chat` 只有全屏这一套界面了（原先那套行式交互已经删掉）：它靠
    备用屏幕、鼠标无关的按键和实时重绘，**必须有真终端**。非终端场景
    （重定向、管道、CI）请用 `agent run`——那是另一条路：一次任务、行式输出。
    """
    if not sys.stdout.isatty():
        return "当前不是终端（重定向 / 管道 / CI）"
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return "没装 prompt_toolkit（跑一次 `uv sync`）"
    return None


def session_choices(sessions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """会话列表 → 选择框的条目：值取 session_id，标签是人看的那一行。

    标签里的每一栏都用 `_fit` 补成固定显示宽度（中文两列），所以选单是齐的。
    第二栏是**最后一句用户消息**——标题只记第一句，常常是"你好"，认不出人。
    """
    choices: list[tuple[str, str]] = []
    for row in sessions:
        session_id = str(row.get("id", ""))
        title = " ".join(str(row.get("title") or "").split()) or "(未命名)"
        last = " ".join(str(row.get("last_user") or "").split()) or "-"
        when = str(row.get("updated_at") or "").replace("T", " ")[:16]
        workspace = str(row.get("workspace") or "-")
        label = (
            f"{_fit(title, 18)} │ {_fit(last, 44)} │ {_fit(when, 16)} │ "
            f"…{session_id[-6:]} │ {_fit(workspace, 18, keep_tail=True)}"
        )
        choices.append((session_id, label))
    return choices


class _SessionPicker:
    """挑历史会话的小界面（全屏自成一体，跟后面的 TUI 一前一后跑）。

    比 `radiolist_dialog` 多做一件事：**能删**。按 `Delete` 或 `d` 先"上膛"，
    再按一次才真删——删除不可逆，多一次确认不亏；Esc 取消上膛，再按一次退出。
    """

    def __init__(
        self,
        sessions: list[dict[str, Any]],
        deleter: Callable[[str], Awaitable[Any]],
    ) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from prompt_toolkit.styles import Style

        self._sessions = list(sessions)
        self._deleter = deleter
        self._index = 0
        self._armed = ""  # 已经上膛、等着二次确认的 session_id
        self._message = "↑/↓ 选择 · Enter 继续 · Delete/d 删除 · Esc 取消"

        bindings = KeyBindings()

        @bindings.add("up")
        @bindings.add("c-p")
        def _up(event: Any) -> None:
            self._move(-1)

        @bindings.add("down")
        @bindings.add("c-n")
        def _down(event: Any) -> None:
            self._move(1)

        @bindings.add("pageup")
        def _page_up(event: Any) -> None:
            self._move(-10)

        @bindings.add("pagedown")
        def _page_down(event: Any) -> None:
            self._move(10)

        @bindings.add("delete")
        @bindings.add("d")
        def _delete(event: Any) -> None:
            session = self._current()
            if session is None:
                return
            session_id = str(session["id"])
            if self._armed != session_id:
                self._armed = session_id
                self._message = f"再按一次 Delete/d 确认删除「{self._session_title(session)}」"
                event.app.invalidate()
                return
            event.app.create_background_task(self._do_delete(session_id))

        @bindings.add("escape")
        @bindings.add("c-c")
        def _cancel(event: Any) -> None:
            if self._armed:  # 先取消上膛，再按一次才走人
                self._armed = ""
                self._message = "已取消删除"
                event.app.invalidate()
                return
            self._leave(None)

        @bindings.add("enter")
        def _accept(event: Any) -> None:
            if self._armed:  # 上膛状态下回车 = 不删，继续挑
                self._armed = ""
                self._message = "已取消删除"
                event.app.invalidate()
                return
            session = self._current()
            self._leave(str(session["id"]) if session else None)

        self._list = Window(FormattedTextControl(self._rows), height=Dimension(weight=1))
        self.app: Any = Application(
            layout=Layout(
                HSplit(
                    [
                        Window(
                            FormattedTextControl("继续哪个会话？"),
                            height=1,
                            style="class:title",
                        ),
                        self._list,
                        Window(FormattedTextControl(self._hint), height=1, style="class:hint"),
                    ]
                )
            ),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "title": "bold #7aa2f7",
                    "hint": "#8b93a7",
                    "selected": "reverse",
                    "danger": "bg:#5c1f1f #ffcccc bold",
                }
            ),
            full_screen=True,
        )

    def _session_title(self, session: dict[str, Any]) -> str:
        title = str(session.get("title") or "").strip()
        return title or "(未命名)"

    def _leave(self, result: str | None) -> None:
        """请求退出。跟 `ChatTUI.exit()` 一样要能重复调用——测试里会不跑界面直接用它。"""
        future = self.app.future
        if future is not None and not future.done():
            self.app.exit(result=result)

    def _current(self) -> dict[str, Any] | None:
        if not self._sessions:
            return None
        return self._sessions[min(self._index, len(self._sessions) - 1)]

    def _move(self, delta: int) -> None:
        if not self._sessions:
            return
        self._index = max(0, min(len(self._sessions) - 1, self._index + delta))
        # 别让选中项跑出可视区：往上顶到底、往下留一屏
        visible = max(1, self.app.output.get_size().rows - 2)
        if self._index < self._list.vertical_scroll:
            self._list.vertical_scroll = self._index
        elif self._index >= self._list.vertical_scroll + visible:
            self._list.vertical_scroll = self._index - visible + 1
        self._armed = ""
        self._message = "↑/↓ 选择 · Enter 继续 · Delete/d 删除 · Esc 取消"
        self.app.invalidate()

    def _rows(self) -> list[tuple[str, str]]:
        fragments: list[tuple[str, str]] = []
        for index, session in enumerate(self._sessions):
            if index:
                fragments.append(("", "\n"))
            selected = index == self._index
            style = "class:danger" if self._armed == str(session["id"]) else ""
            if selected and not style:
                style = "class:selected"
            label = session_choices([session])[0][1]
            fragments.append((style, f"{'❯' if selected else ' '} {label}"))
        return fragments

    def _hint(self) -> str:
        return f" {self._message}"

    async def _do_delete(self, session_id: str) -> None:
        try:
            await self._deleter(session_id)
        except Exception as exc:  # 删不掉要说清原因，别静默失败
            self._armed = ""
            self._message = f"删除失败：{exc}"
            self.app.invalidate()
            return
        self._sessions = [row for row in self._sessions if str(row["id"]) != session_id]
        self._index = min(self._index, max(0, len(self._sessions) - 1))
        self._armed = ""
        if not self._sessions:
            self._leave(None)  # 全删光了，没得挑了
            return
        self._message = f"已删除 …{session_id[-6:]}（还剩 {len(self._sessions)} 个）"
        self.app.invalidate()


async def _pick_session(
    sessions: list[dict[str, Any]], *, deleter: Callable[[str], Awaitable[Any]]
) -> str | None:
    """弹出选择框让用户挑历史会话。取消（Esc）返回 None。"""
    if not sessions:
        return None
    return await _SessionPicker(sessions, deleter).app.run_async()


def format_event_plain(event: Any) -> str:
    """事件 → 纯文本。

    常驻界面里**不能再让 rich 插手**：它和界面渲染会抢同一块终端，
    上次就是这么把 ANSI 序列打进正文的（屏幕上出现 `?[2m` 那种）。
    所以这条路只产出纯文本，界面自己负责上色和布局。
    """
    data = event.data
    kind = event.type.value
    if kind in {"text.delta", "reasoning.delta"}:
        return str(data.get("text", ""))
    if kind == "text.done":
        return ""  # 分片已经逐字打过了
    if kind == "tool.call":
        args = json.dumps(data.get("args", {}), ensure_ascii=False)
        return f"→ {data.get('name')} {args}\n"
    if kind == "tool.result":
        return f"  ← {data.get('status')}\n"
    if kind == "approval.required":
        return f"需要确认：{data.get('name')}\n"
    if kind == "context.compressed":
        # 过程信息里给一行就够：指针化了几条、摘要了几条（细节在事件里）
        return (
            f"  ⇣ 压缩上下文 {data.get('ratio')}："
            f"指针化 {len(data.get('pointerized') or [])} 条，"
            f"摘要 {data.get('summarized', 0)} 条\n"
        )
    if kind == "error":
        return f"错误：{data.get('message')}\n"
    return ""


def _event_line_kind(kind: str) -> str:
    """事件类型 → 日志行类型（决定怎么上色、能不能折进"过程"组）。"""
    if kind == "error":
        return "error"
    if kind == "reasoning.delta":
        return "reason"
    if kind in {"tool.call", "tool.result", "approval.required", "context.compressed"}:
        return "tool"
    return "assistant"


#: 界面配色。全部用 ANSI 名字而不是写死的 RGB——深浅两种终端配色都能看清。
#: 只有状态栏是例外：它有固定底色（下面两行注释说明），所以颜色也写死。
TUI_STYLE = {
    # 状态栏：固定深色底 + 浅灰字。写死是为了不管终端深浅都成立——
    # `reverse` 在深色终端上会翻成一条白条，又晃眼又单调。
    "bar": "bg:#20242c #8b93a7",
    "bar.model": "#7aa2f7 bold",  # 模型名：蓝
    "bar.session": "#9ece6a",  # 会话号：绿
    "bar.dim": "#565f89",  # 分隔符和次要数字：退到背景里
    "bar.usage.ok": "#9ece6a",  # 上下文占用：绿 → 黄 → 红
    "bar.usage.warn": "#e0af68",
    "bar.usage.hot": "#f7768e bold",
    "bar.warn": "#e0af68 bold",  # 重复调用告警：人也要看得见
    "bar.busy": "#e0af68 bold",  # 执行中的阶段：黄
    "bar.idle": "#9ece6a",  # 空闲：绿点
    "bar.hint": "#bb9af7",  # 回滚提示：紫
    "prompt": "#7aa2f7 bold",  # 输入提示符，跟模型名一个色系
    "user": "bold ansibrightblue",
    # 正文写死成近白：终端默认前景色在不少深色主题里是灰的，
    # 一眼看去跟"过程信息"（暗灰）分不开。浅色终端把这一行改成 "" 就跟随主题。
    "assistant": "#e6e6e6",
    "tool": "ansibrightblack",  # Codex 那种暗灰：过程信息一律退到背景里去
    "reason": "ansibrightblack italic",  # 模型自带的思考：同色系，用斜体区分
    "warn": "#e0af68 bold",  # 需要人工确认：黄，必须显眼
    "error": "ansired bold",
    # 围栏里的内容也要用亮色：模型经常把整段正文（诗、故事、表格）放进 ``` 里，
    # 发灰就会被误当成"过程信息"。真代码靠下面的高亮分色，注释才该退到背景里。
    "code": "#d8dee9",
    "code.keyword": "ansibrightmagenta",
    "code.string": "ansigreen",
    "code.number": "ansibrightyellow",
    "code.comment": "ansibrightblack italic",
    "code.func": "ansicyan",
    "code.builtin": "ansibrightcyan",
    "code.op": "#a7b0bd",
}

#: 日志行的类型 → 样式类。空串表示"用终端默认前景色"。
LINE_CLASS = {
    "user": "class:user",
    "assistant": "class:assistant",
    "tool": "class:tool",
    "reason": "class:reason",
    "warn": "class:warn",
    "error": "class:error",
}

#: 归到一个"过程"组里的行类型：工具调用和模型思考都属于"过程信息"
PROCESS_KINDS = frozenset({"tool", "reason"})


def approval_lines(data: dict[str, Any]) -> list[str]:
    """审批提示的几行纯文本（样式由界面统一给）。

    `shell_exec` 这类单参数工具直接把命令摊开——审批的意义就是让人看清要跑什么。
    """
    args = data.get("args") or {}
    detail = args.get("command") if isinstance(args, dict) else None
    if not detail:
        detail = json.dumps(args, ensure_ascii=False)
    lines = [f"⚠ 需要确认：{data.get('name')}", f"  {detail}"]
    if data.get("reason"):
        lines.append(f"  原因：{data['reason']}")
    lines.append("  y 放行 · 回车或 n 拒绝 · 超时按拒绝")
    return lines


def _group_of(kind: str) -> str:
    """把行归类，用来决定"哪里该空一行"和"哪些行能折叠到一起"。"""
    if kind.startswith("code:"):
        return "code"
    if kind in PROCESS_KINDS:
        return "process"
    return kind


def _needs_separator(
    items: list[tuple[str, list[tuple[str, str]]]], previous: str | None, group: str
) -> bool:
    """换组要空一行，但**别空两行**：正文里本来就有空行时按它自己的来。"""
    if previous is None or previous == group:
        return False
    return not items or bool(items[-1][1])


def _code_class(token: Any) -> str:
    """pygments 的 token 类型 → 我们的样式类。"""
    from pygments.token import Token

    for base, name in (
        (Token.Comment, "code.comment"),
        (Token.Literal.String, "code.string"),
        (Token.Literal.Number, "code.number"),
        (Token.Keyword, "code.keyword"),
        (Token.Name.Builtin, "code.builtin"),
        (Token.Name.Function, "code.func"),
        (Token.Name.Class, "code.func"),
        (Token.Name.Decorator, "code.func"),
        (Token.Operator, "code.op"),
    ):
        if token in base:
            return f"class:{name}"
    return "class:code"


def _code_fragments(text: str, lang: str) -> list[tuple[str, str]]:
    """给一行代码上色。

    有 pygments 就用它正经分词；没有就整行一个暗灰样式——高亮是锦上添花，
    不能因为它缺依赖就把整个界面弄挂（pygments 目前只是传递依赖，没写进
    pyproject）。逐行分词拿不到跨行状态，三引号字符串/块注释会掉色，
    聊天窗口里可以接受。
    """
    if not text:
        return []
    try:
        from pygments import lex
        from pygments.lexers import get_lexer_by_name
    except ImportError:
        return [("class:code", text)]

    try:
        lexer = get_lexer_by_name(lang or "text", stripnl=False)
    except Exception:
        return [("class:code", text)]

    fragments: list[tuple[str, str]] = []
    for token, value in lex(text, lexer):
        fragments.append((_code_class(token), value))
    if fragments and fragments[-1][1].endswith("\n"):  # lexer 会补一个结尾换行
        fragments[-1] = (fragments[-1][0], fragments[-1][1][:-1])
    return [fragment for fragment in fragments if fragment[1]] or [("class:code", text)]


def _line_fragments(kind: str, text: str) -> list[tuple[str, str]]:
    """一行日志 → 带样式的片段。代码行走高亮，其余整行一个样式。"""
    if kind.startswith("code:"):
        return _code_fragments(text, kind[5:])
    if not text:
        return []
    return [(LINE_CLASS.get(kind, ""), text)]


def _wrap_fragments(line: list[tuple[str, str]], width: int) -> list[list[tuple[str, str]]]:
    """把一行（带样式的片段）按**显示宽度**折开，中文算 2 列。

    界面要自己取"最后几行"填满日志区，折行就不能交给终端：算宽一点，
    取到的尾巴会多出半行；算窄一点，行尾会被截掉。
    """
    from prompt_toolkit.utils import get_cwidth

    width = max(4, width)
    wrapped: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    used = 0
    for style, raw in line:
        piece = raw.replace("\t", "    ").replace("\r", "")
        for char in piece:
            size = get_cwidth(char)
            if current and used + size > width:
                wrapped.append(current)
                current, used = [], 0
            if current and current[-1][0] == style:  # 相邻同样式合并，减少片段数
                current[-1] = (style, current[-1][1] + char)
            else:
                current.append((style, char))
            used += size
    wrapped.append(current)
    return wrapped


def _fit(text: str, width: int, *, keep_tail: bool = False) -> str:
    """把一段文字补到**固定显示宽度**（中文算 2 列），太长就截断加省略号。

    状态栏靠它排"制表位"：阶段名一长一短、token 数一进位，整条就不该跟着抖。
    `keep_tail=True` 从尾巴留（路径这种后缀比前缀有用）。
    """
    from prompt_toolkit.utils import get_cwidth

    width = max(1, width)
    if get_cwidth(text) > width:
        budget = width - 1  # 给省略号留一列
        if keep_tail:
            kept = ""
            for char in reversed(text):
                if get_cwidth(kept) + get_cwidth(char) > budget:
                    break
                kept = char + kept
            text = "…" + kept
        else:
            kept = ""
            for char in text:
                if get_cwidth(kept) + get_cwidth(char) > budget:
                    break
                kept += char
            text = kept + "…"
    return text + " " * max(0, width - get_cwidth(text))


def _process_summary(run: list[tuple[str, list[tuple[str, str]]]], columns: int) -> str:
    """折叠后那一行：**固定成一行、钉住开头**，不随流式变化。

    两条都是踩出来的：

    1. 别拿"最后一句思考"当进度。那样每来一个 token 这行都在变，看着像抽搐；
       现在只露第一个思考行的开头，想看全文按 Ctrl-O。
    2. 整行宽度必须**与终端宽度挂钩且固定**。以前它有时折成两行、有时一行，
       行数一变，日志区的"最后几行"就整体挪位——表现就是"没人动它，屏幕自己
       往下滚"。所以这里先把行数右对齐（`66` 和 `9` 一样宽），再按剩余宽度裁。
    """
    from prompt_toolkit.utils import get_cwidth

    hint = "   …（Ctrl-O 展开）"
    room = max(20, columns - get_cwidth(hint))
    head = f"▸ 过程 {len(run):>3} 行"
    first = ""
    for kind, fragments in run:
        if kind == "reason":
            first = "".join(text for _style, text in fragments)
            break
    if first:
        head += "：" + _fit(first, 36).rstrip()
    return _fit(head, room).rstrip() + hint


class ChatTUI:
    """全屏常驻界面：日志在界面内部滚动，状态栏 + 输入行钉在窗口最后两行。

    这是"Codex 那种手感"：底栏**永远**在窗口最底下，跟这一轮输出了多少行
    无关；输出在界面自己的日志区里往上滚。代价是走终端的**备用屏幕**
    （alternate screen），退出后本次对话不会留在终端 scrollback 里——这是
    全屏换来的，不是 bug。

    渲染只剩**一个写者**：界面自己。日志只存在 `_log` / `_pending`，由
    `_transcript_fragments()` 按当前窗口尺寸取"最后几行"画出来，一个字节都
    不写 stdout。于是既没有 `patch_stdout` 那套"擦掉 → 打印 → 画回来"，
    也没有"写出去半行、底栏被拽到正文中间"的问题（前两版就是那么坏的）；
    rich 更是完全不参与——模块级 `console` 在导入时就抓死了原始 stdout，
    上一版花屏正是它绕过界面直接写造成的。
    """

    #: 两次真正重画之间至少隔这么久（秒）。流式输出一秒能来几百个分片，
    #: 每个都重画会把终端刷爆——看着就像"屏幕自己在清"。
    PAINT_INTERVAL = 0.05

    def __init__(self, status: ChatStatus) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.history import InMemoryHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import Dimension, HSplit, Layout, VSplit, Window
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.styles import Style

        self.status = status
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._log: list[tuple[str, str]] = []  # (行类型, 文本)，一行一条
        self._pending = ""  # 还没换行的那一行（流式 token 正往里长）
        self._pending_kind = "assistant"
        self._code: str | None = None  # 代码围栏的语言；None = 不在代码块里
        self._folded = True  # 过程信息（工具调用）默认折叠，跟 Codex 一样
        self._scroll = 0  # 从底部往上回滚了几屏行；0 = 跟着最新输出走
        self._frozen: list[tuple[str, str]] | None = None  # 回滚时的日志快照
        self._frozen_pending = ""
        self._asking = False  # 正在问"放行还是拒绝"
        self._deferred: list[str] = []  # 打字打早了：先记着，这一轮完了再当消息发
        self._dirty = False  # 有改动等着画
        self._last_paint = 0.0  # 上次真正重画的时间（限流用）
        self._trace_path = os.environ.get("AGENT_TUI_TRACE", "")
        # 多行 + 历史。Enter 仍然发送（见下面的绑定），换行用 Ctrl-J；历史给
        # prompt_toolkit 自带的 Ctrl-R 增量搜索用。
        self.input = Buffer(
            multiline=True,
            history=InMemoryHistory(),
            accept_handler=self._on_accept,
        )

        bindings = KeyBindings()

        @bindings.add("c-c")
        @bindings.add("c-d")
        def _quit(event: Any) -> None:
            self._lines.put_nowait(None)  # None = 用户要走，别当成空行
            event.app.exit()

        @bindings.add("pageup")
        def _page_up(event: Any) -> None:
            self._scroll_by(self._page_size())

        @bindings.add("pagedown")
        def _page_down(event: Any) -> None:
            self._scroll_by(-self._page_size())

        @bindings.add("up")
        def _line_up(event: Any) -> None:
            self._scroll_by(1)

        @bindings.add("down")
        def _line_down(event: Any) -> None:
            self._scroll_by(-1)

        @bindings.add("end")
        def _to_bottom(event: Any) -> None:
            self._scroll = 0
            self._frozen = None  # 解冻：回到跟着最新输出走
            self._frozen_pending = ""
            self.refresh()

        @bindings.add("enter")
        def _submit(event: Any) -> None:
            # 应用级绑定比默认绑定优先级高，所以多行 Buffer 的"回车换行"被这里顶掉
            self.input.validate_and_handle()

        @bindings.add("c-j")  # Enter 发 CR、Ctrl-J 发 LF，终端层面一定分得开
        @bindings.add("escape", "enter")
        def _newline(event: Any) -> None:
            self.input.insert_text("\n")

        @bindings.add("c-o")
        def _toggle_fold(event: Any) -> None:
            self._folded = not self._folded
            self.refresh()

        style = Style.from_dict(TUI_STYLE)

        self.app = Application(
            layout=Layout(
                HSplit(
                    [
                        # 日志区吃掉除底栏之外的全部高度，自己滚动
                        Window(
                            FormattedTextControl(self._transcript_fragments),
                            height=Dimension(weight=1),
                            wrap_lines=False,
                        ),
                        Window(
                            FormattedTextControl(self._status_fragments),
                            height=1,
                            wrap_lines=False,
                            # 样式挂在窗口上，prompt_toolkit 才会把**整行**填满，
                            # 而不是只反白我们写出去的那几个字
                            style="class:bar",
                        ),
                        VSplit(
                            [
                                # 提示符常驻在最左边：平时"你 > "，问审批时换文案
                                Window(
                                    FormattedTextControl(self._prompt_fragments),
                                    width=6,
                                    height=1,
                                ),
                                Window(BufferControl(buffer=self.input), height=Dimension(min=1)),
                            ],
                            # 多行输入最多长到 6 行，再长自己在框里滚；日志区让位
                            height=Dimension(min=1, max=6),
                        ),
                    ]
                )
            ),
            key_bindings=bindings,
            style=style,
            full_screen=True,  # 备用屏幕：底栏钉在窗口最底，退出后终端复原
        )
        if self._trace_path:
            self._trace_erases()

    def _effective_pending_kind(self) -> str:
        """还在流式的那一行属于什么类型（代码块里就是代码行）。"""
        if self._code is not None and self._pending_kind == "assistant":
            return f"code:{self._code}"
        return self._pending_kind

    def _view(self) -> tuple[list[tuple[str, str]], str]:
        """这一帧该画哪份日志：跟着最新走时是实时的，回滚时是**冻结的快照**。

        为什么要冻：滚动位置是"离底部多少行"，而底部一直在长（思考一秒几十行），
        不冻的话你一往上滚，新输出就把视窗推着走，看着就像屏幕自己在动。
        """
        if self._frozen is not None:
            return self._frozen, self._frozen_pending
        return self._log, self._pending

    def _build_items(
        self,
        start: int,
        columns: int,
    ) -> list[tuple[str, list[tuple[str, str]]]]:
        """把日志从 `start` 开始整理成"要画的行"。

        做两件事：换组时插空行（正文 / 工具 / 用户各自成段）、把连着的过程信息
        折成一行（折叠后 `columns` 决定那一行裁多宽）。
        """
        log, pending = self._view()
        # 折叠得看到整段的开头，否则会把"延续行"当成第一行
        while (
            self._folded
            and start > 0
            and _group_of(log[start][0]) == "process"
            and _group_of(log[start - 1][0]) == "process"
        ):
            start -= 1

        items: list[tuple[str, list[tuple[str, str]]]] = []
        previous = _group_of(log[start - 1][0]) if start > 0 else None
        for kind, text in log[start:]:
            if _needs_separator(items, previous, _group_of(kind)):
                items.append(("", []))
            fragments = _line_fragments(kind, text)
            if not fragments and items and not items[-1][1]:
                continue  # 上一行已经是空行，别再堆一行
            items.append((kind, fragments))
            previous = _group_of(kind)

        if pending:
            # 冻结时那行是"当时"的尾巴，类型跟着快照走，用 reason 兜底（过程信息）
            kind = self._effective_pending_kind() if self._frozen is None else "reason"
            if _needs_separator(items, previous, _group_of(kind)):
                items.append(("", []))
            items.append((kind, _line_fragments(kind, pending)))

        return self._fold(items, columns)

    def _total_lines(self, columns: int) -> int:
        """当前视图折行后一共有多少屏行（只算一次，滚动时用）。"""
        total = 0
        for _kind, fragments in self._build_items(0, columns):
            total += len(_wrap_fragments(fragments, columns))
        return total

    def _fold(
        self, items: list[tuple[str, list[tuple[str, str]]]], columns: int
    ) -> list[tuple[str, list[tuple[str, str]]]]:
        """折叠：连着的过程信息（思考 + 工具调用）压成一行。"""
        if not self._folded:
            return items
        folded: list[tuple[str, list[tuple[str, str]]]] = []
        index = 0
        while index < len(items):
            if _group_of(items[index][0]) != "process":
                folded.append(items[index])
                index += 1
                continue
            start = index
            while index < len(items) and _group_of(items[index][0]) == "process":
                index += 1
            run = items[start:index]
            if len(run) == 1:
                folded.extend(run)
                continue
            folded.append(("tool", [("class:tool", _process_summary(run, columns))]))
        return folded

    def _render_lines(self, rows: int, columns: int) -> list[list[tuple[str, str]]]:
        """日志区这一帧要画的行（片段化），折行后取最后 (rows-2) 行。

        **按屏行取，不按源日志条数取**——这是踩过的坑：折叠会把几百条思考压成
        1 屏行，如果按"从尾部取 N 条源日志"来取窗口，窗口会整个落在那段被折叠的
        内容里，折完只剩一行，屏幕看着就像被清空了。所以这里从尾部起取，**不够
        就往前翻倍扩**，直到凑够一屏（或者翻到日志开头）。
        """
        height = max(1, rows - 2)  # 让出状态栏和输入行
        log, _pending = self._view()
        want = height + self._scroll
        start = max(0, len(log) - (want + 1))
        while True:
            wrapped: list[list[tuple[str, str]]] = []
            for _kind, fragments in self._build_items(start, columns):
                wrapped.extend(_wrap_fragments(fragments, columns))
            if len(wrapped) >= want or start == 0:
                break
            start = max(0, start - max(want + 1, len(log) - start))  # 往前翻倍
        end = min(len(wrapped), max(0, len(wrapped) - self._scroll))
        return wrapped[max(0, end - height) : end]

    def _page_size(self) -> int:
        """一屏能看多少行日志（让出底栏两行）。"""
        return max(1, self.app.output.get_size().rows - 3)

    def _scroll_by(self, delta: int) -> None:
        """滚动日志区。`delta > 0` 往上看历史，负数往回走；到头/到底自动夹住。

        往上滚的第一下会**冻结当前内容**：之后流式输出再多，视窗也不动，
        只在底栏上告诉你"有多少行新输出"。滚回底部（或按 End）自动解冻。
        """
        size = self.app.output.get_size()
        height = max(1, size.rows - 2)
        if delta > 0 and self._frozen is None:
            self._frozen = list(self._log)
            self._frozen_pending = self._pending
        limit = max(0, self._total_lines(size.columns) - height)
        self._scroll = min(limit, max(0, self._scroll + delta))
        if self._frozen is not None and self._scroll == 0:
            self._frozen = None  # 回到底部了，继续跟最新
            self._frozen_pending = ""
        self.refresh()

    def _transcript_fragments(self) -> list[tuple[str, str]]:
        size = self.app.output.get_size()
        fragments: list[tuple[str, str]] = []
        for index, line in enumerate(self._render_lines(size.rows, size.columns)):
            if index:
                fragments.append(("", "\n"))
            fragments.extend(line)
        return fragments

    def _status_fragments(self) -> list[tuple[str, str]]:
        """底栏内容（带样式）。交给 prompt_toolkit 渲染，**不经过 rich**。

        底色是靠 `Window(style="class:bar")` 铺满整行的，这里只管每段的前景色。
        """
        if self.status.phase:
            # 等用户回话时不转圈：那会儿干活的是人，不是 agent
            marker = "⚠" if self._asking else self.status.spinner()
            lead = ("class:bar.busy", f" {marker} {self.status.phase} ")
        else:
            lead = ("class:bar.idle", " ● 空闲 ")
        # 徽标补到固定宽度：不然"空闲"和"执行 fs_read"一换，后面全跟着平移
        fragments = [(lead[0], _fit(lead[1], 20)), *self.status.segments(aligned=True)]
        if self._scroll:
            fresh = len(self._log) - len(self._frozen) if self._frozen is not None else 0
            hint = f" │ ↑ 浏览历史 {self._scroll} 行"
            if fresh:
                hint += f" · {fresh} 行新输出"
            fragments.append(("class:bar.hint", hint + "（End 回底）"))
        return fragments

    def _prompt_fragments(self) -> list[tuple[str, str]]:
        """输入行左边那一小格：问审批的时候要换文案，宽度保持一样免得抖。"""
        if self._asking:
            return [("class:warn", _fit("允许? ", 6))]
        return [("class:prompt", _fit("你 > ", 6))]

    def _on_accept(self, buffer: Any) -> bool:
        """回车：把这行交给主循环，**返回 False 让 prompt_toolkit 自己清空输入框**。

        它清空之前会先 `append_to_history()`——我们抢着 reset 的话，历史里记下的
        就是空串，`Ctrl-R` 搜索等于白装。
        """
        self._lines.put_nowait(buffer.text)
        return False

    def write(self, text: str, kind: str = "assistant") -> None:
        """追加日志。**不写 stdout、不走 rich**：界面自己画（见类文档）。

        `kind` 决定这一行怎么上色、能不能折叠：`assistant`（正文）、`tool`
        （工具调用/结果，暗灰 + 可折叠）、`error`。没换行的尾巴留在 `_pending`
        里继续长，所以流式 token 是逐字出现在日志区最后一行的。
        """
        if not text:
            return
        if not self._pending:
            self._pending_kind = kind
        self._pending += text
        if "\n" not in self._pending:
            self.refresh()
            return
        complete, self._pending = self._pending.rsplit("\n", 1)
        pending_kind, self._pending_kind = self._pending_kind, kind
        for line in complete.split("\n"):
            self._push(pending_kind, line)
        self.refresh()

    def _push(self, kind: str, text: str) -> None:
        """收下一整行。代码围栏在这里处理，因为围栏只认整行。"""
        if kind == "assistant":
            stripped = text.strip()
            if stripped.startswith("```"):
                # 开/闭围栏自己不上屏——只留代码本体，省得占两行
                self._code = None if self._code is not None else stripped[3:].strip()
                return
            if self._code is not None:
                self._log.append((f"code:{self._code}", text))
                return
        self._log.append((kind, text))

    def end_line(self) -> None:
        """把还在长的那行收进日志（它后面要接别的内容了）。"""
        if self._pending:
            kind, text = self._effective_pending_kind(), self._pending
            self._pending = ""
            self._push(kind, text)

    def echo_user(self, line: str) -> None:
        """把用户刚才那句回显到日志里。

        上一轮的回答末尾可能没有换行（模型就这么吐的），不先收一下的话，
        `你 > ...` 会接在回答最后一行屁股后面。

        顺便回到"跟着最新走"：既然你在发新消息，就该看最新那条，而不是停在
        之前翻到的历史位置。
        """
        self._scroll = 0
        self._frozen = None
        self._frozen_pending = ""
        self.end_line()
        self._push("user", f"你 > {line}")
        self.refresh()

    async def run(self) -> None:
        await self.app.run_async()

    def exit(self) -> None:
        """请求退出。**必须能重复调用**。

        Ctrl-C / Ctrl-D 的绑定里已经 `app.exit()` 过一次了，主循环收尾时会再
        调一次；prompt_toolkit 第二次会抛 "Return value already set"——所以这里
        先看 future：没跑起来是 `None`，跑完了是 `done()`，两种都说明无事可做。
        """
        future = self.app.future
        if future is not None and not future.done():
            self.app.exit()

    def refresh(self) -> None:
        """请求重画。**限流**，见 `PAINT_INTERVAL`。"""
        self._dirty = True
        self._paint()

    def _paint(self, *, force: bool = False) -> None:
        """真正把这一帧画出去（或者因为限流先攒着）。

        为什么限流：思考/正文一秒能来几百个分片，每个分片都 `invalidate()` 的话，
        终端会被持续重画——全屏下看着就是"屏幕自己在闪/在清"。攒到下一帧一起画，
        体感一样流畅，终端负载差一个数量级。
        """
        now = time.monotonic()
        if not force and now - self._last_paint < self.PAINT_INTERVAL:
            return
        self._last_paint = now
        self._dirty = False
        self._trace_frame()
        self.app.invalidate()

    def tick(self) -> None:
        """由 `_spin` 每 0.1 秒推一次：转圈 + 把攒下的改动一次性画出去。"""
        if self.status.phase:
            self.status.tick()
        if self._dirty or self.status.phase:
            self._paint(force=True)

    def _trace_frame(self) -> None:
        """排障用：`AGENT_TUI_TRACE=/tmp/tui.jsonl` 时，每画一帧记一行账。

        记的是"这一屏由多少行组成"——怀疑某处在反复重排时，看这个文件就知道
        行数到底有没有抖，不用靠猜。
        """
        if not self._trace_path:
            return
        try:
            size = self.app.output.get_size()
            rows = self._render_lines(size.rows, size.columns)
            self._trace(
                "frame",
                rows=size.rows,
                columns=size.columns,
                log_lines=len(self._log),
                pending=len(self._pending),
                scroll=self._scroll,
                phase=self.status.phase,
                visible_lines=len(rows),
            )
        except OSError:
            self._trace_path = ""  # 写不进去就别再试了

    def _trace_erases(self) -> None:
        """排障用：把每一次"擦屏"连调用栈一起记下来。

        清屏序列只可能从 `renderer.erase()` 发出，所以这条记录能直接回答
        "到底是谁把屏幕擦掉的"——是用户代码写终端（`patch_stdout` 那条路），
        还是框架自己。
        """
        original = self.app.renderer.erase

        def traced(*args: Any, **kwargs: Any) -> Any:
            stack = [frame.function for frame in traceback.extract_stack()[-7:-1]]
            self._trace("erase", stack=stack)
            return original(*args, **kwargs)

        self.app.renderer.erase = traced  # type: ignore[method-assign]

    def _trace(self, kind: str, **fields: Any) -> None:
        """往 `AGENT_TUI_TRACE` 追加一行 JSON。排障专用，默认关。"""
        if not self._trace_path:
            return
        record = {"t": round(time.monotonic(), 3), "kind": kind, **fields}
        try:
            with open(self._trace_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            self._trace_path = ""

    async def next_line(self) -> str | None:
        """等用户提交一行；`None` 表示 Ctrl-C / Ctrl-D。

        先看 `_deferred`：那是"agent 还在干活时用户就打好的字"，被审批提问撞见后
        先寄存在这里，等这一轮结束再按原样当消息发出去。
        """
        if self._deferred:
            return self._deferred.pop(0)
        return await self._lines.get()

    async def confirm(self) -> bool | None:
        """在界面里问一句：放行还是拒绝。

        返回 `True` 放行、`False` 拒绝、`None` 表示用户按了 Ctrl-C（他要走）。
        **默认是拒绝**：直接回车、按 n、或者答非所问地打了别的内容，都不会放行；
        打成消息的那种会先寄存起来（`_deferred`），不吞掉用户打的字。
        """
        self._asking = True
        self.refresh()
        try:
            while True:
                # 直接等新输入，不走 next_line()：不然刚寄存的那行会被自己再取回来
                raw = await self._lines.get()
                if raw is None:
                    return None
                answer = raw.strip().lower()
                if answer in {"y", "yes", "是", "允许"}:
                    return True
                if not answer or answer in {"n", "no", "否", "拒绝"}:
                    return False
                self._deferred.append(raw)  # 这是用户想说的话，不是答案
                self.write("这里是问 y/n：y 放行、回车或 n 拒绝（你那句话等这轮完再发）\n", "warn")
        finally:
            self._asking = False
            self.refresh()


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
    ctx.db = db  # recall（取回被压缩的工具原文）要用
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
        context_limit=settings.context_limit,
        compress_enabled=settings.compress_enabled,
        compress_lossless_ratio=settings.compress_lossless_ratio,
        compress_summary_ratio=settings.compress_summary_ratio,
        compress_keep_recent=settings.compress_keep_recent_tool_results,
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


class _TuiOutput:
    """会话期间把 stdout/stderr 收进界面的日志区。

    为什么不用 `patch_stdout()`：它把"往终端写"变成"擦屏 → 打印 → 重画"，
    而全屏界面下这一擦就是**整屏**——只要有任何东西（库、警告、忘了关的 print）
    写一次，用户看到的就是"屏幕闪一下/清一次"。与其要求"谁都不许写"，不如
    把管道换掉：谁写都进界面日志（暗灰 `·` 行），终端只剩渲染器一个写者。
    """

    def __init__(self, tui: ChatTUI) -> None:
        self._tui = tui
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._tui.write(f"· {line}\n", "tool")
        return len(text)

    def flush(self) -> None:
        return None

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("界面对话期间没有 fileno")

    @property
    def encoding(self) -> str:
        return "utf-8"


class _TuiLogHandler(logging.Handler):
    """把日志塞进界面自己的日志区。

    为什么需要：`logging` 的 handler 在 `configure_logging()` 时就把**原始** stderr
    抓在手里了，`patch_stdout()` 换不掉它——一旦有日志，它就绕过界面直接往
    备用屏幕上写，界面被涂花/被迫整体重画（"屏幕自己在清"就有它一份）。
    TUI 跑着的时候终端只能有一个写者，所以日志也走界面。
    """

    def __init__(self, tui: ChatTUI) -> None:
        super().__init__(level=logging.INFO)
        self._tui = tui

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._tui.write(f"· {record.getMessage()}\n", "tool")
        except Exception:  # 日志出问题绝不能把会话带走
            pass


async def _loop_tui(client: Any, session_id: str, status: ChatStatus, auto_approve: bool) -> None:
    """全屏界面主循环。

    界面占满整个窗口：上面是日志区（内部滚动，每帧只画最后几行），最后两行
    是状态栏 + 输入行。输出全部走 `tui.write()` 进界面缓冲区，一个字节都不写
    stdout，所以底栏**永远**在窗口最底下，跟这一轮输出了多少行无关。

    进来先把库里的历史画一遍（`follow=False` 拿完就断）——`-s` 和 resume 都靠它，
    不然恢复会话是"空屏 + 等你发第一句话才哗啦倒出来"。

    会话期间 `sys.stdout` / `sys.stderr` 被换成 `_TuiOutput`（写进来的东西进界面
    日志区）：全屏下"擦屏"只可能来自有东西往终端写，把管道换掉就没人能擦了。
    """
    tui = ChatTUI(status)
    last_seq = await _restore_backlog(tui, client, session_id)
    # 会话期间日志也进界面：别让任何东西从旁边往备用屏幕上写
    log_handler = _TuiLogHandler(tui)
    logging.getLogger().addHandler(log_handler)
    guard = _TuiOutput(tui)
    with contextlib.redirect_stdout(guard), contextlib.redirect_stderr(guard):  # type: ignore[arg-type]
        runner = asyncio.create_task(tui.run())
        ticker = asyncio.create_task(_spin(tui))
        try:
            while True:
                raw = await tui.next_line()
                if raw is None:  # Ctrl-C / Ctrl-D：界面已退出，必须跟着走
                    break
                line = raw.strip()
                if not line:
                    continue
                if line.lower() in {"exit", "quit", ":q"}:
                    break
                tui.echo_user(line)
                status.begin_turn()
                status.set_phase("思考中")
                tui.refresh()
                try:
                    sent = await client.send_message(session_id, line)
                except Exception as exc:  # 发一条消息失败不该把整个会话带走
                    tui.write(f"发送失败：{exc}\n")
                    status.clear_phase()
                    tui.refresh()
                    continue
                turn_id = str(sent["turn_id"])
                quitting = False
                try:
                    async for event in client.stream_events(session_id, after_seq=last_seq):
                        last_seq = event.seq
                        kind = event.type.value
                        if kind == "approval.required":
                            # 在界面上直接问：y 放行 / 回车拒绝 / Ctrl-C 连人带会话一起走
                            granted: bool | None = (
                                True if auto_approve else await _ask_in_tui(tui, status, event.data)
                            )
                            try:
                                await client.approve(
                                    session_id,
                                    str(event.data.get("call_id", "")),
                                    granted=bool(granted),
                                )
                            except Exception as exc:  # 服务端可能已经等到超时了
                                tui.write(f"答复没能送达服务端：{exc}\n", "error")
                            if granted is None:
                                quitting = True
                                break
                            continue
                        if kind == "tool.call":
                            status.set_phase(f"执行 {event.data.get('name')}")
                            status.count_tool(str(event.data.get("name", "")))
                        elif kind == "tool.result":
                            status.set_phase("思考中")
                        elif kind == "reasoning.delta":
                            status.set_phase("思考中")
                        elif kind == "text.delta":
                            status.set_phase("输出中")
                        # 过程信息（工具调用/等待确认）走暗灰并可折叠；其余是正文
                        tui.write(format_event_plain(event), _event_line_kind(kind))
                        if kind == "turn.done" and event.turn_id == turn_id:
                            status.track(event.data.get("usage") or {})
                            break
                finally:
                    # 正常收尾和流中途断掉都要收回空闲：底栏要是一直转圈，
                    # 那它显示的就不是状态，是谎话
                    status.clear_phase()
                    tui.refresh()
                if quitting:
                    break
        finally:
            ticker.cancel()
            tui.exit()
            await asyncio.gather(runner, ticker, return_exceptions=True)
    logging.getLogger().removeHandler(log_handler)


async def _restore_backlog(tui: ChatTUI, client: Any, session_id: str) -> int:
    """把库里已有的对话画进日志区，返回最后一条事件的 seq。

    用 `follow=False`（服务端早就支持，回放完就断）先取一批历史：这样 `-s`
    和 resume 进来时历史是**摆好的**，而不是空屏等你发第一句话才哗啦倒出来。
    只看正文/思考/工具/错误——历史里的 `turn.done` 没必要再演一遍，
    历史里的审批请求也不该在恢复时又冒出来一次。
    """
    last_seq = 0
    async for event in client.stream_events(session_id, after_seq=0, follow=False):
        last_seq = event.seq
        kind = event.type.value
        if kind == "turn.started":
            prompt = str((event.data or {}).get("prompt", ""))
            if prompt:
                tui.echo_user(prompt)
            continue
        if kind in {"turn.done", "approval.required"}:
            continue
        text = format_event_plain(event)
        if text:
            tui.write(text, _event_line_kind(kind))
    return last_seq


async def _ask_in_tui(tui: ChatTUI, status: ChatStatus, data: dict[str, Any]) -> bool | None:
    """把审批请求摆到界面上，等用户拍板。

    返回 `True` 放行、`False` 拒绝、`None` 表示用户按了 Ctrl-C 要走。
    """
    for line in approval_lines(data):
        tui.write(line + "\n", "warn")
    status.set_phase("等待确认")
    tui.refresh()
    try:
        granted = await tui.confirm()
    finally:
        status.set_phase("思考中")  # 这一轮还没完，别显示成空闲
        tui.refresh()
    tui.write("已允许\n" if granted else "已拒绝\n", "tool")
    return granted


async def _spin(tui: ChatTUI) -> None:
    """每 0.1 秒推一次界面：转圈 + 把攒下的改动画出去（限流在 `ChatTUI._paint`）。

    输出不再"每个分片画一帧"，而是攒到这里的节拍上——一秒最多十帧，
    终端不会被刷爆。
    """
    while True:
        await asyncio.sleep(0.1)
        tui.tick()


if __name__ == "__main__":
    app()
