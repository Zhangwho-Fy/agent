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

    def track(self, usage: dict[str, Any]) -> None:
        self.last_input = int(usage.get("input_tokens", 0))
        self.last_output = int(usage.get("output_tokens", 0))
        self.total += self.last_input + self.last_output
        self.turns += 1

    def text(self) -> str:
        """整条状态栏的纯文本。行式底栏（`PromptSession` 的 toolbar）要的就是它。"""
        return "".join(text for _style, text in self.segments())

    def segments(self, *, aligned: bool = False) -> list[tuple[str, str]]:
        """状态栏的分段内容（样式类, 文本）。

        这里只负责"哪一段是什么"，颜色在 `TUI_STYLE` 里定；纯文本由 `text()`
        拼出来，两条路共用同一份内容，不会各写一遍。

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

        return [
            ("class:bar.model", f" {model}"),
            ("class:bar.dim", " │ 会话 "),
            ("class:bar.session", session),
            ("class:bar.dim", f" │ {self.workspace} │ "),
            (usage, used),
            ("class:bar.dim", f" │ {history} │ {total} │ {turns}"),
        ]

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


def tui_enabled() -> bool:
    """是否启用"常驻界面"。

    **默认关**。教训是：界面这种东西我这边没有真实终端可验，所以它必须是
    opt-in——坏了你只要去掉环境变量就回到行式输出，不用改代码、不用回滚。

    想试：`AGENT_CHAT_UI=tui .venv/bin/agent chat ...`
    """
    import os

    if os.environ.get("AGENT_CHAT_UI", "").lower() != "tui":
        return False
    if not sys.stdout.isatty():
        return False
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return False
    return True


def format_event_plain(event: Any) -> str:
    """事件 → 纯文本。

    常驻界面里**不能再让 rich 插手**：它和界面渲染会抢同一块终端，
    上次就是这么把 ANSI 序列打进正文的（屏幕上出现 `?[2m` 那种）。
    所以这条路只产出纯文本，界面自己负责上色和布局。
    """
    data = event.data
    kind = event.type.value
    if kind == "text.delta":
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
    if kind == "error":
        return f"错误：{data.get('message')}\n"
    return ""


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
    "bar.busy": "#e0af68 bold",  # 执行中的阶段：黄
    "bar.idle": "#9ece6a",  # 空闲：绿点
    "bar.hint": "#bb9af7",  # 回滚提示：紫
    "prompt": "#7aa2f7 bold",  # 输入提示符，跟模型名一个色系
    "user": "bold ansibrightblue",
    "tool": "ansibrightblack",  # Codex 那种暗灰：过程信息一律退到背景里去
    "error": "ansired bold",
    "code": "ansibrightblack",
    "code.keyword": "ansibrightmagenta",
    "code.string": "ansigreen",
    "code.number": "ansibrightyellow",
    "code.comment": "ansibrightblack italic",
    "code.func": "ansicyan",
    "code.builtin": "ansibrightcyan",
    "code.op": "",
}

#: 日志行的类型 → 样式类。空串表示"用终端默认前景色"。
LINE_CLASS = {"user": "class:user", "assistant": "", "tool": "class:tool", "error": "class:error"}


def _group_of(kind: str) -> str:
    """把行归类，用来决定"哪里该空一行"和"哪些行能折叠到一起"。"""
    if kind.startswith("code:"):
        return "code"
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

    def __init__(self, status: ChatStatus) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
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
        self.input = Buffer(multiline=False, accept_handler=self._on_accept)

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
            self.refresh()

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
                                # 提示符常驻在最左边，光标钉在最后一行
                                Window(
                                    FormattedTextControl("› ", style="class:prompt"),
                                    width=2,
                                    height=1,
                                ),
                                Window(BufferControl(buffer=self.input), height=1),
                            ]
                        ),
                    ]
                )
            ),
            key_bindings=bindings,
            style=style,
            full_screen=True,  # 备用屏幕：底栏钉在窗口最底，退出后终端复原
        )

    def _effective_pending_kind(self) -> str:
        """还在流式的那一行属于什么类型（代码块里就是代码行）。"""
        if self._code is not None and self._pending_kind == "assistant":
            return f"code:{self._code}"
        return self._pending_kind

    def _display_items(self, limit: int) -> list[tuple[str, list[tuple[str, str]]]]:
        """把最后 `limit` 条日志整理成"这一帧要画的行"。

        做两件事：换组时插一个空行（正文 / 工具 / 用户各自成段），以及把连着的
        过程信息折成一行。只处理尾巴，所以每帧的工作量跟对话长短无关。
        """
        start = max(0, len(self._log) - limit)
        # 折叠得看到整段的开头，否则会把"延续行"当成第一行
        while (
            self._folded
            and start > 0
            and self._log[start][0] == "tool"
            and self._log[start - 1][0] == "tool"
        ):
            start -= 1

        items: list[tuple[str, list[tuple[str, str]]]] = []
        previous = _group_of(self._log[start - 1][0]) if start > 0 else None
        for kind, text in self._log[start:]:
            if _needs_separator(items, previous, _group_of(kind)):
                items.append(("", []))
            fragments = _line_fragments(kind, text)
            if not fragments and items and not items[-1][1]:
                continue  # 上一行已经是空行，别再堆一行
            items.append((kind, fragments))
            previous = _group_of(kind)

        if self._pending:
            kind = self._effective_pending_kind()
            if _needs_separator(items, previous, _group_of(kind)):
                items.append(("", []))
            items.append((kind, _line_fragments(kind, self._pending)))

        return self._fold(items)

    def _fold(
        self, items: list[tuple[str, list[tuple[str, str]]]]
    ) -> list[tuple[str, list[tuple[str, str]]]]:
        """折叠：连着的过程信息（工具调用）只留第一行，其余收起来。"""
        if not self._folded:
            return items
        folded: list[tuple[str, list[tuple[str, str]]]] = []
        index = 0
        while index < len(items):
            if items[index][0] != "tool":
                folded.append(items[index])
                index += 1
                continue
            start = index
            while index < len(items) and items[index][0] == "tool":
                index += 1
            run = items[start:index]
            if len(run) == 1:
                folded.extend(run)
                continue
            first = "".join(text for _style, text in run[0][1])
            folded.append(("tool", [("class:tool", f"{first}   …共 {len(run)} 行，Ctrl-O 展开")]))
        return folded

    def _render_lines(self, rows: int, columns: int) -> list[list[tuple[str, str]]]:
        """日志区这一帧要画的行（片段化），折行后取最后 (rows-2) 行。

        只处理最后 (rows-2 + 回滚行数) 条**源日志**：一条源日志至少占一屏行，
        再往前的那些一定看不见。回滚时窗口整体往上挪 `_scroll` 屏行。
        """
        height = max(1, rows - 2)  # 让出状态栏和输入行
        wrapped: list[list[tuple[str, str]]] = []
        for _kind, fragments in self._display_items(height + self._scroll + 1):
            wrapped.extend(_wrap_fragments(fragments, columns))
        end = min(len(wrapped), max(height, len(wrapped) - self._scroll))
        return wrapped[max(0, end - height) : end]

    def _page_size(self) -> int:
        """一屏能看多少行日志（让出底栏两行）。"""
        return max(1, self.app.output.get_size().rows - 3)

    def _scroll_by(self, delta: int) -> None:
        """滚动日志区。`delta > 0` 往上看历史，负数往回走；到头/到底自动夹住。"""
        size = self.app.output.get_size()
        height = max(1, size.rows - 2)
        total = sum(
            len(_wrap_fragments(fragments, size.columns))
            for _kind, fragments in self._display_items(10_000)
        )
        self._scroll = min(max(0, total - height), max(0, self._scroll + delta))
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
            lead = ("class:bar.busy", f" {self.status.spinner()} {self.status.phase} ")
        else:
            lead = ("class:bar.idle", " ● 空闲 ")
        # 徽标补到固定宽度：不然"空闲"和"执行 fs_read"一换，后面全跟着平移
        fragments = [(lead[0], _fit(lead[1], 20)), *self.status.segments(aligned=True)]
        if self._scroll:
            fragments.append(("class:bar.hint", f" │ ↑回滚 {self._scroll} 行（End 回底）"))
        return fragments

    def _on_accept(self, buffer: Any) -> bool:
        self._lines.put_nowait(buffer.text)
        buffer.reset()
        return True

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
        """
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
        self.app.invalidate()

    async def next_line(self) -> str | None:
        return await self._lines.get()


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
        if tui_enabled():
            await _loop_tui(client, session_id, status, auto_approve)
            console.print("\n[dim]再见[/dim]")
            return 0
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


async def _loop_tui(client: Any, session_id: str, status: ChatStatus, auto_approve: bool) -> None:
    """全屏界面主循环。

    界面占满整个窗口：上面是日志区（内部滚动，每帧只画最后几行），最后两行
    是状态栏 + 输入行。输出全部走 `tui.write()` 进界面缓冲区，一个字节都不写
    stdout，所以底栏**永远**在窗口最底下，跟这一轮输出了多少行无关。

    `patch_stdout()` 只当保险留着：这条路自己不该有任何 stdout 写入，但库偶尔
    会写（比如日志），让它走界面渲染器，总比直接糊在备用屏幕上好。
    """
    from prompt_toolkit.patch_stdout import patch_stdout

    tui = ChatTUI(status)
    with patch_stdout():
        runner = asyncio.create_task(tui.run())
        ticker = asyncio.create_task(_spin(tui))
        last_seq = 0
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
                try:
                    async for event in client.stream_events(session_id, after_seq=last_seq):
                        last_seq = event.seq
                        kind = event.type.value
                        if kind == "approval.required":
                            # 界面里的交互式审批还没做：目前只认 --yes，否则按拒绝
                            tui.write(f"\n需要确认：{event.data.get('name')}（此处暂按拒绝）\n")
                            await client.approve(
                                session_id,
                                str(event.data.get("call_id", "")),
                                granted=bool(auto_approve),
                            )
                            continue
                        if kind == "tool.call":
                            status.set_phase(f"执行 {event.data.get('name')}")
                        elif kind == "tool.result":
                            status.set_phase("思考中")
                        elif kind == "text.delta":
                            status.set_phase("输出中")
                        # 过程信息（工具调用/等待确认）走暗灰并可折叠；其余是正文
                        line_kind = (
                            "error"
                            if kind == "error"
                            else "tool"
                            if kind in {"tool.call", "tool.result", "approval.required"}
                            else "assistant"
                        )
                        tui.write(format_event_plain(event), line_kind)
                        if kind == "turn.done" and event.turn_id == turn_id:
                            status.track(event.data.get("usage") or {})
                            break
                finally:
                    # 正常收尾和流中途断掉都要收回空闲：底栏要是一直转圈，
                    # 那它显示的就不是状态，是谎话
                    status.clear_phase()
                    tui.refresh()
        finally:
            ticker.cancel()
            tui.exit()
            await asyncio.gather(runner, ticker, return_exceptions=True)


async def _spin(tui: ChatTUI) -> None:
    """执行期间让底栏逐帧转圈——它得"看着是活的"，而不是一张静止的图。

    只在有阶段时刷新，空闲时一个 tick 都不发，省得白白重绘。
    """
    while True:
        await asyncio.sleep(0.1)
        if tui.status.phase:
            tui.status.tick()
            tui.refresh()


def _ask_approval(data: Any) -> bool:
    """终端里问一句。默认拒绝：直接回车不放行。"""
    args = json.dumps(data.get("args", {}), ensure_ascii=False)
    console.print(f"[dim]原因：{data.get('reason', '')}[/dim]")
    answer = input(f"  允许执行 {args} 吗？[y/N] ").strip().lower()
    return answer in {"y", "yes", "是"}


if __name__ == "__main__":
    app()
