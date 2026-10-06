"""界面渲染：状态栏、配色、纯文本格式化、审批块。

从 `main.py` 拆出来的。这里只放**不依赖命令行框架**的东西：

- `EventRenderer`：`agent run` 那条路的富文本行式渲染；
- `ChatStatus`：全屏界面底栏的数据（工具往返、上下文占用、重复告警）；
- 一组纯函数：`format_event_plain` / `approval_lines` / `_fit` / 代码高亮 / 折行。

界面本身（`ChatTUI`、会话选择器、日志重定向、审批交互）在 `tui.py`。
依赖方向只有一条：`main.py` → `tui.py` → `ui.py`。纯函数放这里是为了能直接测。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

console = Console()


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
    #: 本轮的**工具往返**次数（不是单次工具调用：一批里调三个工具也只算一次往返）
    #: 与上限，以及"同一个工具被反复调用"的告警（第 4 节）。
    #: 全部由事件本地计数——和给模型的状态块同源，但**渲染是两份**（4.6）。
    tool_rounds: int = 0
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
        self.tool_rounds = 0
        self.repeat = ""
        self._tool_counts = {}

    def count_tool(self, name: str, *, round_no: int | None = None) -> None:
        """记一次工具调用。

        `round_no` 由服务端给（`tool.call` 事件里的 `round`）——**往返次数必须用服务端的数**，
        否则一批里调三个工具，客户端会数成 3，而真正的硬上限只走了 1。
        老服务端不带这个字段时退化成"见了 tool.call 就 +1"。
        """
        self.tool_rounds = round_no if round_no is not None else self.tool_rounds + 1
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
            f"工具往返 {self.tool_rounds:>2}/{self.tool_limit}"
            if self.tool_limit
            else f"工具往返 {self.tool_rounds:>2}"
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


#: 审批块里的标签列宽（"命令" / "原因" 都是 4 列中文）
_APPROVAL_LABEL = 4

#: 审批块最多摊开几行命令。超了给明确省略标记——完整内容永远在事件里
#: （`agent replay <会话>` 能看到全文），屏幕上一屏糊满反而看不清要批什么。
_APPROVAL_MAX_LINES = 8


def approval_lines(
    data: dict[str, Any], *, width: int = 100, max_lines: int = _APPROVAL_MAX_LINES
) -> list[str]:
    """审批提示的几行纯文本（样式由界面统一给）。

    这里要同时满足两件冲突的事：**让人看清要跑什么** 和 **别糊一屏**。取舍：

    - 命令逐行摊开（`shell_exec` 不用自己去解 JSON），但每行裁到终端宽度、
      整体最多 `max_lines` 行；裁掉多少写清楚，不装作没裁。
    - 标签列对齐，答复提示放在**最后一行**，紧贴输入框。
    - "怎么答"同时常驻底栏（`_status_fragments`），滚动时也看得见。
    """
    args = data.get("args") or {}
    detail = args.get("command") if isinstance(args, dict) else None
    if not detail:
        detail = json.dumps(args, ensure_ascii=False)
    detail = str(detail)

    limit = max(20, width - _APPROVAL_LABEL - 2)
    raw_lines = detail.splitlines() or [""]

    def row(label: str, text: str) -> str:
        """标签列对齐：`命令` / `原因` / 续行都从同一列开始。"""
        return f"  {_fit(label, _APPROVAL_LABEL)}  {_fit(text, limit).rstrip()}"

    lines = [f"⚠ 需要确认  {data.get('name')}"]
    for index, raw in enumerate(raw_lines[:max_lines]):
        lines.append(row("命令" if index == 0 else "", raw))
    if len(raw_lines) > max_lines:
        lines.append(
            f"  {' ' * _APPROVAL_LABEL}  …（命令共 {len(raw_lines)} 行 / {len(detail)} 字符，"
            f"只显示前 {max_lines} 行；全文见 agent replay）"
        )
    if data.get("reason"):
        lines.append(row("原因", str(data["reason"])))
    lines.append("  回车 允许 · n 拒绝 · 超时按拒绝")
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
