"""全屏交互界面：会话选择器、常驻底栏、日志重定向、界面内审批。

从 `main.py` 拆出来的。依赖 `ui.py`（状态栏数据与纯渲染函数），
**不反向依赖 main.py**——命令层想要界面就 `from .tui import ChatTUI`。
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any

from .ui import (
    TUI_STYLE,
    ChatStatus,
    _event_line_kind,
    _fit,
    _group_of,
    _line_fragments,
    _needs_separator,
    _process_summary,
    _wrap_fragments,
    approval_lines,
    format_event_plain,
    session_choices,
)


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
        if self._asking:
            # 「怎么答」常驻底栏：日志滚上去之后，不用再回去找那一行
            fragments.append(("class:bar.warn", " · 回车 允许 / n 拒绝"))
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

        **回车 = 允许**（和 Codex 一致，一次按键就够）。协议里的"默认拒绝"说的是
        **拿不到答复**的情况——超时、没有审批通道、Ctrl-C——那些仍然一律不放行。
        打了别的内容不算答复：那多半是用户想说的话，先寄存起来（`_deferred`），
        不吞掉他打的字。
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
                if not answer or answer in {"y", "yes", "是", "允许"}:
                    return True
                if answer in {"n", "no", "否", "拒绝"}:
                    return False
                self._deferred.append(raw)  # 这是用户想说的话，不是答案
                self.write("这里在问审批：回车允许、n 拒绝（你那句话等这轮完再发）\n", "warn")
        finally:
            self._asking = False
            self.refresh()


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
                            # 在界面上直接问：回车放行 / n 拒绝 / Ctrl-C 连人带会话一起走
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
                            raw_round = event.data.get("round")
                            status.count_tool(
                                str(event.data.get("name", "")),
                                round_no=int(raw_round) if raw_round is not None else None,
                            )
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
    for line in approval_lines(data, width=tui.app.output.get_size().columns):
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
