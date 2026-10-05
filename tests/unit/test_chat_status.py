"""交互式状态栏的格式与降级逻辑。

底栏本身渲染不了（测试里没有终端），但"算出来的字符串对不对""没装依赖时会不会
悄悄炸掉"这两件事可以直接测——它们才是真正容易错的部分。
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from agent.client.main import ChatStatus, ChatTUI


def _plain(status: ChatStatus) -> str:
    """状态栏的纯文本形态：把分段拼起来（界面里是带样式的同一份内容）。"""
    return "".join(text for _style, text in status.segments())


def test_status_summarizes_usage_and_context() -> None:
    status = ChatStatus(
        model="deepseek-v4-flash",
        workspace="/mnt/g/code/agent",
        session_id="sess_f4f8206a790244658e20949066692bd3",
        context_limit=1000,
    )
    status.track({"input_tokens": 120, "output_tokens": 30})
    status.track({"input_tokens": 200, "output_tokens": 50})

    text = _plain(status)

    assert "deepseek-v4-flash" in text
    assert "/mnt/g/code/agent" in text
    assert "…692bd3" in text, "会话只显示后 6 位，别把整屏塞满"
    assert "上下文 200/1000（20%）" in text, "用最近一次调用的输入量表示占用"
    assert "本轮 ↑200 ↓50" in text
    assert "累计 400" in text, "两轮合计 (120+30)+(200+50)"
    assert "第 2 轮" in text


def test_status_without_context_limit_still_reads_well() -> None:
    status = ChatStatus(model="m", session_id="sess_1")
    status.track({"input_tokens": 7})
    assert "上下文 7" in _plain(status)
    assert "/" not in _plain(status).split("上下文")[1].split("│")[0]


def test_chat_needs_a_real_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """非终端里开不出全屏界面，要说人话并指路 `agent run`，而不是吐一堆控制字符。"""
    from agent.client.main import tui_problem

    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    problem = tui_problem()
    assert problem is not None and "终端" in problem

    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert tui_problem() is None


def _bar(tui: ChatTUI) -> str:
    return "".join(text for _style, text in tui._status_fragments())


def test_status_bar_has_its_own_palette() -> None:
    """状态栏要有固定底色：`reverse` 在深色终端上会翻成一条白条，单调又晃眼。"""
    from agent.client.main import TUI_STYLE

    assert "bg:" in TUI_STYLE["bar"], "底色挂在 bar 上，Window 才能铺满整行"
    assert "reverse" not in TUI_STYLE["bar"]


def test_status_segments_colour_by_meaning() -> None:
    """模型名/会话号/上下文占用各自一段：占用越高越红。"""
    status = ChatStatus(model="m", session_id="sess_abcdef", context_limit=100)

    def styles() -> list[str]:
        return [style for style, _text in status.segments()]

    assert "class:bar.model" in styles()
    assert "class:bar.session" in styles()

    status.track({"input_tokens": 10})
    assert "class:bar.usage.ok" in styles()
    status.track({"input_tokens": 60})
    assert "class:bar.usage.warn" in styles()
    status.track({"input_tokens": 95})
    assert "class:bar.usage.hot" in styles(), "快到上下文上限了要显眼"

    assert _plain(status), "拼起来就是给人看的那一行，不能是空串"


def test_status_fragments_swap_the_phase_badge() -> None:
    tui = ChatTUI(ChatStatus(model="m", session_id="sess_1"))
    assert tui._status_fragments()[0][0] == "class:bar.idle"
    assert "class:bar.model" in [style for style, _text in tui._status_fragments()]

    tui.status.set_phase("思考中")
    assert tui._status_fragments()[0][0] == "class:bar.busy"


def test_status_bar_keeps_its_columns_when_things_change() -> None:
    """阶段名变长、token 进位，后面的列不能跟着挪（就是"制表位"）。"""
    tui = ChatTUI(
        ChatStatus(
            model="deepseek-v4-flash",
            workspace="/mnt/g/code/agent",
            session_id="sess_aabbcc",
            context_limit=128000,
        )
    )
    tui.status.track({"input_tokens": 7, "output_tokens": 1})
    idle = _bar(tui)

    tui.status.set_phase("执行 shell_exec")
    busy = _bar(tui)
    tui.status.track({"input_tokens": 12345, "output_tokens": 6789})
    grown = _bar(tui)

    assert len(idle) == len(busy) == len(grown), "整条的宽度都该是稳的"
    assert idle.index("deepseek-v4-flash") == busy.index("deepseek-v4-flash")
    assert idle.index("│ 本轮") == busy.index("│ 本轮") == grown.index("│ 本轮")
    assert "执行 shell_exec" in busy, "徽标够宽，别把阶段名截了"


def test_tui_bar_is_always_drawn_and_shows_phase() -> None:
    """底栏必须**一直在**：空闲时也画，执行时多一个转圈的阶段。"""
    status = ChatStatus(model="m", session_id="sess_1")
    tui = ChatTUI(status)

    assert "空闲" in _bar(tui)
    assert status.spinner() not in _bar(tui), "空闲时不该有 spinner"

    status.set_phase("执行 fs_read")
    status.tick()
    busy = _bar(tui)
    assert "执行 fs_read" in busy
    assert status.spinner() in busy

    status.clear_phase()
    assert "空闲" in _bar(tui), "跑完一轮只是不再转圈，底栏不消失"


async def test_tui_enter_hands_the_line_to_the_loop() -> None:
    """回车把整行交给主循环；清空输入行交给 prompt_toolkit 自己做。

    返回 False 是**故意的**：它清空之前会先把内容记进历史，我们抢着 reset
    的话历史里就是空串，Ctrl-R 搜索等于白装。
    """
    tui = ChatTUI(ChatStatus())

    tui.input.insert_text("你好")
    tui.input.validate_and_handle()

    assert tui.input.text == "", "prompt_toolkit 会清空输入框"
    assert tui.input.history.get_strings() == ["你好"], "历史要留一份，Ctrl-R 才搜得到"
    assert await tui.next_line() == "你好"


def test_tui_never_writes_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """界面自己画：日志进缓冲区，stdout 一个字节都不能有。

    写 stdout 就得跟终端的光标位置较劲，上一版底栏被拽进正文正是这么来的。
    """
    tui = ChatTUI(ChatStatus())

    tui.write("你好，世界\n继续")
    assert capsys.readouterr().out == ""
    assert tui._log == [("assistant", "你好，世界")], "换好行的进 _log"
    assert tui._pending == "继续", "没换行的尾巴继续长，不能被当成一整行"

    tui.write("，还有更多\n")
    assert tui._log == [("assistant", "你好，世界"), ("assistant", "继续，还有更多")]
    assert tui._pending == ""


def _texts(lines: list[list[tuple[str, str]]]) -> list[str]:
    return ["".join(text for _style, text in line) for line in lines]


def test_repaints_are_throttled() -> None:
    """流式一秒能来几百个分片，不能每个都重画——攒到节拍上一起画。"""

    class _App:
        def __init__(self) -> None:
            self.paints = 0

        def invalidate(self) -> None:
            self.paints += 1

    tui = ChatTUI(ChatStatus())
    fake = _App()
    tui.app = fake

    for _ in range(200):
        tui.write("思考中…\n", "reason")

    assert len(tui._log) == 200, "内容一条都不能丢"
    assert fake.paints <= 3, f"200 个分片只该画几帧，实际 {fake.paints} 帧"

    before = fake.paints
    tui.tick()
    assert fake.paints == before + 1, "节拍到了要把攒下的改动画出去"


def test_logging_is_routed_into_the_transcript() -> None:
    """日志得走界面：logging 的 handler 抓的是原始 stderr，会绕过界面把屏幕涂花。"""
    import logging

    from agent.client.main import _TuiLogHandler

    tui = ChatTUI(ChatStatus())
    record = logging.LogRecord(
        name="agent.server.app",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="恢复：1 个没跑完的 turn 已标记为 interrupted",
        args=(),
        exc_info=None,
    )
    _TuiLogHandler(tui).emit(record)

    assert tui._log == [("tool", "· 恢复：1 个没跑完的 turn 已标记为 interrupted")]


def test_stray_stdout_writes_land_in_the_transcript() -> None:
    """会话期间谁写 stdout 都进界面日志，而不是去擦屏（那正是"屏幕闪一下"的来源）。"""
    from agent.client.main import _TuiOutput

    tui = ChatTUI(ChatStatus())
    guard = _TuiOutput(tui)

    assert guard.isatty() is False, "别让 prompt_toolkit 以为这是终端"
    guard.write("一条库打印的提示")
    assert tui._log == [], "没换行先攒着"
    guard.write("\n第二行\n")
    assert tui._log == [("tool", "· 一条库打印的提示"), ("tool", "· 第二行")]
    guard.flush()


def test_transcript_renders_only_the_tail_that_fits() -> None:
    """日志区只画最后几行——底栏在下面两行，不能被日志挤走。"""
    tui = ChatTUI(ChatStatus())
    tui.write("".join(f"第{index}行\n" for index in range(20)))

    tail = _texts(tui._render_lines(5, 20))
    assert tail == ["第17行", "第18行", "第19行"], "5 行窗口 → 日志区 3 行"
    assert _texts(tui._render_lines(3, 20)) == ["第19行"]


def test_scroll_looks_back_and_clamps() -> None:
    """回滚看前文：越往上越早，到顶/到底自动停住。"""
    tui = ChatTUI(ChatStatus())
    tui.write("".join(f"第{index}行\n" for index in range(200)))

    tui._scroll_by(10_000)
    top = tui._scroll
    assert top > 0, "应该滚得动"
    tui._scroll_by(10_000)
    assert tui._scroll == top, "到最上面就该停住，不能无限加"
    tui._scroll_by(-10_000)
    assert tui._scroll == 0, "回到底部跟着最新输出走"
    assert tui._frozen is None, "回到底部要解冻"

    small = ChatTUI(ChatStatus())
    small.write("".join(f"第{index}行\n" for index in range(60)))
    assert _texts(small._render_lines(5, 20)) == ["第57行", "第58行", "第59行"]
    small._scroll_by(2)  # 往上两行：底部三行是 57/58/59，再往上就是 55/56/57
    assert _texts(small._render_lines(5, 20)) == ["第55行", "第56行", "第57行"]
    assert "浏览历史 2 行" in _bar(small), "回滚时底栏要说明现在不在最新位置"


def test_scrolled_view_is_frozen_while_output_keeps_coming() -> None:
    """滚上去之后视窗要冻住：否则新输出会把视图推着走（看着像屏幕自己在动）。"""
    tui = ChatTUI(ChatStatus())
    tui.write("".join(f"第{index}行\n" for index in range(40)))
    tui._scroll_by(3)
    before = _texts(tui._render_lines(8, 20))

    tui.write("".join(f"新{index}行\n" for index in range(5)))

    assert _texts(tui._render_lines(8, 20)) == before, "滚上去之后视窗不该再动"
    assert "5 行新输出" in _bar(tui), "底栏要告诉你攒了多少新内容"

    tui._scroll_by(-10_000)  # 回到底部：解冻，跟最新
    assert tui._frozen is None
    assert _texts(tui._render_lines(8, 20))[-1] == "新4行"


def test_a_huge_folded_run_does_not_blank_the_screen() -> None:
    """折叠把几百行思考压成 1 行时，取窗口不能整个落在那一段里——那会让屏幕看着像被清空。"""
    tui = ChatTUI(ChatStatus())
    tui.echo_user("第一问")
    tui.write("这是回答\n", "assistant")
    tui.write("".join(f"思考第{index}行\n" for index in range(500)), "reason")

    lines = _texts(tui._render_lines(20, 100))  # 20 行窗口 → 日志区 18 行

    body = "\n".join(lines)
    assert "你 > 第一问" in body, f"问题被折没了：{lines}"
    assert "这是回答" in body
    assert body.index("你 > 第一问") < body.index("这是回答") < body.index("▸ 过程")
    assert len(lines) >= 4, f"窗口不该只剩折叠那一行：{lines}"


def test_exit_is_safe_to_call_twice() -> None:
    """Ctrl-C 的绑定已经退过一次，收尾再调一次不该炸（曾经的 Traceback 来源）。"""
    tui = ChatTUI(ChatStatus())
    tui.exit()
    tui.exit()


def test_echo_user_closes_the_previous_line() -> None:
    """上一轮回答没换行时，新的提问不能接在它屁股后面。"""
    first = ChatTUI(ChatStatus())
    first.echo_user("第一问")
    assert first._log == [("user", "你 > 第一问")]

    tui = ChatTUI(ChatStatus())
    tui.write("上一轮的回答没换行")
    tui.echo_user("第二问")
    assert tui._log == [("assistant", "上一轮的回答没换行"), ("user", "你 > 第二问")]
    assert tui._pending == ""


def test_line_colors_distinguish_user_tool_and_body() -> None:
    """用户消息、正文、过程信息各自成段、各自上色（过程信息是暗灰）。"""
    tui = ChatTUI(ChatStatus())
    tui.echo_user("你好")
    tui.write("这是正文\n", "assistant")
    tui.write("→ fs_read\n", "tool")

    items = tui._build_items(0, 100)
    assert [kind for kind, _frags in items] == ["user", "", "assistant", "", "tool"]
    assert [frags[0][0] for _kind, frags in items if frags] == [
        "class:user",
        "class:assistant",
        "class:tool",
    ]


def test_tool_runs_fold_and_unfold() -> None:
    """过程信息默认折成一行；展开后原样显示（Ctrl-O 就是切这个开关）。"""
    tui = ChatTUI(ChatStatus())
    tui.write('→ fs_read {"path": "a.py"}\n', "tool")
    tui.write("  ← ok\n", "tool")
    tui.write("正文\n", "assistant")

    assert _texts(tui._render_lines(10, 100)) == [
        "▸ 过程   2 行   …（Ctrl-O 展开）",
        "",
        "正文",
    ]

    tui._folded = False
    assert _texts(tui._render_lines(10, 100)) == [
        '→ fs_read {"path": "a.py"}',
        "  ← ok",
        "",
        "正文",
    ]


def test_process_folds_reasoning_and_tools_together() -> None:
    """思考与工具调用同属"过程"：折成一行，**钉住开头那句**，不随流式抖。"""
    tui = ChatTUI(ChatStatus())
    tui.write("我们需要先看 builder.py\n", "reason")
    tui.write("再确认条件边\n", "reason")
    tui.write("→ fs_read\n", "tool")

    folded = _texts(tui._render_lines(12, 100))
    assert folded[0].startswith("▸ 过程   3 行：我们需要先看 builder.py"), folded[0]
    assert "再确认条件边" not in folded[0], "别拿『最后在想什么』当预览，那会每来一个 token 都在变"

    # 同一段过程里又来了一句思考：预览一个字都不该动，只有行数变
    tui.write("第三句思考\n", "reason")
    again = _texts(tui._render_lines(12, 100))
    assert again[0].split("：")[1] == folded[0].split("：")[1], "预览必须钉在开头那句"
    assert "4 行" in again[0], again[0]

    tui.write("正文\n", "assistant")
    assert _texts(tui._render_lines(12, 100))[1:] == ["", "正文"]

    # 宽度也得固定成一行：否则行数一变，日志区的"最后几行"会整体挪位（屏幕自己滚）
    for columns in (40, 60, 120):
        lines = _texts(tui._render_lines(12, columns))
        assert len(lines) == 3, f"{columns} 列下折成了别的行数：{lines}"

    tui._folded = False
    assert _texts(tui._render_lines(12, 100))[:4] == [
        "我们需要先看 builder.py",
        "再确认条件边",
        "→ fs_read",
        "第三句思考",
    ]


def test_code_fences_are_hidden_and_highlighted() -> None:
    """围栏本身不上屏，代码本体按 pygments 分词上色（没有 pygments 也能跑）。"""
    tui = ChatTUI(ChatStatus())
    tui.write("看代码：\n```python\ndef f(x):\n    return x + 1\n```\n完事\n", "assistant")

    assert [kind for kind, _text in tui._log] == [
        "assistant",
        "code:python",
        "code:python",
        "assistant",
    ], "开/闭围栏自己不占行"

    code = next(frags for kind, frags in tui._build_items(0, 100) if kind == "code:python")
    styles = {style for style, _text in code}
    assert "class:code.keyword" in styles, "def 是关键字"
    assert "class:code.func" in styles, "f 是函数名"


def test_wrapping_counts_chinese_as_two_columns() -> None:
    """折行要按显示宽度算，否则取到的"最后几行"会对不上。"""
    from agent.client.main import _wrap_fragments

    assert _texts(_wrap_fragments([("", "中文字")], 4)) == ["中文", "字"]
    assert _texts(_wrap_fragments([], 10)) == [""]
    assert _texts(_wrap_fragments([("", "ab\tcd")], 8)) == ["ab    cd"], "制表符展开后正好占满"


def test_fit_pads_and_truncates_by_display_width() -> None:
    """状态栏的"制表位"：补到固定宽度，太长就截断（中文算 2 列）。"""
    from agent.client.main import _fit

    assert _fit("abc", 6) == "abc   "
    assert _fit("abcdef", 4) == "abc…", "留一列给省略号"
    assert _fit("中文字", 4) == "中… ", "中文两列，只放得下一个字加省略号"
    assert _fit("abcdef", 4) == "abc…"
    assert _fit("/very/long/path/x", 8, keep_tail=True) == "…/path/x"


def test_approval_lines_show_what_will_run() -> None:
    """审批的意义就是让人看清要跑什么：shell 命令直接摊开，不用自己解 JSON。"""
    from agent.client.main import approval_lines

    lines = approval_lines(
        {"name": "shell_exec", "args": {"command": "uv run pytest -q"}, "reason": "写操作"}
    )
    assert lines[0] == "⚠ 需要确认：shell_exec"
    assert lines[1] == "  uv run pytest -q"
    assert "写操作" in lines[2]
    assert "回车" in lines[-1], "得写清楚回车=拒绝"

    other = approval_lines({"name": "fs_write", "args": {"path": "a.py"}})
    assert other[1] == '  {"path": "a.py"}'
    assert len(other) == 3, "没有 reason 就不占一行"


def test_prompt_tells_you_when_it_is_asking() -> None:
    """输入行左边那一格要能从"你 > "变成"允许? "，宽度还得一样。"""
    from prompt_toolkit.utils import get_cwidth

    tui = ChatTUI(ChatStatus())
    normal = tui._prompt_fragments()
    assert normal[0][0] == "class:prompt"

    tui._asking = True
    asking = tui._prompt_fragments()
    assert asking[0][0] == "class:warn"
    assert "允许?" in "".join(text for _style, text in asking)
    assert get_cwidth("".join(t for _s, t in normal)) == get_cwidth(
        "".join(t for _s, t in asking)
    ), "换文案不能让输入行整体挪位"


async def test_confirm_y_n_and_ctrl_c() -> None:
    """y 放行、回车/N 拒绝、Ctrl-C 表示"我要走"。"""
    tui = ChatTUI(ChatStatus())

    tui._lines.put_nowait("y")
    assert await tui.confirm() is True
    tui._lines.put_nowait("")
    assert await tui.confirm() is False, "直接回车必须按拒绝算"
    tui._lines.put_nowait("N")
    assert await tui.confirm() is False
    tui._lines.put_nowait(None)
    assert await tui.confirm() is None
    assert tui._asking is False, "问完要恢复输入行文案"


async def test_confirm_does_not_swallow_a_typed_ahead_message() -> None:
    """agent 干活时用户先打好的那句，不能被审批提问当成答案吞掉。"""
    tui = ChatTUI(ChatStatus())
    tui._lines.put_nowait("帮我看看 README")
    tui._lines.put_nowait("y")

    assert await tui.confirm() is True
    assert await tui.next_line() == "帮我看看 README", "那句话要留着，这一轮完再发"


async def test_ask_in_tui_renders_request_and_answer() -> None:
    """整条链路：请求上屏 → 等回答 → 结果落地 → 回到"这一轮还在跑"的状态。"""
    from agent.client.main import _ask_in_tui

    tui = ChatTUI(ChatStatus())
    status = ChatStatus()
    tui._lines.put_nowait("y")

    granted = await _ask_in_tui(
        tui, status, {"name": "shell_exec", "args": {"command": "uv run pytest -q"}}
    )

    assert granted is True
    texts = [text for _kind, text in tui._log]
    assert any("需要确认" in text for text in texts)
    assert texts[-1] == "已允许"
    assert status.phase == "思考中", "审批完这一轮还没结束，别显示成空闲"


def test_session_choices_are_aligned() -> None:
    """resume 选单：值取 session_id，标签按显示宽度对齐（中文两列）。"""
    from prompt_toolkit.utils import get_cwidth

    from agent.client.main import session_choices

    choices = session_choices(
        [
            {
                "id": "sess_abcdef123456",
                "title": "看看 builder.py",
                "updated_at": "2026-10-05T09:30:00+08:00",
                "workspace": "/mnt/g/code/agent",
            },
            {"id": "sess_ffffffffffffff", "title": "", "updated_at": "", "workspace": None},
        ]
    )

    assert [value for value, _label in choices] == ["sess_abcdef123456", "sess_ffffffffffffff"]
    labels = [label for _value, label in choices]
    assert "…123456" in labels[0], "只露 id 尾巴，别把整行塞满"
    assert "(未命名)" in labels[1]
    assert all("│" in label for label in labels)
    assert get_cwidth(labels[0]) == get_cwidth(labels[1]), "两行宽度得一样，选单才是齐的"


class _FakeBacklog:
    """假客户端：只吐一批预置事件，模拟 `follow=False` 那种"放完就断"。"""

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.follow_calls: list[bool] = []

    async def _stream(self, session_id: str, *, after_seq: int = 0, follow: bool = True) -> Any:
        self.follow_calls.append(follow)
        for event in self._events:
            yield event

    def stream_events(self, session_id: str, *, after_seq: int = 0, follow: bool = True) -> Any:
        return self._stream(session_id, after_seq=after_seq, follow=follow)


def _event(seq: int, kind: str, data: dict[str, Any] | None = None) -> Any:
    from agent.core.events import Event, EventType

    return Event.create(session_id="sess_1", seq=seq, type=EventType(kind), data=data or {})


async def test_restore_backlog_draws_the_history() -> None:
    """恢复会话时先画历史，而且用跟实时一样的行类型（用户/思考/工具/正文）。"""
    from agent.client.main import _restore_backlog

    events = [
        _event(1, "turn.started", {"prompt": "第一问"}),
        _event(2, "reasoning.delta", {"text": "先想一下\n"}),
        _event(3, "tool.call", {"name": "fs_read", "args": {"path": "a.py"}}),
        _event(4, "text.delta", {"text": "答案\n"}),
        _event(5, "turn.done", {"usage": {}}),
    ]
    client = _FakeBacklog(events)
    tui = ChatTUI(ChatStatus())

    last_seq = await _restore_backlog(tui, client, "sess_1")

    assert client.follow_calls == [False], "拿历史必须 follow=False，否则会挂在那儿等新事件"
    assert last_seq == 5, "last_seq 推到最新，后面接实时流才不会重放"
    assert tui._log == [
        ("user", "你 > 第一问"),
        ("reason", "先想一下"),
        ("tool", '→ fs_read {"path": "a.py"}'),
        ("assistant", "答案"),
    ]


async def test_session_picker_deletes_after_arming() -> None:
    """选择框里删会话：上膛才变红、真删要等第二次按；删不掉要显示原因。"""
    from agent.client.main import _SessionPicker

    deleted: list[str] = []

    async def deleter(session_id: str) -> None:
        deleted.append(session_id)

    sessions = [
        {"id": "sess_aaa111", "title": "第一问"},
        {"id": "sess_bbb222", "title": "第二问"},
    ]
    picker = _SessionPicker(list(sessions), deleter)
    assert picker._current() is not None and picker._current()["id"] == "sess_aaa111"

    picker._armed = "sess_aaa111"  # 第一次按 Delete/d 只上膛
    rows = picker._rows()
    assert any(style == "class:danger" for style, _text in rows), "上膛的那行要变红"

    await picker._do_delete("sess_aaa111")
    assert deleted == ["sess_aaa111"]
    assert [row["id"] for row in picker._sessions] == ["sess_bbb222"]
    assert "已删除" in picker._hint()

    async def boom(_session_id: str) -> None:
        raise RuntimeError("服务端 409")

    failing = _SessionPicker(list(sessions), boom)
    await failing._do_delete("sess_aaa111")
    assert len(failing._sessions) == 2, "删失败时列表不能动"
    assert "删除失败" in failing._hint()


async def test_session_picker_survives_deleting_the_last_session() -> None:
    """删到最后一个也不能炸：`_leave` 在界面没跑的时候要安静跳过。"""
    from agent.client.main import _SessionPicker

    async def deleter(_session_id: str) -> None:
        return None

    picker = _SessionPicker([{"id": "sess_only", "title": "唯一一个"}], deleter)
    await picker._do_delete("sess_only")
    assert picker._sessions == []
