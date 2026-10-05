"""交互式状态栏的格式与降级逻辑。

底栏本身渲染不了（测试里没有终端），但"算出来的字符串对不对""没装依赖时会不会
悄悄炸掉"这两件事可以直接测——它们才是真正容易错的部分。
"""

from __future__ import annotations

import sys

import pytest

from agent.client.main import ChatStatus, ChatTUI, make_asker


def test_status_summarizes_usage_and_context() -> None:
    status = ChatStatus(
        model="deepseek-v4-flash",
        workspace="/mnt/g/code/agent",
        session_id="sess_f4f8206a790244658e20949066692bd3",
        context_limit=1000,
    )
    status.track({"input_tokens": 120, "output_tokens": 30})
    status.track({"input_tokens": 200, "output_tokens": 50})

    text = status.text()

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
    assert "上下文 7" in status.text()
    assert "/" not in status.text().split("上下文")[1].split("│")[0]


def test_missing_prompt_toolkit_falls_back_to_plain_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """依赖没装时必须退化成 `input()`，而不是抛异常把 chat 弄挂。"""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert make_asker(ChatStatus()).__name__ == "plain"


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

    assert status.text() == "".join(text for _style, text in status.segments())


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
    """回车把整行交给主循环、并清空输入行；Ctrl-C 用 None 哨兵表示退出。"""
    tui = ChatTUI(ChatStatus())

    class _Buffer:
        text = "你好"
        reset_called = False

        def reset(self) -> None:
            self.reset_called = True

    buffer = _Buffer()
    assert tui._on_accept(buffer) is True, "返回 True 表示继续跑，而不是关掉界面"
    assert buffer.reset_called is True
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

    small = ChatTUI(ChatStatus())
    small.write("".join(f"第{index}行\n" for index in range(20)))
    assert _texts(small._render_lines(5, 20)) == ["第17行", "第18行", "第19行"]
    small._scroll = 2  # 往上两行：底部三行是 17/18/19，再往上就是 15/16/17
    assert _texts(small._render_lines(5, 20)) == ["第15行", "第16行", "第17行"]
    assert "回滚 2 行" in _bar(small), "回滚时底栏要说明现在不在最新位置"


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

    items = tui._display_items(100)
    assert [kind for kind, _frags in items] == ["user", "", "assistant", "", "tool"]
    assert [frags[0][0] for _kind, frags in items if frags] == [
        "class:user",
        "",
        "class:tool",
    ]


def test_tool_runs_fold_and_unfold() -> None:
    """过程信息默认折成一行；展开后原样显示（Ctrl-O 就是切这个开关）。"""
    tui = ChatTUI(ChatStatus())
    tui.write('→ fs_read {"path": "a.py"}\n', "tool")
    tui.write("  ← ok\n", "tool")
    tui.write("正文\n", "assistant")

    assert _texts(tui._render_lines(10, 100)) == [
        '→ fs_read {"path": "a.py"}   …共 2 行，Ctrl-O 展开',
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

    code = next(frags for kind, frags in tui._display_items(100) if kind == "code:python")
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
