"""交互式状态栏的格式与降级逻辑。

底栏本身渲染不了（测试里没有终端），但"算出来的字符串对不对""没装依赖时会不会
悄悄炸掉"这两件事可以直接测——它们才是真正容易错的部分。
"""

from __future__ import annotations

import sys

import pytest

from agent.client.main import ChatStatus, make_plain_asker, tui_available


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


def test_no_tty_means_no_tui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有终端（重定向、CI、管道）时必须退回行式，否则会吐一屏控制字符。"""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert tui_available() is False
    assert make_plain_asker().__name__ == "plain"
