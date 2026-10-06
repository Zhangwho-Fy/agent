"""状态块（L3）的渲染规则：条件出现、说行动不说数字、有硬上限。"""

from __future__ import annotations

from datetime import datetime

from agent.core.status import MAX_CHARS, StatusSnapshot, ToolStat

NOW = datetime(2026, 10, 6, 16, 40)


def snapshot(**overrides: object) -> StatusSnapshot:
    base: dict[str, object] = dict(now=NOW, rounds=2, tool_rounds=3, max_tool_rounds=12)
    base.update(overrides)
    return StatusSnapshot(**base)  # type: ignore[arg-type]


def test_time_and_progress_are_always_there() -> None:
    text = snapshot().render_for_model()

    assert text.startswith("<agent_state ")
    assert "<time>2026-10-06 16:40</time>" in text
    assert 'tools="3" limit="12"' in text
    assert "还能再做 9 次" in text


def test_remaining_never_goes_negative_at_the_limit() -> None:
    text = snapshot(tool_rounds=12).render_for_model()
    assert "还能再做 0 次" in text


def test_single_call_does_not_earn_a_warning() -> None:
    """低于阈值不占 token——这是它"条件出现"的意义。"""
    text = snapshot(tool_stats={"fs_read": ToolStat(calls=1)}).render_for_model()
    assert "<repeat" not in text


def test_repeat_warning_reports_counts_and_advice() -> None:
    text = snapshot(tool_stats={"fs_read": ToolStat(calls=3, failures=2)}).render_for_model()

    assert 'tool="fs_read" count="3" failures="2"' in text
    assert "别原样重试" in text


def test_repeat_lines_are_capped() -> None:
    stats = {f"tool{index}": ToolStat(calls=5) for index in range(10)}
    text = snapshot(tool_stats=stats).render_for_model()
    assert text.count("<repeat") == 3


def test_context_hint_only_when_almost_full_and_says_no_number() -> None:
    """D34：数字会引发上下文焦虑，所以只说"该收尾了"。"""
    quiet = snapshot(context_tokens=500, context_limit=1000).render_for_model()
    loud = snapshot(context_tokens=850, context_limit=1000).render_for_model()

    assert "<context>" not in quiet
    assert "<context>" in loud
    assert "85%" not in loud
    assert "该收尾了" in loud


def test_render_stays_within_budget() -> None:
    stats = {f"tool{index}": ToolStat(calls=9, failures=4) for index in range(10)}
    text = snapshot(tool_stats=stats, context_tokens=990, context_limit=1000).render_for_model()

    assert len(text) <= MAX_CHARS
    assert "</agent_state>" in text, "砍到只剩骨架也要是一个完整的块"
