"""流式块与工具调用装配器的测试。

这一层最容易藏隐蔽 bug：并行调用、分片 JSON、模型给出坏参数，
都是真实模型会干的事。
"""

from __future__ import annotations

from agent.core.chunks import ToolCallBuilder, ToolCallDelta


def test_assembles_tool_call_split_across_chunks() -> None:
    builder = ToolCallBuilder()
    builder.add(ToolCallDelta(index=0, id="call_1", name="fs.read"))
    builder.add(ToolCallDelta(index=0, arguments='{"pa'))
    builder.add(ToolCallDelta(index=0, arguments='th": "src/app.py"}'))

    calls = builder.build()

    assert len(calls) == 1
    assert calls[0].id == "call_1"
    assert calls[0].name == "fs.read"
    assert calls[0].arguments == {"path": "src/app.py"}
    assert calls[0].parse_error is None


def test_assembles_parallel_calls_in_index_order() -> None:
    builder = ToolCallBuilder()
    # 两个调用交错到达，真实模型的并行调用就是这样
    builder.add(ToolCallDelta(index=1, id="call_b", name="shell.exec", arguments='{"cmd": "ls"}'))
    builder.add(ToolCallDelta(index=0, id="call_a", name="fs.list", arguments='{"path": "."}'))

    calls = builder.build()

    assert [call.id for call in calls] == ["call_a", "call_b"]
    assert [call.name for call in calls] == ["fs.list", "shell.exec"]


def test_invalid_json_is_reported_instead_of_raised() -> None:
    builder = ToolCallBuilder()
    builder.add(ToolCallDelta(index=0, id="call_1", name="fs.read", arguments='{"path": '))

    calls = builder.build()

    assert calls[0].arguments == {}
    assert calls[0].parse_error is not None
    assert "JSON" in calls[0].parse_error


def test_non_object_json_is_rejected() -> None:
    builder = ToolCallBuilder()
    builder.add(ToolCallDelta(index=0, id="call_1", name="fs.read", arguments='["a", "b"]'))

    calls = builder.build()

    assert calls[0].arguments == {}
    assert calls[0].parse_error == "参数不是 JSON 对象"


def test_missing_name_is_reported() -> None:
    builder = ToolCallBuilder()
    builder.add(ToolCallDelta(index=0, arguments="{}"))

    calls = builder.build()

    assert calls[0].name == ""
    assert calls[0].parse_error == "模型没有给出工具名"


def test_builder_reports_whether_any_call_seen() -> None:
    builder = ToolCallBuilder()
    assert builder.has_calls is False
    builder.add(ToolCallDelta(index=0, name="fs.read"))
    assert builder.has_calls is True
