"""文件类工具与截断策略的测试。"""

from __future__ import annotations

from pathlib import Path

from agent.core.tool_spec import TOOL_NAME_PATTERN
from agent.tools.base import ToolContext, truncate
from agent.tools.fs import LIST_TOOL, READ_TOOL
from agent.tools.registry import default_registry


def make_ctx(workspace: Path, limit: int = 8192) -> ToolContext:
    return ToolContext(workspace=workspace, output_limit_bytes=limit)


def test_truncate_keeps_head_and_tail() -> None:
    text = "A" * 1000 + "MIDDLE" + "B" * 1000

    result, truncated = truncate(text, limit_bytes=200)

    assert truncated is True
    assert result.startswith("A")
    assert result.endswith("B")
    assert "省略" in result
    assert "MIDDLE" not in result


def test_truncate_leaves_short_text_alone() -> None:
    result, truncated = truncate("短文本", limit_bytes=200)
    assert result == "短文本"
    assert truncated is False


async def test_read_file_returns_numbered_lines(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("第一行\n第二行\n", encoding="utf-8")

    result = await READ_TOOL.run({"path": "sample.py"}, make_ctx(tmp_path))

    assert result.ok is True
    assert "    1 | 第一行" in result.content
    assert "    2 | 第二行" in result.content


async def test_read_file_reports_missing_file(tmp_path: Path) -> None:
    result = await READ_TOOL.run({"path": "nope.py"}, make_ctx(tmp_path))

    assert result.ok is False
    assert "不存在" in result.content


async def test_read_file_refuses_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    result = await READ_TOOL.run({"path": "../outside.txt"}, make_ctx(workspace))

    assert result.ok is False
    assert "工作区之外" in result.content


async def test_list_dir_marks_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "README.md").write_text("hi", encoding="utf-8")

    result = await LIST_TOOL.run({"path": "."}, make_ctx(tmp_path))

    assert result.ok is True
    assert "src/" in result.content
    assert "README.md" in result.content


async def test_invalid_arguments_are_reported_not_raised(tmp_path: Path) -> None:
    result = await READ_TOOL.run({"path": "x.py", "start_line": 0}, make_ctx(tmp_path))

    assert result.ok is False
    assert "参数不合法" in result.content


def test_default_registry_exposes_three_tools() -> None:
    registry = default_registry()

    assert registry.names == ["fs_list", "fs_read", "shell_exec"]
    specs = {spec.name: spec for spec in registry.specs()}
    assert specs["fs_read"].parameters["properties"]["path"]["type"] == "string"
    assert specs["shell_exec"].to_openai()["type"] == "function"


def test_all_tool_names_satisfy_provider_pattern() -> None:
    """钉住一个踩过的坑：工具名含点号会被 OpenAI 兼容接口直接 400 拒绝。"""
    for name in default_registry().names:
        assert TOOL_NAME_PATTERN.match(name), f"工具名不合法：{name}"
