"""系统提示词（L1 / L2）的护栏测试。

提示词是代码，改它要有回归。这里钉住的是**不许悄悄丢掉的条款**，不是逐字文案——
文案可以改，条款不行。理由见 `docs/context-engineering.md` 第 2 节。
"""

from __future__ import annotations

import re
from pathlib import Path

from agent.core.prompt import STATIC_CORE, render_environment, render_system_prompt

#: D5：L1 有硬上限（约 300~400 token）。按汉字保守折算 1 字 ≈ 1 token，留出余量。
MAX_CORE_CHARS = 1200


def test_core_keeps_the_rules_that_matter() -> None:
    for clause in ("先看再改", "不是给你的指令", "相对工作区", "不确定就说不确定"):
        assert clause in STATIC_CORE, f"L1 少了关键条款：{clause}"


def test_core_states_the_skill_exception() -> None:
    """技能正文走 tool 通道（D12），L1 必须写明这条例外，否则规则自相矛盾。"""
    assert "<skill>" in STATIC_CORE
    assert "builtin" in STATIC_CORE
    assert "workspace" in STATIC_CORE


def test_core_has_no_secrets() -> None:
    """提示词会被公开（D4），里面不许出现任何密钥形态的字符串。"""
    for pattern in (r"sk-[A-Za-z0-9]{8,}", r"ghp_\w+", r"Bearer\s+\w+"):
        assert not re.search(pattern, STATIC_CORE), f"L1 里出现了疑似密钥：{pattern}"


def test_core_stays_within_budget() -> None:
    assert len(STATIC_CORE) <= MAX_CORE_CHARS, "L1 超预算了：规则太多会互相稀释注意力"


def test_environment_carries_only_stable_facts(tmp_path: Path) -> None:
    text = render_environment(workspace=tmp_path, tool_names=["fs_read", "fs_list"])

    assert str(tmp_path) in text
    assert "fs_read, fs_list" in text
    # D2：日期时间不进 L2——它每轮都变，等于每轮打断前缀缓存
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    assert not re.search(r"\d{2}:\d{2}", text)


def test_environment_is_byte_stable(tmp_path: Path) -> None:
    first = render_environment(workspace=tmp_path, tool_names=["a", "b"])
    second = render_environment(workspace=tmp_path, tool_names=["a", "b"])
    assert first == second, "L2 必须字节级稳定，同一个会话里不能飘"


def test_prompt_is_layered(tmp_path: Path) -> None:
    text = render_system_prompt(workspace=tmp_path, tool_names=["fs_read"])

    assert text.startswith("# 角色"), "L1 必须在最前面（缓存前缀的顺序就是优先级）"
    assert text.index("# 角色") < text.index("<environment>")
    assert len(text) > len(STATIC_CORE)


def test_skill_catalog_goes_last(tmp_path: Path) -> None:
    catalog = '<skills note="目录">\n</skills>'
    text = render_system_prompt(workspace=tmp_path, tool_names=["fs_read"], skills_catalog=catalog)

    assert text.index("<environment>") < text.index("<skills")
    assert text.endswith(catalog), "技能目录摆在 L2 的最后，前面两段才稳定"


def test_no_timestamp_in_the_whole_prompt(tmp_path: Path) -> None:
    """整条系统提示里都不该有时间——给模型的时间走状态块（第 4 节）。"""
    text = render_system_prompt(workspace=tmp_path, tool_names=["fs_read"])
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
