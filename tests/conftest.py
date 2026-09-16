"""测试公共设施。

这里最关键的是 `_clean_env`：它保证测试不受开发机上环境变量影响。
没有它，你本机导出的 AGENT_MODEL 会让 CI 上全绿、本地必挂（或反过来）。
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """清掉所有 AGENT_ 开头的环境变量，让每个测试从干净的起点开始。"""
    for name in list(os.environ):
        if name.startswith("AGENT_"):
            monkeypatch.delenv(name, raising=False)
