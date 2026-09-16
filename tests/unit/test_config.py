"""配置层测试。

注意每个用例都用 `Settings(_env_file=None)`：显式关掉 .env 加载。
否则测试会读到开发者本地那份真实 .env，结果就不确定了。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.config import Settings


def test_defaults_match_design() -> None:
    settings = Settings(_env_file=None)
    assert settings.provider == "deepseek"
    assert settings.model == "deepseek-v4-flash"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.max_tool_rounds == 12
    assert settings.output_limit_bytes == 8192
    assert settings.host == "127.0.0.1"


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("AGENT_PORT", "9000")
    monkeypatch.setenv("AGENT_TOOL_TIMEOUT_S", "5.5")

    settings = Settings(_env_file=None)
    assert settings.model == "deepseek-v4-pro"
    assert settings.port == 9000
    assert settings.tool_timeout_s == 5.5


def test_out_of_range_port_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, port=70000)


def test_describe_never_leaks_secrets() -> None:
    settings = Settings(_env_file=None, api_key="sk-super-secret-value")
    described = settings.describe()

    assert "sk-super-secret-value" not in " ".join(described.values())
    assert described["api_key"] == "<已设置，21 字符>"


def test_describe_reports_missing_secret() -> None:
    described = Settings(_env_file=None, api_key="").describe()
    assert described["api_key"] == "<未设置>"


def test_workspace_is_expanded_and_absolute() -> None:
    settings = Settings(_env_file=None, workspace=Path("~/some/repo"))
    resolved = settings.resolved_workspace
    assert resolved.is_absolute()
    assert "~" not in str(resolved)
