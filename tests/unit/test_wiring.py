"""会话装配：三处调用方拿到的是同一套东西，开关只有一份。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage

from agent.config import Settings
from agent.core.bus import EventBus
from agent.core.reliability import EventEmitter
from agent.graph.wiring import build_session_graph


class _Model:
    def bind_tools(self, tools: Any) -> _Model:
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:  # pragma: no cover - 不会真的调
        return AIMessage(content="ok")


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, api_key="", **overrides)


def test_wiring_exposes_the_expected_pieces(tmp_path: Path) -> None:
    wiring = build_session_graph(
        make_settings(max_tool_rounds=7, context_limit=1000),
        workspace=tmp_path,
        emitter=EventEmitter("sess_w", EventBus()),
        model=_Model(),
    )

    assert wiring.ctx.workspace == tmp_path
    assert "recall" in wiring.registry.names
    assert wiring.policy.workspace == tmp_path.resolve()
    assert wiring.graph is not None


def test_wiring_without_db_has_no_recall_source(tmp_path: Path) -> None:
    """评测不接库：`recall` 得给出人话，而不是崩。"""
    wiring = build_session_graph(
        make_settings(),
        workspace=tmp_path,
        emitter=EventEmitter("sess_w", EventBus()),
        model=_Model(),
    )

    assert wiring.ctx.fetch_tool_call is None
