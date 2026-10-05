"""服务端验收测试。

全部走 **ASGI 直连 + 回放模型**：不联网、不要密钥。覆盖阶段 4 的三条验收：

1. 走完"建会话 → 发消息 → 收事件 → 审批 → 拿结果"；
2. 按 seq 补齐（断线续传语义），不丢不重；
3. 同一条消息发两次不重复执行。

一个环境限制：httpx 的 `ASGITransport` 会**把响应缓冲到结束**才交给客户端，
所以无限 SSE 在它上面读不出实时性。因此分两路测——
端到端用"发消息 + 轮询库里的事件"，实时流部分直接驱动 `event_stream()` 生成器。
真正带 socket 的 `curl` 验收要在有网络的机器上跑（见 AGENTS.md）。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from agent.client.api import AgentClient
from agent.config import Settings
from agent.core.bus import EventBus
from agent.core.events import Event, EventType
from agent.server.app import create_app, event_stream
from agent.store import repo
from agent.store.db import Database

REPO_ROOT = Path(__file__).resolve().parents[2]
TOKEN = "test-token"


def make_settings(tmp_path: Path, *, case: str, trace: Path | None = None) -> Settings:
    """按评测夹具构造设置：回放模型 + 临时库 + 固定令牌。"""
    return Settings(
        _env_file=None,
        provider="replay",
        trace_path=trace or REPO_ROOT / "evals" / "recordings" / f"{case}.jsonl",
        workspace=REPO_ROOT / "evals" / "workspaces" / case,
        db_path=tmp_path / "agent.db",
        auth_token=TOKEN,
        api_key="",
        approval_timeout_s=5,
    )


def doubled_trace(tmp_path: Path, case: str) -> Path:
    """把一份录制复制两遍，够跑两轮用。

    一轮要消耗 N 次模型调用，两轮就要 2N 条——回放是严格按顺序取的，
    这本身也是它的优点：少一条就当场报错，而不是悄悄跑偏。
    """
    source = REPO_ROOT / "evals" / "recordings" / f"{case}.jsonl"
    lines = source.read_text(encoding="utf-8").strip().splitlines()
    target = tmp_path / f"{case}-x2.jsonl"
    target.write_text("\n".join([*lines, *lines]) + "\n", encoding="utf-8")
    return target


@pytest.fixture()
async def server(tmp_path: Path) -> AsyncIterator[tuple[AgentClient, Any, Database]]:
    """起一个 in-process 服务端，返回（客户端, app, 库句柄）。"""
    settings = make_settings(tmp_path, case="read-sample")
    app = create_app(settings)
    # ASGITransport 不会自己跑 lifespan，得手动进一次
    async with app.router.lifespan_context(app):
        client = AgentClient("http://test", TOKEN, transport=httpx.ASGITransport(app=app))
        try:
            yield client, app, app.state.agent.db
        finally:
            await client.close()


async def wait_for_event(
    db: Database,
    session_id: str,
    predicate: Callable[[Event], bool],
    *,
    budget: float = 20.0,
) -> Event:
    """轮询库里的事件，直到出现满足条件的那条。

    真实客户端是订阅 SSE；这里用轮询是为了绕开 ASGITransport 的缓冲限制，
    验的东西一样：**事件先落库，所以事后一定查得到**。
    """
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        for row in await repo.list_events(db, session_id):
            event = repo.row_to_event(row)
            if predicate(event):
                return event
        await asyncio.sleep(0.05)
    msg = f"等不到满足条件的事件：{session_id}"
    raise TimeoutError(msg)


def turn_done(turn_id: str) -> Callable[[Event], bool]:
    return lambda event: event.type is EventType.TURN_DONE and event.turn_id == turn_id


async def test_health_needs_no_token(server: tuple[AgentClient, Any, Database]) -> None:
    client, _, _ = server
    assert (await client.health())["status"] == "ok"


async def test_wrong_token_is_rejected(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path, case="read-sample"))
    async with app.router.lifespan_context(app):
        bad = AgentClient("http://test", "wrong", transport=httpx.ASGITransport(app=app))
        try:
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await bad.list_sessions()
            assert excinfo.value.response.status_code == 401
        finally:
            await bad.close()


async def test_full_flow_returns_an_answer(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """建会话 → 发消息 → 收事件 → 拿结果。"""
    client, _, db = server
    session_id = (await client.create_session(title="验收"))["session_id"]

    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    assert sent["duplicate"] is False

    done = await wait_for_event(db, session_id, turn_done(sent["turn_id"]))
    assert done.data["status"] == "done"

    events = [repo.row_to_event(row) for row in await repo.list_events(db, session_id)]
    kinds = [event.type.value for event in events]
    assert kinds[0] == "turn.started"
    assert "text.delta" in kinds, "应该有流式分片"
    assert "tool.call" in kinds, "应该真的调了工具"
    assert kinds[-1] == "turn.done"

    text = "".join(
        str(event.data.get("text", "")) for event in events if event.type is EventType.TEXT_DELTA
    )
    assert "add" in text and "is_even" in text
    assert done.data["usage"]["input_tokens"] > 0


async def test_seq_is_monotonic_and_unique(
    server: tuple[AgentClient, Any, Database],
) -> None:
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    seqs = [row["seq"] for row in await repo.list_events(db, session_id)]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


async def test_sse_frames_carry_id_event_and_data(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """SSE 帧格式：`id:` 供断线续传，`event:` 给客户端分派，`data:` 是事件本体。"""
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    # follow=false：只回放已有事件就结束（curl 看历史、测试断言都用它）
    response = await client._http.get(f"/sessions/{session_id}/events?follow=false")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "event: turn.started" in body
    assert "event: turn.done" in body
    assert "id: 1\n" in body
    assert '"type":"turn.started"' in body.replace(" ", "")


async def test_last_event_id_resumes_after_that_seq(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """断线续传：带上 Last-Event-ID，服务端只补这之后的事件。"""
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    response = await client._http.get(
        f"/sessions/{session_id}/events?follow=false",
        headers={"Last-Event-ID": "2"},
    )
    ids = [
        int(line.split(":", 1)[1]) for line in response.text.splitlines() if line.startswith("id:")
    ]
    assert ids, "应当有事件"
    assert min(ids) == 3, "必须从断点后一条接着补"


async def test_event_stream_merges_history_and_live_without_duplicates() -> None:
    """直接驱动流生成器：历史补齐 → 实时投递 → 按 seq 去重。

    这段是 SSE 的核心逻辑，绕开 HTTP 单独测，正好补上 ASGITransport 测不到的部分。
    """
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "a.db")
        db.connect()
        bus = EventBus()
        await repo.create_session(db, session_id="sess_x", profile="code")
        for seq in (1, 2):
            await repo.append_event(db, _event(seq))

        disconnected = False

        async def is_disconnected() -> bool:
            return disconnected

        stream = event_stream(
            db=db,
            bus=bus,
            session_id="sess_x",
            after_seq=0,
            is_disconnected=is_disconnected,
        )
        # 历史两条
        assert "id: 1" in await anext(stream)
        assert "id: 2" in await anext(stream)

        # 实时的第 3 条
        bus.publish(_event(3))
        assert "id: 3" in await anext(stream)

        # 同一条再推一遍（比如历史补齐和实时流的接缝）不能重复投递
        bus.publish(_event(3))
        bus.publish(_event(4))
        assert "id: 4" in await anext(stream), "重复的 seq 应当被跳过"

        disconnected = True
        with pytest.raises(StopAsyncIteration):
            await anext(stream)
        await db.close()


def _event(seq: int) -> Event:
    return Event.create(
        session_id="sess_x", seq=seq, type=EventType.TEXT_DELTA, data={"text": str(seq)}
    )


async def test_same_idempotency_key_does_not_run_twice(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """客户端超时重发是常态：同一个键第二次必须直接返回原 turn，不重复执行。"""
    client, _, _ = server
    session_id = (await client.create_session())["session_id"]
    payload = "sample.py 里定义了哪几个函数？只列函数名。"

    first = await client.send_message(session_id, payload, idempotency_key="k-1")
    second = await client.send_message(session_id, payload, idempotency_key="k-1")

    assert first["duplicate"] is False
    assert second["duplicate"] is True
    assert second["turn_id"] == first["turn_id"]

    detail = (await client._http.get(f"/sessions/{session_id}")).json()
    assert len(detail["turns"]) == 1, "第二次不该产生新的 turn"


async def test_tool_calls_and_results_are_paired(
    server: tuple[AgentClient, Any, Database],
) -> None:
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    events = [repo.row_to_event(row) for row in await repo.list_events(db, session_id)]
    calls = [e for e in events if e.type is EventType.TOOL_CALL]
    results = [e for e in events if e.type is EventType.TOOL_RESULT]
    assert calls and len(calls) == len(results)
    assert {c.data["call_id"] for c in calls} == {r.data["call_id"] for r in results}


async def test_approval_blocks_a_write_until_the_client_decides(tmp_path: Path) -> None:
    """审批验收：模型要写文件 → 服务端挂起 → 客户端拒绝 → 工作区纹丝不动。"""
    settings = make_settings(tmp_path, case="write-approval")
    workspace = Path(settings.resolved_workspace)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        client = AgentClient("http://test", TOKEN, transport=httpx.ASGITransport(app=app))
        try:
            session_id = (await client.create_session())["session_id"]
            sent = await client.send_message(
                session_id, "在工作区根目录创建一个 notes.txt，内容写 hello"
            )

            request = await wait_for_event(
                app.state.agent.db,
                session_id,
                lambda e: e.type is EventType.APPROVAL_REQUIRED,
            )
            assert request.data["name"] == "shell_exec"
            assert "expires_at" in request.data, "审批事件要带过期时间"
            await client.approve(session_id, str(request.data["call_id"]), granted=False)

            done = await wait_for_event(app.state.agent.db, session_id, turn_done(sent["turn_id"]))
            assert done.data["status"] == "done"
            assert not (workspace / "notes.txt").exists(), "拒绝了就不能动文件"

            events = [
                repo.row_to_event(row)
                for row in await repo.list_events(app.state.agent.db, session_id)
            ]
            assert any(
                e.type is EventType.TOOL_RESULT and e.data["status"] == "approval" for e in events
            )
        finally:
            await client.close()


async def test_approval_timeout_counts_as_denied(tmp_path: Path) -> None:
    """超时按拒绝：没人理会审批，服务端不能一直挂着。"""
    settings = make_settings(tmp_path, case="write-approval")
    settings.approval_timeout_s = 0.3
    workspace = Path(settings.resolved_workspace)
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        client = AgentClient("http://test", TOKEN, transport=httpx.ASGITransport(app=app))
        try:
            session_id = (await client.create_session())["session_id"]
            sent = await client.send_message(
                session_id, "在工作区根目录创建一个 notes.txt，内容写 hello"
            )
            done = await wait_for_event(app.state.agent.db, session_id, turn_done(sent["turn_id"]))
            assert done.data["status"] == "done"
            assert not (workspace / "notes.txt").exists()
        finally:
            await client.close()


async def test_two_turns_in_one_session(tmp_path: Path) -> None:
    """同一会话的第二轮：新 turn、新事件，互不干扰。"""
    settings = make_settings(
        tmp_path, case="read-sample", trace=doubled_trace(tmp_path, "read-sample")
    )
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        client = AgentClient("http://test", TOKEN, transport=httpx.ASGITransport(app=app))
        db = app.state.agent.db
        try:
            session_id = (await client.create_session())["session_id"]
            first = await client.send_message(session_id, "第一次问")
            await wait_for_event(db, session_id, turn_done(first["turn_id"]))
            second = await client.send_message(session_id, "第二次问")
            await wait_for_event(db, session_id, turn_done(second["turn_id"]))

            detail = (await client._http.get(f"/sessions/{session_id}")).json()
            assert len(detail["turns"]) == 2
            assert all(turn["status"] == "done" for turn in detail["turns"])
            stored = await repo.list_messages(db, session_id)
            assert [row["role"] for row in stored] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]
        finally:
            await client.close()


async def test_running_out_of_recording_fails_the_turn_but_keeps_the_server_up(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """回放用完（图的行为变了）时：这一轮标记失败，服务端本身不能倒。"""
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    first = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(first["turn_id"]))

    second = await client.send_message(session_id, "再来一次")
    done = await wait_for_event(db, session_id, turn_done(second["turn_id"]))
    assert done.data["status"] == "failed"

    events = [repo.row_to_event(row) for row in await repo.list_events(db, session_id)]
    assert any(event.type is EventType.ERROR for event in events)
    assert (await client.health())["status"] == "ok", "服务端不能跟着倒"


async def test_messages_are_persisted_as_rows(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """消息要落库：重放和审计看的是它，不只是事件流。"""
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    rows = await repo.list_messages(db, session_id)
    assert [row["role"] for row in rows] == ["user", "assistant"]
    assert "sample.py" in rows[0]["content"]
    assert rows[1]["content"], "助手回答也要落库"

    session = await repo.get_session(db, session_id)
    assert session is not None
    assert session["title"].startswith("sample.py"), "第一句话就是标题：resume 列表靠它认人"


async def test_follow_false_replays_the_backlog_and_stops(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """`follow=false` 是"恢复历史"的地基：把已有事件放一遍就结束，不会挂在那儿等。"""
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    replayed = [event async for event in client.stream_events(session_id, follow=False)]

    assert replayed, "得能拿到历史"
    assert replayed[0].type is EventType.TURN_STARTED, "从第一条开始，客户端才知道整段的开头"
    assert replayed[-1].type is EventType.TURN_DONE, "放完已有事件就收尾"
    assert [event.seq for event in replayed] == sorted(event.seq for event in replayed)


async def test_session_list_carries_the_last_user_message(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """会话列表要能认人：`GET /sessions` 带上最后一句用户消息和 turn 数。"""
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    question = "sample.py 里定义了哪几个函数？只列函数名。"
    sent = await client.send_message(session_id, question)
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    row = next(item for item in await client.list_sessions() if item["id"] == session_id)

    assert row["last_user"] == question
    assert row["turns"] == 1
    assert row["title"].startswith("sample.py"), "标题仍是第一句，两个字段配合着用"


async def test_delete_session_removes_it_completely(
    server: tuple[AgentClient, Any, Database],
) -> None:
    """删会话：列表里没了、事件也没了；再删一次是 404，不是静默成功。"""
    client, _, db = server
    session_id = (await client.create_session())["session_id"]
    sent = await client.send_message(session_id, "sample.py 里定义了哪几个函数？只列函数名。")
    await wait_for_event(db, session_id, turn_done(sent["turn_id"]))

    assert await client.delete_session(session_id) == {"deleted": session_id}
    assert [row["id"] for row in await client.list_sessions()] == []
    assert await repo.list_events(db, session_id) == [], "事件不能留成孤儿数据"

    with pytest.raises(httpx.HTTPStatusError) as caught:
        await client.delete_session(session_id)
    assert caught.value.response.status_code == 404
