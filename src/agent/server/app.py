"""HTTP + SSE 服务端。

两条实现原则：

1. **先落库、再推送**——由 `EventEmitter` 保证，服务端和 CLI 走的是同一条路。
2. **断线续传靠 seq，不靠连接状态**——重连时带上 `Last-Event-ID`，服务端从库里
   补齐缺口再接上实时流。seq 会话内单调递增，所以接缝处即使重复投递也能去重。

并发模型（设计文档 2.3）：收到消息立刻 `create_task` 驱动图并返回 202，不阻塞请求；
同一会话一把锁串行，不同会话互不干扰。
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import __version__
from ..config import Settings
from ..core.bus import EventBus
from ..core.ids import new_id
from ..graph.checkpointer import open_checkpointer
from ..store import repo
from ..store.db import Database
from .runtime import SessionRuntime

logger = logging.getLogger(__name__)

#: SSE 心跳间隔：长时间没事件时发个注释行，免得中间的代理把连接掐了
KEEPALIVE_SECONDS = 15.0


class CreateSession(BaseModel):
    profile: Literal["chat", "code"] = "code"
    workspace: str | None = None
    title: str = ""


class SendMessage(BaseModel):
    content: str
    idempotency_key: str | None = None


class ApprovalDecision(BaseModel):
    granted: bool


@dataclass
class AppState:
    settings: Settings
    db: Database
    bus: EventBus = field(default_factory=EventBus)
    checkpointer: Any = None
    runtimes: dict[str, SessionRuntime] = field(default_factory=dict)
    tasks: set[asyncio.Task[Any]] = field(default_factory=set)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    state = AppState(settings=settings, db=Database(settings.resolved_db_path))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state.db.connect()
        # 崩溃恢复：进程都重启了，还挂在 running 的 turn 不可能还在跑
        recovered = await repo.interrupt_running_turns(state.db)
        if recovered:
            logger.warning("恢复：%d 个没跑完的 turn 已标记为 interrupted", recovered)
        state.checkpointer = open_checkpointer(settings.resolved_db_path)
        try:
            yield
        finally:
            for runtime in state.runtimes.values():
                await runtime.cancel_pending()
            for task in list(state.tasks):
                task.cancel()
            if state.tasks:
                await asyncio.gather(*state.tasks, return_exceptions=True)
            await state.db.close()

    app = FastAPI(title="agent", version=__version__, lifespan=lifespan)
    # 挂到 app.state 上：调试和测试想看一眼内部状态（库、总线、运行时）时用得上
    app.state.agent = state

    async def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        """校验 `Authorization: Bearer <token>`。

        `auth_token` 没配就不校验——本地开发图省事；而 `agent serve` 启动时会自动
        生成一个并打印出来，所以正常路径上始终是有校验的。

        **必须是 `async def`**：FastAPI 会把同步依赖丢进线程池执行，而受限容器里
        线程池的任务交接会失败（见 AGENTS.md 已知坑"整个进程静默卡死"）——
        表现就是每个带鉴权的请求都挂住，`/health` 却正常。同一个根因第三次踩。
        """
        if not settings.auth_token:
            return
        if authorization != f"Bearer {settings.auth_token}":
            raise HTTPException(status_code=401, detail="缺少或错误的 Authorization")

    async def runtime_for(session: sqlite3.Row) -> SessionRuntime:
        session_id = str(session["id"])
        runtime = state.runtimes.get(session_id)
        if runtime is None:
            workspace = session["workspace"] or str(settings.resolved_workspace)
            runtime = SessionRuntime(
                settings=settings,
                db=state.db,
                bus=state.bus,
                checkpointer=state.checkpointer,
                session_id=session_id,
                workspace=workspace,
                start_seq=await repo.last_event_seq(state.db, session_id),
            )
            state.runtimes[session_id] = runtime
        return runtime

    async def load_session(session_id: str) -> sqlite3.Row:
        row = await repo.get_session(state.db, session_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"会话不存在：{session_id}")
        return row

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.post("/sessions", status_code=201, dependencies=[Depends(require_token)])
    async def create_session(body: CreateSession) -> dict[str, Any]:
        # 只在建会话时解析一次路径，微秒级；为它开线程反而更贵（见 AGENTS.md 已知坑）
        raw = body.workspace or settings.resolved_workspace
        resolved = Path(raw).expanduser().resolve()  # noqa: ASYNC240
        if not resolved.is_dir():
            raise HTTPException(status_code=400, detail=f"工作区不存在：{resolved}")
        session_id = new_id("sess")
        await repo.create_session(
            state.db,
            session_id=session_id,
            profile=body.profile,
            workspace=str(resolved),
            title=body.title,
        )
        return {"session_id": session_id, "profile": body.profile, "workspace": str(resolved)}

    @app.get("/sessions", dependencies=[Depends(require_token)])
    async def list_sessions(limit: int = 20) -> dict[str, Any]:
        rows = await repo.list_sessions(state.db, limit=limit)
        return {"sessions": [dict(row) for row in rows]}

    @app.get("/sessions/{session_id}", dependencies=[Depends(require_token)])
    async def get_session(session_id: str) -> dict[str, Any]:
        row = await load_session(session_id)
        turns = await repo.list_turns(state.db, session_id)
        return {"session": dict(row), "turns": [dict(turn) for turn in turns]}

    @app.delete("/sessions/{session_id}", dependencies=[Depends(require_token)])
    async def remove_session(session_id: str) -> dict[str, Any]:
        """删掉一个会话，连同它的事件、消息、turn、checkpoint。

        正在跑的任务不给删：那条路上还在往这个会话里写事件，删了就是半截状态。
        """
        await load_session(session_id)
        runtime = state.runtimes.get(session_id)
        if runtime is not None and runtime.lock.locked():
            raise HTTPException(status_code=409, detail="这个会话正在跑，等它结束再删")
        await repo.delete_session(state.db, session_id)
        state.runtimes.pop(session_id, None)
        return {"deleted": session_id}

    async def _drive(runtime: SessionRuntime, session_id: str, turn_id: str, prompt: str) -> None:
        """后台驱动一轮：同会话一把锁，结束后把结果落库。"""
        async with runtime.lock:
            try:
                result = await runtime.run_turn(prompt, turn_id=turn_id)
            except Exception:
                logger.exception("turn 执行失败：%s", turn_id)
                await repo.finish_turn(state.db, turn_id, status="failed")
                return
            await repo.append_message(
                state.db,
                session_id=session_id,
                seq=await repo.next_message_seq(state.db, session_id),
                role="assistant",
                content=result.text,
            )
            await repo.finish_turn(
                state.db,
                turn_id,
                status="done" if result.status == "done" else "failed",
                input_tokens=int(result.usage.get("input_tokens", 0)),
                output_tokens=int(result.usage.get("output_tokens", 0)),
            )
            await repo.touch_session(state.db, session_id)

    @app.post(
        "/sessions/{session_id}/messages",
        status_code=202,
        dependencies=[Depends(require_token)],
    )
    async def post_message(session_id: str, body: SendMessage) -> dict[str, Any]:
        """发一条消息：立刻返回 202，任务在后台跑。"""
        session = await load_session(session_id)
        if not body.content.strip():
            raise HTTPException(status_code=400, detail="内容不能为空")

        if body.idempotency_key:
            hit = await repo.lookup_idempotency(state.db, body.idempotency_key)
            if hit is not None:
                # 客户端超时重发：返回上一轮的 turn_id，**不重复执行工具**
                return {"turn_id": hit["turn_id"], "duplicate": True}

        runtime = await runtime_for(session)
        turn_id = new_id("turn")
        await repo.start_turn(state.db, turn_id=turn_id, session_id=session_id)
        # 第一句话就是标题：resume 列表里得看得出这是哪次对话
        await repo.title_session_if_empty(state.db, session_id, body.content)
        await repo.append_message(
            state.db,
            session_id=session_id,
            seq=await repo.next_message_seq(state.db, session_id),
            role="user",
            content=body.content,
        )
        if body.idempotency_key:
            try:
                await repo.remember_idempotency(
                    state.db,
                    key=body.idempotency_key,
                    session_id=session_id,
                    turn_id=turn_id,
                )
            except sqlite3.IntegrityError:
                hit = await repo.lookup_idempotency(state.db, body.idempotency_key)
                return {"turn_id": hit["turn_id"] if hit else turn_id, "duplicate": True}

        task = asyncio.create_task(_drive(runtime, session_id, turn_id, body.content))
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
        return {"turn_id": turn_id, "duplicate": False}

    @app.get("/sessions/{session_id}/events", dependencies=[Depends(require_token)])
    async def stream_events(
        session_id: str,
        request: Request,
        after_seq: int = 0,
        follow: bool = True,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> Response:
        """会话事件流（SSE）。

        `Last-Event-ID` 就是断线续传：客户端上次收到的最后一个 seq，服务端从这里往后补。

        `follow=false` 只把已有事件回放一遍就结束——`curl` 看历史、脚本里做断言都靠它。
        """
        await load_session(session_id)
        resume_from = after_seq
        if last_event_id and last_event_id.isdigit():
            resume_from = max(resume_from, int(last_event_id))
        return StreamingResponse(
            event_stream(
                db=state.db,
                bus=state.bus,
                session_id=session_id,
                after_seq=resume_from,
                is_disconnected=request.is_disconnected,
                follow=follow,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post(
        "/sessions/{session_id}/approvals/{call_id}",
        dependencies=[Depends(require_token)],
    )
    async def decide_approval(
        session_id: str, call_id: str, body: ApprovalDecision
    ) -> dict[str, Any]:
        """兑现一个挂起的审批。"""
        runtime = state.runtimes.get(session_id)
        if runtime is None:
            raise HTTPException(status_code=404, detail="这个会话当前没有在跑的任务")
        if not runtime.resolve_approval(call_id, body.granted):
            raise HTTPException(status_code=409, detail="这个审批已经超时或已被处理")
        return {"call_id": call_id, "granted": body.granted, "resolved": True}

    return app


async def event_stream(
    *,
    db: Database,
    bus: EventBus,
    session_id: str,
    after_seq: int,
    is_disconnected: Callable[[], Awaitable[bool]],
    follow: bool = True,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
) -> AsyncIterator[str]:
    """会话事件流的核心（抽成模块级函数是为了能直接测）。

    **先订阅、再补历史**：反过来的话，两步之间产生的事件会丢。先订阅的代价是
    补历史时可能重复收到刚补过的那几条，所以按 seq 去重——这正是 seq 必须单调的原因。
    """
    subscription = bus.subscribe(session_id)
    last = after_seq
    try:
        for row in await repo.list_events(db, session_id, after_seq=after_seq):
            last = int(row["seq"])
            yield repo.row_to_event(row).to_sse()
        if not follow:
            return
        while not await is_disconnected():
            try:
                event = await asyncio.wait_for(subscription.get(), timeout=keepalive_seconds)
            except TimeoutError:
                yield ": keep-alive\n\n"
                continue
            if event is None:  # 订阅被关闭
                break
            if event.seq <= last:  # 历史补齐时已经发过了
                continue
            last = event.seq
            yield event.to_sse()
    finally:
        subscription.close()
