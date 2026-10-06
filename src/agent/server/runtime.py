"""一个会话的全部运行时零件。

**为什么按会话建一套**：每个会话有自己的工作区、自己的 checkpointer thread_id，
而"同一会话串行、不同会话并行"这条并发约定也需要一把每会话的锁。把模型、图、
事件发射器都绑在会话上，这三件事就自然成立了。

**审批为什么用 Future 等着**：`interrupt()` 挂起的是图，恢复要靠 `Command(resume=...)`。
中间这段等待发生在服务端——工具节点挂起后，我们在这里等一个 HTTP POST 把它兑现。
超时由 `stream_turn` 统一处理（超时按拒绝），这里只负责等待和收尾。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

from ..config import Settings
from ..core.bus import EventBus
from ..core.reliability import EventEmitter
from ..graph.bridge import TurnResult, recursion_limit_for, stream_turn
from ..graph.builder import build_graph
from ..models.factory import build_chat_model
from ..store import repo
from ..store.db import Database
from ..tools.base import ToolContext
from ..tools.policy import Policy
from ..tools.registry import default_registry


class SessionRuntime:
    """某个会话的模型、图、事件与审批等待区。"""

    def __init__(
        self,
        *,
        settings: Settings,
        db: Database,
        bus: EventBus,
        checkpointer: Any,
        session_id: str,
        workspace: str,
        start_seq: int,
    ) -> None:
        self.settings = settings
        self.db = db
        self.session_id = session_id
        self.workspace = workspace
        self.lock = asyncio.Lock()  # 同一会话串行
        self._pending: dict[str, asyncio.Future[bool]] = {}

        ctx = ToolContext(
            workspace=Path(workspace).expanduser().resolve(),
            timeout_s=settings.tool_timeout_s,
            output_limit_bytes=settings.output_limit_bytes,
            db=db,
        )
        policy = Policy(ctx.workspace)
        registry = default_registry()
        # 事件先落库、再推送：顺序反了会出现"客户端看到过、事后查不到"
        self.emitter = EventEmitter(
            session_id,
            bus,
            start_seq=start_seq,
            sink=lambda event: repo.append_event(db, event),
        )
        self.graph = build_graph(
            model=build_chat_model(settings),
            registry=registry,
            policy=policy,
            ctx=ctx,
            emitter=self.emitter,
            max_tool_rounds=settings.max_tool_rounds,
            context_limit=settings.context_limit,
            compress_enabled=settings.compress_enabled,
            compress_lossless_ratio=settings.compress_lossless_ratio,
            compress_summary_ratio=settings.compress_summary_ratio,
            compress_keep_recent=settings.compress_keep_recent_tool_results,
            checkpointer=checkpointer,
            db=db,
        )

    async def run_turn(self, prompt: str, *, turn_id: str) -> TurnResult:
        """跑一轮。调用方负责把它的返回值落库（或用返回的 usage 更新 turn 行）。"""
        return await stream_turn(
            graph=self.graph,
            prompt=prompt,
            emitter=self.emitter,
            session_id=self.session_id,
            turn_id=turn_id,
            recursion_limit=recursion_limit_for(self.settings.max_tool_rounds),
            approver=self._ask_for_approval,
            approval_timeout_s=self.settings.approval_timeout_s,
        )

    async def _ask_for_approval(self, requests: list[dict[str, Any]]) -> dict[str, bool]:
        """挂起点在这里等 HTTP：每个待批调用一个 Future，谁先来兑现听谁的。"""
        loop = asyncio.get_running_loop()
        waits: dict[str, asyncio.Future[bool]] = {}
        for request in requests:
            call_id = str(request["call_id"])
            future: asyncio.Future[bool] = loop.create_future()
            self._pending[call_id] = future
            waits[call_id] = future
        try:
            decisions = await asyncio.gather(*waits.values())
        finally:
            # 超时或取消时也要清干净，否则残留的 Future 会让人以为"还挂着"
            for call_id in waits:
                self._pending.pop(call_id, None)
        return dict(zip(waits, decisions, strict=True))

    def resolve_approval(self, call_id: str, granted: bool) -> bool:
        """兑现一个等待中的审批。返回 False 表示没有这个待批项（已超时或已处理）。"""
        future = self._pending.get(call_id)
        if future is None or future.done():
            return False
        future.set_result(granted)
        return True

    async def cancel_pending(self) -> None:
        """服务端收尾时把还挂着的审批一律按拒绝兑现，别让图悬着。"""
        for future in list(self._pending.values()):
            if not future.done():
                future.set_result(False)
        with contextlib.suppress(Exception):
            await asyncio.sleep(0)
