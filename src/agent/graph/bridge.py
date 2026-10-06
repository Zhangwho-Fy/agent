"""事件桥：把框架的流映射成我们对外的事件。

这是全项目最关键的适配层。**框架负责"怎么跑"，这里负责"对外怎么说"**：

- 框架的 `stream_mode="messages"` → 我们的 `text.delta`
- 工具节点自己发 `tool.call` / `tool.result`（分级与拦截都在那一层）
- 图挂起（`interrupt()`）→ 发 `approval.required`，等决策，再 `Command(resume=...)` 接着跑
- 一轮结束 → `text.done` / `turn.done`

审批的**等待**也在这层，不在图里：图只负责挂起，"问谁、等多久、超时怎么办"
是对外行为。这样换客户端（CLI → HTTP）时，改的只有这一处。

把映射集中在一个文件里，是为了让框架升级或换实现时，改动有明确边界。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ..core.events import EventType
from ..core.reliability import EventEmitter

logger = logging.getLogger(__name__)

#: 审批回调：拿到待批的调用清单，返回 `{call_id: 是否放行}`（也接受一个 bool 表示整批）
Approver = Callable[[list[dict[str, Any]]], Awaitable[dict[str, bool] | bool]]


@dataclass(slots=True)
class TurnResult:
    """一轮对话的结果。落库、日志、断言都用它，避免调用方再去解析事件流。"""

    text: str = ""
    status: str = "done"
    usage: dict[str, int] = field(default_factory=dict)
    duration_ms: int = 0


def recursion_limit_for(max_tool_rounds: int) -> int:
    """按"工具往返次数"换算出 LangGraph 的 `recursion_limit`。

    一次工具往返要占两个 superstep（agent 节点 + tools 节点），再加进入图与收尾的余量。
    这只是**护栏**（防止图停不下来），真正的业务上限在模型节点里按 `tool_rounds` 数。
    """
    return max_tool_rounds * 2 + 4


async def stream_turn(
    *,
    graph: Any,
    prompt: str,
    emitter: EventEmitter,
    session_id: str,
    turn_id: str,
    recursion_limit: int = 40,
    approver: Approver | None = None,
    approval_timeout_s: float = 120.0,
) -> TurnResult:
    """驱动一轮对话：流式发事件，遇到审批挂起就等人，再恢复。"""
    started = time.perf_counter()
    await emitter.emit(EventType.TURN_STARTED, {"prompt": prompt}, turn_id=turn_id)

    # 每轮都重置计数器：`rounds` / `tool_rounds` 的语义是"这一轮用了多少次"，
    # 接了 checkpointer 之后状态会跨轮保留，不显式归零的话第二轮会继承上一轮的计数。
    inputs: dict[str, Any] = {
        "messages": [HumanMessage(content=prompt)],
        "rounds": 0,
        "tool_rounds": 0,
        "tool_stats": {},
        "halt": None,
        "turn_id": turn_id,
        "approvals": {},
    }
    config: dict[str, Any] = {
        "configurable": {"thread_id": session_id},  # 与 checkpointer 关联的线程 ID
        "recursion_limit": recursion_limit,  # 循环次数硬上限，防止停不下来
    }

    text_parts: list[str] = []
    #: 兜底文本：收尾消息（例如轮数用尽时模型节点直接给的那条）不是流式来的，
    #: 只能从 updates 里拿；正常情况下这里用不上。
    fallback_text = ""
    usage: dict[str, int] = {}
    status = "done"
    halt: dict[str, Any] | None = None
    pending: Any = inputs

    try:
        while True:
            suspended: Any = None
            async for mode, payload in graph.astream(
                pending, config=config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, metadata = payload
                    if metadata.get("langgraph_node") != "agent":
                        continue  # 只要模型节点的增量，工具节点的输出另有事件
                    # 推理模型的"思考"是**模型自带**的另一路流（DeepSeek 放在
                    # additional_kwargs 里），我们只负责搬运，不加工、不伪造。
                    thinking = (getattr(chunk, "additional_kwargs", None) or {}).get(
                        "reasoning_content"
                    )
                    if isinstance(thinking, str) and thinking:
                        await emitter.emit(
                            EventType.REASONING_DELTA, {"text": thinking}, turn_id=turn_id
                        )
                    piece = getattr(chunk, "content", "")
                    if isinstance(piece, str) and piece:
                        text_parts.append(piece)
                        await emitter.emit(EventType.TEXT_DELTA, {"text": piece}, turn_id=turn_id)
                    continue

                payload = payload or {}
                if "__interrupt__" in payload:
                    suspended = payload["__interrupt__"]
                    continue
                for node_update in payload.values():
                    node_update = node_update or {}
                    if node_update.get("usage") is not None:
                        usage = node_update["usage"]
                    if node_update.get("halt"):
                        halt = node_update["halt"]
                    for message in node_update.get("messages") or []:
                        content = getattr(message, "content", "")
                        if (
                            isinstance(message, AIMessage)
                            and not getattr(message, "tool_calls", None)
                            and isinstance(content, str)
                            and content
                        ):
                            fallback_text = content

            if not suspended:
                break
            decisions, approval_note = await _resolve_approvals(
                suspended,
                emitter=emitter,
                turn_id=turn_id,
                approver=approver,
                timeout_s=approval_timeout_s,
            )
            pending = Command(resume=decisions)  # 带着决策回去，图接着跑
    except Exception as exc:  # 基础设施级故障：模型调用失败、图出错
        status = "failed"
        await emitter.emit(
            EventType.ERROR,
            {"kind": type(exc).__name__, "message": str(exc), "retryable": False},
            turn_id=turn_id,
        )

    final_text = "".join(text_parts) or fallback_text
    if halt:
        # 审批没过 = 这一轮就停在这儿。用 text.delta 发出去，界面才会画出来
        # （text.done 只给非流式消费者和回放看）。说清是"人拒绝"还是"没人应答"——
        # 超时/没通道的时候不该冤枉成"你拒绝了"。
        note = (
            f"\n已停下：{halt.get('name')} 需要人工确认，但{approval_note}。\n"
            if approval_note
            else f"\n已停下：你拒绝了 {halt.get('name')}，这一轮不再继续。"
            "要换个做法，直接说一句就行。\n"
        )
        await emitter.emit(EventType.TEXT_DELTA, {"text": note}, turn_id=turn_id)
        final_text = f"{final_text}{note}"
        status = "stopped"
    duration_ms = int((time.perf_counter() - started) * 1000)
    await emitter.emit(EventType.TEXT_DONE, {"text": final_text, "usage": usage}, turn_id=turn_id)
    await emitter.emit(
        EventType.TURN_DONE,
        {
            "status": status,
            "usage": usage,
            "duration_ms": duration_ms,
            "chars": len(final_text),
        },
        turn_id=turn_id,
    )
    return TurnResult(text=final_text, status=status, usage=usage, duration_ms=duration_ms)


async def _resolve_approvals(
    suspended: Any,
    *,
    emitter: EventEmitter,
    turn_id: str,
    approver: Approver | None,
    timeout_s: float,
) -> tuple[dict[str, bool], str]:
    """把挂起原因翻译成 `approval.required` 事件，拿到决策。

    **拿不到决策一律按拒绝**：没人可问、超时、回调自己报错，都是"不放行"。
    审批这种地方默认值必须是拒绝，不能是放行。

    返回 `(决策, 为什么没拿到决策)`——第二个值只在"不是人拒绝的"时候非空
    （超时 / 没通道 / 回调出错），调用方用来说清这一轮为什么停下，别冤枉用户。
    """
    requests: list[dict[str, Any]] = []
    for item in suspended or ():
        value = getattr(item, "value", None) or {}
        requests.extend(value.get("requests") or [])

    if not requests:
        return {}, ""

    expires_at = (datetime.now(UTC) + timedelta(seconds=timeout_s)).isoformat(timespec="seconds")
    for request in requests:
        await emitter.emit(
            EventType.APPROVAL_REQUIRED,
            {**request, "expires_at": expires_at},
            turn_id=turn_id,
        )

    denied = {request["call_id"]: False for request in requests}
    if approver is None:
        logger.warning("有待审批的调用但没有审批通道，按拒绝处理：%s", list(denied))
        return denied, "当时没有审批通道"

    try:
        async with asyncio.timeout(timeout_s):
            granted = await approver(requests)
    except TimeoutError:
        logger.warning("审批超时（%ss），按拒绝处理：%s", timeout_s, list(denied))
        return denied, f"等了 {timeout_s:g} 秒没人应答"
    except Exception:
        logger.exception("审批回调出错，按拒绝处理")
        return denied, "审批通道出错了"

    if isinstance(granted, bool):
        return {request["call_id"]: granted for request in requests}, ""
    if isinstance(granted, dict):
        return {
            request["call_id"]: bool(granted.get(request["call_id"], False)) for request in requests
        }, ""
    return denied, "答复的形状无法识别"
