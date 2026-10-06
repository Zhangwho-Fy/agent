"""图的节点：模型节点、审批节点、工具节点。

为什么不用 LangGraph 预置的 `ToolNode`：预置节点直接执行工具，中间没有位置插
"分级 → 拦截 → 记事件"。而这几步正是本项目要展示的部分，所以工具节点自己写。

**为什么审批单独一个节点**：被 `interrupt()` 挂起的节点，恢复时是**从头重跑**的。
如果工具节点自己挂起，重跑就意味着已经执行过的工具会被再执行一遍、事件会被重复发一遍。
所以把"会挂起"的部分拆成审批节点，它只做纯计算（分级判断）；工具节点永远不挂起，
也就永远不会重跑。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from ..core.events import EventType
from ..core.guard import scan_suspicious, strip_invisible, wrap_untrusted
from ..core.prompt import STATIC_CORE
from ..core.reliability import EventEmitter
from ..core.status import StatusSnapshot, ToolStat
from ..logging import log_extra
from ..store import repo
from ..store.db import Database
from ..tools.base import ToolContext
from ..tools.policy import Decision, Policy, PolicyDecision
from ..tools.registry import ToolRegistry
from .state import AgentState

logger = logging.getLogger(__name__)


def _strip_reasoning(message: Any) -> Any:
    """摘掉消息上的 `additional_kwargs["reasoning_content"]`。

    为什么摘：思考内容已经由 bridge 作为 `reasoning.delta` 事件发出去了（界面、
    回放、录放都用那一份）。留在消息里的话，它会跟着 checkpointer 进图状态，
    而**每个 superstep 都会把整份消息列表再存一遍**——一轮长思考就能让 checkpoint
    多出几十 KB 的重复副本（实测一次 33 行的回答，60 个 checkpoint 里塞了约 2.6MB）。

    **不是因为会被发回模型**：查过 `langchain_openai._convert_message_to_dict` 与
    `langchain_deepseek._get_request_payload`，它们只搬 `tool_calls` / `function_call`
    / `audio`，不会转发这个字段，所以它没有多花输入 token。纯粹是不让运行态里堆副本。

    没有思考内容时原样返回（别多造对象：`add_messages` 会按 id 去重，
    同一对象重复返回等于没追加，这条坑在 AGENTS.md 里记着）。
    """
    if not (getattr(message, "additional_kwargs", None) or {}).get("reasoning_content"):
        return message
    clean = message.model_copy(deep=True)
    clean.additional_kwargs = {
        key: value for key, value in clean.additional_kwargs.items() if key != "reasoning_content"
    }
    return clean


def build_model_node(
    model: BaseChatModel,
    registry: ToolRegistry,
    *,
    system_prompt: str | Callable[[AgentState], str] | None = None,
    max_tool_rounds: int = 12,
    context_limit: int = 0,
    clock: Callable[[], datetime] | None = None,
) -> Any:
    """模型节点：把工具清单绑给模型，收到回复就追加进状态。

    `system_prompt` 允许传一个**接收 state 的可调用对象**，不是洁癖：技能目录按轮冻结
    （D11），每次模型调用都要拿到当期的那一份，而"当期"只能从 state 里的 `turn_id` 看出来。
    传字符串则固定不变。

    **循环上限在这里兜底**：工具往返次数用尽后不再调用模型，直接给一条收尾消息——
    它没有 `tool_calls`，条件边 `_route` 看到就会走到 END，循环停住。
    为什么不用 `recursion_limit` 表达这个语义：那数的是 superstep（`agent→tools`
    一次往返算 2 步），换算关系藏在框架里；这里数的是"工具往返次数"，和配置项同名同义。
    """
    bound_model = model.bind_tools(registry.to_openai_tools())

    def system_message_text(state: AgentState) -> str:
        if system_prompt is None:
            return STATIC_CORE  # 没给环境信息时退化成 L1；测试与嵌入式用法走这条
        return system_prompt(state) if callable(system_prompt) else system_prompt

    def status_message(state: AgentState) -> SystemMessage:
        """状态块：**只拼进这一次请求**，不写回 state（D19）。"""
        snapshot = StatusSnapshot(
            now=(clock or datetime.now)(),
            rounds=int(state.get("rounds") or 0),
            tool_rounds=int(state.get("tool_rounds") or 0),
            max_tool_rounds=max_tool_rounds,
            tool_stats={
                name: ToolStat(
                    calls=int(value.get("calls", 0)),
                    failures=int(value.get("failures", 0)),
                )
                for name, value in (state.get("tool_stats") or {}).items()
            },
            context_tokens=int(state.get("context_tokens") or 0),
            context_limit=context_limit,
        )
        return SystemMessage(content=snapshot.render_for_model())

    async def call_model(state: AgentState) -> dict[str, Any]:
        if (state.get("tool_rounds") or 0) >= max_tool_rounds:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"已达到本轮工具调用上限（{max_tool_rounds} 次往返），先停下来汇报："
                            "上面这些工具结果就是目前能拿到的全部信息。"
                        )
                    )
                ]
            }

        messages = [
            SystemMessage(content=system_message_text(state)),
            *state["messages"],
            status_message(state),
        ]
        response = _strip_reasoning(await bound_model.ainvoke(messages))

        usage = getattr(response, "usage_metadata", None) or {}
        previous = state.get("usage") or {}
        input_tokens = int(usage.get("input_tokens", 0)) or int(state.get("context_tokens") or 0)
        return {
            "messages": [response],
            "rounds": (state.get("rounds") or 0) + 1,
            # 这一次调用实际塞进去的量，就是"上下文有多大"最直接的锚点
            "context_tokens": input_tokens,
            "usage": {
                "input_tokens": previous.get("input_tokens", 0) + int(usage.get("input_tokens", 0)),
                "output_tokens": previous.get("output_tokens", 0)
                + int(usage.get("output_tokens", 0)),
            },
        }

    return call_model


def build_approval_node(registry: ToolRegistry, policy: Policy) -> Any:
    """审批节点：把本批需要人工确认的调用一次性挂起，等恢复。

    **这个节点必须保持没有副作用**——它会被 `interrupt()` 挂起，恢复时从头重跑。
    所以这里只做分级判断，一个事件都不发；`approval.required` 事件由 bridge 发，
    因为 bridge 能看到挂起的原因，而且它在图外面，不会重跑。

    恢复值支持两种形状：`True/False`（整批一个答案）或 `{call_id: bool}`（逐个答复）。
    """

    async def approve(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        pending: list[dict[str, Any]] = []
        for call in getattr(last, "tool_calls", None) or []:
            name = str(call.get("name", ""))
            tool = registry.get(name)
            if tool is None:
                continue  # 未知工具交给工具节点回填"未知工具"，不需要审批
            decision = policy.classify(tool, dict(call.get("args") or {}))
            if decision.decision is Decision.APPROVAL:
                pending.append(
                    {
                        "call_id": str(call.get("id", "")),
                        "name": name,
                        "args": dict(call.get("args") or {}),
                        "reason": decision.reason,
                    }
                )

        if not pending:
            return {"approvals": {}}

        granted = interrupt({"requests": pending})
        return {"approvals": _normalize_approvals(pending, granted)}

    return approve


def _normalize_approvals(pending: list[dict[str, Any]], granted: Any) -> dict[str, bool]:
    """把恢复值规整成 `{call_id: bool}`。

    无法识别的形状一律当拒绝：审批这种地方，默认值必须是"不放行"。
    """
    if isinstance(granted, bool):
        return {item["call_id"]: granted for item in pending}
    if isinstance(granted, dict):
        return {item["call_id"]: bool(granted.get(item["call_id"], False)) for item in pending}
    return {item["call_id"]: False for item in pending}


def build_tool_node(
    *,
    registry: ToolRegistry,
    policy: Policy,
    ctx: ToolContext,
    emitter: EventEmitter,
    db: Database | None = None,
) -> Any:
    """工具节点：分级 → 执行或拒绝 → 发事件 → 把结果回填给模型。

    传入 `db` 时，每次调用会往 `tool_calls` 表记一条审计：
    **全量输出入库，模型只看到截断版**——人要排查时能看全文。
    """

    async def call_tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        approvals = state.get("approvals") or {}
        turn_id = state.get("turn_id") or None
        results: list[AnyMessage] = []
        halt: dict[str, Any] | None = None
        # 这一批是第几次工具往返。同上，是给客户端和状态块看的硬上限计数。
        round_no = (state.get("tool_rounds") or 0) + 1
        stats: dict[str, dict[str, int]] = {
            name: dict(value) for name, value in (state.get("tool_stats") or {}).items()
        }

        def count(name: str, *, failed: bool) -> None:
            """计数器由代码维护——状态块里的数字一个都不能让模型自己算（4.1）。"""
            entry = stats.setdefault(name, {"calls": 0, "failures": 0})
            entry["calls"] += 1
            entry["failures"] += int(failed)

        for call in getattr(last, "tool_calls", None) or []:
            name = str(call.get("name", ""))
            args = dict(call.get("args") or {})
            call_id = str(call.get("id", ""))
            await emitter.emit(
                EventType.TOOL_CALL,
                {"call_id": call_id, "name": name, "args": args, "round": round_no},
            )

            tool = registry.get(name)
            if tool is None:
                count(name, failed=True)
                content = f"未知工具：{name}"
                results.append(ToolMessage(content=content, tool_call_id=call_id))
                await emitter.emit(
                    EventType.TOOL_RESULT,
                    {"call_id": call_id, "name": name, "status": "unknown_tool"},
                )
                continue

            decision = policy.classify(tool, args)
            recorded = db is not None and turn_id is not None
            if recorded:
                await repo.start_tool_call(
                    db,
                    call_id=call_id,
                    session_id=emitter.session_id,
                    turn_id=turn_id,
                    name=name,
                    args=args,
                    tier=tool.tier.value,
                )

            # 拒绝与"未获批准"都走同一条路：不执行，把原因回填给模型
            refusal = _refusal_reason(decision, bool(approvals.get(call_id)))
            if refusal is not None:
                count(name, failed=True)
                if decision.decision is Decision.APPROVAL and not approvals.get(call_id):
                    # 人工拒绝（或没人可问）：**这一轮到此为止**。
                    # 不这么做的话，模型会拿着"未获批准"换一条命令再问一次，变成一直弹审批。
                    halt = {"call_id": call_id, "name": name, "reason": decision.reason}
                results.append(ToolMessage(content=refusal, tool_call_id=call_id))
                await emitter.emit(
                    EventType.TOOL_RESULT,
                    {
                        "call_id": call_id,
                        "name": name,
                        "status": decision.decision.value,
                        "preview": refusal,
                    },
                )
                if recorded:
                    await repo.finish_tool_call(
                        db,
                        call_id,
                        status=decision.decision.value,
                        decision=decision.decision.value,
                    )
                continue

            result = await tool.run(args, ctx)
            count(name, failed=not result.ok)
            if result.wrap == "none":
                # 工具自己包好了容器（技能正文是**操作说明**，走 <skill> 块，见 3.5）。
                # 不做可疑扫描：那是我们自己发布/用户确认过的说明文字，扫它只会制造噪音。
                content = strip_invisible(result.content)
                flagged: list[str] = []
            else:
                # 外部内容统一打上来源标记：工具结果一律当**数据**看（guard 模块文档）。
                # 命中的可疑模式只作为属性记录，不改内容、不拦截。
                flagged = scan_suspicious(result.content)
                detail = {
                    key: args[key] for key in ("path", "cwd") if isinstance(args.get(key), str)
                }
                content = wrap_untrusted(
                    result.content,
                    source=name,
                    suspicious=",".join(flagged) if flagged else None,
                    **detail,
                )
                if flagged:
                    logger.warning(
                        "工具结果命中可疑模式",
                        extra=log_extra(
                            session_id=emitter.session_id,
                            turn_id=turn_id,
                            tool=name,
                            rules=flagged,
                        ),
                    )
            results.append(ToolMessage(content=content, tool_call_id=call_id))
            await emitter.emit(
                EventType.TOOL_RESULT,
                {
                    "call_id": call_id,
                    "name": name,
                    "status": "ok" if result.ok else "error",
                    "exit_code": result.exit_code,
                    "truncated": result.truncated,
                    "duration_ms": result.duration_ms,
                    "preview": result.content[:400],
                    "suspicious": flagged,
                },
            )
            if recorded:
                await repo.finish_tool_call(
                    db,
                    call_id,
                    status="ok" if result.ok else "error",
                    decision="auto",
                    result=result.content,  # 全量入库；模型只看到截断版
                    exit_code=result.exit_code,
                )

        return {
            "messages": results,
            "tool_rounds": (state.get("tool_rounds") or 0) + 1,
            "tool_stats": stats,
            "halt": halt,
            "approvals": {},  # 本批用完了，清空，别影响下一批
        }

    return call_tools


def _refusal_reason(decision: PolicyDecision, approved: bool) -> str | None:
    """不执行时返回要回填给模型的文本；该执行则返回 None。

    三种不执行的情况合并在这里：危险拒绝、需要审批但没批、审批超时（bridge 会
    把超时折成"未批准"，所以到这儿看不出区别——这也是我们想要的：对模型而言
    都是"这条路走不通，换个做法"）。
    """
    if decision.decision is Decision.DENY:
        return (
            f"未执行（拒绝）：{decision.reason}\n"
            "别再重试同一个命令；换个做法，或者把这里被拦下来的情况告诉用户。"
        )
    if decision.decision is Decision.APPROVAL and not approved:
        return (
            f"未执行（人工未批准）：{decision.reason}\n"
            "别原样重试；先说明你想做什么、为什么需要它，或者换一条不需要审批的路。"
        )
    return None
