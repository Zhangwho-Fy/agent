"""教学脚本：把"发给模型的原始报文"和"模型回来的原始报文"打印出来。

跑一次就能看清楚四件事：

1. 工具清单（tools）是怎么传过去的——它就是个 JSON 数组
2. 模型返回的"工单"（tool_calls）长什么样，含 id / name / arguments
3. 流式下工单是**碎着来的**：arguments 会分好几块，拼起来才是合法 JSON
4. 结果怎么回填（role=tool + tool_call_id 配对），模型拿到结果后才给出答案

这里用的是**真实的 HTTP 报文**（`with_raw_response` 拿到 status/text/headers，
`http_request` 拿到请求体），不是 SDK 反序列化后的对象，所以你会看到协议原貌。

用法：

    .venv/bin/python scripts/dump_raw.py

只发两次极短请求，token 消耗可以忽略；密钥不会打印。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import OpenAI

from agent.config import Settings
from agent.tools.base import ToolContext
from agent.tools.registry import default_registry

#: 明确要求它调工具，免得模型直接回答、看不到工单
PROMPT = "请调用 fs_read 读取 src/agent/graph/builder.py 的前 10 行。不要直接回答。"

MAX_DUMP_CHARS = 4000


def show(title: str, payload: Any) -> None:
    """分段打印，长内容截断，避免刷屏。"""
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
    if not isinstance(payload, str):
        payload = json.dumps(payload, ensure_ascii=False, indent=2)
    if len(payload) > MAX_DUMP_CHARS:
        payload = payload[:MAX_DUMP_CHARS] + f"\n... [已截断，共 {len(payload)} 字符]"
    print(payload)


def mask_headers(headers: dict[str, str]) -> dict[str, str]:
    """请求头里的 Authorization 不能原样打印。"""
    out = {}
    for key, value in headers.items():
        if key.lower() in {"authorization", "api-key", "x-api-key"}:
            out[key] = "Bearer sk-****（已打码）"
        else:
            out[key] = value
    return out


def main() -> int:
    settings = Settings()
    if not settings.api_key:
        print("没有密钥，先在 .env 里填 AGENT_API_KEY")
        return 2

    client = OpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=60.0)
    registry = default_registry()
    tools = registry.to_openai_tools()
    messages: list[dict[str, Any]] = [{"role": "user", "content": PROMPT}]
    payload: dict[str, Any] = {"model": settings.model, "messages": messages, "tools": tools}

    # ---- ① 我们要发出去的东西 ----
    show(
        "① 请求体：tools 就是这么传过去的（注意每个工具都有 name/description/parameters）", payload
    )

    # ---- ② 非流式：拿到真实 HTTP 响应 ----
    print("\n" + "=" * 72)
    print("② 非流式调用：真实 HTTP 报文")
    print("=" * 72)
    try:
        raw = client.chat.completions.with_raw_response.create(**payload)
    except Exception as exc:
        print(f"调用失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"HTTP 状态：{raw.status_code}")
    request_headers = getattr(getattr(raw, "http_request", None), "headers", None)
    if request_headers:
        show("真实的请求头（Authorization 已打码）", mask_headers(dict(request_headers)))
    request_body = getattr(getattr(raw, "http_request", None), "content", None)
    if request_body:
        show("真实的请求体（字节级原文）", request_body.decode("utf-8", errors="replace"))
    show("真实的响应体（字节级原文）", raw.text)

    completion = raw.parse()
    choice = completion.choices[0]
    message = choice.message
    print(f"\nfinish_reason = {choice.finish_reason!r}（调工具时是 'tool_calls'）")
    print(f"message.content = {message.content!r}（调工具时通常是空）")
    tool_calls = message.tool_calls or []
    print(f"工单数量：{len(tool_calls)}")
    for call in tool_calls:
        print(f"  - id={call.id!r}  name={call.function.name!r}")
        print(f"    arguments（字符串！）= {call.function.arguments!r}")

    if not tool_calls:
        print("\n这次模型没有递工单，后面的步骤跳过。")
        return 0

    # ---- ③ 流式：工单是碎着来的 ----
    show(
        "③ 流式调用：下面是每个 chunk 的原貌（重点看 arguments 怎么被切开）",
        "(只显示 model_dump 后的 JSON，逐条打印)",
    )
    fragments: dict[int, list[str]] = {}
    try:
        stream = client.chat.completions.create(**payload, stream=True)
        for index, chunk in enumerate(stream):
            print(f"chunk[{index}] {chunk.model_dump_json(exclude_none=True)[:400]}")
            if not chunk.choices:
                continue
            for call in chunk.choices[0].delta.tool_calls or []:
                if call.function and call.function.arguments:
                    fragments.setdefault(call.index, []).append(call.function.arguments)
    except Exception as exc:
        print(f"流式调用失败：{type(exc).__name__}: {exc}")

    if fragments:
        print("\n把 arguments 的分片拼起来：")
        for index, parts in fragments.items():
            joined = "".join(parts)
            print(f"  工单 {index} 分了 {len(parts)} 块 → {joined!r}")
            try:
                print(f"  解析成对象 → {json.loads(joined)}")
            except json.JSONDecodeError as exc:
                print(f"  解析失败（分片阶段本来就可能不是合法 JSON）：{exc.msg}")

    # ---- ④ 回填结果，让模型给出最终答案 ----
    print("\n" + "=" * 72)
    print("④ 我们执行工具，把结果按 tool_call_id 配对回填，再问一次")
    print("=" * 72)
    call = tool_calls[0]
    tool = registry.get(call.function.name)
    if tool is None:
        print(f"未知工具：{call.function.name}")
        return 1

    args = json.loads(call.function.arguments or "{}")
    ctx = ToolContext(
        workspace=settings.resolved_workspace,
        timeout_s=settings.tool_timeout_s,
        output_limit_bytes=settings.output_limit_bytes,
    )
    result = asyncio.run(tool.run(args, ctx))
    print(
        f"本地工具执行结果：ok={result.ok} truncated={result.truncated} 耗时={result.duration_ms}ms"
    )

    follow_up = [
        *messages,
        {"role": "assistant", "content": None, "tool_calls": [c.model_dump() for c in tool_calls]},
        {"role": "tool", "tool_call_id": call.id, "content": result.content},
    ]
    show("回填后的请求体（注意 role=tool 那条带着 tool_call_id）", {"messages": follow_up})

    final = client.chat.completions.create(
        model=settings.model,
        messages=follow_up,
        tools=tools,  # 工具清单每次都要带上
    )
    answer = final.choices[0].message.content
    show("模型的最终回答", answer or "(空)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
