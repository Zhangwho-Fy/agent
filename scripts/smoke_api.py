"""阶段 1.1：模型接口连通性烟测。

目的不是"跑通一个 demo"，而是确定三件事（它们决定 provider 怎么写）：

1. base_url 与 key 是否有效——用 `models.list` 最小代价验证；
2. DeepSeek 走哪种协议形态：`chat.completions` 还是 `responses`；
3. 工具调用（tool calls）能否正常返回——这是主循环的地基。

用法：

    .venv/bin/python scripts/smoke_api.py

三次调用都是极短请求，token 消耗可以忽略。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import AsyncOpenAI

from agent.config import Settings

#: 一个确定性极强的工具：只回显入参，便于断言模型确实发起了调用
ECHO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "回显传入的文本，无副作用",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的文本"}},
            "required": ["text"],
        },
    },
}

TOOL_PROMPT = "请调用 echo 工具，把 text 设为 ping。不要直接回答。"


def line(label: str, value: str) -> None:
    print(f"  {label:<26} {value}")


async def check_models(client: AsyncOpenAI) -> bool:
    print("[1/3] models.list —— 验证 base_url 与密钥")
    started = time.perf_counter()
    try:
        page = await client.models.list()
    except Exception as exc:
        line("结果", f"失败：{type(exc).__name__}: {exc}")
        return False

    ids = [model.id for model in page.data]
    line("结果", f"OK（{time.perf_counter() - started:.1f}s）")
    line("可见模型数", str(len(ids)))
    line("模型示例", ", ".join(ids[:5]) if ids else "(空)")
    return True


async def check_chat_completions(client: AsyncOpenAI, model: str) -> bool:
    print("[2/3] chat.completions + 工具调用（流式）")
    started = time.perf_counter()
    text_parts: list[str] = []
    calls: list[str] = []
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": TOOL_PROMPT}],
            tools=[ECHO_TOOL],
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                text_parts.append(delta.content)
            for call in delta.tool_calls or []:
                name = call.function.name if call.function else None
                if name:
                    calls.append(name)
    except Exception as exc:
        line("结果", f"失败：{type(exc).__name__}: {exc}")
        return False

    line("结果", f"OK（{time.perf_counter() - started:.1f}s）")
    line("文本输出", "".join(text_parts)[:60] or "(无)")
    line("工具调用", ", ".join(calls) if calls else "(无)")
    return bool(calls)


async def check_responses(client: AsyncOpenAI, model: str) -> bool:
    print("[3/3] responses —— 确认另一种协议形态是否可用")
    started = time.perf_counter()
    try:
        response = await client.responses.create(model=model, input="只回答两个字：可用")
    except Exception as exc:
        line("结果", f"不可用：{type(exc).__name__}: {str(exc)[:120]}")
        return False

    line("结果", f"OK（{time.perf_counter() - started:.1f}s）")
    line("输出", (getattr(response, "output_text", "") or "")[:60])
    return True


async def main() -> int:
    settings = Settings()
    print("配置：")
    line("provider / model", f"{settings.provider} / {settings.model}")
    line("base_url", settings.base_url)
    line("api_key", f"已设置（{len(settings.api_key)} 字符）" if settings.api_key else "未设置")
    print()

    if not settings.api_key:
        print("没有密钥，无法烟测。请先在 .env 里填 AGENT_API_KEY。")
        return 2

    client = AsyncOpenAI(base_url=settings.base_url, api_key=settings.api_key, timeout=60.0)
    try:
        auth_ok = await check_models(client)
        if not auth_ok:
            print("\n结论：认证或地址有问题，先解决这一步，后面的协议判定没有意义。")
            return 1

        print()
        chat_ok = await check_chat_completions(client, settings.model)
        print()
        responses_ok = await check_responses(client, settings.model)
    finally:
        await client.close()

    print("\n结论：")
    line("chat.completions", "可用" + ("（含工具调用）" if chat_ok else "（但未返回工具调用）"))
    line("responses", "可用" if responses_ok else "不可用")
    if chat_ok:
        line("provider 实现", "走 chat.completions")
    elif responses_ok:
        line("provider 实现", "走 responses")
    else:
        line("provider 实现", "两条路都有问题，需要人工确认")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
