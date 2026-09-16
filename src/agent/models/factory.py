"""构建 chat model。

只做一件事：把 `Settings` 翻译成 LangChain 的模型对象。单独一层的理由——
测试里要换成回放模型、阶段 3 要在这里挂录制包装，而图本身不该知道差别。
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel

from ..config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.provider == "deepseek":
        return _build_deepseek(settings)
    raise ValueError(f"暂不支持的 provider：{settings.provider}")


def _build_deepseek(settings: Settings) -> BaseChatModel:
    from langchain_deepseek import ChatDeepSeek

    kwargs: dict[str, Any] = {"model": settings.model, "temperature": 0}
    if settings.api_key:
        kwargs["api_key"] = settings.api_key

    # 不同版本的适配类对自定义端点的字段名不一致（api_base / base_url），
    # 按实际存在的字段挑一个，而不是猜一个然后让人去 debug。
    fields = getattr(ChatDeepSeek, "model_fields", {})
    for candidate in ("api_base", "base_url", "openai_api_base"):
        if candidate in fields:
            kwargs[candidate] = settings.base_url
            break

    return ChatDeepSeek(**kwargs)
