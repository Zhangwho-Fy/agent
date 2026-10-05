"""与 `agent serve` 对话的瘦客户端。

只做三件事：建会话、发消息、收事件流（含兑现审批）。渲染留给调用方——
这样 CLI 和测试能用同一套客户端，只是渲染方式不同。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..core.events import Event


class AgentClient:
    """HTTP 客户端。`base_url` 形如 `http://127.0.0.1:8765`。"""

    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # transport 可注入：测试里换成 ASGI 直连，不开真实端口
        self._http = httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> AgentClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def close(self) -> None:
        await self._http.aclose()

    async def health(self) -> dict[str, Any]:
        return (await self._http.get("/health")).raise_for_status().json()

    async def create_session(
        self, *, workspace: str | None = None, profile: str = "code", title: str = ""
    ) -> dict[str, Any]:
        response = await self._http.post(
            "/sessions", json={"workspace": workspace, "profile": profile, "title": title}
        )
        return response.raise_for_status().json()

    async def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        response = await self._http.get("/sessions", params={"limit": limit})
        return list(response.raise_for_status().json()["sessions"])

    async def send_message(
        self, session_id: str, content: str, *, idempotency_key: str | None = None
    ) -> dict[str, Any]:
        """发一条消息。返回 `{turn_id, duplicate}`——`duplicate=True` 说明幂等键命中，
        服务端没有重复执行。"""
        response = await self._http.post(
            f"/sessions/{session_id}/messages",
            json={"content": content, "idempotency_key": idempotency_key},
        )
        return response.raise_for_status().json()

    async def approve(self, session_id: str, call_id: str, *, granted: bool) -> dict[str, Any]:
        response = await self._http.post(
            f"/sessions/{session_id}/approvals/{call_id}", json={"granted": granted}
        )
        return response.raise_for_status().json()

    async def stream_events(self, session_id: str, *, after_seq: int = 0) -> AsyncIterator[Event]:
        """订阅会话事件流。

        `after_seq` 对应服务端的 `Last-Event-ID` 语义：断线重连时传上次收到的最后一个
        seq，服务端会把缺口补齐再接着推。超时设成 None——事件之间可能隔很久。
        """
        url = f"/sessions/{session_id}/events"
        async with self._http.stream(
            "GET", url, params={"after_seq": after_seq}, timeout=None
        ) as response:
            response.raise_for_status()
            frame: dict[str, str] = {}
            async for line in response.aiter_lines():
                if not line:
                    payload = frame.get("data")
                    if payload:
                        yield Event.model_validate_json(payload)
                    frame = {}
                    continue
                if line.startswith(":"):
                    continue  # keep-alive 注释行
                field_name, _, value = line.partition(":")
                frame[field_name.strip()] = value.lstrip()
