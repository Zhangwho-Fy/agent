"""ID 生成。

统一前缀是为了在日志和事件流里一眼区分对象类型；用 uuid4 而不是数据库自增，
是为了不依赖存储、不必回查、多端并发也不会撞号。
"""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """生成形如 `sess_1f2e3d...` 的标识符。"""
    return f"{prefix}_{uuid4().hex}"
