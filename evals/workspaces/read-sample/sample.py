"""一个小样例模块，用作评测夹具。

刻意保持稳定：夹具变了，录制好的回放就对不上了。
"""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """返回两个数之和。"""
    return a + b


def greet(name: str) -> str:
    """打个招呼。"""
    return f"你好，{name}"


def is_even(number: int) -> bool:
    """判断是不是偶数。"""
    return number % 2 == 0
