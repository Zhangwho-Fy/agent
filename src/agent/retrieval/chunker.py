"""切块：按行开窗，带重叠。

**为什么不用 LangChain 的 splitter**：这里要的能力很小（按行窗口 + 重叠），
而 `langchain-text-splitters` 没装、沙箱里也装不了，引进来只会让默认路径依赖
一个下不了的东西。接口就是 `list[Chunk]`，将来要换成语法树切分，换实现即可。

按行切对代码是合适的：一个函数、一段注释天然就是若干行，行号还是最好的定位符
（给模型看、给人看都用得上）。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: 只索引这些扩展名：别的要么是二进制，要么对代码检索没意义
SOURCE_SUFFIXES: frozenset[str] = frozenset(
    {".py", ".md", ".toml", ".txt", ".json", ".yaml", ".yml", ".sql", ".sh", ".ini", ".cfg"}
)

#: 这些目录不必进：依赖、缓存、构建产物，索引它们只会淹没有效结果
SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "dist",
        "build",
        "recordings",
    }
)

#: 超过这个大小就不切了——一个文件撑爆索引不值得
MAX_FILE_BYTES = 512 * 1024


@dataclass(frozen=True, slots=True)
class Chunk:
    """一块源码：位置 + 内容。行号是 1 起、闭区间。"""

    path: str  # 相对工作区，用 / 分隔，跨平台一致
    start_line: int
    end_line: int
    content: str


def iter_source_files(root: Path) -> Iterator[Path]:
    """遍历工作区里的源码文件。

    用 `os.walk` 而不是 `rglob`：前者能在**进入之前**剪掉 `SKIP_DIRS`，
    否则光是把 `.venv` 列一遍就够慢的（在挂载盘上尤其明显）。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.suffix.lower() in SOURCE_SUFFIXES:
                yield path


def chunk_file(
    path: Path,
    root: Path,
    *,
    max_lines: int = 60,
    overlap: int = 10,
) -> list[Chunk]:
    """把一个文件切成若干块；读不了（二进制/太大）就返回空列表。"""
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    lines = text.splitlines()
    if not lines:
        return []
    step = max(max_lines - overlap, 1)
    relative = path.relative_to(root).as_posix()

    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        window = lines[start : start + max_lines]
        body = "\n".join(window)
        if body.strip():  # 全空白的窗口没有检索价值
            chunks.append(
                Chunk(
                    path=relative,
                    start_line=start + 1,
                    end_line=start + len(window),
                    content=body,
                )
            )
        if start + max_lines >= len(lines):
            break
        start += step
    return chunks
