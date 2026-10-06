"""防泄露：仓库里不许出现真实形态的密钥。

**踩过的坑**：`tests/unit/test_cli.py` 里有一条断言，本意是"这个值看起来像真 key、
不该被当成占位符"——但写的人直接把**真实 API key** 粘了进去。那个文件随着仓库推到了
**公开的** GitHub 上，key 被爬虫捡走刷掉了余额。

这条测试把"以后别再犯"变成 CI 里的一条硬约束：扫描所有 **git 跟踪的文件**
（也就是所有会被公开的东西），命中真实密钥形态就失败。

为什么只扫"跟踪的文件"：`.env` 这类只在本地、被 `.gitignore` 挡住的文件不在扫描范围——
它们本来就不会公开，扫了只会天天误报。

为什么不是"什么都拦"：占位符要放行。判据是**形态**——真 key 有固定形状
（`sk-` 后面跟 32 位 hex 等），而 `sk-xxxxxxxx` / `sk-your-key-here` 不是。
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

#: 真实密钥的形态。占位符故意都躲开这些形状：
#: `xxxx…` 不是 hex（x 不在 0-9a-f），`your-key-here` 长度也不够。
SECRET_SHAPES: dict[str, re.Pattern[str]] = {
    # DeepSeek / 其它 `sk-` + 32 位 hex 的服务
    "sk- + 32 位 hex": re.compile(r"sk-[0-9a-fA-F]{32}"),
    # OpenAI 那种更长的 key
    "sk- 长串": re.compile(r"sk-[A-Za-z0-9]{40,}"),
    "GitHub token": re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    "AWS access key id": re.compile(r"AKIA[0-9A-Z]{16}"),
}

#: 这些文件不扫：它们是"描述密钥长什么样"的规则本身，扫了只会自伤
SKIP_FILES = {"tests/unit/test_no_secrets.py"}


def tracked_files() -> list[str]:
    """git 跟踪的文件列表——也就是"会被公开的东西"。"""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_tracked_files_contain_no_real_looking_secrets() -> None:
    hits: list[str] = []
    for relative in tracked_files():
        if relative in SKIP_FILES:
            continue
        path = REPO_ROOT / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue  # 二进制或读不了，跳过
        for name, pattern in SECRET_SHAPES.items():
            for match in pattern.finditer(text):
                # 只报前 8 个字符：失败信息本身不能又变成一次泄露
                hits.append(f"{relative}｜{name}｜{match.group()[:8]}…")

    assert not hits, (
        "仓库里出现了真实形态的密钥（这些文件会被推到公开仓库）：\n  "
        + "\n  ".join(hits)
        + "\n\n把它换成明显的占位符；如果它曾经是真实密钥，先去控制台吊销。"
    )


def test_placeholder_shapes_are_not_flagged() -> None:
    """占位符必须放行，否则这条测试自己就会把正常开发卡死。"""
    samples = [
        "sk-your-key-here",
        "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "sk-super-secret-value",
        "sk-test-fixture-not-a-real-key",
        "",
    ]
    for sample in samples:
        for pattern in SECRET_SHAPES.values():
            assert not pattern.search(sample), f"占位符被误判成密钥：{sample}"
