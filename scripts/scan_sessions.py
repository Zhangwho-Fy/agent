"""会话审计 CLI：扫真实会话库，找异常模式（见 `docs/design.md` 第 10.2 节）。

用法：

    .venv/bin/python scripts/scan_sessions.py
    .venv/bin/python scripts/scan_sessions.py --db ~/.local/share/agent/agent.db --limit 10

**只读打开**，不改任何数据；不联网、不要密钥。它是运营式自检，不是 CI 门槛——
发现异常返回 0，找不到库才返回 1。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agent.audit import scan
from agent.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="扫会话库找异常模式（只读）")
    parser.add_argument("--db", default="", help="会话库路径；默认取配置里的 db_path")
    parser.add_argument("--limit", type=int, default=5, help="每类发现最多列几条")
    args = parser.parse_args()

    settings = Settings()
    path = Path(args.db).expanduser() if args.db else settings.resolved_db_path
    try:
        report = scan(path, limit=args.limit)
    except FileNotFoundError as exc:
        print(exc)
        print("提示：先跑一次 agent run / agent serve，才会有会话库。")
        return 1
    print(report.as_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
