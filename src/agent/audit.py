"""会话审计：从真实会话库里找异常模式。

设计见 `docs/design.md` 第 10.2 节（L2 的成本与效率）。四条约定：

1. **只读**：用 sqlite 的 `mode=ro` 打开，绝不写——诊断不该改变被诊断的对象（D65）。
2. **不造数据**：只聚合 `turns` / `events` / `tool_calls` 里已有的东西，不新增采集链路（D62）。
3. **给行动**：每条发现都要能回答"接下来该看什么"，不堆原始行。
4. **不是评测**：它不给通过率、不阻塞 CI。看它更像看监控，而不是看考试成绩。

建议的阅读顺序：先看异常 turn 状态，再看"哪一轮最忙"，最后看反复失败的工具——
前者说明"出过事"，后两者说明"事情没做完还在硬撑"。找到之后用
`agent replay <会话>` 复盘那一条。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

#: 一轮里工具往返超过它就值得看一眼（正常任务很少超过）
BUSY_ROUNDS = 8

#: 同一个工具在同一轮里失败几次算"卡住"
REPEAT_FAILURES = 2

#: 一次报告里每类发现最多列几条
DEFAULT_LIMIT = 5


@dataclass(frozen=True, slots=True)
class Finding:
    """一条发现。`kind` 机器可读，`detail` 给人看。"""

    kind: str
    detail: str


@dataclass(slots=True)
class AuditReport:
    db_path: str
    sessions: int = 0
    turns: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    findings: list[Finding] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"会话审计：{self.db_path}",
            f"  会话 {self.sessions}｜轮次 {self.turns}"
            f"｜token {self.input_tokens}+{self.output_tokens}",
        ]
        if not self.findings:
            lines.append("  没有发现异常（工具往返、失败重试、审批拒绝都在正常范围）")
            return "\n".join(lines)
        for finding in self.findings:
            lines.append(f"  ⚠️ [{finding.kind}] {finding.detail}")
        lines.append("  复盘：agent replay <会话 id>")
        return "\n".join(lines)


def scan(db_path: str | Path, *, limit: int = DEFAULT_LIMIT) -> AuditReport:
    """扫描一个会话库，返回异常清单。只读，不改动任何数据。"""
    path = Path(db_path).expanduser()
    if not path.exists():
        msg = f"会话库不存在：{path}"
        raise FileNotFoundError(msg)
    # 只读 URI：诊断不该有机会改库。WAL 模式下只读打开也能看到已提交的数据。
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        report = AuditReport(db_path=str(path))
        _totals(conn, report)
        _abnormal_turns(conn, report, limit)
        _busy_turns(conn, report, limit)
        _stuck_tools(conn, report, limit)
        _denied_approvals(conn, report)
        _compression(conn, report)
        return report
    finally:
        conn.close()


def _totals(conn: sqlite3.Connection, report: AuditReport) -> None:
    report.sessions = int(_scalar(conn, "SELECT COUNT(*) FROM sessions") or 0)
    row = conn.execute(
        "SELECT COUNT(*) AS turns,"
        " COALESCE(SUM(input_tokens), 0) AS inp,"
        " COALESCE(SUM(output_tokens), 0) AS outp"
        " FROM turns"
    ).fetchone()
    if row is not None:
        report.turns = int(row["turns"])
        report.input_tokens = int(row["inp"])
        report.output_tokens = int(row["outp"])


def _abnormal_turns(conn: sqlite3.Connection, report: AuditReport, limit: int) -> None:
    """出过事的轮次：failed / interrupted，或崩了之后没来得及收尾的 running。"""
    rows = conn.execute(
        "SELECT session_id, id, status FROM turns"
        " WHERE status IN ('failed', 'interrupted', 'running')"
        " ORDER BY started_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    for row in rows:
        report.findings.append(
            Finding(
                kind="abnormal-turn",
                detail=f"{row['status']}｜{row['session_id']}｜{row['id']}",
            )
        )


def _busy_turns(conn: sqlite3.Connection, report: AuditReport, limit: int) -> None:
    """最忙的几轮：工具调用多到不正常的，多半在原地打转。"""
    rows = conn.execute(
        "SELECT session_id, turn_id, COUNT(*) AS calls FROM tool_calls"
        " GROUP BY turn_id HAVING calls >= ?"
        " ORDER BY calls DESC LIMIT ?",
        (BUSY_ROUNDS, limit),
    ).fetchall()
    for row in rows:
        report.findings.append(
            Finding(
                kind="busy-turn",
                detail=(
                    f"{row['calls']} 次工具调用｜{row['session_id']}｜{row['turn_id']}"
                    "（看看是不是在同一个地方反复试）"
                ),
            )
        )


def _stuck_tools(conn: sqlite3.Connection, report: AuditReport, limit: int) -> None:
    """卡住的工具：同一轮里同一个工具反复失败。"""
    rows = conn.execute(
        "SELECT session_id, turn_id, name,"
        " SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS failures,"
        " COUNT(*) AS calls"
        " FROM tool_calls GROUP BY turn_id, name"
        " HAVING failures >= ? ORDER BY failures DESC LIMIT ?",
        (REPEAT_FAILURES, limit),
    ).fetchall()
    for row in rows:
        report.findings.append(
            Finding(
                kind="stuck-tool",
                detail=(
                    f"{row['name']} 失败 {row['failures']}/{row['calls']} 次"
                    f"｜{row['session_id']}｜{row['turn_id']}"
                ),
            )
        )


def _denied_approvals(conn: sqlite3.Connection, report: AuditReport) -> None:
    """被拒绝的审批：这是"拦住了"的证据，也是"模型想干危险事"的信号。"""
    total = int(
        _scalar(
            conn,
            "SELECT COUNT(*) FROM tool_calls WHERE status = 'denied' OR decision = 'deny'",
        )
        or 0
    )
    if total:
        report.findings.append(Finding(kind="denied", detail=f"{total} 次调用被拒绝或直接拦下"))


def _compression(conn: sqlite3.Connection, report: AuditReport) -> None:
    """压缩触发次数：频繁触发说明单个会话太长了，值得看看是不是绕。"""
    total = int(_scalar(conn, "SELECT COUNT(*) FROM events WHERE type = 'context.compressed'") or 0)
    if total:
        report.findings.append(Finding(kind="compressed", detail=f"上下文压缩触发 {total} 次"))


def _scalar(conn: sqlite3.Connection, sql: str) -> object | None:
    row = conn.execute(sql).fetchone()
    return None if row is None else row[0]
