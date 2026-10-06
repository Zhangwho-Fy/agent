"""golden 评测集：把"改完是变好还是变坏"变成一个数字。

agent 是概率系统：改一行提示词、换一个模型、调一个参数，靠感觉判断不了好坏——
你只会记得最近一次翻车。golden 集给的是**通过率**。

一个 case 由三部分组成：

1. **任务**（`prompt`）—— 一句话，跟真人会敲的一样；
2. **夹具**（`trace`）—— 这个任务录制好的模型响应，跑的时候走回放，不花钱不联网；
3. **判定**（`checks`）—— 断言规则。因为 LLM 输出不是逐字确定的，判定要挑
   那些**稳定的信号**：调用了哪个工具、答案里有没有关键结论、有没有真的动文件。

判定规则刻意做得朴素（见 `CHECKS`）：能用规则判的绝不请模型来判。
模型当裁判又贵又飘，留到真的需要评开放题时再说。
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Case:
    """一条 golden 用例。"""

    id: str
    prompt: str
    trace: Path
    checks: list[dict[str, Any]] = field(default_factory=list)
    #: 需要审批的用例：true = 自动放行，false = 一律拒绝（默认拒绝，才能验"拦得住"）
    approve: bool = False
    #: 相对仓库根的工作区；默认就是仓库根
    workspace: str = "."


@dataclass(slots=True)
class CaseResult:
    case: Case
    passed: bool
    failures: list[str]
    text: str = ""
    duration_ms: int = 0
    tools: list[str] = field(default_factory=list)
    status: str = "done"
    error: str | None = None
    #: L2：成本与效率。数据来自 `TurnResult.usage` 和事件流，不额外采集（D62）
    input_tokens: int = 0
    output_tokens: int = 0
    tool_rounds: int = 0
    max_tool_rounds: int = 0

    @property
    def hit_round_limit(self) -> bool:
        """这一轮是不是因为工具往返用尽才收尾的——"过了但代价不对"的典型信号。"""
        return self.max_tool_rounds > 0 and self.tool_rounds >= self.max_tool_rounds


@dataclass(slots=True)
class Report:
    results: list[CaseResult]

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    @property
    def rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def totals(self) -> dict[str, int]:
        """整套用例的总量：通过率之外的代价（L2）。"""
        return {
            "input_tokens": sum(r.input_tokens for r in self.results),
            "output_tokens": sum(r.output_tokens for r in self.results),
            "tool_rounds": sum(r.tool_rounds for r in self.results),
            "duration_ms": sum(r.duration_ms for r in self.results),
        }

    def as_text(self) -> str:
        lines = [f"golden 通过率：{self.passed}/{len(self.results)} = {self.rate:.0%}"]
        for result in self.results:
            mark = "✅" if result.passed else "❌"
            tools = ",".join(result.tools) or "-"
            cost = (
                f"{result.duration_ms}ms｜工具 {result.tool_rounds} 轮({len(result.tools)} 次)"
                f"｜token {result.input_tokens}+{result.output_tokens}"
            )
            if result.hit_round_limit:
                cost += "｜⚠️ 撞上轮数上限"
            lines.append(f"  {mark} {result.case.id}｜{cost}")
            lines.append(f"      工具序列 {tools}")
            for failure in result.failures:
                lines.append(f"      - {failure}")
        totals = self.totals
        lines.append(
            f"  合计｜{totals['duration_ms']}ms｜工具 {totals['tool_rounds']} 轮"
            f"｜token {totals['input_tokens']}+{totals['output_tokens']}"
        )
        return "\n".join(lines)


def load_cases(directory: str | Path) -> list[Case]:
    """从目录读 `*.json`，一个文件一条用例。文件名就是 id 的兜底。"""
    folder = Path(directory)
    cases: list[Case] = []
    for path in sorted(folder.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append(
            Case(
                id=payload.get("id", path.stem),
                prompt=payload["prompt"],
                trace=Path(payload["trace"]),
                checks=payload.get("checks", []),
                approve=bool(payload.get("approve", False)),
                workspace=payload.get("workspace", "."),
            )
        )
    return cases


async def run_case(case: Case, *, repo_root: Path) -> CaseResult:
    """跑一条用例：回放模型 + 真实图 + 真实工具，最后按 `checks` 判定。"""
    from .config import Settings
    from .core.bus import EventBus
    from .core.events import Event
    from .core.reliability import EventEmitter
    from .graph.bridge import recursion_limit_for as _recursion_limit_for
    from .graph.bridge import stream_turn
    from .graph.checkpointer import open_checkpointer
    from .graph.wiring import build_session_graph
    from .models.factory import build_chat_model

    workspace = (repo_root / case.workspace).resolve()
    trace = case.trace if case.trace.is_absolute() else (repo_root / case.trace)
    settings = Settings(
        _env_file=None,  # 评测不受本机 .env 影响
        provider="replay",
        trace_path=trace,
        workspace=workspace,
        api_key="",
    )

    emitter = EventEmitter("sess_eval", EventBus())
    events: list[Event] = []
    emitter.on_event = events.append

    with tempfile.TemporaryDirectory() as tmp:
        # 评测必须与开发机隔离：记忆库和索引都放临时目录。否则同一个用例在
        # "有记忆的机器"和 CI 上跑出两种结果，而且会把开发机的真实记忆喂给模型。
        settings.memory_db_path = Path(tmp) / "memory.db"
        settings.index_dir = Path(tmp) / "index"
        wiring = build_session_graph(
            settings,
            workspace=workspace,
            model=build_chat_model(settings),
            emitter=emitter,
            # 评测要确定性：压缩会改写消息序列，把录放夹具打乱（D32）
            compress_enabled=False,
            checkpointer=open_checkpointer(Path(tmp) / "ckpt.db"),
        )
        graph = wiring.graph

        async def approver(requests: list[dict[str, Any]]) -> dict[str, bool]:
            return {request["call_id"]: case.approve for request in requests}

        result = await stream_turn(
            graph=graph,
            prompt=case.prompt,
            emitter=emitter,
            session_id="sess_eval",
            turn_id="turn_eval",
            approver=approver,
            approval_timeout_s=5,
            recursion_limit=_recursion_limit_for(settings.max_tool_rounds),
        )

    tools = [str(event.data.get("name", "")) for event in events if event.type.value == "tool.call"]
    # 工具往返次数：事件里的 round 是工具节点自增的，比 len(tools) 更准
    # （一次模型回复可以带多个工具调用，那算一轮）
    rounds = [
        int(event.data.get("round", 0)) for event in events if event.type.value == "tool.call"
    ]
    usage = result.usage or {}
    outcome = CaseResult(
        case=case,
        passed=False,
        failures=[],
        text=result.text,
        duration_ms=result.duration_ms,
        tools=tools,
        status=result.status,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        tool_rounds=max(rounds, default=0),
        max_tool_rounds=settings.max_tool_rounds,
    )
    outcome.failures = _check(case.checks, outcome, workspace, events)
    outcome.passed = not outcome.failures
    return outcome


def _check(
    checks: list[dict[str, Any]],
    result: CaseResult,
    workspace: Path,
    events: list[Any],
) -> list[str]:
    """逐条判定，返回失败原因（空列表 = 通过）。"""
    failures: list[str] = []
    for check in checks:
        kind = check.get("kind")
        if kind == "tool_called":
            if check["name"] not in result.tools:
                failures.append(f"期望调用 {check['name']}，实际调用了 {result.tools or '无'}")
        elif kind == "tool_not_called":
            if check["name"] in result.tools:
                failures.append(f"不该调用 {check['name']}")
        elif kind == "answer_contains":
            if check["text"] not in result.text:
                failures.append(f"答案里没有 {check['text']!r}：{result.text[:120]!r}")
        elif kind == "answer_not_contains":
            if check["text"] in result.text:
                failures.append(f"答案里不该出现 {check['text']!r}")
        elif kind == "answer_nonempty":
            if not result.text.strip():
                failures.append("答案为空")
        elif kind == "max_tool_calls":
            if len(result.tools) > int(check["n"]):
                failures.append(f"工具调用 {len(result.tools)} 次，超过上限 {check['n']}")
        elif kind == "status":
            if result.status != check["value"]:
                failures.append(f"状态是 {result.status}，期望 {check['value']}")
        elif kind == "tool_result_status":
            statuses = [
                event.data.get("status") for event in events if event.type.value == "tool.result"
            ]
            if check["value"] not in statuses:
                failures.append(f"没有出现状态为 {check['value']} 的工具结果：{statuses}")
        elif kind == "file_absent":
            if (workspace / check["path"]).exists():
                failures.append(f"文件不该存在：{check['path']}")
        elif kind == "file_exists":
            if not (workspace / check["path"]).exists():
                failures.append(f"文件应当存在：{check['path']}")
        else:
            failures.append(f"未知的判定类型：{kind}")
    return failures


async def run_all(cases: list[Case], *, repo_root: Path) -> Report:
    results = []
    for case in cases:
        try:
            results.append(await run_case(case, repo_root=repo_root))
        except Exception as exc:  # 用例自身炸了（缺夹具、回放对不上）也算失败
            results.append(
                CaseResult(
                    case=case,
                    passed=False,
                    failures=[f"{type(exc).__name__}: {exc}"],
                    error=str(exc),
                )
            )
    return Report(results)
