"""golden 评测集：跑回放，给出通过率。

**不需要联网、不需要密钥**——模型响应全部来自 `evals/recordings/*.jsonl`。
所以它既能在本地随手跑，也能直接放进 CI。

改了提示词、图结构、工具清单之后，这里会告诉你"整体是变好还是变坏"；
如果哪条挂了，看失败原因，再决定是修代码还是重新录一份夹具。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from agent.eval import load_cases, run_all

pytestmark = pytest.mark.eval

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_DIR = REPO_ROOT / "evals" / "cases"


async def test_golden_set_passes() -> None:
    cases = load_cases(CASES_DIR)
    assert cases, f"没有找到 golden 用例：{CASES_DIR}"

    report = await run_all(cases, repo_root=REPO_ROOT)
    print("\n" + report.as_text())

    assert report.failed == 0, "有 golden 用例没通过：\n" + report.as_text()


async def test_golden_report_carries_cost_and_efficiency() -> None:
    """L2（docs/design.md 第 10.2 节）：报告要能看出"过了，但代价不对"。

    断言的是**报告里真的有这些数字**，不是某个具体阈值——阈值要等积累几次基线再定。
    """
    report = await run_all(load_cases(CASES_DIR), repo_root=REPO_ROOT)
    text = report.as_text()

    assert "token" in text, "报告没有 token 列"
    assert "轮" in text, "报告没有工具往返次数"
    assert "合计" in text, "报告没有总量行"
    totals = report.totals
    assert totals["input_tokens"] > 0, "回放夹具带了 usage，聚合不应该是 0"
    # 每条用例都要有轮数上限的依据，否则 hit_round_limit 永远是 False
    assert all(result.max_tool_rounds > 0 for result in report.results)


@pytest.mark.skipif(shutil.which("git") is None, reason="没有 git，跳过仓库检查")
def test_fixtures_are_tracked_by_git() -> None:
    """夹具必须真的在仓库里，而不只是在本机磁盘上。

    踩过的坑：`evals/traces/` 被 `.gitignore` 里那条 `traces/`（本意是忽略运行产物）
    一起吞掉了——本地跑全绿，CI 一上来三条全挂，因为 CI 机器上**只有仓库里的东西**。
    这条测试把"你以为提交了"变成"当场就知道"。（该目录现已改名 `recordings/`。）
    """
    for case in load_cases(CASES_DIR):
        relative = case.trace if not case.trace.is_absolute() else case.trace.relative_to(REPO_ROOT)
        assert (REPO_ROOT / relative).exists(), f"夹具不存在：{relative}"
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(relative)],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        assert tracked.returncode == 0, (
            f"夹具没有被 git 跟踪，CI 上会读不到：{relative}"
            "（检查它是不是撞上了 .gitignore 的规则）"
        )
