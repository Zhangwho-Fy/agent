"""golden 评测集：跑回放，给出通过率。

**不需要联网、不需要密钥**——模型响应全部来自 `evals/traces/*.jsonl`。
所以它既能在本地随手跑，也能直接放进 CI。

改了提示词、图结构、工具清单之后，这里会告诉你"整体是变好还是变坏"；
如果哪条挂了，看失败原因，再决定是修代码还是重新录一份夹具。
"""

from __future__ import annotations

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
