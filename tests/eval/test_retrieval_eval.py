"""检索评测：给出一组可比较的数字（recall@5 / MRR）。

**不需要联网、不需要密钥**：默认嵌入后端是离线的确定性实现。

注意：离线后端只做字面级匹配，**这些数字不能代表真模型的效果**。
它的价值在于：换切块策略、改融合参数之后，能立刻看出是变好还是变坏。
真模型（`AGENT_EMBED_BACKEND=fastembed`）跑同一套用例，数字会明显更高。
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent.retrieval import EvalCase, RetrievalIndex, build_embedder, evaluate, hybrid_search
from agent.store.db import Database

pytestmark = pytest.mark.eval

REPO_ROOT = Path(__file__).resolve().parents[2]
CASES_FILE = REPO_ROOT / "evals" / "retrieval" / "cases.json"

#: 回归底线。刻意不设 100%——离线后端是兜底实现，
#: 追求"全对"会逼着人去调用例，而不是调代码。
MIN_RECALL = 0.5


async def test_retrieval_quality_meets_the_floor() -> None:
    payload = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    cases = [
        EvalCase(query=row["query"], expected=tuple(row["expected"])) for row in payload["cases"]
    ]

    with TemporaryDirectory() as tmp:
        db = Database(Path(tmp) / "index.db")
        db.connect()
        index = RetrievalIndex(db, build_embedder("offline"), REPO_ROOT / payload["root"])
        indexed = await index.rebuild()
        assert indexed > 0, "索引是空的：检查 root 和切块"

        async def search(query: str, limit: int) -> list[str]:
            hits = await hybrid_search(db, index.embedder, query, limit=limit)
            # 同一文件的多个块算一次命中：指标关心的是"文件有没有被找到"
            paths: list[str] = []
            for hit in hits:
                if hit.path not in paths:
                    paths.append(hit.path)
            return paths

        report = await evaluate(search, cases, k=5)
        print("\n" + report.as_text())
        await db.close()

    assert report.recall >= MIN_RECALL, report.as_text()
