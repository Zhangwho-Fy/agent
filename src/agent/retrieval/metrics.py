"""检索指标：recall@k 与 MRR。

**为什么必须有数字**：检索改一个参数（切块大小、融合权重、换模型）之后，
"感觉好像好一点"没有任何说服力。两个指标分别回答两个问题：

- **recall@k**：正确答案有没有进前 k 条？（覆盖能力）
- **MRR**：第一个正确答案排在第几？（排序质量）

一个高一个低是很常见的事：recall@5 很高但 MRR 很低，说明东西找得到、但排得靠后——
模型仍然得在一堆噪声里挑。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

#: 检索函数签名：给一个查询和前 k 条，返回命中的文件路径（按排名）
SearchFn = Callable[[str, int], Awaitable[list[str]]]


@dataclass(frozen=True, slots=True)
class EvalCase:
    """一条检索用例：一个问题 + 期望命中的文件（任意一个命中就算对）。"""

    query: str
    expected: tuple[str, ...]


@dataclass(slots=True)
class CaseOutcome:
    case: EvalCase
    hits: list[str] = field(default_factory=list)  # 命中的文件路径，按排名
    rank: int | None = None  # 第一个命中的名次（1 起），没命中为 None

    @property
    def hit(self) -> bool:
        return self.rank is not None


@dataclass(slots=True)
class EvalReport:
    k: int
    outcomes: list[CaseOutcome]

    @property
    def recall(self) -> float:
        if not self.outcomes:
            return 0.0
        return sum(1 for outcome in self.outcomes if outcome.hit) / len(self.outcomes)

    @property
    def mrr(self) -> float:
        """平均倒数排名：命中记 1/rank，没命中记 0。"""
        if not self.outcomes:
            return 0.0
        total = sum(1.0 / outcome.rank for outcome in self.outcomes if outcome.rank)
        return total / len(self.outcomes)

    def as_text(self) -> str:
        lines = [
            f"检索评测：{len(self.outcomes)} 条查询｜"
            f"recall@{self.k} = {self.recall:.0%}｜MRR = {self.mrr:.3f}"
        ]
        for outcome in self.outcomes:
            if outcome.hit:
                lines.append(f"  ✅ rank {outcome.rank}｜{outcome.case.query}")
            else:
                top = "、".join(outcome.hits[:3]) or "无结果"
                lines.append(f"  ❌ 未命中｜{outcome.case.query}｜前三条是 {top}")
        return "\n".join(lines)


async def evaluate(
    search: SearchFn,
    cases: list[EvalCase],
    *,
    k: int = 5,
) -> EvalReport:
    """跑一批用例，算出 recall@k 与 MRR。

    `search` 是个协程函数：`(query, limit) -> list[命中路径]`——
    只依赖"给我路径列表"，所以词法、语义、混合检索都能用同一套指标。
    """
    outcomes: list[CaseOutcome] = []
    for case in cases:
        paths = await search(case.query, k)
        expected = set(case.expected)
        rank = next((i for i, path in enumerate(paths, start=1) if path in expected), None)
        outcomes.append(CaseOutcome(case=case, hits=paths, rank=rank))
    return EvalReport(k=k, outcomes=outcomes)
