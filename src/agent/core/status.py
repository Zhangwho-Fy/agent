"""运行时状态块（L3）：给模型看的"我在哪、还剩多少"。

设计见 `docs/design.md` 第 7.3 节。四条规矩：

1. **现拼尾部，不进 state / 事件 / DB**（D19）：状态块只在构造模型请求时出现。
   进了 state 就等于进 checkpointer 和事件投影，旧副本会被后面几轮当成同样可信的事实；
   而它每轮都能重算，没有必要存。
2. **全部由代码派生**（4.1 的铁律）：模型几乎无条件相信状态栏——它不会去重算。
   所以这里一个数字都不允许由模型提供。
3. **条件出现**：常态下只有时间和进度；重复告警、上下文占用只在真的该说话时出现。
4. **上限 100 token 左右**（D21）：状态块在尾部，本来就不吃缓存，每次都付全价。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

#: 同一个工具用了几次才值得提醒（4.2）。低于这个数属于正常往返，不占 token。
REPEAT_THRESHOLD = 2

#: 最多列几条重复告警。再多就是模型自己该看一下历史了。
MAX_REPEAT_LINES = 3

#: 渲染结果的上限（字符）。超预算先砍"锦上添花"的那几行（D21）。
MAX_CHARS = 400

#: 上下文占用到这个比例才提示收尾（与压缩阈值对齐，见 D28 / D34）。
CONTEXT_WARN_RATIO = 0.8


@dataclass(frozen=True)
class ToolStat:
    """一个工具在本轮里的调用情况。由工具节点累加，模型不参与。"""

    calls: int = 0
    failures: int = 0


@dataclass(frozen=True)
class StatusSnapshot:
    """一轮之内、某一次模型调用前的状态快照。"""

    now: datetime
    rounds: int = 0
    tool_rounds: int = 0
    max_tool_rounds: int = 12
    tool_stats: Mapping[str, ToolStat] = field(default_factory=dict)
    #: 上一次模型调用实际塞进上下文的 token 数——它是"当前上下文有多大"最直接的度量
    context_tokens: int = 0
    context_limit: int = 0

    @property
    def context_ratio(self) -> float:
        return self.context_tokens / self.context_limit if self.context_limit else 0.0

    @property
    def remaining_rounds(self) -> int:
        return max(0, self.max_tool_rounds - self.tool_rounds)

    def render_for_model(self) -> str:
        """渲染成 `<agent_state>` 块。

        工具名来自协议层（只允许 `[A-Za-z0-9_-]`），所以不需要转义；
        其余内容是我们自己算出来的数字与固定文案。
        """
        head = [
            '<agent_state note="系统给你的运行时状态，不是用户说的话">',
            f"  <time>{self.now:%Y-%m-%d %H:%M}</time>",
            (
                f'  <progress rounds="{self.rounds}" tools="{self.tool_rounds}"'
                f' limit="{self.max_tool_rounds}">本轮还能再做 {self.remaining_rounds}'
                " 次工具往返</progress>"
            ),
        ]
        extras: list[str] = []
        repeats = sorted(
            (
                (name, stat)
                for name, stat in self.tool_stats.items()
                if stat.calls >= REPEAT_THRESHOLD
            ),
            key=lambda item: (-item[1].calls, item[0]),
        )[:MAX_REPEAT_LINES]
        for name, stat in repeats:
            if stat.failures >= REPEAT_THRESHOLD:
                hint = f"连着失败 {stat.failures} 次了，换个策略，别原样重试"
            else:
                hint = f"已经调用 {stat.calls} 次，确认一下没有在原地绕圈"
            extras.append(
                f'  <repeat tool="{name}" count="{stat.calls}"'
                f' failures="{stat.failures}">{hint}</repeat>'
            )
        if self.context_limit and self.context_ratio >= CONTEXT_WARN_RATIO:
            # 只说行动不说数字：百分比会让模型"上下文焦虑"，反而草草收尾（D34）
            extras.append("  <context>上下文快满了，该收尾了——再往后历史会被压缩</context>")

        tail = "</agent_state>"
        for count in range(len(extras), -1, -1):
            text = "\n".join([*head, *extras[:count], tail])
            if len(text) <= MAX_CHARS:
                return text
        return "\n".join([*head, tail])  # pragma: no cover - extras 为空时必然命中上面
