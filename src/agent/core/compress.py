"""上下文压缩的纯逻辑：阶梯、指针化、摘要请求。

设计见 `docs/context-engineering.md` 第 5 节。这里只做**纯计算**，不碰图、不调模型、
不落库——节点在 `graph/compress.py`，找回通道在 `tools/recall.py`。

### 相对设计文档的一处修正

书里的第一级是"噪声直接删除"。**在我们的消息结构里不能删**：`tool_call` 与
`tool result` 必须配对，删掉一条 `ToolMessage` 会让整段消息序列不合法（协议直接报错）。
所以"删"在这里一律是**换内容、留指针**：原文本来就在 `tool_calls` 表里，模型需要时
用 `recall(call_id=…)` 取回。阶梯因此变成：

    无损（指针化） → 有损（摘要） → 隔离（子 agent，暂不实现）

两档阈值不变：占用 >60% 只做指针化（零 LLM 调用），>80% 才动摘要。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from xml.sax.saxutils import quoteattr

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage

#: 指针里保留多少原文（字符）。太少模型不知道那是什么，太多就白压了。
POINTER_HEAD_CHARS = 200

#: 短于这个长度的工具结果不值得指针化——省不下多少，还把上下文搞碎。
MIN_COMPRESS_CHARS = 400

#: 摘要最多喂多少字符进去。超长时只取头尾：中间往往是同一份输出的重复部分。
SUMMARY_INPUT_CHARS = 24_000

SUMMARY_PROMPT = """你是上下文压缩器。把下面这段对话历史压成结构化事实，供后续继续干活用。

必须**原样保留**（不许改写、不许概括）：
- 文件路径、函数名、变量名、类名
- 错误信息原文
- 版本号、日期、数字
- 用户提出的要求与约束
- 已经做出的决定及理由
- 未完成的事项与下一步

可以丢弃：工具输出的原文、重复的目录列表、成功命令的完整输出、模型的中间推理。
可以提一句"某些工具输出已压缩，需要时用 recall(call_id) 取回"，但不要编造 call_id。

按这个结构输出，没有内容的小节写"（无）"：
## 任务
## 已确认
## 改动
## 未完成
## 教训

只输出正文，不要复述提示词，不要解释你在做什么。"""


@dataclass(frozen=True)
class CompressionPlan:
    """这一次该做哪一档。"""

    pointerize: bool = False
    summarize: bool = False

    @property
    def active(self) -> bool:
        return self.pointerize or self.summarize


def plan(
    ratio: float, *, lossless_ratio: float = 0.6, summary_ratio: float = 0.8
) -> CompressionPlan:
    """按占用率决定跑哪几级。

    分两档的意义：指针化是零成本的（无 LLM、无等待），所以早做；摘要有成本也有风险，
    靠近上限再批量做，别频繁打断缓存（D28）。
    """
    if ratio >= summary_ratio:
        return CompressionPlan(pointerize=True, summarize=True)
    if ratio >= lossless_ratio:
        return CompressionPlan(pointerize=True)
    return CompressionPlan()


def _pointer_content(content: str, call_id: str) -> str:
    head = content[:POINTER_HEAD_CHARS].rstrip()
    if head.endswith("</untrusted>"):  # 短内容会连闭合标签一起进来，先摘掉
        head = head[: -len("</untrusted>")].rstrip()
    quota = quoteattr(call_id)
    return (
        f"<compressed call_id={quota}>\n{head}\n"
        f"...（原文已移出上下文；需要时用 recall(call_id={quota}) 取回全文）\n</compressed>"
    )


def pointerize(
    messages: Sequence[AnyMessage], *, keep_recent: int = 4
) -> tuple[list[AnyMessage], list[str]]:
    """把较早的工具结果换成指针，返回 (新消息列表, 被压缩的 call_id 列表)。

    只换内容，**不删消息**——配对关系必须保住（见模块文档）。
    """
    tool_indexes = [
        index for index, message in enumerate(messages) if isinstance(message, ToolMessage)
    ]
    keep = set(tool_indexes[-keep_recent:]) if keep_recent > 0 else set()
    compressed: list[str] = []
    result: list[AnyMessage] = []
    for index, message in enumerate(messages):
        if (
            index in tool_indexes
            and index not in keep
            and isinstance(message.content, str)
            and len(message.content) > MIN_COMPRESS_CHARS
            and not message.content.startswith("<compressed ")
        ):
            call_id = str(message.tool_call_id or "")
            result.append(
                ToolMessage(
                    content=_pointer_content(message.content, call_id),
                    tool_call_id=message.tool_call_id,
                    id=message.id,
                )
            )
            compressed.append(call_id)
        else:
            result.append(message)
    return result, compressed


def split_for_summary(
    messages: Sequence[AnyMessage], *, keep_recent: int = 6
) -> tuple[list[AnyMessage], list[AnyMessage]]:
    """把消息切成"该进摘要的"和"原样保留的"两段。

    切点必须落在**配对边界**上：不能把 `AIMessage(tool_calls)` 和它的 `ToolMessage`
    切开。做法是从目标位置往回退，直到遇到一条不是工具结果、且前一条也没带工具调用的位置。
    """
    if len(messages) <= keep_recent:
        return [], list(messages)
    cut = len(messages) - keep_recent
    while cut > 0:
        if not isinstance(messages[cut], ToolMessage) and not getattr(
            messages[cut - 1], "tool_calls", None
        ):
            break
        cut -= 1
    return list(messages[:cut]), list(messages[cut:])


def summary_request(prefix: Sequence[AnyMessage]) -> list[AnyMessage]:
    """构造压缩用的请求：任务意图 + 对话原文（超出预算就取头尾）。"""
    lines: list[str] = []
    for message in prefix:
        kind = getattr(message, "type", "?")
        content = message.content if isinstance(message.content, str) else str(message.content)
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            names = ", ".join(str(call.get("name")) for call in tool_calls)
            content = f"{content}\n[调用工具] {names}"
        lines.append(f"[{kind}] {content}")
    body = "\n\n".join(lines)
    if len(body) > SUMMARY_INPUT_CHARS:
        half = SUMMARY_INPUT_CHARS // 2
        body = f"{body[:half]}\n\n...（中间省略）...\n\n{body[-half:]}"
    return [SystemMessage(content=SUMMARY_PROMPT), HumanMessage(content=body)]


def memory_message(summary: str) -> SystemMessage:
    """摘要产物的容器。用 system 角色：它是系统给的事实底稿，不是用户说的话。"""
    return SystemMessage(
        content=(
            '<memory note="更早的对话已压缩成这些事实；需要原文时用 recall(call_id=…)">\n'
            f"{summary.strip()}\n</memory>"
        )
    )
