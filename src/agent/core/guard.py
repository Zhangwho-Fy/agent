"""外部内容的来源标记与清洗。

两条原则（见 `docs/design.md` 6.2 与 6.3）：

1. **标记**：工具结果来自工作区文件、命令输出、第三方仓库，统统是**数据不是指令**。
   统一包一层 `<untrusted source=…>`，给模型一个稳定的锚点，也方便测试断言。
2. **清洗只做确定性的事**：零宽字符与 bidi 控制符直接剥掉——它们混进代码本来就会
   把 diff 和编译搞坏，属于顺手的收益。可疑短语**只记录不拦截**：措辞变体太容易绕过，
   误伤真实代码注释的代价却是实打实的。

这里**不是**安全边界。真正的边界在工具层：危险命令 DENY、写操作审批、路径 realpath
后才判越界。包装只是让模型知道"这段话该当资料看"。
"""

from __future__ import annotations

import re
from typing import Any
from xml.sax.saxutils import quoteattr

#: 零宽字符、bidi 控制符、BOM。混进代码里会造成"看起来一样但编译不过"。
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")

#: 提前关掉容器的序列——内容里出现它，就能伪装成"容器外面"的文字。
_CLOSING_TAG = re.compile(r"</\s*untrusted", re.IGNORECASE)

#: 可疑模式的规则表。
#:
#: **故意做得少而精**：宁可漏（漏了还有审批兜底），不要误报（误报会污染正常代码注释）。
#: 命中只用于记录（事件里的标记 + 日志），**从不据此拦截内容**。
SUSPICIOUS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "覆盖指令",
        re.compile(r"(忽略|无视|忘记)(以上|上述|前面|之前|所有)*(的)*(指令|要求|规则|设定)"),
    ),
    (
        "override-instructions",
        re.compile(r"ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+\w+", re.I),
    ),
    (
        "伪造角色标记",
        re.compile(r"<\|[a-z_]+\|>|\[/?INST\]", re.I),
    ),
)


def strip_invisible(text: str) -> str:
    """剥掉零宽字符与 bidi 控制符。信息量不变，可读性和 diff 都变好。"""
    return _INVISIBLE.sub("", text)


def scan_suspicious(text: str) -> list[str]:
    """返回命中的规则名列表。**调用方不要据此拦截内容**，只记录。"""
    return [name for name, pattern in SUSPICIOUS_PATTERNS if pattern.search(text)]


def _neutralize_boundary(text: str) -> str:
    """只中和"提前关掉容器"的那一个序列。

    为什么不做全量 XML 转义：内容里全是代码，把 `<` 统统变成 `&lt;` 会让代码没法读。
    容器本身只是提示而不是安全边界（真正的边界在沙箱与审批），所以只需挡住这一种
    最容易伪装的写法。
    """
    return _CLOSING_TAG.sub("&lt;/untrusted", text)


def wrap_untrusted(content: str, *, source: str, **attrs: Any) -> str:
    """把外部内容包成 `<untrusted>`。

    属性值用 `quoteattr` 转义（属性是我们生成的，必须挡住内容里的引号）；
    正文只做零宽字符剥离与容器边界中和，其余原样保留，代码才读得下去。
    """
    attributes: dict[str, str] = {"source": source}
    for key, value in attrs.items():
        if value is not None and value != "":
            attributes[key] = str(value)
    head = " ".join(f"{key}={quoteattr(value)}" for key, value in attributes.items())
    body = _neutralize_boundary(strip_invisible(content))
    return f"<untrusted {head}>\n{body}\n</untrusted>"
