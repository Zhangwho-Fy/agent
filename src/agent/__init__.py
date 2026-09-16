"""本地代码库助手 Agent。

分层（依赖方向单向，内层不知道外层的存在）：

    client / server  →  graph  →  {core, models, tools, retrieval, store}

编排由 LangGraph 承担（`graph/`），核心的自研部分在框架**不覆盖**的语义层：
事件日志与对外契约、幂等与恢复、沙箱与审批、录放与评测。
"""

__version__ = "0.1.0"
