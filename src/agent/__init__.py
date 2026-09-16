"""本地代码库助手 Agent。

分层（依赖方向单向，内层不知道外层的存在）：

    client / server  →  core  →  {providers, tools, retrieval, store}

core 只依赖 Protocol 抽象，具体实现在启动时装配，这样测试里可以用回放
provider 顶替真实模型，CI 不需要联网也不需要密钥。
"""

__version__ = "0.1.0"
