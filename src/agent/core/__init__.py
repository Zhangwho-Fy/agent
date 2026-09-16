"""内核层：纯逻辑，不碰网络、不碰文件系统、不 import 具体 provider。

依赖方向单向：client / server → graph → {core, models, tools, retrieval, store}。
"""
