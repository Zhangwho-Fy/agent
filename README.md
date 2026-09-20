# agent

一个本地运行的代码库助手 Agent：在指定仓库里读代码、跑命令、改文件，闭环完成真实任务；兼有纯聊天模式。

设计目标不是"能调模型"，而是把模型外面那圈工程做扎实：**可持久化、可恢复、可重放、可评测**。

## 文档

- [AGENTS.md](AGENTS.md) — **交接文档**：当前状态、换机器继续的步骤、代码地图、设计决定、已知坑
- [docs/design.md](docs/design.md) — 目标、非目标、里程碑、验收标准
- [docs/detailed-design.md](docs/detailed-design.md) — 选型、架构、目录结构、接口、数据模型、配置、测试策略
- [docs/stage-1.md](docs/stage-1.md) — 阶段 1 总结：做成了什么、一次 run 的完整流程、为什么 agent 需要"图"
- [docs/knowledge.md](docs/knowledge.md) — 知识点与面试考点（面试前只读这一份）
- [docs/protocol.md](docs/protocol.md) — 事件与 HTTP 协议契约（第 4 阶段冻结）

## 当前状态

阶段 0（设计与骨架）与阶段 1（最小闭环）已完成：`agent run` 能真实读代码、跑命令、流式回答。
阶段 2（可靠性层：SQLite 持久化、幂等、崩溃恢复、checkpointer）是下一步，进度见 [AGENTS.md](AGENTS.md) 第 1 节。

## 快速开始

```bash
# 没有 uv 就先装：curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                     # 建 .venv；国内网络慢时加 UV_HTTP_TIMEOUT=300
                            # 系统只有 3.13/3.14 时用 uv sync --python 3.12
cp .env.example .env        # 填入 AGENT_API_KEY，该文件不进 git
uv run agent doctor         # 体检：Python、依赖、配置、密钥
uv run agent run "用一句话说明 src/agent/graph/builder.py 里的图是怎么流转的"
```

当前只有单进程 CLI（`run` / `doctor` / `config` / `version`）。`agent serve`（HTTP + SSE 服务端）属于阶段 4，尚未实现。
