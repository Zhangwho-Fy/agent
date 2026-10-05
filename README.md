# agent

一个本地运行的代码库助手 Agent：在指定仓库里读代码、跑命令、改文件，闭环完成真实任务。

设计目标不是"能调模型"，而是把模型外面那圈工程做扎实：**可持久化、可恢复、可重放、可评测**。

## 现在能做什么

- **真读真跑**：三个工具（`fs_read` / `fs_list` / `shell_exec`），模型自己决定调哪个、调几次，结果回填后继续推理
- **分级管控**：只读命令自动执行；有副作用的命令挂起等你点头；危险命令（`rm -rf`、`sudo`、`git push`…）直接拒绝，并把原因回填给模型让它换做法
- **会话持久化**：事件、消息、工具调用、审批、token 用量全部落 SQLite
- **崩溃恢复**：跑到一半 `kill -9`，重启后把没跑完的 turn 标成 `interrupted`，同一会话可以接着聊
- **可重放**：`agent replay` 不调模型，把一次会话的事件流按 seq 原序还原
- **可评测**：golden 集走回放执行，**不联网、不要密钥、几秒出通过率**
- **服务化**：`agent serve` 提供 HTTP + SSE（任务在服务端跑，客户端只是订阅者），断线重连按 `Last-Event-ID` 补齐；`agent chat` 是配套的交互式 CLI
- **代码检索**：切块 + FTS5 词法 + 向量语义 + RRF 混合排序，10 条查询 **recall@5 = 90%、MRR = 0.758**（离线跑，1.7 秒）
- **CI**：每次 push 自动跑 ruff + pytest，全程不注入密钥

## 快速开始

需要 `uv`（含 Python 3.12 下载能力）、DeepSeek API key、网络。

```bash
# 没有 uv 就先装：curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                     # 建 .venv；国内网络慢时加 UV_HTTP_TIMEOUT=300
                            # 系统只有 3.13/3.14 时用 uv sync --python 3.12
cp .env.example .env        # 填入 AGENT_API_KEY，该文件不进 git
uv run agent doctor         # 体检：Python、依赖、配置、密钥

# 跑一个真实任务
uv run agent run "用一句话说明 src/agent/graph/builder.py 里的图是怎么流转的"
# 接着刚才那个会话继续问（checkpointer 会把历史带回来）
uv run agent run -s <会话 id> "那个文件的异常处理是怎么做的？"

# 交互式：一个终端跑服务端，另一个终端聊天
uv run agent serve                 # 打印访问令牌
uv run agent chat --token <令牌>    # 多轮对话，写操作会停下来问你
```

## 命令

| 命令 | 作用 |
| --- | --- |
| `agent run "任务"` | 跑一轮任务，终端流式显示 |
| `agent run -s <会话> "任务"` | 接着已有会话跑 |
| `agent run -y "任务"` | 自动批准所有写操作（无人值守） |
| `agent run -w <目录> "任务"` | 换个工作区 |
| `agent sessions` | 列出最近的会话 |
| `agent replay <会话> [--raw]` | 重放事件流，**不调模型** |
| `agent serve` | 启动 HTTP + SSE 服务端 |
| `agent chat [-s 会话] [--token ...]` | 交互式多轮对话（全屏界面，走服务端） |
| `agent resume [--last]` | 列出历史会话、挑一个继续（带上下文，历史先画出来；列表里 `Delete`/`d` 删会话） |
| `agent doctor` / `agent config` / `agent version` | 环境与配置自检（不依赖 langgraph） |

## 开发

```bash
uv run pytest -q                       # 全部测试：不需要联网、不需要密钥
uv run pytest tests/eval -s            # golden 集，打印通过率
uv run ruff check . && uv run ruff format --check .
```

**录放**是这套测试的地基：把真实模型调用录成 JSONL，之后离线回放，测试因此既快又确定。

```bash
# 录：真实调用，同时把每次 (请求, 响应) 追加进文件
AGENT_TRACE_PATH=evals/recordings/my-case.jsonl uv run agent run "任务"
# 放：完全离线
AGENT_PROVIDER=replay AGENT_TRACE_PATH=evals/recordings/my-case.jsonl uv run agent run "任务"
```

## 状态

| 阶段 | 状态 |
| --- | --- |
| 0 设计 / 1 最小闭环 | ✅ |
| 2 可靠性层（持久化、幂等、崩溃恢复、审批、checkpointer） | ✅ 验收已跑通 |
| 3 工程化（录放、golden 集、CI） | ✅ |
| 4 服务化（HTTP + SSE、断线续传、幂等、交互式 CLI） | ✅ |
| 5 检索（切块、混合检索、recall@5 评测） | ✅ |
| 6 扩展（MCP 适配器、可选 C++ 工具） | 下一步 |

编排用 LangGraph，模型接入用 LangChain；自研的是框架**不覆盖**的那层语义：对外事件契约、幂等与恢复、沙箱与审批策略、录放与评测。详细进度与踩过的坑见 [AGENTS.md](AGENTS.md)。

## 文档

- [AGENTS.md](AGENTS.md) — **交接文档**：当前状态、换机器继续的步骤、代码地图、设计决定、已知坑

> 更细的设计文档（`docs/design.md`、`docs/detailed-design.md`、`docs/stage-1.md`、`docs/knowledge.md`）
> 是**本地文档**，已从版本控制移除，克隆这个仓库不会有它们。
