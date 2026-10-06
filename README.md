# agent

一个本地运行的代码库助手 Agent：在指定仓库里读代码、跑命令、改文件，闭环完成真实任务。

设计目标不是"能调模型"，而是把模型之外的工程做扎实：**可持久化、可恢复、可重放、可评测**。

编排用 LangGraph，模型接入用 LangChain；自研部分是框架不覆盖的那层语义：对外事件契约、
幂等与恢复、执行边界与审批、录放与评测。

## 功能

| 方向 | 说明 |
| --- | --- |
| 工具调用 | 文件读写、命令执行、代码检索、技能加载、跨会话记忆等一组工具，由模型自主决定调用对象与轮次 |
| 权限分级 | 只读操作自动执行，有副作用的操作需人工确认，危险操作直接拒绝并把原因返回模型 |
| 执行边界 | 路径解析后校验工作区边界；命令超时终止整个进程组；子进程只透传白名单环境变量；工具输出截断后再回填 |
| 会话持久化 | 事件、消息、工具调用、审批记录与 token 用量全部落 SQLite |
| 崩溃恢复 | 轮次执行中进程被中断（崩溃、被 kill、Ctrl-C）时，该轮次会在下次启动时标记为 `interrupted`；会话历史保留，可以继续 |
| 历史回放 | `agent replay` 只读地打印一次历史会话的事件流：不调用模型、不执行工具，用于复盘与排查 |
| 可评测 | 分四档按代价递进：L1 回放断言、L2 成本与效率、L3 语义判分、L4 真实任务成功率；L1 / L2 已实现 |
| 服务化 | `agent serve` 提供 HTTP + SSE；任务在服务端执行，客户端断线后按 `Last-Event-ID` 补齐 |
| 交互式 CLI | `agent chat` 提供多轮对话界面，支持流式输出与过程信息折叠；`agent resume` 选择历史会话继续 |
| 代码检索 | `search_code` 工具：切块 + FTS5 词法 + 向量语义 + RRF 融合，只把命中的片段送入上下文 |
| 上下文工程 | 提示词分层；外部内容统一来源标记；技能目录渐进式披露；运行时状态块；压缩阶梯（指针化 → 摘要）配 `recall` 找回 |
| 跨会话记忆 | 偏好、项目事实、决策与未完成事项存入独立 `memory.db`；按来源分级信任；同 key 的更新走 supersede 链；每轮按重要性注入摘要 |

**检索评测**：固定 10 条查询，正确文件进入前 5 条的比例（recall@5）为 100%，平均排名 MRR = 0.787。
默认向量后端是离线的确定性实现（不是语义模型），这组数字只用于回归对比。

## 快速开始

需要 `uv`（含 Python 3.12 下载能力）、DeepSeek API key、网络。

```bash
# 没有 uv 就先装：curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync                     # 建 .venv；国内网络慢时加 UV_HTTP_TIMEOUT=300
                            # 系统只有 3.13/3.14 时用 uv sync --python 3.12
cp .env.example .env        # 填入 AGENT_API_KEY；服务端/客户端共用的令牌写 AGENT_AUTH_TOKEN
uv run agent doctor         # 体检：Python、依赖、配置、密钥

# 跑一个真实任务
uv run agent run "用一句话说明 src/agent/graph/builder.py 里的图是怎么流转的"
# 继续该会话（checkpointer 会带上前文）
uv run agent run -s <会话 id> "那个文件的异常处理是怎么做的？"

# 交互式：一个终端跑服务端，另一个终端聊天
uv run agent serve                 # 没配 AGENT_AUTH_TOKEN 时会随机生成一个并打印
uv run agent chat                  # 多轮对话；有副作用的操作会等待人工确认
uv run agent resume                # 列出历史会话，选择一个继续
```

## 命令

| 命令 | 作用 |
| --- | --- |
| `agent run "任务"` | 跑一轮任务，终端流式显示 |
| `agent run -s <会话> "任务"` | 接着已有会话跑 |
| `agent run -y "任务"` | 自动批准所有写操作（无人值守） |
| `agent run -w <目录> "任务"` | 换个工作区 |
| `agent sessions [--json]` | 列出最近的会话（含每个会话最后一句用户消息），`--json` 供脚本消费 |
| `agent replay <会话> [--raw]` | 重放事件流，**不调模型** |
| `agent serve` | 启动 HTTP + SSE 服务端 |
| `agent chat [-s 会话] [--token ...]` | 交互式多轮对话（走服务端） |
| `agent resume [--last]` | 列出历史会话并继续其中一个；列表中可删除会话 |
| `agent doctor` / `agent config` / `agent version` | 环境与配置自检（不依赖 langgraph） |

## 开发

```bash
uv run pytest -q                       # 全部测试：不需要联网、不需要密钥
uv run pytest tests/eval -s            # golden 集，打印通过率
uv run python scripts/scan_sessions.py # 扫会话库找异常模式（只读）
uv run ruff check . && uv run ruff format --check .
```

**录放**：把一次真实模型调用（请求与响应）录成 JSONL；之后用回放模式重跑同一段，
不需要网络和密钥。测试依赖这种方式保持离线。

```bash
# 录：真实调用，同时把每次 (请求, 响应) 追加进文件
AGENT_TRACE_PATH=evals/recordings/my-case.jsonl uv run agent run "任务"
# 放：完全离线
AGENT_PROVIDER=replay AGENT_TRACE_PATH=evals/recordings/my-case.jsonl uv run agent run "任务"
```

**日志**：一行一条 JSON 写 stderr；要留档就设 `AGENT_LOG_PATH`（按 2MB × 3 份轮转）。
`agent serve` 没配时默认写到会话库旁边的 `agent.log`。日志只记录异常与降级路径；
正常路径的事实都在事件里（`agent replay` 看得到）。
