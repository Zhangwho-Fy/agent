# agent

一个本地运行的代码库助手 Agent：在指定仓库里读代码、跑命令、改文件，闭环完成真实任务。

设计目标不是"能调模型"，而是把模型外面那圈工程做扎实：**可持久化、可恢复、可重放、可评测**。

## 现在能做什么

- **九个工具，真读真跑**：`fs_read` / `fs_list` / `shell_exec` / `recall` / `skill_load` / `skill_create` / `search_code` / `memory_write` / `memory_search`——模型自己决定调哪个、调几次，结果回填后继续推理
- **分级管控**：只读命令自动执行；有副作用的命令挂起等你点头；危险命令（`rm -rf`、`sudo`、`git push`…）直接拒绝，并把原因回填给模型让它换做法
- **会话持久化**：事件、消息、工具调用、审批、token 用量全部落 SQLite
- **崩溃恢复**：跑到一半 `kill -9`，重启后把没跑完的 turn 标成 `interrupted`，同一会话可以接着聊
- **可重放**：`agent replay` 不调模型，把一次会话的事件流按 seq 原序还原
- **可评测**：golden 集走回放执行，**不联网、不要密钥、几秒出通过率**
- **服务化**：`agent serve` 提供 HTTP + SSE（任务在服务端跑，客户端只是订阅者），断线重连按 `Last-Event-ID` 补齐；`agent chat` 是配套的交互式 CLI
- **交互式界面**：`agent chat` 是全屏常驻界面（状态栏 + 输入行钉在最底，日志在内滚动）；思考过程暗灰可折叠、代码高亮、写操作当场问你、`agent resume` 挑历史会话并先把历史画出来
- **代码检索（RAG）**：已接成 `search_code` 工具——agent 自己会搜代码，先拿到"文件 + 行号 + 片段"（L0），再决定要不要读全文。切块 + FTS5 词法 + 向量语义 + RRF 混合排序，10 条查询 **recall@5 = 100%、MRR = 0.787**（离线跑，不联网；这个数只当回归基线）
- **上下文工程**：系统提示词分静态核心 + 会话环境两层；外部内容统一打来源标记（`<untrusted>`），只有技能正文是操作说明（`<skill>`）；技能目录渐进式披露（目录进系统提示、正文按需加载）；模型每轮看到一张"还剩多少预算"的状态块；上下文到 60% 把旧工具结果换成指针（原文可用 `recall` 取回），到 80% 才做结构化摘要
- **跨会话记忆**：值得长期保留的东西（用户偏好、项目事实、决策与理由、未完成事项、带因果的经历）存进独立的 `memory.db`；来源分四种信任等级，工作区来的内容一律按数据（`<untrusted>`）处理；同 `key` 的新记忆让旧的失效但不删除，随时能回原始出处核查；每轮按重要性挑几条拼在请求尾部，模型也能按需 `memory_search`（查看 / 编辑的 CLI 还没做）
- **CI**：每次 push 自动跑 ruff + pytest，全程不注入密钥

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
# 接着刚才那个会话继续问（checkpointer 会把历史带回来）
uv run agent run -s <会话 id> "那个文件的异常处理是怎么做的？"

# 交互式：一个终端跑服务端，另一个终端聊天
uv run agent serve                 # 没配 AGENT_AUTH_TOKEN 时会随机生成一个并打印
uv run agent chat                  # 多轮对话（全屏界面），写操作停下来问你：回车允许 / n 拒绝
uv run agent resume                # 列出历史会话，挑一个继续（历史先画出来）
```

## 命令

| 命令 | 作用 |
| --- | --- |
| `agent run "任务"` | 跑一轮任务，终端流式显示 |
| `agent run -s <会话> "任务"` | 接着已有会话跑 |
| `agent run -y "任务"` | 自动批准所有写操作（无人值守） |
| `agent run -w <目录> "任务"` | 换个工作区 |
| `agent sessions [--json]` | 列出最近的会话（含每个会话最后一句用户消息），`--json` 给脚本用 |
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

**日志**：一行一条 JSON 写 stderr；要留档就设 `AGENT_LOG_PATH`（按 2MB × 3 份轮转）。
`agent serve` 没配时默认写到会话库旁边的 `agent.log`。日志只装异常与降级路径——
正常路径的事实都在事件里（`agent replay` 看得到）。

## 状态

| 阶段 | 状态 |
| --- | --- |
| 0 设计 / 1 最小闭环 | ✅ |
| 2 可靠性层（持久化、幂等、崩溃恢复、审批、checkpointer） | ✅ 验收已跑通 |
| 3 工程化（录放、golden 集、CI） | ✅ |
| 4 服务化（HTTP + SSE、断线续传、幂等、交互式 CLI） | ✅ |
| 5 检索（切块、混合检索、recall@5 评测） | ✅ |
| 上下文工程（提示词分层、技能、状态块、压缩） | ✅ |
| 检索接入 R0（`search_code` 工具、索引懒建 / mtime 增量） | ✅ |
| 用户记忆（独立 `memory.db`、写入 / 检索 / 每轮摘要） | ✅ P0 + P1 机制；P2 整理与三层评测待做 |
| 6 扩展（MCP 适配器、可选 C++ 工具） | 暂缓：没有真实需求前不引入（MCP 的价值是接第三方生态，代价是沙箱语义变弱） |

编排用 LangGraph，模型接入用 LangChain；自研的是框架**不覆盖**的那层语义：对外事件契约、幂等与恢复、沙箱与审批策略、录放与评测。

## 文档

设计文档（`docs/design.md`，含事件与 HTTP 契约、上下文工程、用户记忆与 RAG，以及
D1 ~ D60 决策记录）、图文版（`docs/context-engineering.html`，**两页**：
① 上下文工程 ② 用户记忆与 RAG）和**交接文档**（`AGENTS.md`：当前状态、换机器继续的步骤、
代码地图、踩过的坑）都是**本地文档**，已从版本控制移除——克隆这个仓库不会有它们，
仓库里只有代码、测试和这份 README。
