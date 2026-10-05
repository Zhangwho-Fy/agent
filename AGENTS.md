# 交接文档（人和 AI 助手都读这份）

> 文件名用 `AGENTS.md` 而不是 `AGENT.md`：这是约定俗成的名字，Codex 之类的编码助手
> 进仓库时会自动读它，所以换一台机器、换一个助手都能无缝接上。

## 0. 一句话

一个本地运行的**代码库助手 Agent**：在你指定的仓库里读代码、跑命令、改文件。
编排交给 **LangGraph**（行业标准件），自研的重点是框架**不覆盖**的部分：
对外事件契约、幂等与恢复、沙箱与审批、录放与评测。

## 1. 当前状态

| 项 | 状态 |
| --- | --- |
| 阶段 0（设计） | ✅ 完成 |
| 阶段 1（最小闭环） | ✅ 代码完成，**4/4 条验收已验证** |
| 阶段 2（可靠性层） | ✅ 完成，**验收已跑通**（见"已实测"③④⑤） |
| 阶段 3（工程化） | ✅ 完成：录放 + golden 集 + CI |
| 阶段 4（服务化） | ✅ 代码完成：`agent serve`（HTTP + SSE）、`agent chat`（交互式）、断线续传、幂等接线。socket 级 `curl` 验收待在有网络的机器上跑（见 7.2） |
| 阶段 5（检索） | ✅ 完成：切块 + FTS5 词法 + 向量语义 + RRF 混合 + 指标评测 |
| 交互式界面 | ✅ `agent chat` 的常驻底栏（阶段 4 的延伸）：**全屏** + 配色 + 代码高亮 + 过程信息折叠（含模型自带的思考）+ 回滚 + 界面内审批。它是 chat 的**唯一**界面，没有开关；行式输出只在 `agent run` 那条路上（见 7.5） |
| 已实测 | ① 流式逐 token：691 字符的回答产生 399 个 `text.delta`，跨度 0.913s，与直连 SDK 的 0.901s 一致；② 工具报错自愈：故意读错路径 → 模型自己 `fs_list` → `find` 定位 → 读到正确文件；③ **崩溃恢复**：跑到一半 `kill -9`，重启后打出"恢复：1 个没跑完的 turn 已标记为 interrupted"，同会话续跑并答对了上文相关问题；④ **重放**：`agent replay <会话>` 不调模型，把 223 条事件按 seq 原序还原；⑤ **审批**：没通道 / 被拒 / 超时三种情况都不放行；⑥ **golden 集**：3/3 通过，**3 秒、不联网、不要密钥**；⑦ CI 上全部测试在空密钥下同样全绿；⑧ **服务端**：14 个测试覆盖阶段 4 三条验收；⑨ **检索**：10 条查询 **recall@5 = 90%、MRR = 0.758**（离线兜底嵌入，1.7 秒跑完，不联网）；⑩ **常驻底栏**：用户实测全屏布局、配色、折叠、滚动、Ctrl-C 退出、界面内审批；⑪ **推理流**：探针实测该端点返回 `reasoning_content`（思考 106 字符 / 正文 3 字符），思考已能上屏 |
| 下一步 | 阶段 6：扩展（MCP 适配器；C++ 工具可选） |
| 代码量 | 源码约 5200 行，测试 155 个（全绿），ruff 干净 |
| 语言 | **纯 Python，没有任何 C++ 代码**（C++ 是阶段 6 的可选加分项，见第 5 节第 6 条） |

已跑通的实际效果：

```bash
$ agent run "用一句话说明 src/agent/graph/builder.py 里的图是怎么流转的"
→ fs_read {"path": "src/agent/graph/builder.py"}
  ← ok
从 START 进入 agent 节点（调用模型），条件边 _route 检查最新消息……
```

## 2. 在另一台机器上继续

需要：`uv`（含 Python 3.12 下载能力）、DeepSeek API key、网络。

```bash
git clone git@github.com:Zhangwho-Fy/agent.git
cd agent
# 没有 uv 时先装（官方脚本，装到 ~/.local/bin，之后 source ~/.local/bin/env 或重开终端）：
#   curl -LsSf https://astral.sh/uv/install.sh | sh
cp .env.example .env          # 填入 AGENT_API_KEY（从原机器 ~/.codex/config.toml 里
                              # [model_providers.deepseek] 段复制 sk-... ，或去控制台新建）
UV_HTTP_TIMEOUT=300 uv sync   # 国内网络到 PyPI 慢，超时参数必须加；
                              # 系统 Python 不是 3.12 时用 uv sync --python 3.12
.venv/bin/agent doctor        # 体检：Python 版本、依赖、配置、密钥
.venv/bin/agent run "用一句话说明 src/agent/graph/builder.py 的作用"
```

`.env` 已在 `.gitignore` 里，**key 永远不要提交**。若 `uv sync` 反复卡在某个大 wheel：

```bash
rm -f uv.lock && uv sync --default-index https://mirrors.aliyun.com/pypi/simple/
```

## 3. 常用命令

| 命令 | 作用 |
| --- | --- |
| `.venv/bin/agent doctor` | 环境与配置体检（不依赖 langgraph，缺依赖也能跑） |
| `.venv/bin/agent config` | 打印解析后的配置（密钥只显示长度） |
| `.venv/bin/agent run "任务"` | 跑一次真实任务，终端流式显示 |
| `.venv/bin/agent run -s <会话> "任务"` | 接着已有会话跑（checkpointer 会带上历史） |
| `.venv/bin/agent sessions` | 列出最近的会话 |
| `.venv/bin/agent replay <会话> [--raw]` | 按原序重放事件流，**不调模型**（`--raw` 看逐条分片） |
| `.venv/bin/agent serve` | 启动 HTTP + SSE 服务端（会打印访问令牌） |
| `.venv/bin/agent chat [-s 会话]` | 交互式多轮对话（**全屏界面**），走服务端 |
| `.venv/bin/agent resume [--last]` | 列出历史会话、挑一个继续（`--last` 直接接最近一个）；等价于 `agent chat --resume` |
| `AGENT_TRACE_PATH=x.jsonl agent run ...` | 边跑边把模型调用录成夹具（阶段 3） |
| `AGENT_PROVIDER=replay AGENT_TRACE_PATH=x.jsonl agent run ...` | 用录制文件离线跑，不联网 |
| `.venv/bin/python -m pytest tests/eval -s` | 跑 golden 集并打印通过率（不联网、不要密钥） |
| `.venv/bin/python -m pytest -q` | 全量测试（**不需要联网、不需要 key**） |
| `.venv/bin/ruff check . && .venv/bin/ruff format --check .` | 静态检查，提交前必须过 |
| `.venv/bin/python scripts/smoke_api.py` | 模型连通性烟测（真实调用，手动跑） |

## 4. 代码地图

```
src/agent/
├── config.py           pydantic-settings 读 .env + 环境变量，密钥打码
├── logging.py          结构化日志（一行一 JSON），压掉 httpx 之类噪音
├── graph/              编排（LangGraph）
│   ├── state.py        图状态；messages 用 add_messages 归约器
│   ├── nodes.py        模型节点 + 审批节点 + 工具节点
│   ├── builder.py      组装 START → agent ⇄ approve → tools → agent
│   ├── checkpointer.py 检查点存储（同步 SqliteSaver + 异步薄适配）
│   └── bridge.py       把框架的流翻译成我们的事件（关键适配层，含审批等待）
├── core/
│   ├── events.py       事件模型 + SSE 编码（seq 作 SSE id，供断线续传）
│   ├── bus.py          会话内事件分发（有界队列，慢了丢最老）
│   ├── reliability.py  事件编号、幂等键（阶段 2 换成落库实现）
│   ├── prompt.py       系统提示
│   ├── messages.py     消息/工具调用模型
│   ├── tool_spec.py    工具契约 + 工具名校验（协议只允许 [A-Za-z0-9_-]）
│   └── errors.py       异常层次
├── models/
│   ├── factory.py      构建模型；按 provider 决定「真实 / 边跑边录 / 回放」
│   └── trace.py        录放：JSONL 录制器与回放器（阶段 3）
├── eval.py             golden 评测集：加载用例、跑回放、判定、算通过率
├── retrieval/          检索（阶段 5）
│   ├── chunker.py      按行开窗切块，带重叠；遍历源码文件时先剪掉依赖目录
│   ├── embeddings.py   嵌入接口 + 离线确定性兜底 + fastembed 后端
│   ├── index.py        SQLite 索引：chunks 表 + FTS5(trigram) + mtime 增量
│   ├── search.py       词法 / 语义 / RRF 混合
│   └── metrics.py      recall@k 与 MRR
├── server/             服务端（阶段 4）
│   ├── app.py          FastAPI 路由：会话、消息、SSE 事件流、审批
│   └── runtime.py      每会话一套：模型、图、事件发射器、审批等待区
├── tools/
│   ├── base.py         ToolContext/ToolResult、工作区路径边界、输出截断
│   ├── fs.py           fs_read / fs_list
│   ├── shell.py        shell_exec（超时杀进程组、环境变量白名单）
│   ├── policy.py       三级分级：只读自动 / 写操作审批 / 危险拒绝
│   └── registry.py     工具注册与 schema 导出
├── store/              持久化：SQLite 里的事实源（阶段 2）
│   ├── schema.sql      六张表的建表语句（events 是事实源，messages 是投影）
│   ├── db.py           Database：连接、WAL、串行化访问
│   └── repo.py         各表读写：seq 分配、按 seq 补事件、崩溃恢复标记……
└── client/main.py      CLI：version / doctor / config / run / sessions / replay

evals/                  golden 用例（cases/）、夹具工作区（workspaces/）、录制文件（traces/）
.github/workflows/ci.yml  CI：uv sync + ruff + pytest，全程不注入密钥
```

## 5. 五条设计决定（别走回头路）

1. **编排用 LangGraph**，不自研状态机——编排是行业标准件，本项目差异化在语义层。
2. **对外契约自己写**：事件日志、seq 续传、幂等键、审计明细，框架没有对应抽象。
3. **checkpointer 与事件日志并存**：前者是图运行态（可丢），后者是对外事实源（可重放）。
4. **错误是给模型的信息**：工具失败不抛异常，编码成结果回填；只有基础设施故障才向上抛。
5. **没写进图的东西故意留在图外**：沙箱、审批策略、幂等——它们不是控制流，塞进状态会变成状态袋。
6. **当前不混编 C++**：差异化全在语义层（Python 侧），引 C++ 要付 CMake、工具链、CI、双路径测试的成本，收益只有"跨语言叙事"。真要用时只有一种形式——**独立进程 + CLI**（被 `shell_exec` 或注册表当普通工具调用），不是 pybind11 的 ABI 混编，且必须可降级。触发条件：有数据支撑的性能热点、或要复用已有 C++ 库。

## 6. 文档索引

| 文档 | 什么时候看 |
| --- | --- |
| [docs/design.md](docs/design.md) | 目标、非目标、里程碑、验收标准 |
| [docs/detailed-design.md](docs/detailed-design.md) | 选型、架构、目录、数据模型、接口、测试策略 |
| [docs/stage-1.md](docs/stage-1.md) | 阶段 1 总结；**"为什么 agent 需要图"**的完整回答 |
| [docs/knowledge.md](docs/knowledge.md) | 知识点与面试考点（面试前只读这一份） |
| [docs/protocol.md](docs/protocol.md) | 事件与 HTTP 契约（阶段 4 冻结，暂未创建） |

> 注意：`docs/` 是**本地文档**，已从版本控制移除（`.gitignore` 里忽略），
> 只存在于开发机上。上面的链接在本机可用，克隆到别处则没有这些文件。

## 7. 阶段 2 做了什么（已完成）

| # | 事项 | 落点 |
| --- | --- | --- |
| 1 | SQLite 六张表 + 读写函数 | `store/schema.sql`、`store/db.py`、`store/repo.py` |
| 2 | 事件**先落库再推送** | `core/reliability.py` 的 `sink`，接口仍是 `emit()`（变成 async） |
| 3 | LangGraph checkpointer | `graph/checkpointer.py`（`SqliteSaver` + 异步薄适配；为什么不用 `AsyncSqliteSaver` 见文件说明） |
| 4 | 审批流 | `graph/nodes.py` 的 `approve` 节点 `interrupt()`；`graph/bridge.py` 负责等待与 `Command(resume=...)`；超时/无通道按拒绝 |
| 5 | 崩溃恢复 | `repo.interrupt_running_turns()`，CLI 启动时调用 |
| 6 | 验收 | `agent run` / `kill -9` / `--session` 续跑 / `agent replay` 四步已实测 |

### 7.1 阶段 3 做了什么（已完成）

| # | 事项 | 落点 |
| --- | --- | --- |
| 1 | 录放 | `models/trace.py`（`RecordingChatModel` / `ReplayChatModel` / JSONL 读写），`models/factory.py` 按 provider 分流 |
| 2 | golden 集 | `eval.py`（加载用例 + 判定 + 通过率）、`evals/cases/*.json`、`evals/workspaces/`、`evals/recordings/*.jsonl` |
| 3 | CI | `.github/workflows/ci.yml`：`uv sync --frozen` → ruff → pytest，**不注入任何密钥** |
| 4 | 验收 | golden 3/3；110 个测试在空密钥下全绿 |

录放的三种用法：

```bash
# 录：真实调用，同时把每次 (请求, 响应) 追加到 JSONL
AGENT_TRACE_PATH=evals/recordings/xxx.jsonl .venv/bin/agent run "任务"
# 放：完全离线，不联网不要密钥
AGENT_PROVIDER=replay AGENT_TRACE_PATH=evals/recordings/xxx.jsonl .venv/bin/agent run "任务"
# 评测：跑 golden 集并打印通过率
.venv/bin/python -m pytest tests/eval -s
```

### 7.2 阶段 4 做了什么（已完成）

| # | 事项 | 落点 |
| --- | --- | --- |
| 1 | HTTP + SSE 服务端 | `server/app.py`（`create_app`）、`server/runtime.py`（每会话一套模型/图/审批等待区） |
| 2 | 交互式 CLI | `agent chat`：一次启动多轮对话，走服务端；`agent serve` 启动服务端 |
| 3 | 断线续传 | SSE 的 `Last-Event-ID` / `?after_seq=` → `repo.list_events(after_seq=...)`；**先订阅再补历史**，靠 seq 去重 |
| 4 | 幂等 | `idempotency` 表接线：同一个键第二次直接返回原 `turn_id`，不重复执行工具 |
| 5 | 验收 | `tests/unit/test_server.py`（14 个）：建会话→发消息→收事件→审批→结果；按 seq 补齐不丢不重；同键不重复执行 |

**一个环境限制**：httpx 的 `ASGITransport` 会把响应**缓冲到结束**才交给客户端，
所以无限 SSE 读不出实时性，测试改成"发消息 + 轮询库里的事件"，实时流那段直接
驱动 `event_stream()` 生成器。**socket 级的 `curl` 验收要在有网络的机器上跑**：

```bash
agent serve &                       # 记下打印出来的令牌
curl -s localhost:8765/health
SID=$(curl -s -X POST localhost:8765/sessions -H "Authorization: Bearer $TOKEN" \
      -H 'Content-Type: application/json' -d '{}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["session_id"])')
curl -s -X POST localhost:8765/sessions/$SID/messages -H "Authorization: Bearer $TOKEN" \
      -H 'Content-Type: application/json' -d '{"content":"用一句话说明这个项目"}'
curl -N localhost:8765/sessions/$SID/events -H "Authorization: Bearer $TOKEN"   # 实时流
curl -s "localhost:8765/sessions/$SID/events?follow=false" -H "Authorization: Bearer $TOKEN"  # 只回放
```

### 7.3 阶段 5 做了什么（已完成）

| # | 事项 | 落点 |
| --- | --- | --- |
| 1 | 切块 | `retrieval/chunker.py`：按行开窗 + 重叠，遍历时先剪掉 `.venv` 等目录 |
| 2 | 索引 | `retrieval/index.py`：`chunks` 表 + `chunks_fts`（FTS5 **trigram**，中文注释才检索得到）+ 按 mtime 增量重建 |
| 3 | 混合检索 | `retrieval/search.py`：词法（FTS5/BM25）+ 语义（余弦）+ **RRF 融合**（只看名次，不用调权重） |
| 4 | 指标 | `retrieval/metrics.py`：recall@k、MRR；`evals/retrieval/cases.json` 10 条查询 |
| 5 | 验收 | **recall@5 = 90%、MRR = 0.758**，1.7 秒跑完、不联网不要密钥（`pytest tests/eval -s`） |

**嵌入模型这件事**：DeepSeek 没有 embeddings 接口，而沙箱里下不了权重，
所以默认后端是 `HashingEmbedder`——**字面级匹配的确定性兜底，不是语义模型**，
上面那两个数字只能用来做回归对比（改切块、改融合参数后看涨跌），不代表真模型效果。
换真模型只要一步（首次运行会下模型权重）：

```bash
uv pip install fastembed
AGENT_EMBED_BACKEND=fastembed .venv/bin/python -m pytest tests/eval -s
```

（`config.py` 里还有 `embed_backend` / `embed_model` 两个开关，默认 `offline`。）

### 7.4 下一步：阶段 6

1. MCP 客户端适配器：把现成的 MCP server 当工具源接进注册表
2. （可选）C++ 工具：只在有数据支撑的性能热点处引入，且必须可降级
3. 顺手可做的小账：把检索接成 `search_code` 工具（注册表加一项即可），
   让 agent 自己会搜代码，而不是只会 `fs_list` + `fs_read`

1. **先定嵌入模型**：DeepSeek 没有 embeddings 接口，只有两条路——本地 `bge-m3`
   （免费离线，要下几百 MB 权重）或别家 API（通义 / 硅基流动，OpenAI 兼容）
2. 切块 + 索引构建与增量更新（用 LangChain 的 splitter，但检索指标自己写）
3. 混合检索（词法层可直接用 SQLite FTS5，语义层用向量）
4. golden 集扩一档：检索类用例 + recall@5 / MRR 的实测对比写进 README

## 8. 已知坑（现象 → 原因 → 做法）

| 现象 | 原因 | 做法 |
| --- | --- | --- |
| 首次真实调用 400：`Invalid 'tools[0].function.name'` | 协议要求函数名只匹配 `[A-Za-z0-9_-]`，用了 `fs.read` | 工具名用下划线；校验器前移到构造期 + 测试钉住 |
| CLI 输出里混进 JSON 行 | httpx 在 info 级打印每个请求 | 噪音 logger 压到 warning |
| `uv sync` 卡在大 wheel | 国内到 files.pythonhosted.org 慢 | `UV_HTTP_TIMEOUT=300`，或换镜像并删 `uv.lock` |
| 测试结果受本机影响 | 读到了真实 `.env` / 环境变量 | `Settings(_env_file=None)` + autouse fixture 清 `AGENT_*` |
| 路径检查被绕过 | 只做字符串前缀判断，没解析软链接 | 一律 `realpath` 后再判边界 |
| 命令超时后仍有残留进程 | 只杀了父进程 | `start_new_session=True` + `os.killpg` 杀整组 |
| 循环上限配置不生效 | `AGENT_MAX_TOOL_ROUNDS` 定义了但没人读，`bridge.py` 写死 `recursion_limit=40` | 上限改成在模型节点按 `tool_rounds` 判断（到顶就不再调模型、直接收尾一句），`recursion_limit` 只当护栏，用 `recursion_limit_for()` 按"一轮 2 步"换算（已修，`tests/unit/test_graph.py` 钉住） |
| 图跑到一半就没声音了（不像报错，也不占 CPU） | 框架会把**同步**可调用对象丢进线程池执行；受限容器里线程池的任务交接不可用，于是永久等待 | 条件边路由函数写成 `async`（图里其余部分本来就全是 async）。已验证：改完在沙箱内 8 秒跑完全量测试 |
| 假模型/回放模型返回的消息"没被追加" | `add_messages` 归约器按 id 去重：**同一个消息对象重复返回等于没追加**（id 都是 `None` 时它按"是否已在列表里"判断），于是最后一条停在 ToolMessage，条件边直接收工 | 每次返回**新的消息对象**（`model_copy(deep=True)` 或每次新建）。阶段 3 写回放模型时必须注意 |
| 整个进程静默卡死，不报错也不占 CPU（`asyncio.to_thread` 相关） | 受限容器里，把阻塞调用丢进线程池后，工作线程**反向唤醒事件循环**这一步会失败。判据：`to_thread` 里跑 `time.sleep` / 写文件 / 任何 sqlite 语句都挂住，换成手写 `threading.Thread` 就正常——差别只在要不要唤醒事件循环 | 别用 `to_thread` 包装阻塞调用。数据库这类"单条语句微秒级"的操作直接在事件循环里同步执行（`store/db.py` 就是这么做并写明了理由）；真要异步 IO 就找原生异步驱动 |
| 一接 checkpointer 就卡死 | 官方 `AsyncSqliteSaver` 基于 `aiosqlite`，而 aiosqlite 用后台线程 + 事件循环回调，撞上的是同一条限制 | 用同步 `SqliteSaver`（它的逻辑本来就在当前线程），只补一层把异步方法接到同步实现的薄适配：`graph/checkpointer.py` |
| 审批通过后工具被执行了两遍 / 事件重复推送 | `interrupt()` 挂起的节点，**恢复时会从头重跑**。挂起点和副作用放在同一个节点里，重跑就会重复执行、重复发事件 | 把"会挂起"的部分拆成独立的**纯计算**节点（本项目是 `approve`）：它只做分级判断，一个事件都不发；执行留在永远不会挂起的 `tools` 节点。`tests/unit/test_approval.py` 钉住 |
| 录一遍再放一遍，结果对不上 | 评测夹具用了**会变的目录**：录制文件写在被 `fs_list` 的工作区里，第二次跑时文件大小变了，工具输出自然不同 | 夹具工作区要独立且稳定（`evals/workspaces/<id>/`），录制文件、临时文件一律放工作区**外面**。`tests/unit/test_trace.py` 的注释里记着这条 |
| 带鉴权的接口全部挂住，`/health` 却正常 | FastAPI 把**同步**依赖（`def`）丢进线程池执行，撞上"线程池任务交接失败"那条限制——同一个根因第三次踩（前两次：LangGraph 同步可调用对象、`asyncio.to_thread`） | 框架会把可调用对象丢线程池的地方，一律写成 `async def`：FastAPI 依赖、LangGraph 节点与条件边、路由函数。`server/app.py` 的 `require_token` 有注释 |
| 本地测试全绿，CI 一上来全挂：读不到评测夹具 | `.gitignore` 里的 `traces/`（本意是忽略运行产物）把 `evals/traces/` 一起吞了，文件只存在本机、没进仓库；而 **CI 机器上只有仓库里的东西** | 夹具目录改名成 `evals/recordings/` 避开这条规则，`.gitignore` 里也留了警告；另加 `test_fixtures_are_tracked_by_git`，用 `git ls-files` 在本地就把"没提交"揪出来 |
| 每次运行要等 30 秒以上，且前 30 秒屏幕上没有任何输出 | 代码和 `.venv` 都在 Windows 盘（`/mnt/g`，9p 挂载）。`agent run` 要读 3885 个 `.py` 文件，跨文件系统每次读都是往返。实测：`import openai` 从 `/mnt/g` 要 12.8s，从 Linux 侧只要 1.6s | `.venv` 移到 Linux 文件系统，原位置留软链接（`.gitignore` 里的规则写成 `.venv` 不带斜杠，否则软链接匹配不到）。`agent version` 从十几秒降到 1 秒 |
| 交互界面花屏：正文里出现 `?[2m` 这类字符、状态栏文字混进输出 | **rich 和 prompt_toolkit 抢同一块终端**。rich 的分片写入（`console.print(..., end="")`）被界面重绘打断，ANSI 序列没被终端解释，当成普通字符打了出来 | **同一时刻只能有一套东西写终端**。TUI 模式下 rich 完全不参与：输出走 `format_event_plain()` 进界面缓冲区，颜色由界面自己的 style 出 |
| 底栏被拽到正文中间、永远滚不到屏幕底部 | 往终端写**半行**（`text.delta` 直接就写）。prompt_toolkit 那套"擦掉 → 打印 → 画回来"要求每次写完光标停在行首，停在行中间它就把界面贴在那儿画 | 行内那版改不动，**换全屏**：日志全部留在界面缓冲区，`stdout` 一个字节都不写（见 7.5） |
| 状态栏在深色终端上是一条白条，又晃眼又单调 | `reverse` 是把前景/背景对调；深色终端下等于白底黑字 | 状态栏别用 `reverse`，给它**固定深色底 + 分段前景色**（写死 hex），深浅终端都成立 |
| 状态栏内容一变（"空闲"→"执行 fs_read"、token 进位），整条跟着左右抖 | 字段宽度是内容撑出来的，前面一长后面全平移 | 会变宽的字段（阶段徽标、各数字）**补到固定显示宽度**（中文按 2 列算），就是给状态栏排"制表位"：`_fit()` |
| 屏幕上"隔一阵清一次"，就在思考的时候；`AGENT_TUI_TRACE` 里 `erase` 却是 0 次 | 取窗口按"从尾部取 N 条**源日志**"算，而折叠会把几百行思考压成 **1 屏行**——窗口整个落进被折叠的那段，折完只剩一行，屏幕看着就像被清空 | 窗口按**折行后的屏行**取：不够一屏就往前翻倍扩（`_render_lines`）。另外回滚时**冻结内容**（`_frozen`），否则新输出会把视窗推着走 |
| 模型写的故事/诗整段发暗灰，看着像"思考过程" | 模型把整段正文放进了 ``` 围栏，而代码的兜底样式是 `ansibrightblack`；围栏本身又不显示，于是看着就是"正文变灰了" | 围栏里的内容用**亮色**（`TUI_STYLE["code"]`），真代码靠关键字/字符串分色、只有注释退到背景里。判据：查库里 `text.delta` 里有没有 ```（`reasoning.delta` 才是思考） |
| `Ctrl-C` 退出时抛 `Return value already set. Application.exit() failed.` | 按键绑定里已经 `app.exit()` 过一次，主循环收尾又调一次，prompt_toolkit 第二次会抛 | `ChatTUI.exit()` 先看 `app.future`：`None`（没跑起来）或 `done()`（已经退了）就直接返回——**退出动作要能重复调用** |
| 删掉一个方法后，服务端每一轮都"跑不完"，测试里 10 个 `TimeoutError`，而且整轮从 13 秒变成 3 分半 | 删 `pending_approvals` 时把它头上的 **`@property` 装饰器留下了**，于是装饰器落到了紧跟其后的 `run_turn` 上，`run_turn` 变成 property，调用即失败 | 删函数要连**装饰器一起删**；这类错误 ruff 查不出来，只有全量测试才照得出来（表现是"慢 + 超时"，不是"报错"）。教训：删完立刻跑全量，别只看单文件 |
| 删会话时报 `FOREIGN KEY constraint failed` | SQLite **是开着外键的**，删表顺序不对：先删 `turns` 时，`tool_calls` 里还有行引用它 | 子表在前、父表在后：`tool_calls → events → messages → idempotency → turns → sessions`（`repo._SESSION_TABLES` 的注释里写着理由）。单测里塞一条 `tool_calls` 把这个顺序钉住，否则没有工具调用的会话测不出问题 |
| 客户端报 `405 Method Not Allowed`（或新接口 404），代码里明明有这个路由 | **服务端还是改之前启的那个进程**。Python 服务端不热加载，`agent serve` 起一次就固定住了；测试里用的是 in-process app，所以全绿 | 改完服务端代码**重启 `agent serve`**。快速判定：`curl -s -o /dev/null -w '%{http_code}\n' -X DELETE -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/sessions/nope` —— 405 是旧进程，404 才是新进程（路由在、会话不在） |

### 7.5 `agent chat` 的常驻底栏（已完成）

目标是 Codex 那种手感：**底栏钉在窗口最后两行、状态栏常驻、执行期间也不消失**。

- **没有开关了**：`agent chat` 只有这一套界面（原先那套行式交互已删除）。
  `tui_problem()` 负责前置检查：**非终端**（重定向 / 管道 / CI）或没装
  prompt_toolkit 时打印一句人话并指路 `agent run "问题"`，不硬上控制字符。
- **形态**：**全屏**（走终端备用屏幕）。上面是日志区（界面内部滚动），最后两行是
  状态栏 + 输入行；底栏位置跟这一轮输出了多少行无关。代价：退出后本次对话不在
  终端 scrollback 里。
- **渲染只有一个写者**：日志存在界面自己的缓冲区（`_log` / `_pending`），由
  `_transcript_fragments()` 按当前窗口尺寸取"最后几行"画出来，**一个字节都不写
  stdout**。会话期间 `sys.stdout` / `sys.stderr` 被换成 `_TuiOutput`（谁写都进界面
  日志区，显示成暗灰 `·` 行），日志另外挂 `_TuiLogHandler` 也进界面——**终端只剩
  渲染器一个写者**。这样就不存在 `patch_stdout()` 那套"写一次 → 擦屏 → 重画"，
  也就没有"写半行把底栏拽走""屏幕被清一下"的问题；rich 完全不参与。
- **配色**：用户消息亮蓝、正文近白（`#e6e6e6`）、过程信息（工具调用 + 模型思考）暗灰、
  思考用斜体；状态栏固定深底 + 分段颜色（模型名蓝、会话号绿、上下文占用
  绿/黄/红、执行中黄、空闲绿点）。正文写死是给深色终端：浅色终端把
  `TUI_STYLE["assistant"]` 改成 `""` 就跟随主题；代码块（``` 里的内容）也是亮色，
  只有注释退到背景里——**模型经常把整段正文放进围栏**，发灰会被误当成过程信息。
- **代码高亮**：``` 围栏本身不占行，代码本体用 **pygments** 分词上色；pygments 目前
  只是传递依赖，缺了就整行暗灰降级（`_code_fragments`），绝不让高亮把界面弄挂。
- **思考过程**：模型自带的 `reasoning_content`（DeepSeek 放在
  `additional_kwargs` 里）经 bridge 翻成 **`reasoning.delta`** 事件，和正文各走各的。
  界面里思考与工具调用同属"过程"组，**默认折成一行**并露出最后一截思考当进度
  （`▸ 过程 6 行：思考…我们需要先算`），`Ctrl-O` 展开看全文。
- **折叠**：`Ctrl-O` 全局展开/收起整个"过程"组。
- **滚动**：`PgUp`/`PgDn` 翻页、`↑`/`↓` 逐行、`End` 回到底部；回滚时状态栏右侧
  显示 `↑ 浏览历史 N 行 · M 行新输出（End 回底）`。**往上一滚就冻结当前内容**
  （`_frozen`）：之后流式输出再多，视窗也不动，只在底栏计数——不冻的话"离底部
  多少行"这个锚点会被新内容推着走，看着就像屏幕自己在滚。滚到底或 `End` 解冻。
  （滚轮能翻日志是白捡的：终端把滚轮翻成 ↑/↓ 键，而我们正好把 ↑/↓ 绑成了逐行滚动。
  别开 `mouse_support`，那会截走文本选择。）
- **输入**：Enter 发送、**`Ctrl-J` 换行**（Enter 发 CR、Ctrl-J 发 LF，终端层面一定
  分得开；Codex 的 Shift+Enter 在多数终端里和 Enter 无法区分，实测直接发送）；
  输入区最多长到 6 行，再长自己在框里滚。历史用 `InMemoryHistory`，
  `Ctrl-R` 增量搜索是 prompt_toolkit 自带的。
- **恢复历史会话**：`agent resume`（或 `chat --resume`）——拉服务端会话列表、
  弹一个选择框（自己写的 `_SessionPicker`，比 radiolist 多一个删除）、选中就带着
  历史进界面。列表里 `Delete`/`d` **删会话**（先上膛再确认一次，Esc 取消上膛）；
  删除走 `DELETE /sessions/{id}`，服务端连带清掉事件/消息/turn/工具调用/幂等键
  和图 checkpoint，正在跑的任务拒删（409）。**进去先画历史**：
  客户端用 `stream_events(follow=False)` 把已有事件放一遍（`_restore_backlog`），
  再把 `last_seq` 推到最新接实时流；`turn.done` / `approval.required` 这类
  "当时的收尾动作"不在恢复时重演。会话标题取**第一句用户输入**
  （`repo.title_session_if_empty`，占位符 `chat` 也当空），否则列表里全是一模一样的
  `chat`，认不出哪次是哪次。想直接看文本用 `agent replay <会话>`。
- **审批**：`approval.required` 就地展开黄块（工具名、要跑的**具体命令**、原因、怎么答），
  状态栏变 `⚠ 等待确认`、输入行变 `允许? `；`y` 放行、回车/n 拒绝、`Ctrl-C` 拒掉这一单
  并退出。**默认拒绝**；用户在 agent 干活时先打好的那句会寄存，不会被当成审批答案吞掉。
- **已验证**：用户实测（全屏、配色、折叠、滚动、Ctrl-C 退出、审批、resume、删会话）；
  175 个测试全绿。
- **待做**：鼠标滚动/多行相关的小细节（滚轮走方向键这条路已经够用）。
- **改这里的注意事项**（都是踩出来的）：
  1. **别让 rich 和界面同时写终端**——模块级 `console` 在导入时就抓死了原始 stdout，
     `patch_stdout()` 换不掉它手里的文件对象。TUI 路径一律 `format_event_plain()`；
  2. **别往 stdout 写半行**——行内渲染那版就死在这。现在日志全部留在界面里；
  3. **退出动作要幂等**（`ChatTUI.exit()` 看 `app.future`），按键绑定和收尾都会调；
  4. 沙箱里**没有 tty**，改完自己验不了，必须让用户在真实终端跑一次；
  5. 非终端里开不出全屏界面——由 `tui_problem()` 前置拦住并指路 `agent run`，
     别让它吐一堆控制字符。

## 9. 协作约定

- **分工**：框架接线、可靠性层、store、server、测试、CLI、检索、文档 → 助手写；
  工具边界（`shell_exec` 的截断与白名单）、评测任务设计、每阶段 review → 自己写。
- **提交纪律**：写完先汇报，**得到"提交"授权再 commit**；commit 信息只写一行英文主题（conventional commits 风格），不写 body。
- **提交前门槛**：`pytest` 全绿 + `ruff check` 干净，两条都过才提交。
- **推送**：本地提交与推送分开；push 需要联网授权。
- **测试原则**：单元测试与回放测试不需要联网、不需要 key；真实调用只在手动冒烟与演示时做。
