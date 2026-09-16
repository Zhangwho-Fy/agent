# Agent 详细设计

> 上位文档：[design.md](./design.md)（目标、里程碑、验收标准）。
> 本文负责技术细节：选型、架构、目录结构、接口、数据模型、配置、测试策略。

## 1. 技术选型

| 位置 | 选型 | 备选 | 为什么 |
| --- | --- | --- | --- |
| 语言 | Python 3.12 | 3.10 / 3.11 | `X \| None`、`StrEnum`、`TaskGroup`、`asyncio.timeout` 等现代语法；系统自带的 3.8 已停止维护 |
| 环境与依赖 | uv | pip + venv | 装依赖快一个量级，顺带管 Python 版本 |
| 模型接入 | `openai` SDK（OpenAI 兼容协议） | 各家自研 SDK | DeepSeek 兼容 OpenAI 协议，换供应商只改 `base_url` + 模型名 |
| 数据模型 / 配置 | pydantic v2 + pydantic-settings | dataclass | 校验、序列化、从环境变量读配置全都免费 |
| HTTP 服务 | FastAPI + uvicorn | 标准库 `http.server` | 流式响应（SSE）实现干净，类型标注即接口文档 |
| CLI | Typer | argparse | 类型标注即参数定义，风格与 pydantic 统一 |
| 终端渲染 | Rich | 手写 ANSI | 流式文本、代码块高亮、工具调用折叠 |
| 存储 | SQLite（标准库 `sqlite3`） | Postgres / 内存字典 | 单机单用户零运维；FTS5 直接兼做检索的词法层 |
| 并发 | asyncio | 多线程 | IO 密集型，一个 turn 一个 Task 最自然 |
| 测试 | pytest + pytest-asyncio | unittest | 生态标准，插件齐 |
| 静态检查 | ruff | flake8 + black + isort | 一个工具顶三个，配置短 |

## 2. 架构

### 2.1 组件职责

| 组件 | 职责 | 不做什么 |
| --- | --- | --- |
| `core` | 消息 / 事件模型、主循环、prompt 组装、上下文压缩 | 不碰网络、不碰文件系统、不 import 具体 provider |
| `providers` | 调用模型 API、重试与退避、超时、用量统计 | 不做 prompt 组装、不执行工具 |
| `tools` | 工具注册、参数 schema、执行、超时、输出截断 | 不判定审批（交给 `policy`） |
| `policy` | 工具三级分类、审批请求、危险操作拦截 | 不执行命令 |
| `retrieval` | 切块、索引构建与增量更新、混合检索 | 不直接调模型 |
| `store` | 会话 / 消息 / 事件 / 工具调用持久化 | 不含业务逻辑 |
| `server` | HTTP 路由、SSE、token 认证 | 不含 agent 逻辑 |
| `client` | CLI 交互、流式渲染、审批提示 | 不直接访问数据库 |

依赖方向单向：`client / server → core → {providers, tools, retrieval, store}`。
`core` 只依赖 Protocol 抽象，具体实现在启动时装配（依赖注入），这也是能用回放 provider 替换真实 provider 的原因。

### 2.2 一次 turn 的数据流

1. CLI 发 `POST /sessions/{id}/messages`，带 `idempotency_key`。
2. server 校验 token → 查幂等键，命中则直接返回原 `turn_id`（不重复执行）。
3. 用户消息落库并分配会话内递增 `seq`。
4. 写 `turn.started` 事件 → **先落库，再推送** → 返回 `202 {"turn_id": ...}`。
5. core 组装上下文（系统提示 + 历史消息 + 本次输入）并附上工具清单。
6. `provider.stream()` 逐块产出：文本增量 → 发 `text.delta`；工具调用增量 → 累积成完整 `ToolCall`。
7. 若本轮有工具调用：发 `tool.call` → `policy` 分类：
   - 只读 → 直接执行
   - 写操作 → 发 `approval.required`，等客户端 `POST /approvals/{call_id}`（超时按拒绝处理）
   - 危险 → 直接拒绝，把拒绝原因回填给模型
8. 执行工具 → 发 `tool.result`（**截断版进模型上下文，全量入库**）。
9. 工具结果作为 `tool` 消息追加进上下文，回到第 5 步，最多 `max_tool_rounds` 轮。
10. 模型不再请求工具 → 发 `text.done` 与 `turn.done`（含 token 用量、耗时）。
11. SSE 消费者按 `seq` 推送；客户端断线重连带 `Last-Event-ID`，服务端从 DB 补齐缺口。
12. CLI 渲染：文本流式打印，工具调用折叠成一行摘要，审批请求变成交互式确认。

### 2.3 并发模型

- **单事件循环。** server 收到消息后 `asyncio.create_task(loop.run_turn(...))`，立刻返回 202，不等结果。
- **会话内串行**：每个会话一把 `asyncio.Lock`，同一会话的 turn 不会交错。
- **会话间并行**：不同会话互不阻塞。
- **工具执行**：`asyncio.create_subprocess_exec` + `asyncio.timeout`；stdout/stderr 边读边截断，避免大输出把内存吃满。
- **数据库**：`sqlite3` 同步驱动 + `asyncio.to_thread`，所有写操作经单一 `asyncio.Queue` 串行化（SQLite 是单写者模型，串行化比加锁简单且不会死锁）。
- **事件分发**：`EventBus` 每会话维护一组订阅者队列；队列有界，满了丢最老的并记日志——客户端总能从 DB 按 `seq` 补齐，所以丢推送不等于丢数据。

### 2.4 错误与恢复

| 情况 | 处理 |
| --- | --- |
| 模型 API 429 / 5xx / 连接错误 | 指数退避重试（上限 N 次），每次记 `error` 事件 |
| 模型 API 4xx（参数错、欠费） | 不重试，终止本轮，`turn.done` 带失败原因 |
| 工具超时 | 杀进程组，回填"超时"给模型，让它换个做法 |
| 工具非零退出 | 不当作异常，把 stderr 回填给模型（这是它的信息源） |
| 服务进程崩溃 | 会话状态从事件日志恢复；未完成的 turn 标记为 `interrupted` |
| 客户端断线 | 无影响，任务继续；重连按 `seq` 补齐 |

### 2.5 关键决策与理由

1. **事件先落库再推送**——否则客户端补齐时会缺事件；顺序反了就出现"看到过但查不到"。
2. **SSE 而非 WebSocket**——本场景是服务端单向推流，SSE 更简单、可 `curl` 调试、断线重连语义天然（`Last-Event-ID`）。
3. **工具输出截断但全量入库**——模型只看头尾各 4KB，省钱且避免上下文被一条 `ls -R` 冲爆；人要排查时能从库里看全文。
4. **审批超时视为拒绝**——安全默认值，避免"没人看所以放行"。
5. **不引 LangChain**——主循环是本项目的核心价值，引框架则面试无从谈起；只借 `openai` SDK 的 HTTP 客户端。

## 3. 目录结构

```
agent/
├── README.md                 项目说明、快速开始、架构图、已知限制
├── pyproject.toml            依赖、入口点、pytest / ruff 配置
├── .env.example              配置样例（不含真实 key）
├── .gitignore
├── docs/
│   ├── design.md             规划层：目标、里程碑、验收
│   ├── detailed-design.md    本文
│   └── protocol.md           事件与 HTTP 协议契约（第 4 阶段冻结）
├── src/agent/
│   ├── __init__.py
│   ├── config.py              Settings（pydantic-settings），从 .env + 环境变量加载
│   ├── logging.py             结构化日志配置（JSON 行）
│   ├── core/
│   │   ├── events.py          事件类型、Event 模型、序列化
│   │   ├── messages.py        Message / ToolCall / ToolResult 模型
│   │   ├── bus.py             EventBus：订阅、广播、有界队列
│   │   ├── loop.py            AgentLoop：主循环（模型↔工具）
│   │   ├── prompt.py          系统提示组装、上下文压缩
│   │   └── errors.py          异常层次
│   ├── providers/
│   │   ├── base.py            Provider Protocol、Chunk 类型
│   │   ├── openai_compat.py   OpenAI 兼容实现（DeepSeek）
│   │   ├── replay.py          录制回放实现（测试与离线演示）
│   │   └── recorder.py        把真实响应录成 fixtures
│   ├── tools/
│   │   ├── base.py            Tool Protocol、ToolSpec、ToolContext、ToolResult
│   │   ├── registry.py        注册与查找、导出模型可用的 schema
│   │   ├── fs.py              fs.read / fs.list / fs.write
│   │   ├── shell.py           shell.exec（超时、进程组、输出截断）
│   │   ├── search.py          代码检索工具（第 5 阶段接入 retrieval）
│   │   └── policy.py          三级分类、审批流转、危险命令拦截
│   ├── retrieval/             第 5 阶段
│   │   ├── chunk.py           按符号切块（tree-sitter 优先，正则兜底）
│   │   ├── index.py           FTS5 索引构建与增量更新
│   │   └── hybrid.py          词法 + 向量，RRF 融合
│   ├── store/
│   │   ├── schema.sql         建表语句
│   │   ├── db.py              连接、迁移、串行写队列
│   │   └── session.py         会话/消息/事件/工具调用的读写与投影
│   ├── server/
│   │   ├── app.py             FastAPI 应用与路由
│   │   ├── sse.py             事件流编码、Last-Event-ID 续传
│   │   └── auth.py            token 认证依赖
│   └── client/
│       ├── main.py            Typer 入口：run / serve / sessions / replay / eval
│       ├── render.py          Rich 渲染：流式文本、工具卡片、审批提示
│       └── transport.py       HTTP + SSE 客户端封装
├── tests/
│   ├── conftest.py            公共 fixture（临时工作区、内存库、回放 provider）
│   ├── unit/                  纯函数单测（策略、截断、切块、RRF）
│   └── integration/           回放驱动的端到端测试（断言事件序列）
├── fixtures/                  录制的模型响应（进 git，CI 用它跑）
└── eval/
    ├── cases/                 golden 任务集（端到端）
    ├── retrieval/             检索 golden 查询集与指标
    └── runner.py              评测执行器与报告
```

## 4. 数据模型

### 4.1 SQLite schema

```sql
CREATE TABLE sessions (
  id          TEXT PRIMARY KEY,          -- sess_ + uuid4 hex
  title       TEXT NOT NULL DEFAULT '',
  profile     TEXT NOT NULL,             -- chat | code
  workspace   TEXT,                      -- 工作区根目录（code profile 必填）
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);

CREATE TABLE messages (
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL REFERENCES sessions(id),
  seq         INTEGER NOT NULL,          -- 会话内单调递增
  role        TEXT NOT NULL,             -- system | user | assistant | tool
  content     TEXT NOT NULL,
  tool_call_id TEXT,
  created_at  TEXT NOT NULL,
  UNIQUE (session_id, seq)
);

CREATE TABLE events (                     -- 只追加的事实源（source of truth）
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL,
  turn_id     TEXT,
  seq         INTEGER NOT NULL,           -- 会话内单调递增
  type        TEXT NOT NULL,
  data        TEXT NOT NULL,              -- JSON
  created_at  TEXT NOT NULL,
  UNIQUE (session_id, seq)
);

CREATE TABLE turns (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  status       TEXT NOT NULL,             -- running | done | failed | interrupted
  input_tokens  INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  started_at   TEXT NOT NULL,
  ended_at     TEXT
);

CREATE TABLE tool_calls (
  id          TEXT PRIMARY KEY,           -- 模型给出的 call_id
  session_id  TEXT NOT NULL,
  turn_id     TEXT NOT NULL,
  name        TEXT NOT NULL,
  args        TEXT NOT NULL,              -- JSON
  tier        TEXT NOT NULL,              -- read | write | dangerous
  decision    TEXT,                       -- auto | allow | deny | timeout
  status      TEXT NOT NULL,              -- pending | running | ok | error | timeout | denied
  result      TEXT,                       -- 全量输出（模型只看到截断版）
  exit_code   INTEGER,
  started_at  TEXT,
  ended_at    TEXT
);

CREATE TABLE idempotency (
  key         TEXT PRIMARY KEY,           -- 客户端提供的幂等键
  session_id  TEXT NOT NULL,
  turn_id     TEXT NOT NULL,
  created_at  TEXT NOT NULL
);
```

`messages` 是从 `events` 投影出来的物化视图：删掉它可以从事件日志重建，这正是 `agent replay --rebuild` 的验证方式。

### 4.2 事件

| 事件 | 关键字段 |
| --- | --- |
| `turn.started` | `turn_id` |
| `text.delta` | `text` |
| `text.done` | `text`（完整文本）、`usage` |
| `tool.call` | `call_id`、`name`、`args` |
| `approval.required` | `call_id`、`name`、`args`、`reason`、`expires_at` |
| `tool.result` | `call_id`、`status`、`exit_code`、`truncated`、`preview` |
| `turn.done` | `status`、`usage`、`duration_ms` |
| `error` | `kind`、`message`、`retryable` |

统一的信封（SSE 的 `data:` 就是它）：

```json
{
  "id": "evt_01J...",
  "seq": 42,
  "session_id": "sess_01J...",
  "turn_id": "turn_01J...",
  "type": "tool.call",
  "data": {"call_id": "call_1", "name": "fs.read", "args": {"path": "src/app.py"}},
  "ts": "2026-09-16T10:00:00.123Z"
}
```

### 4.3 HTTP 契约

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/healthz` | 存活探针 |
| POST | `/sessions` | 建会话 `{profile, workspace?, title?}` → `201 {id}` |
| GET | `/sessions` | 会话列表（分页） |
| GET | `/sessions/{id}` | 会话详情（含 topic、workspace、状态） |
| GET | `/sessions/{id}/messages?after_seq=` | 消息历史（增量拉取） |
| POST | `/sessions/{id}/messages` | `{content, idempotency_key}` → `202 {turn_id}` |
| GET | `/sessions/{id}/events?last_event_id=` | SSE 事件流（`Last-Event-ID` 头亦可） |
| POST | `/sessions/{id}/interrupt` | 中断当前 turn |
| POST | `/sessions/{id}/approvals/{call_id}` | `{decision: allow\|deny}` → `200` |
| GET | `/sessions/{id}/turns/{turn_id}` | 单轮用量与耗时 |

## 5. 模块接口

```python
# core/messages.py
class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    role: Role
    content: str
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]


# providers/base.py
class TextDelta(BaseModel):
    text: str


class ToolCallDelta(BaseModel):
    index: int
    id: str | None
    name: str | None
    args_json: str


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


Chunk = TextDelta | ToolCallDelta | Usage


class Provider(Protocol):
    async def stream(
        self, *, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> AsyncIterator[Chunk]: ...


# tools/base.py
class Tool(Protocol):
    spec: ToolSpec  # name / description / params JSON schema
    tier: Tier  # read | write | dangerous

    async def run(self, args: BaseModel, ctx: ToolContext) -> ToolResult: ...


# core/loop.py
class AgentLoop:
    def __init__(
        self,
        *,
        provider: Provider,
        registry: ToolRegistry,
        policy: Policy,
        store: SessionStore,
        bus: EventBus,
        settings: Settings,
    ) -> None: ...
    async def run_turn(self, session_id: str, user_text: str) -> str: ...  # 返回 turn_id
```

## 6. 配置

从 `.env` 与环境变量读取，前缀 `AGENT_`，由 `pydantic-settings` 校验。

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `AGENT_PROVIDER` | `deepseek` | `deepseek` / `replay` |
| `AGENT_MODEL` | `deepseek-v4-flash` | 模型名 |
| `AGENT_BASE_URL` | `https://api.deepseek.com` | 兼容 OpenAI 协议的地址 |
| `AGENT_API_KEY` | 无 | 密钥，只从环境读，绝不入 git |
| `AGENT_WORKSPACE` | 当前目录 | code profile 的工作区根 |
| `AGENT_MAX_TOOL_ROUNDS` | `12` | 单轮最多几次工具往返 |
| `AGENT_TOOL_TIMEOUT_S` | `60` | 单个工具超时 |
| `AGENT_OUTPUT_LIMIT_BYTES` | `8192` | 工具输出进上下文的截断阈值 |
| `AGENT_APPROVAL_TIMEOUT_S` | `120` | 审批等待上限，超时视为拒绝 |
| `AGENT_DB_PATH` | `~/.local/share/agent/agent.db` | 会话库 |
| `AGENT_HOST` / `AGENT_PORT` | `127.0.0.1` / `8765` | 监听地址，默认只本机 |
| `AGENT_AUTH_TOKEN` | 自动生成 | 客户端调用凭证 |
| `AGENT_LOG_LEVEL` | `info` | 日志级别 |

## 7. 工具与安全策略

| 级别 | 例子 | 行为 |
| --- | --- | --- |
| `read` | `fs.read`、`fs.list`、`search_code`、`shell.exec` 只读白名单（`ls`、`cat`、`git status`、`git diff`、`grep`） | 自动执行 |
| `write` | `fs.write`、`git add/commit`、跑测试、装依赖 | 发审批事件，等客户端确认 |
| `dangerous` | `rm -rf`、`sudo`、`git push --force`、写工作区外路径、读 `.env`/密钥文件、管道下载执行 | 直接拒绝，回填拒绝原因 |

沙箱约束（始终生效，与级别无关）：

- 路径经 `realpath` 解析后必须落在工作区内，软链接逃逸同样拦
- 子进程超时后杀整个进程组，避免留下孤儿
- 子进程环境变量白名单（不透传 API key 等敏感变量）
- 输出按 `AGENT_OUTPUT_LIMIT_BYTES` 截断，头尾各半，中间标注省略字节数

## 8. 检索（RAG）细化

- **切块**：按符号边界（函数/类/方法），块内保留 `path`、`start_line`、`end_line`、`symbol`、`language`；tree-sitter 缺失时退回正则/缩进切分。
- **索引**：FTS5 建在 `symbol`、`path`、`content` 上；向量层可插拔，先 numpy 暴力算，需要时再换 faiss / sqlite-vec；按文件 hash 增量重建。
- **融合**：两路结果用 RRF 融合，避免调权重。
- **使用方式**：作为 `search_code` 工具让模型自己调；另留一个可选的预填模式做对比。
- **评测**：golden 查询集 + recall@5 / MRR，对比纯词法 / 纯向量 / 混合三种配置，结论写进 README。
- **降级**：向量不可用时退化为纯词法并在事件里提示，绝不因此让检索失败。

## 9. 测试策略

| 层次 | 覆盖内容 | 是否联网 | 是否需 key |
| --- | --- | --- | --- |
| 单元测试 | 策略判定、输出截断、切块、RRF、事件序列化 | 否 | 否 |
| 回放测试 | 主循环、工具往返、审批流转、恢复逻辑（用 `fixtures/` 录制响应） | 否 | 否 |
| 评测 | golden 任务集通过率、检索指标 | 否（回放） | 否 |
| 冒烟测试 | 真实 API 走一次最小任务 | 是 | 是（手动触发） |

CI 只跑前三层，因此**永远不需要密钥**；真实调用只在本地手动跑。

## 10. 阶段 1 任务拆解

| 编号 | 任务 | 负责人 |
| --- | --- | --- |
| 1.0 | 项目骨架：`pyproject.toml`、包结构、`.env.example`、日志、config | 我 |
| 1.1 | 模型接口连通性烟测（确认 DeepSeek 的端点形态） | 我 |
| 1.2 | `Provider` Protocol + `replay` 实现 | 我 |
| 1.3 | `openai_compat` 真实实现（流式 + 工具调用增量拼装） | 我 |
| 1.4 | 事件模型 + `EventBus`（内存版） | 我 |
| 1.5 | 工具：`fs.read` / `fs.list` / `shell.exec` | `shell.exec` 你写，其余我写 |
| 1.6 | 工具策略与三级分类的骨架 | 我 |
| 1.7 | `AgentLoop.run_turn` 主循环 | 你写（我给骨架和讲解） |
| 1.8 | CLI 最小版：单会话、流式打印、工具调用折叠显示 | 我 |
| 1.9 | 一次真实任务的冒烟演示 + 首次 commit | 一起 |

## 11. 风险

| 风险 | 应对 |
| --- | --- |
| DeepSeek 端点形态与预期不符（`chat.completions` vs `responses`） | 1.1 先烟测，再定 provider 实现 |
| 工具输出过大冲爆上下文 | 截断策略 + 全量入库，单测覆盖边界 |
| 审批流程把演示卡住 | 只读工具自动执行，演示主线不依赖审批 |
| 范围膨胀（阶段 5、6 都是大块） | 严格按里程碑推进，每阶段可独立演示 |
| 你只在 review、不动手 → 面试讲不透 | core loop / 工具执行 / 评测三块必须你写 |
