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
| 已实测 | ① 流式逐 token：691 字符的回答产生 399 个 `text.delta`，跨度 0.913s，与直连 SDK 的 0.901s 一致；② 工具报错自愈：故意读错路径 → 模型自己 `fs_list` → `find` 定位 → 读到正确文件；③ **崩溃恢复**：跑到一半 `kill -9`，重启后打出"恢复：1 个没跑完的 turn 已标记为 interrupted"，同会话续跑并答对了上文相关问题；④ **重放**：`agent replay <会话>` 不调模型，把 223 条事件按 seq 原序还原；⑤ **审批**：没通道 / 被拒 / 超时三种情况都不放行（`tests/unit/test_approval.py` 钉住）；⑥ **golden 集**：3/3 通过，**3 秒跑完、不联网、不要密钥**；⑦ 110 个测试在 `AGENT_API_KEY` 为空时同样全绿（CI 场景） |
| 下一步 | 阶段 4：服务化（HTTP + SSE + 协议冻结）；交互式 `agent chat` 也放在这里 |
| 代码量 | 源码约 3000 行，测试 110 个（全绿），ruff 干净 |
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
| 2 | golden 集 | `eval.py`（加载用例 + 判定 + 通过率）、`evals/cases/*.json`、`evals/workspaces/`、`evals/traces/*.jsonl` |
| 3 | CI | `.github/workflows/ci.yml`：`uv sync --frozen` → ruff → pytest，**不注入任何密钥** |
| 4 | 验收 | golden 3/3；110 个测试在空密钥下全绿 |

录放的三种用法：

```bash
# 录：真实调用，同时把每次 (请求, 响应) 追加到 JSONL
AGENT_TRACE_PATH=evals/traces/xxx.jsonl .venv/bin/agent run "任务"
# 放：完全离线，不联网不要密钥
AGENT_PROVIDER=replay AGENT_TRACE_PATH=evals/traces/xxx.jsonl .venv/bin/agent run "任务"
# 评测：跑 golden 集并打印通过率
.venv/bin/python -m pytest tests/eval -s
```

### 7.2 下一步：阶段 4

1. `agent serve`：FastAPI + SSE，让 CLI 从"直接驱动图"改成"订阅事件流"
2. 断线续传：`Last-Event-ID` → `repo.list_events(after_seq=...)`（存储层已就绪，缺接线）
3. 幂等：客户端重发同一条消息时用 `idempotency` 表挡住（表和读写函数已就绪，缺接线）
4. 交互式 CLI（`agent chat`）：一次启动多轮对话，把启动开销从"每轮都付"降到"只付一次"
5. 协议冻结 → `docs/protocol.md`

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
| 每次运行要等 30 秒以上，且前 30 秒屏幕上没有任何输出 | 代码和 `.venv` 都在 Windows 盘（`/mnt/g`，9p 挂载）。`agent run` 要读 3885 个 `.py` 文件，跨文件系统每次读都是往返。实测：`import openai` 从 `/mnt/g` 要 12.8s，从 Linux 侧只要 1.6s | `.venv` 移到 Linux 文件系统，原位置留软链接（`.gitignore` 里的规则写成 `.venv` 不带斜杠，否则软链接匹配不到）。`agent version` 从十几秒降到 1 秒 |

## 9. 协作约定

- **分工**：框架接线、可靠性层、store、server、测试、CLI、检索、文档 → 助手写；
  工具边界（`shell_exec` 的截断与白名单）、评测任务设计、每阶段 review → 自己写。
- **提交纪律**：写完先汇报，**得到"提交"授权再 commit**；commit 信息只写一行英文主题（conventional commits 风格），不写 body。
- **提交前门槛**：`pytest` 全绿 + `ruff check` 干净，两条都过才提交。
- **推送**：本地提交与推送分开；push 需要联网授权。
- **测试原则**：单元测试与回放测试不需要联网、不需要 key；真实调用只在手动冒烟与演示时做。
