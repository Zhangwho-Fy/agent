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
| 阶段 1（最小闭环） | ✅ 代码完成，**2/4 条验收已验证** |
| 未验证 | ① 流式是逐 token 输出还是被缓冲；② 工具报错后模型能否自我纠正 |
| 下一步 | 阶段 2：可靠性层（SQLite 持久化、幂等、崩溃恢复、checkpointer） |
| 代码量 | 源码约 1600 行，测试 57 个（全绿），ruff 干净 |
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
cp .env.example .env          # 填入 AGENT_API_KEY（从原机器 ~/.codex/config.toml 里
                              # [model_providers.deepseek] 段复制 sk-... ，或去控制台新建）
UV_HTTP_TIMEOUT=300 uv sync   # 国内网络到 PyPI 慢，超时参数必须加
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
│   ├── nodes.py        模型节点 + 工具节点（分级/拦截/发事件都在工具节点里）
│   ├── builder.py      组装 START → agent ⇄ tools → END
│   └── bridge.py       把框架的流翻译成我们的事件（关键适配层）
├── core/
│   ├── events.py       事件模型 + SSE 编码（seq 作 SSE id，供断线续传）
│   ├── bus.py          会话内事件分发（有界队列，慢了丢最老）
│   ├── reliability.py  事件编号、幂等键（阶段 2 换成落库实现）
│   ├── prompt.py       系统提示
│   ├── messages.py     消息/工具调用模型
│   ├── tool_spec.py    工具契约 + 工具名校验（协议只允许 [A-Za-z0-9_-]）
│   └── errors.py       异常层次
├── models/factory.py   构建 ChatDeepSeek（换供应商改这里）
├── tools/
│   ├── base.py         ToolContext/ToolResult、工作区路径边界、输出截断
│   ├── fs.py           fs_read / fs_list
│   ├── shell.py        shell_exec（超时杀进程组、环境变量白名单）
│   ├── policy.py       三级分级：只读自动 / 写操作审批 / 危险拒绝
│   └── registry.py     工具注册与 schema 导出
└── client/main.py      CLI：run / doctor / config / version
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

## 7. 阶段 2 要做什么（下一步）

1. 建 SQLite 表：`sessions` / `messages` / `events` / `turns` / `tool_calls` / `idempotency`（schema 见 detailed-design 4.1）
2. `EventEmitter` 改成**先落库再推送**（接口不变，换实现）
3. 接 LangGraph 的 checkpointer：需新依赖 `langgraph-checkpoint-sqlite`（`uv add` 走网络，由人执行）
4. 接审批流：工具节点用 `interrupt()` 挂起，`Command(resume=...)` 恢复，**超时按拒绝**
5. 崩溃恢复：重启后把 `running` 的 turn 标成 `interrupted`，会话历史可读、可继续
6. 验收：跑到一半 `kill -9`，重启后会话还在、能续跑；`agent replay <trace>` 不调模型也能还原事件序列

## 8. 已知坑（现象 → 原因 → 做法）

| 现象 | 原因 | 做法 |
| --- | --- | --- |
| 首次真实调用 400：`Invalid 'tools[0].function.name'` | 协议要求函数名只匹配 `[A-Za-z0-9_-]`，用了 `fs.read` | 工具名用下划线；校验器前移到构造期 + 测试钉住 |
| CLI 输出里混进 JSON 行 | httpx 在 info 级打印每个请求 | 噪音 logger 压到 warning |
| `uv sync` 卡在大 wheel | 国内到 files.pythonhosted.org 慢 | `UV_HTTP_TIMEOUT=300`，或换镜像并删 `uv.lock` |
| 测试结果受本机影响 | 读到了真实 `.env` / 环境变量 | `Settings(_env_file=None)` + autouse fixture 清 `AGENT_*` |
| 路径检查被绕过 | 只做字符串前缀判断，没解析软链接 | 一律 `realpath` 后再判边界 |
| 命令超时后仍有残留进程 | 只杀了父进程 | `start_new_session=True` + `os.killpg` 杀整组 |

## 9. 协作约定

- **分工**：框架接线、可靠性层、store、server、测试、CLI、检索、文档 → 助手写；
  工具边界（`shell_exec` 的截断与白名单）、评测任务设计、每阶段 review → 自己写。
- **提交纪律**：写完先汇报，**得到"提交"授权再 commit**；commit 信息只写一行英文主题（conventional commits 风格），不写 body。
- **提交前门槛**：`pytest` 全绿 + `ruff check` 干净，两条都过才提交。
- **推送**：本地提交与推送分开；push 需要联网授权。
- **测试原则**：单元测试与回放测试不需要联网、不需要 key；真实调用只在手动冒烟与演示时做。
