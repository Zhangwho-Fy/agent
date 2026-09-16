# 阶段 1 总结：最小闭环

> 状态：**代码完成，验收 2/4 已验证**（端到端问答 ✅、工具调用可见 ✅；流式增量、工具报错自愈待确认）
>
> 代码量：源码 1585 行（core 461 / tools 492 / graph 250 / client 210 / models 38），测试 57 个全绿。

## 1. 这一阶段做成了什么

一句话：**一句命令行下去，agent 能自己读代码、跑命令，基于真实内容回答问题**。

```
$ agent run "用一句话说明 src/agent/graph/builder.py 里的图是怎么流转的"
你 用一句话说明 src/agent/graph/builder.py 里的图是怎么流转的
agent
→ fs_read {"path": "src/agent/graph/builder.py"}
  ← ok
从 START 进入 agent 节点（调用模型），然后由条件边 _route 检查最新消息：
若带 tool_calls 就转到 tools 执行工具并回到 agent 形成循环，否则直接到 END 结束。
```

交付物按层分：

| 层 | 文件 | 职责 |
| --- | --- | --- |
| 编排 | `graph/state.py` / `nodes.py` / `builder.py` | 图状态、模型节点、工具节点、条件边（LangGraph） |
| 桥接 | `graph/bridge.py` | 把框架的流翻译成我们对外的事件 |
| 模型 | `models/factory.py` | 按配置构建 `ChatDeepSeek` |
| 工具 | `tools/base.py` / `fs.py` / `shell.py` / `policy.py` / `registry.py` | 三个工具 + 路径边界 + 分级策略 + 输出截断 |
| 可靠性 | `core/reliability.py` | 事件编号（seq）、幂等键（阶段 1 为内存实现） |
| 事件 | `core/events.py` / `bus.py` | 事件模型、SSE 编码、会话内分发 |
| 提示 | `core/prompt.py` | 系统提示（行为与边界） |
| 入口 | `client/main.py` | `agent run` / `doctor` / `config` / `version` |

## 2. 一次 `agent run` 到底发生了什么

按执行顺序，每一步都能在代码里找到落点：

1. `client/main.py` 读配置（`.env` + 环境变量）→ 建 `ToolContext`（工作区、超时、输出限额）
2. 组装工具注册表与策略：`registry` 给出三个工具的 schema，`Policy` 持有工作区边界
3. `models/factory.py` 构建 `ChatDeepSeek`，`bind_tools()` 把工具清单绑给模型
4. `graph/builder.py` 编译图：`START → agent ⇄ tools → END`
5. `graph/bridge.py` 用 `astream(stream_mode=["messages","updates"])` 驱动一轮
6. 模型节点产出回复：有 `tool_calls` 就落到 `tools` 节点，没有就到 `END`
7. 工具节点：发 `tool.call` 事件 → `Policy` 分级 → 执行/拦截 → 发 `tool.result` → 回填 `ToolMessage`
8. 结果回到 `agent` 节点，形成循环（`recursion_limit` 兜底）
9. 全程 `EventEmitter` 给事件分配 seq 并推送；CLI 用它渲染流式输出与工具调用

## 3. 为什么 agent 会有"图"这种东西（重点）

### 3.1 先承认：你的直觉是对的

"发送请求 → 拼接消息 → 判断要不要调工具 → 再发送"这个循环（ReAct 循环）**就是 agent 的核心**，用 `while` + `if` 写出来完全正确，而且在"一次跑到底"的前提下它最优——代码短、调试直观。本项目最初的方案也是手写这个循环。

### 3.2 但下面四个需求会把 while 逼到墙角

1. **人工介入**：写文件、提交代码这类操作要等人确认。人可能 10 分钟后才点"同意"。而 `while` 循环的"我在等审批"这个状态是**内存里的一行代码位置**，进程一停就没了。
2. **崩溃恢复**：任务跑到第 7 步进程被杀，重启后要能接着跑，而不是从头再来（工具可能已经改过文件了）。
3. **断线续传与观测**：客户端要能看到"现在在哪一步、在等什么"，而不仅仅是最终文本。
4. **分支与并行**：不同工具走不同确认流程、失败走不同重试路径、多个子任务并行再汇总。用 `if/else` 堆到十几层之后，没人敢改那段代码。

根因只有一个：**`while` 循环里"执行到哪了"这个信息存在 Python 调用栈里**。调用栈不能序列化、不能存库、不能在另一个进程里重建。

### 3.3 图的本质：把控制流从"调用栈"搬到"数据"里

图做的事就是：把"下一步做什么"从代码的跳转，变成一个**显式的数据结构**——状态（`AgentState`）+ 节点（函数）+ 边（条件函数）。

一旦控制流变成了数据，下面这些东西几乎是白送的：

- **检查点**：状态能序列化，就能存下来（LangGraph 的 `checkpointer`）
- **恢复**：能从任意一个检查点继续跑
- **人工介入**：`interrupt()` 把"等审批"变成状态里的一条记录，人回来了再 `Command(resume=...)`
- **时间旅行调试**：回到第 N 步看当时的状态，而不是重放一遍
- **可观测**：状态和边都是数据，能画出来、能打进日志、能推给客户端
- **并行与子流程**：数据是 DAG，不是一条线

一句话记住：**图不是目的，"可暂停、可恢复、可观测"才是目的；图只是把控制流数据化之后自然而然的形式。**

### 3.4 代价与判断标准

| 维度 | `while` 循环 | 图 |
| --- | --- | --- |
| 上手成本 | 极低 | 要理解状态、归约器、条件边 |
| 调试 | 单进程栈，直观 | 跨层：框架 → 节点 → 你的代码 |
| 暂停 / 恢复 | 要自己发明（序列化调用栈） | 框架内置 |
| 分支 / 并行 / 子流程 | `if/else` 迅速失控 | 加节点与边即可 |
| 可观测 | 自己打日志 | 状态可导出、可回放 |
| 一次性脚本 | **更合适** | 过度设计 |

判断标准很干脆：**只要"跑一半要停、停了要能接着跑、要让人看见在跑什么"，就该用图；只要求一次跑完的脚本，`while` 更好。**

### 3.5 本项目里的具体形状

```python
START → agent ──(最后一条消息有 tool_calls)──→ tools ──→ agent
          └────(没有 tool_calls)────────────→ END
```

- `agent` 节点：调用模型，返回增量（`messages` 追加、`rounds` 累加、token 用量累加）
- `tools` 节点：发事件 → 分级 → 执行或拦截 → 回填 `ToolMessage`
- 条件边 `_route`：等价于手写版里的 `if not tool_calls: return`
- `recursion_limit`：循环上限，护栏必须有

**注意我们并没有把所有东西都塞进图**：事件日志、沙箱边界、幂等键、审批策略都在图之外。因为它们是**对外契约与安全语义**，不是控制流。这个边界划错，图会变成一坨什么都往里装的状态袋。

## 4. 本阶段的知识点（面试向）

### 4.1 LangGraph

| 概念 | 一句话 | 我们代码里的位置 |
| --- | --- | --- |
| `StateGraph` | 用状态 + 节点 + 边描述的图 | `graph/builder.py` |
| 状态归约器 | `Annotated[list, add_messages]` 让节点只返回**增量**，不必读改写全量 | `graph/state.py` |
| 节点 | 一个接收状态、返回增量字典的函数 | `graph/nodes.py` |
| 条件边 | 由函数决定走哪条边（循环是否继续） | `_route` |
| `compile()` | 把图编译成可执行对象 | `builder.py` |
| `checkpointer` | 状态持久化（阶段 2 接 SQLite） | 目前传 `None` |
| `interrupt()` | 暂停等人（阶段 2 接审批流） | 尚未启用 |
| `stream_mode` | `messages`（逐 token）/ `updates`（节点产出）/ `custom` | `graph/bridge.py` |
| `recursion_limit` | 循环次数硬上限 | `bridge.py` |

### 4.2 LangChain 模型层

- `BaseChatModel` 抽象：`ainvoke` / `astream` / `bind_tools`
- `AIMessage.tool_calls` 是**已解析好的结构化列表**（id / name / args），不用自己拼 JSON 分片（这正是框架帮我们省掉的部分）
- `ToolMessage(tool_call_id=...)`：工具结果必须和请求它的那次调用配对，否则模型会混淆
- `usage_metadata`：token 记账的来源

### 4.3 协议约束（踩过的坑）

- **函数名只允许 `[A-Za-z0-9_-]`**：`fs.read` 被 400 拒绝，改成 `fs_read`；并用 pydantic 校验器把约束**前移到构造期**，再加测试钉住
- `tools` 参数是 `{"type": "function", "function": {...}}` 结构，参数用 JSON Schema 描述
- 工具报错不要抛异常，要作为工具结果回填——**错误是给模型的信息**

### 4.4 工程语义（我们自研、框架不提供的部分）

- **路径边界**：`realpath` 之后判前缀，防 `../` 与软链接逃逸（字符串前缀判断会被绕过）
- **进程组超时**：`start_new_session=True` + `os.killpg`，否则 `sleep 100 &` 会留成孤儿
- **输出截断**：头尾各半（报错与结论通常在尾部），全量进库、截断进上下文
- **环境变量白名单**：子进程只拿到 PATH/HOME/LANG 等，密钥不外泄（有测试断言 `env` 里查不到 `AGENT_API_KEY`）
- **分级策略**：只读自动、写操作审批、危险命令拒绝；拒绝时把**原因**回填，模型才知道换做法
- **事件 seq 与幂等键**：为阶段 2 的续传与去重打地基
- **结构化日志**：一行一条 JSON，并把 httpx 之类的噪音压到 warning

### 4.5 工具调用协议：六个关键问题（问答整理）

**Q1：模型具体怎么调用的？用 LangChain 的 SDK 吗？**

LangChain 不是 SDK，是**适配层**。真正跟服务器通信的是 `openai` 官方 SDK（它负责 HTTP、SSE 解析、重试、类型）。三层分工：

```
我们的节点代码  →  ChatDeepSeek（LangChain：抹平各家字段差异）
                →  openai SDK（发请求、解流、重试）
                →  DeepSeek 服务器
```

我们这边只有两行：`model.bind_tools(...)` + `await bound_model.ainvoke(messages)`。LangGraph 完全不碰 HTTP，它只管"什么时候调、调完往哪走"。**能讲清这三层的分工，就说明你知道框架到底在替你干什么。**

**Q2：工具怎么实现？输入输出是什么？**

工具就是**普通的 Python 异步函数**，和模型没有任何关系：

- 输入：① pydantic 模型声明的参数（`model_validate` 校验）；② `ToolContext`（工作区、超时、输出上限）
- 输出：`ToolResult`（`ok` / `content` / `truncated` / `exit_code` / `duration_ms`）
- **最终回填给模型的只有 `content` 那段文本**，其余字段是给我们自己和日志看的

关键认识：**模型不能执行任何东西，它只能"请求执行"**。所以路径限制、危险命令拦截、超时、环境变量白名单都必须落在工具这一层——指望模型守规矩是不成立的。

**Q3：工单是规范输出吗？所有模型统一吗？**

分两说：

- **形状是事实标准**：`tools` 传进去、`tool_calls` 返回出来，来自 OpenAI 的 Chat Completions 协议，DeepSeek / Qwen / Kimi / GLM 都兼容（大家想让用户改个 `base_url` 就能用）。
- **但格式不统一**：Claude 用 `tool_use` block（参数直接是对象），Gemini 又是另一套；**LangChain 在这里的价值就是把差异吃掉**，统一成 `AIMessage.tool_calls`。
- **不是所有模型都支持工具调用**：小模型/老模型可能完全不支持，那就只能"提示里要求输出 JSON，自己解析"——这是另一个流派。

**Q4：工单号怎么设计的？**

**是模型服务端生成的**，不是我们设计的。我们只做一件事：原样拿走，执行完塞回 `ToolMessage(tool_call_id=...)` 完成配对。要点：

- 只依赖"同一次回复内唯一"，算法是厂商内部实现
- **并行调用时顺序不保证**，所以必须按 id 配对，不能按顺序
- 别混三类 ID：

| 名字 | 谁生成 | 作用 | 有效期 |
| --- | --- | --- | --- |
| `call_id`（工单号） | 模型服务端 | 配对"调用"与"结果" | 一次回复内 |
| `seq`（事件序号） | 我们 | 客户端断线后按它补齐事件 | 会话内单调递增 |
| `message_id` / `turn_id` | 我们 | 消息主键 / 一轮对话 | 阶段 2 起落库 |

**Q5：不返回工单就结束了吗？**

基本是。判据只有一条：**最后一条消息里还有没有 `tool_calls`**——我们的条件边 `_route` 就是干这个。四个例外：

1. 我们**拒绝**了工单（危险命令），仍会回填"被拒绝"的结果，循环继续，模型会换个做法
2. 模型输出撞到长度上限（`finish_reason` 是 `length`）
3. 模型反复调同一个工具转圈 → 靠 `recursion_limit` 强制刹车
4. 用户中断

**Q6：为什么裸 HTTP 时从没见过工单号？**

因为**没传 `tools`**。工具调用是个开关：

```json
// 请求（关键就是多出来的 tools 数组）
{"model": "deepseek-v4-flash",
 "messages": [{"role": "user", "content": "..."}],
 "tools": [{"type": "function",
            "function": {"name": "fs_read",
                         "description": "读取工作区内的文本文件",
                         "parameters": {"type": "object",
                                        "properties": {"path": {"type": "string"}},
                                        "required": ["path"]}}}]}

// 响应（content 为空，finish_reason 是 tool_calls）
{"choices": [{"finish_reason": "tool_calls",
              "message": {"role": "assistant", "content": null,
                          "tool_calls": [{"id": "call_00_AbC123", "type": "function",
                                          "function": {"name": "fs_read",
                                                       "arguments": "{\"path\": \"src/app.py\"}"}}]}}]}

// 回填
{"role": "tool", "tool_call_id": "call_00_AbC123", "content": "1 | 文件内容……"}
```

三个容易漏掉的细节：

1. **`arguments` 是 JSON 字符串，不是对象**（双重编码）
2. 不调用工具时，响应里**没有 `tool_calls` 这个字段**（不是 `null`），所以翻响应体找不到
3. **流式下工单是碎片**：`arguments` 会被切成 `{"pa` / `th": "..."}`，要拼完才能解析，中途几乎必然不是合法 JSON

**附加三条（都是面试会追的）：**

- **每次请求都要带完整工具清单**：接口无状态，服务器不记得上一次；工具清单本身也占 token 且每轮重复计费，所以工具别贪多
- **参数有两道关**：JSON Schema 是给模型看的说明书，`model_validate` 是守门员（模型会写错类型、漏必填、编造字段）
- **返回值没有协议规定**：协议只要求带 `tool_call_id` 配对，内容是一段字符串——所以**返回值的可读性就是工具设计的质量**（我们在 `shell_exec` 里特意带上 `$ 命令` 和 `(exit=1)`）

想看真实报文，跑 `scripts/dump_raw.py`：它把字节级的请求体、响应体、流式分片全打出来（密钥已打码）。

## 5. 已知限制（阶段 1 不做，后面补）

- 无持久化：会话、事件、图状态都在内存里，进程重启即丢（阶段 2）
- 无审批流：`APPROVAL` 目前等同"拒绝并告知"（阶段 2 接 `interrupt`）
- 无回放与评测集：还没法离线跑回归（阶段 3）
- 无服务端与 SSE：`agent run` 是单进程 CLI，没有对外契约（阶段 4）
- 没有 `search_code` 工具与检索（阶段 5）

## 6. 面试问答速查（本阶段）

| 问题 | 一句话答案 |
| --- | --- |
| agent 和单轮问答的区别 | 有工具与循环：模型请求工具 → 程序执行 → 结果回填 → 再问，直到不再请求工具 |
| 为什么用图而不是 while 循环 | 图把控制流从调用栈搬到数据里，于是"暂停/恢复/观测/分支"才成立；一次性脚本仍用 while 更合适 |
| 图的代价是什么 | 多一层抽象、调试要穿两层、简单场景更啰嗦——所以只在需要暂停/恢复时用 |
| 状态里的 `add_messages` 干什么 | 归约器：节点只返回增量，框架负责合并进列表 |
| 循环怎么停下来 | 模型不再请求工具（条件边到 END），以及 `recursion_limit` 兜底 |
| 工具执行失败怎么办 | 不抛异常，编码成工具结果回填；只有基础设施故障才向上抛 |
| 怎么防止 agent 乱执行命令 | 三级分级 + 危险模式正则拒绝 + 工作区 realpath 边界 + 进程组超时 + 环境变量白名单 |
| 上下文会不会被输出冲爆 | 工具输出头尾各半截断（默认 8KB），全量入库供人排查 |
| 你的项目里框架负责什么 | 编排（状态机/条件边/检查点/中断/流式）；对外契约与安全语义全是自研 |
| 踩过什么坑 | 工具名带点号被 400 拒绝 → 约束前移到 pydantic 校验器并加测试钉住 |
