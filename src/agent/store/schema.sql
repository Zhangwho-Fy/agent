-- 阶段 2 的事实源。全部用 IF NOT EXISTS，重复执行没有副作用。
--
-- 两张身份说明：
--   events   —— 只追加的事实源（source of truth）：可重放、可评测
--   messages —— 从 events 投影出来的物化视图：删掉可以从事件日志重建
--
-- 时间一律存 UTC 的 ISO 字符串（见 db.now_iso），可读且可直接比较。

CREATE TABLE IF NOT EXISTS sessions (
  id          TEXT PRIMARY KEY,          -- sess_ + uuid4 hex
  title       TEXT NOT NULL DEFAULT '',
  profile     TEXT NOT NULL,             -- chat | code
  workspace   TEXT,                      -- 工作区根目录（code profile 必填）
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL REFERENCES sessions(id),
  seq          INTEGER NOT NULL,         -- 会话内单调递增
  role         TEXT NOT NULL,            -- system | user | assistant | tool
  content      TEXT NOT NULL,
  tool_call_id TEXT,
  created_at   TEXT NOT NULL,
  UNIQUE (session_id, seq)
);

-- 只追加，永不更新：出问题时要能回答"当时到底发生了什么"
CREATE TABLE IF NOT EXISTS events (
  id          TEXT PRIMARY KEY,
  session_id  TEXT NOT NULL REFERENCES sessions(id),
  turn_id     TEXT,
  seq         INTEGER NOT NULL,          -- 会话内单调递增
  type        TEXT NOT NULL,
  data        TEXT NOT NULL,             -- JSON
  created_at  TEXT NOT NULL,
  UNIQUE (session_id, seq)
);

CREATE TABLE IF NOT EXISTS turns (
  id            TEXT PRIMARY KEY,
  session_id    TEXT NOT NULL REFERENCES sessions(id),
  status        TEXT NOT NULL,           -- running | done | failed | interrupted
  input_tokens  INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  started_at    TEXT NOT NULL,
  ended_at      TEXT
);

CREATE TABLE IF NOT EXISTS tool_calls (
  id          TEXT PRIMARY KEY,          -- 模型给出的 call_id
  session_id  TEXT NOT NULL REFERENCES sessions(id),
  turn_id     TEXT NOT NULL REFERENCES turns(id),
  name        TEXT NOT NULL,
  args        TEXT NOT NULL,             -- JSON
  tier        TEXT NOT NULL,             -- read | write | dangerous
  decision    TEXT,                      -- auto | allow | deny | timeout
  status      TEXT NOT NULL,             -- pending | running | ok | error | timeout | denied
  result      TEXT,                      -- 全量输出（模型只看到截断版）
  exit_code   INTEGER,
  started_at  TEXT,
  ended_at    TEXT
);

CREATE TABLE IF NOT EXISTS idempotency (
  key         TEXT PRIMARY KEY,          -- 客户端提供的幂等键
  session_id  TEXT NOT NULL REFERENCES sessions(id),
  turn_id     TEXT NOT NULL,
  created_at  TEXT NOT NULL
);

-- 查询用的索引。会话内的 seq 查询已经被 UNIQUE 约束的隐式索引覆盖，
-- 这里只补那些按"非首列"过滤的路径。
CREATE INDEX IF NOT EXISTS idx_turns_session    ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_status     ON turns(status);
CREATE INDEX IF NOT EXISTS idx_tool_calls_turn  ON tool_calls(turn_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);
