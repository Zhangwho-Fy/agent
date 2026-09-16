"""运行时配置：从环境变量和 .env 加载，用 pydantic 校验。

对照 C++：相当于一个全局配置结构体 + 解析配置文件的函数，区别是校验规则
直接写在类型上（`port: int = Field(ge=1, le=65535)` 就完成了范围检查），
解析失败会明确指出是哪个字段、哪里不对。

优先级：环境变量 > .env 文件 > 代码里的默认值。
`.env` 从当前工作目录读取，所以命令要在项目根目录下执行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 打日志、打印配置摘要时需要打码的字段
SECRET_FIELDS: frozenset[str] = frozenset({"api_key", "auth_token"})


class Settings(BaseSettings):
    """所有运行时可调项。字段名加 AGENT_ 前缀即为环境变量名。"""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里多出来的键不报错，方便加实验性配置
    )

    # ---- 模型接入 ----
    provider: Literal["deepseek", "replay"] = "deepseek"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = Field(default="", description="模型服务密钥，只从环境读取，绝不入库")

    # ---- 工作区与执行限制 ----
    workspace: Path = Field(default=Path("."), description="code profile 的工作区根目录")
    max_tool_rounds: int = Field(default=12, ge=1, le=100, description="单轮最多几次工具往返")
    tool_timeout_s: float = Field(default=60.0, gt=0, description="单个工具超时（秒）")
    output_limit_bytes: int = Field(default=8192, ge=256, description="工具输出进上下文的截断阈值")
    approval_timeout_s: float = Field(default=120.0, gt=0, description="审批等待上限，超时视为拒绝")

    # ---- 服务 ----
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    auth_token: str = Field(default="", description="客户端调用凭证，空则由服务端生成")
    db_path: Path = Path("~/.local/share/agent/agent.db")
    log_level: str = "info"

    # ---- 派生属性 ----
    @property
    def resolved_workspace(self) -> Path:
        """展开 ~ 并转成绝对路径。工具执行前必须拿到绝对路径做边界检查。"""
        return self.workspace.expanduser().resolve()

    @property
    def resolved_db_path(self) -> Path:
        return self.db_path.expanduser()

    def describe(self) -> dict[str, str]:
        """给人看的配置摘要：敏感字段只报告"有没有设置"，不报告内容。"""
        data = self.model_dump(mode="json")
        for name in SECRET_FIELDS:
            raw = str(data.get(name, ""))
            data[name] = f"<已设置，{len(raw)} 字符>" if raw else "<未设置>"
        return {key: str(value) for key, value in data.items()}
