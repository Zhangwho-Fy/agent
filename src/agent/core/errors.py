"""异常层次。

原则：**可预期的失败不抛异常，编码成工具结果回填给模型**（工具超时、非零退出、
参数不合法都属于这一类）；只有基础设施级故障（网络中断、磁盘写不进）才向上抛。
"""

from __future__ import annotations


class AgentError(Exception):
    """本项目所有自定义异常的基类。"""


class PathEscapeError(AgentError):
    """路径解析后落在工作区之外（包含软链接逃逸、`../` 逃逸）。"""


class ToolExecutionError(AgentError):
    """工具执行过程中的基础设施故障（不是"命令返回非零"）。"""


class ToolNotFoundError(AgentError):
    """模型请求了不存在的工具。"""
