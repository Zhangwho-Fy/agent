"""嵌入：一个接口，两种后端。

- `HashingEmbedder`：**离线、确定性、无依赖**。把词和字符 n-gram 哈希到固定维度再归一化。
  它不是语义模型，只是"字面重叠"的一种稠密表示，作用是在下不了权重的环境里
  （CI、受限容器）让检索链路能跑、能测。
- `build_embedder`：真模型走这里（fastembed 的 ONNX 后端，装得上就用）。

为什么要自己定协议而不是直接用 LangChain 的 `Embeddings`：项目的检索层只需要
"文本 → 向量"这一件事，接口越小越好换；以后要接 LangChain 的组件，
在外面包一层适配即可，不影响内部实现。
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

#: 一个"词"：连续的字母数字下划线，或连续的 CJK 字符
_WORD = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+")


class Embedder(Protocol):
    """文本 → 向量。`dim` 只用于校验和存储规划。"""

    dim: int
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def _features(text: str) -> list[str]:
    """把文本拆成特征：词 + 中文二元组 + 英文前缀。

    - 词：`resolve_within` 这种标识符整词保留，代码检索主要靠它；
    - 中文二元组：中文没有空格，切成 bigram 才能让"审批"匹配到"审批流"；
    - 标识符的子串：`chunk_file` 也能被 `chunk` 命中（下划线再切一刀）。
    """
    lowered = text.lower()
    features: list[str] = []
    for token in _WORD.findall(lowered):
        if token.isascii():
            features.append(token)
            features.extend(part for part in token.split("_") if len(part) > 2)
        else:
            features.append(token)
            features.extend(token[i : i + 2] for i in range(len(token) - 1))
    return features


class HashingEmbedder:
    """确定性哈希嵌入：同样的文本永远得到同样的向量，不依赖任何外部资源。"""

    name = "offline-hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature in _features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dim
            # 用同一位哈希的另一个 bit 决定符号：避免所有特征同向叠加
            vector[bucket] += 1.0 if (value >> 63) & 1 else -1.0
        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [round(component / norm, 6) for component in vector]


class FastEmbedEmbedder:
    """真模型后端（ONNX，不需要 torch）。装不上就在构造时明确报错。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - 取决于本机装没装
            msg = (
                "使用了 embed_backend=fastembed 但没有安装依赖。"
                "装一次即可（首次运行会下载模型权重）：\n"
                "    uv pip install fastembed\n"
                "想先用离线兜底后端，把 AGENT_EMBED_BACKEND 设成 offline。"
            )
            raise RuntimeError(msg) from exc
        self._model = TextEmbedding(model_name=model_name)
        self.name = f"fastembed:{model_name}"
        self.dim = len(next(iter(self._model.embed(["探测"]))))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:  # pragma: no cover - 需要权重
        return [list(vector) for vector in self._model.embed(list(texts))]


def build_embedder(backend: str = "offline", model_name: str = "") -> Embedder:
    """按配置造嵌入器。默认离线——CI 和沙箱里没有权重可下。"""
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name or "BAAI/bge-small-zh-v1.5")
    if backend == "offline":
        return HashingEmbedder()
    msg = f"未知的嵌入后端：{backend}（可选 offline / fastembed）"
    raise ValueError(msg)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """余弦相似度。两个向量都归一化过时，这就是点积。"""
    return sum(a * b for a, b in zip(left, right, strict=False))
