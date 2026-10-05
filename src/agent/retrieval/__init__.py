"""检索：把仓库切块、建索引、混合检索，并给出可比较的指标。

分工跟项目其余部分一致：**通用基础设施**（切块、向量存取）能用现成的就用，
**检索策略与评测**（混合排序、recall@5 / MRR）自己写——后者才是要展示的部分。

默认后端是离线的确定性嵌入（见 `embeddings.HashingEmbedder`），原因很实际：
CI 和受限沙箱里下不了模型权重。它**不是语义模型**，指标不能代表真模型，
但足以让整条链路可跑、可测、可回归；换成真模型只要改一个配置。
"""

from .chunker import Chunk, chunk_file, iter_source_files
from .embeddings import Embedder, HashingEmbedder, build_embedder
from .index import RetrievalIndex
from .metrics import EvalCase, EvalReport, evaluate
from .search import SearchHit, hybrid_search

__all__ = [
    "Chunk",
    "Embedder",
    "EvalCase",
    "EvalReport",
    "HashingEmbedder",
    "RetrievalIndex",
    "SearchHit",
    "build_embedder",
    "chunk_file",
    "evaluate",
    "hybrid_search",
    "iter_source_files",
]
