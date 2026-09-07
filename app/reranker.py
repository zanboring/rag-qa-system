"""重排序（Re-rank）模块：在向量召回的候选集上做二次精排。

面试可讲点：
- RAG 检索是「召回（Recall）+ 精排（Re-rank）」两段式：
  向量检索负责从海量片段中快速召回 Top-N 候选（粗排），
  重排负责在候选集上做更精细的打分，把最相关的片段提到最前。
- 本项目用「Reciprocal Rank Fusion（RRF）」融合两个排序信号：
  ① 向量语义相似度排序；② 词法重叠排序（字符 n-gram 命中率）。
  RRF 不依赖分数绝对值、无需归一化，只按「名次」融合，简单且稳健。
- 生产可用 cross-encoder（如 bge-reranker）做深度语义重排；
  这里用零依赖的词法重排演示「融合 + 精排」思想，可离线、可单测。
"""


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    """把文本拆成字符 n-gram 集合，用于词法重叠计算。"""
    text = text.lower()
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def _lexical_overlap(query: str, doc_text: str, n: int = 2) -> float:
    """计算查询与片段之间的词法重叠系数（值域 [0, 1]）。

    用「重叠 n-gram 数 / min(两方 n-gram 数)」做归一化，
    避免长文档因 n-gram 总数大而在交集计数上天然占优。
    """
    q = _char_ngrams(query, n)
    d = _char_ngrams(doc_text, n)
    if not q or not d:
        return 0.0
    return len(q & d) / min(len(q), len(d))


class Reranker:
    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        raise NotImplementedError


class LexicalReranker(Reranker):
    """基于 RRF 的词法重排器（零依赖、确定性）。

    hits 元素约定：{"id": str, "score": float, "metadata": {"text": str}}
    """

    def __init__(self, ngram: int = 2, rrf_k: int = 60):
        self.ngram = ngram
        self.rrf_k = rrf_k

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if not hits:
            return []

        # 排序信号一：向量语义相似度（降序）
        by_vector = sorted(hits, key=lambda h: h["score"], reverse=True)

        # 排序信号二：词法重叠（降序）
        by_lexical = sorted(
            hits,
            key=lambda h: _lexical_overlap(
                query, h.get("metadata", {}).get("text", ""), self.ngram
            ),
            reverse=True,
        )

        # RRF：对每个片段，累加它在两个排序中的名次倒数
        rrf: dict[str, float] = {}
        for rank, hit in enumerate(by_vector):
            key = hit["id"]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, hit in enumerate(by_lexical):
            key = hit["id"]
            rrf[key] = rrf.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)

        return sorted(hits, key=lambda h: rrf[h["id"]], reverse=True)[:top_k]
