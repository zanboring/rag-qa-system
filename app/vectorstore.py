"""向量存储：内存版（余弦相似度）+ ChromaDB 可选。

面试可讲点：
- 向量检索：把 query 也向量化，与库中每个 chunk 向量算余弦相似度，取 Top-K。
- 内存版是 O(n) 暴力检索，够演示；生产数据量大时用 ChromaDB / Milvus / FAISS
  等专用向量库，靠 ANN（近似最近邻）索引把检索复杂度降到亚线性。
"""

import math

from app import config


class VectorStore:
    async def add(self, id: str, vector: list[float], metadata: dict) -> None:
        raise NotImplementedError

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        raise NotImplementedError

    async def delete(self, ids: list[str]) -> int:
        """删除指定 id 的向量，返回实际删除数量。"""
        raise NotImplementedError


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度：值域 [-1, 1]，越接近 1 越相似。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class MemoryVectorStore(VectorStore):
    """内存向量库，暴力余弦检索。"""

    def __init__(self):
        self._items: dict[str, dict] = {}

    async def add(self, id: str, vector: list[float], metadata: dict) -> None:
        self._items[id] = {"vector": vector, "metadata": metadata}

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        scored = []
        for id, item in self._items.items():
            score = _cosine(vector, item["vector"])
            scored.append((score, id, item["metadata"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": id, "score": score, "metadata": meta}
            for score, id, meta in scored[:top_k]
        ]

    async def delete(self, ids: list[str]) -> int:
        removed = 0
        for id in ids:
            if id in self._items:
                del self._items[id]
                removed += 1
        return removed

    def __len__(self) -> int:
        return len(self._items)


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量库（生产可选，需 `pip install chromadb`）。"""

    def __init__(self, collection_name: str = "rag_kb"):
        import chromadb  # 延迟导入

        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)

    async def add(self, id: str, vector: list[float], metadata: dict) -> None:
        self.collection.add(
            ids=[id], embeddings=[vector], metadatas=[metadata]
        )

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        res = self.collection.query(
            query_embeddings=[vector], n_results=top_k
        )
        hits = []
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for id, meta, dist in zip(ids, metas, dists):
            hits.append(
                {"id": id, "score": 1 - dist, "metadata": meta}
            )  # Chroma 默认欧氏距离，转成类相似度
        return hits

    async def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)


def get_vectorstore() -> VectorStore:
    if config.VECTOR_BACKEND == "chroma":
        return ChromaVectorStore()
    return MemoryVectorStore()
