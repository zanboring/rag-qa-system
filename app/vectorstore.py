"""向量存储：内存版（余弦相似度）+ ChromaDB 可选。

面试可讲点：
- 向量检索：把 query 也向量化，与库中每个 chunk 向量算余弦相似度，取 Top-K。
- 内存版是 O(n) 暴力检索，够演示；生产数据量大时用 ChromaDB / Milvus / FAISS
  等专用向量库，靠 ANN（近似最近邻）索引把检索复杂度降到亚线性。
"""

import asyncio
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

    async def get(self, ids: list[str]) -> list[dict]:
        """按 id 批量取回 {vector, metadata}，用于改名/重建索引时获取原文。

        关键修复：旧版没有 get，导致 PUT 改 name 时无法刷新历史 chunk 的 metadata.doc_name。
        """
        raise NotImplementedError

    async def list_all(self) -> list[dict]:
        """列出库中全部片段（含 metadata）。

        用途：启动时从向量库**反向重建文档注册表**，让向量库成为文档元数据的
        单一事实来源，避免"文档表放内存、向量放磁盘"导致重启后两者不一致
        （表现为：检索还能命中，但文档列表却是空的）。
        """
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
            # 过滤负相似度：语义反向的片段对回答无帮助，不应出现在 Top-K 中
            if score > 0:
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

    async def get(self, ids: list[str]) -> list[dict]:
        """取回指定 id 的 {vector, metadata}，保持原顺序；不存在的 id 跳过。"""
        return [
            {"id": id, "vector": self._items[id]["vector"], "metadata": self._items[id]["metadata"]}
            for id in ids if id in self._items
        ]

    def __len__(self) -> int:
        return len(self._items)


class ChromaVectorStore(VectorStore):
    """ChromaDB 向量库（生产可选，需 `pip install chromadb`）。

    两个必须注意的实现细节（都属于"默认行为陷阱"）：

    1. **必须显式指定余弦空间**。Chroma 创建 collection 时默认距离度量为 L2（欧氏距离）。
       若沿用默认值，query 返回的 `distances` 是欧氏距离，而本项目其它后端（内存库）
       用的都是余弦相似度，两者的**数值量纲完全不同**：

           L2 距离 = 0.5   ⇒  实际余弦相似度 ≈ 0.875，而 `1 - 0.5 = 0.5`

       后果是 `score` 字段失去一致语义，任何依赖 score 绝对值的下游计算
       （如置信度归一化、固定阈值判拒答）都会失真。指定 cosine 空间后，
       `distance` 即为 `1 - 余弦相似度`，`1 - distance` 才真正等于余弦相似度，
       与内存库的量纲对齐。

    2. **同步方法必须包进线程**。Chroma 的 Python 客户端是同步阻塞 API，
       直接在 async 函数里调用会阻塞事件循环，使整个服务在查询期间无法响应其它请求。
       这里统一用 asyncio.to_thread 卸载到线程池。
    """

    def __init__(self, collection_name: str = "rag_kb", persist_path: str | None = None):
        import chromadb  # 延迟导入：未安装 chromadb 时不影响其它后端

        if persist_path:
            # 持久化模式：数据落盘，进程/容器重启后仍可检索——部署时的必选项
            self.client = chromadb.PersistentClient(path=persist_path)
        else:
            # 纯内存模式：进程退出即丢，适合本地开发与单元测试
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(self, id: str, vector: list[float], metadata: dict) -> None:
        await asyncio.to_thread(
            self.collection.add, ids=[id], embeddings=[vector], metadatas=[metadata]
        )

    async def search(self, vector: list[float], top_k: int) -> list[dict]:
        res = await asyncio.to_thread(
            self.collection.query, query_embeddings=[vector], n_results=top_k
        )
        hits = []
        ids = res.get("ids", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for id, meta, dist in zip(ids, metas, dists):
            # 已指定 cosine 空间：distance = 1 - cosine_similarity
            hits.append({"id": id, "score": 1 - dist, "metadata": meta})
        return hits

    async def delete(self, ids: list[str]) -> int:
        """删除指定 id，返回**实际删除数**（与 MemoryVectorStore 语义一致）。

        Chroma 的 delete 对不存在的 id 静默忽略，直接返回 len(ids) 会虚报删除数量，
        导致调用方（如 PUT 更新时的日志、DELETE 接口的 removed_chunks）拿到错误数字。
        因此先 get 一次确认真正存在的 id。
        """
        if not ids:
            return 0
        existing = await asyncio.to_thread(self.collection.get, ids=ids)
        existing_ids = existing.get("ids", []) or []
        if not existing_ids:
            return 0
        await asyncio.to_thread(self.collection.delete, ids=existing_ids)
        return len(existing_ids)

    async def get(self, ids: list[str]) -> list[dict]:
        """从 Chroma 取回 {vector, metadata}。

        include 必须显式包含 "embeddings"：Chroma 的 get 默认不返回向量，
        遗漏时 `res["embeddings"]` 为 None，后续重新入库会写入空向量。
        """
        if not ids:
            return []
        res = await asyncio.to_thread(
            self.collection.get, ids=ids, include=["embeddings", "metadatas"]
        )
        out = []
        for id, emb, meta in zip(
            res.get("ids", []), res.get("embeddings") or [], res.get("metadatas") or []
        ):
            out.append({"id": id, "vector": emb, "metadata": meta})
        return out


def get_vectorstore() -> VectorStore:
    if config.VECTOR_BACKEND == "chroma":
        # CHROMA_PATH 非空时启用持久化，让容器重启后数据不丢
        return ChromaVectorStore(persist_path=config.CHROMA_PATH or None)
    return MemoryVectorStore()
