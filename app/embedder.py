"""Embedding 向量化模块。

面试可讲点：
- Embedding 把文本映射成高维向量，语义相近的文本向量距离更近。
- 本项目默认用「字符 n-gram 哈希」做一个确定性、零依赖的 Embedding，
  用于离线演示和单测；生产可切换到 BGE / 智谱 embedding 等真实模型。
- 换模型只需改环境变量，符合「依赖注入 / 可插拔」的设计思想。
"""

import hashlib
import math

import httpx

from app import config


class Embedder:
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class HashEmbedder(Embedder):
    """基于字符 n-gram 哈希的确定性 Embedding（零依赖、可离线跑）。

    原理：把文本拆成字符 n-gram，每个 n-gram 哈希到固定维度桶并累加，
    得到稀疏向量后做 L2 归一化。相似文本共享较多 n-gram，
    因此余弦相似度能粗略反映文本相关性——足够演示 RAG 检索链路。
    """

    def __init__(self, dim: int = 256, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram

    async def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        text = text.lower()
        # 首尾加哨兵字符，保留边界 n-gram 信息
        padded = " " * (self.ngram - 1) + text + " " * (self.ngram - 1)
        for i in range(len(padded) - self.ngram + 1):
            gram = padded[i : i + self.ngram]
            h = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % self.dim] += 1.0
        # L2 归一化，便于用余弦相似度比较
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class BgeEmbedder(Embedder):
    """本地 BGE 模型（需 `pip install sentence-transformers`），生产可选。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        from sentence_transformers import SentenceTransformer  # 延迟导入

        self.model = SentenceTransformer(model_name)

    async def embed(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()


class ZhipuEmbedder(Embedder):
    """智谱云端 Embedding（需 ZHIPU_API_KEY），生产可选。"""

    def __init__(self, api_key: str, model: str = "embedding-3"):
        self.api_key = api_key
        self.model = model

    async def embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://open.bigmodel.cn/api/paas/v4/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]


def get_embedder() -> Embedder:
    """按配置返回 Embedder 实例。"""
    if config.EMBEDDING_BACKEND == "bge":
        return BgeEmbedder()
    if config.EMBEDDING_BACKEND == "zhipu":
        return ZhipuEmbedder(config.ZHIPU_API_KEY, config.ZHIPU_EMBED_MODEL)
    return HashEmbedder(config.HASH_EMBED_DIM)
