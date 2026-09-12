"""Embedding 向量化模块。

可选后端（改环境变量 EMBEDDING_BACKEND 切换）：
    hash    字符 n-gram 哈希，零依赖、确定性，用于离线演示与单测
    bge     本地 BGE 模型（需 sentence-transformers）
    ollama  本地 Ollama embedding 模型（需 ollama serve，默认 bge-m3）
    zhipu   智谱云端 embedding API

面试可讲点：
- Embedding 把文本映射成高维向量，语义相近的文本向量距离更近。
- 默认用 hash 是为了「零依赖可离线」，它做的是**词形匹配而非语义匹配**；
  生产必须换真实语义模型，否则同义改写类的查询会大量漏召回。
- 换模型只需改环境变量，符合「依赖注入 / 可插拔」的设计思想。
- **选模型必须用目标语言验证**：实测 nomic-embed-text 英文正常、中文失效，
  详见 config.OLLAMA_EMBED_MODEL 的说明。
"""

import hashlib
import math

import httpx

from app import config


class Embedder:
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    async def aclose(self) -> None:
        """释放后端持有的资源（如 HTTP 连接池）。

        默认空实现：本地模型与哈希实现不持有外部连接，无需清理。
        持有 httpx.AsyncClient 的后端必须覆写，否则连接池会一直驻留到进程退出。
        """
        return None


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
        # SentenceTransformer.encode 是同步调用，包 asyncio.to_thread 避免阻塞事件循环
        import asyncio
        return await asyncio.to_thread(lambda: self.model.encode(text).tolist())


class OllamaEmbedder(Embedder):
    """Ollama 本地 embedding 模型（需 `ollama serve` 并已 pull 模型）。

    适用场景：数据不能出本地，同时又要真实的语义检索能力——
    这是「零依赖 hash」与「云端 API」之外的第三条路。

    模型选择的坑：**必须用目标语言的语料验证**。
    实测 nomic-embed-text 在英文上区分度良好（同义 0.84 / 无关 0.36），
    但在中文上完全失效（同义 0.67 反而低于无关 0.71）——
    只看模型知名度选型，会在中文场景得到一个"越不相关越相似"的检索器。
    中文建议 bge-m3 / bge-large-zh 这类多语言或中文专用模型。
    """

    def __init__(self, base_url: str, model: str = "bge-m3", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # 复用连接池；本地推理耗时波动大，超时给宽一些
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        # 用新版 /api/embed 接口：input 接受字符串或字符串数组，
        # 响应统一为 embeddings 数组，便于后续扩展为批量调用。
        resp = await self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": [text]},
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]


class ZhipuEmbedder(Embedder):
    """智谱云端 Embedding（需 ZHIPU_API_KEY），生产可选。"""

    def __init__(self, api_key: str, model: str = "embedding-3"):
        if not api_key:
            raise ValueError("使用 zhipu embedding 后端需设置 ZHIPU_API_KEY 环境变量")
        self.api_key = api_key
        self.model = model
        # 复用连接池：避免每次请求都建立新 TCP 连接。
        # 注意配套的 aclose()：连接池不关闭会一直占用文件描述符与连接资源。
        self._client = httpx.AsyncClient(timeout=60)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        resp = await self._client.post(
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
    if config.EMBEDDING_BACKEND == "ollama":
        return OllamaEmbedder(config.OLLAMA_BASE_URL, config.OLLAMA_EMBED_MODEL)
    if config.EMBEDDING_BACKEND == "zhipu":
        return ZhipuEmbedder(config.ZHIPU_API_KEY, config.ZHIPU_EMBED_MODEL)
    return HashEmbedder(config.HASH_EMBED_DIM)
