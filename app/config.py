"""RAG 知识库问答系统 —— 配置模块。

通过环境变量切换后端，默认全部用「零依赖离线版」，
保证本地无 GPU / 无网络 / 无 API Key 也能跑起来和做单测：

- EMBEDDING_BACKEND: hash（确定性字符 n-gram）| bge（本地）| zhipu（云端）
- VECTOR_BACKEND:   memory（内存余弦检索）| chroma（专用向量库）
- LLM_BACKEND:      mock（确定性假模型）| ollama（本地）| glm4（云端）
"""

import os

EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "hash")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "memory")
LLM_BACKEND = os.getenv("LLM_BACKEND", "mock")

# 切片参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "3"))

# HashEmbedder 维度
HASH_EMBED_DIM = int(os.getenv("HASH_EMBED_DIM", "256"))

# Ollama 本地模型
# 变量名兼容 OLLAMA_API_BASE：这是 Ollama 生态里常见的写法，
# 兼容后可自动复用机器上已有的配置，不必再设一份 OLLAMA_BASE_URL。
OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_API_BASE") or "http://localhost:11434"
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# ChromaDB 持久化路径。
# 为空 → 使用纯内存模式，进程退出数据即丢失（适合本地开发与测试）；
# 非空 → 数据落盘到该目录，容器重启后依然可检索（部署时必须设置）。
CHROMA_PATH = os.getenv("CHROMA_PATH", "")

# DeepSeek 云端（OpenAI 兼容协议）。
# 注意：DeepSeek **只提供对话（chat）接口，没有 embedding 接口**。
# 因此它只能用于「生成」和「裁判」两个环节，不能替代向量化后端——
# 需要语义检索能力时，仍须选 bge（本地模型）或 zhipu（云端 embedding）。
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

# 智谱 GLM-4 云端。
# API Key 兼容两种变量名：社区文档常用 ZHIPU_API_KEY，
# 而官方 SDK 与部分安装包写入的是 ZHIPUAI_API_KEY，两个都读以免配置"看起来设了却读不到"。
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY") or os.getenv("ZHIPUAI_API_KEY") or ""
ZHIPU_BASE_URL = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
ZHIPU_EMBED_MODEL = os.getenv("ZHIPU_EMBED_MODEL", "embedding-3")
ZHIPU_CHAT_MODEL = os.getenv("ZHIPU_CHAT_MODEL", "glm-4-flash")
