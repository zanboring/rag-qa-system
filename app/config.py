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
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

# 智谱 GLM-4 云端
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
ZHIPU_EMBED_MODEL = os.getenv("ZHIPU_EMBED_MODEL", "embedding-3")
ZHIPU_CHAT_MODEL = os.getenv("ZHIPU_CHAT_MODEL", "glm-4-flash")
