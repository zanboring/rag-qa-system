"""RAG 知识库问答系统 —— FastAPI 入口。

接口：
    POST /documents         上传文档（切片 + 向量化 + 入库）
    POST /query             提问（检索 + 生成 + 溯源）
    GET  /documents         列出已入库文档（支持分页）
    GET  /documents/{doc_id} 获取单个文档元数据
    DELETE /documents/{doc_id} 删除文档及全部向量片段
    GET  /health            健康检查 + 当前后端

后端可插拔（改环境变量即可，见 config.py）：
    EMBEDDING_BACKEND=hash|bge|zhipu
    VECTOR_BACKEND=memory|chroma
    LLM_BACKEND=mock|ollama|glm4
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from app import config
from app.chunker import chunk_text
from app.embedder import get_embedder
from app.llm import get_llm
from app.rag import RAGPipeline
from app.vectorstore import get_vectorstore

# 文档注册表：doc_id -> {name, chunk_count, created_at}
# 内存存储（演示用），生产可换数据库或文档存储服务
DOCUMENTS: dict[str, dict] = {}

# 列表分页默认值
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# 懒加载单例：避免在 import 时就构建（也便于测试重置）
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(get_embedder(), get_vectorstore(), get_llm())
    return _pipeline


def reset_pipeline() -> None:
    """测试辅助：重置管道与文档表，保证每个测试独立。"""
    global _pipeline
    _pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_pipeline()  # 启动时预热管道
    yield


app = FastAPI(title="RAG 知识库问答系统", version="1.0.0", lifespan=lifespan)


class DocumentRequest(BaseModel):
    name: str
    text: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    rerank: bool = False


@app.post("/documents")
async def upload_document(req: DocumentRequest):
    chunks = chunk_text(req.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档内容为空")

    pipeline = get_pipeline()
    doc_id = uuid.uuid4().hex
    for i, chunk in enumerate(chunks):
        vec = await pipeline.embedder.embed(chunk)
        await pipeline.vectorstore.add(
            id=f"{doc_id}:{i}",
            vector=vec,
            metadata={"doc_name": req.name, "chunk_index": i, "text": chunk},
        )
    DOCUMENTS[doc_id] = {
        "name": req.name,
        "chunk_count": len(chunks),
        "created_at": int(time.time()),
    }
    return {"doc_id": doc_id, "name": req.name, "chunk_count": len(chunks)}


@app.post("/query")
async def query(req: QueryRequest):
    pipeline = get_pipeline()
    return await pipeline.ask(req.question, req.top_k, req.rerank)


@app.get("/documents")
async def list_documents(
    page: int = Query(1, ge=1, description="页码，从 1 开始"),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description="每页条数"),
):
    """分页列出已入库文档，按入库时间倒序。

    返回格式：
        {
          "total": 总条数,
          "page": 当前页,
          "page_size": 每页条数,
          "items": [ {doc_id, name, chunk_count, created_at}, ... ]
        }
    """
    all_items = [
        {"doc_id": k, **v} for k, v in DOCUMENTS.items()
    ]
    # 按 created_at 倒序，created_at 缺失的兜底为 0
    all_items.sort(key=lambda x: x.get("created_at", 0), reverse=True)

    total = len(all_items)
    start = (page - 1) * page_size
    end = start + page_size
    items = all_items[start:end]

    return {"total": total, "page": page, "page_size": page_size, "items": items}


@app.get("/documents/{doc_id}")
async def get_document(doc_id: str):
    """获取单个文档的元数据（不返回向量数据）。"""
    meta = DOCUMENTS.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"doc_id": doc_id, **meta}


@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """删除指定文档及其全部向量片段。

    片段 id 约定为 `{doc_id}:{chunk_index}`，删除时按此规则重建 id 列表，
    从向量库与文档注册表两处一并清除，保证查询不再命中该文档。
    """
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="文档不存在")

    chunk_count = DOCUMENTS[doc_id]["chunk_count"]
    pipeline = get_pipeline()
    ids = [f"{doc_id}:{i}" for i in range(chunk_count)]
    removed = await pipeline.vectorstore.delete(ids)

    del DOCUMENTS[doc_id]
    return {"doc_id": doc_id, "removed_chunks": removed}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backends": {
            "embedding": config.EMBEDDING_BACKEND,
            "vector": config.VECTOR_BACKEND,
            "llm": config.LLM_BACKEND,
        },
    }
