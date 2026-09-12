"""RAG 知识库问答系统 —— FastAPI 入口。

接口：
    POST   /documents         上传文档（切片 + 向量化 + 入库）
    POST   /documents/batch   批量上传文档（部分失败不影响整体）
    POST   /query             提问（检索 + 生成 + 溯源）
    GET    /documents         列出已入库文档（支持分页）
    GET    /documents/{doc_id} 获取单个文档元数据
    PUT    /documents/{doc_id} 更新文档（删旧片段 + 重新入库）
    DELETE /documents/{doc_id} 删除文档及全部向量片段
    GET    /health            健康检查 + 当前后端

后端可插拔（改环境变量即可，见 config.py）：
    EMBEDDING_BACKEND=hash|bge|zhipu
    VECTOR_BACKEND=memory|chroma
    LLM_BACKEND=mock|ollama|glm4
"""

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

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
    top_k: int = Field(3, ge=1, le=50, description="检索片段数，上限 50 防止过大召回")
    rerank: bool = False


class DocumentUpdateRequest(BaseModel):
    """文档更新请求：name 与 text 至少传一个（model_dump(exclude_unset=True) 校验）。"""
    name: str | None = None
    text: str | None = None


class BatchDocumentItem(BaseModel):
    """批量上传的单篇文档。"""
    name: str
    text: str


class BatchDocumentRequest(BaseModel):
    """批量上传请求：documents 为非空列表，单次上限见 BATCH_MAX_SIZE。"""
    documents: list[BatchDocumentItem]


# 批量上传单次上限：防止单次请求过大撑爆内存（每篇还要走 embed + 入库）
BATCH_MAX_SIZE = 50


async def _upload_single_document(req: DocumentRequest) -> dict:
    """上传单篇文档的内部实现（POST /documents 与 /documents/batch 共用）。

    返回 {"doc_id": str, "name": str, "chunk_count": int}；
    文本为空时抛 HTTPException(400)，由调用方按批量场景归入 failed 列表。
    """
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


@app.post("/documents")
async def upload_document(req: DocumentRequest):
    return await _upload_single_document(req)


@app.post("/documents/batch")
async def upload_documents_batch(req: BatchDocumentRequest):
    """批量上传文档：单篇失败不影响整体，分别记入 uploaded / failed。

    设计要点：
    - 容错：单篇切片为空 → 记入 failed（带 index/name/error），不抛 500；
      这样调用方一次能上传多篇而不必重试整批。
    - 限制：单次最多 BATCH_MAX_SIZE（=50），超限立即 400，防止大请求撑爆内存。
    - 顺序：uploaded / failed 数组顺序与请求 documents 一致，便于按 index 对账。

    返回结构：
        {
          "uploaded": [{"index": 0, "doc_id": "...", "name": "...", "chunk_count": N}, ...],
          "failed":   [{"index": 2, "name": "...", "error": "..."}, ...]
        }
    """
    if not req.documents:
        raise HTTPException(status_code=400, detail="documents 列表不能为空")
    if len(req.documents) > BATCH_MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"批量上传单次最多 {BATCH_MAX_SIZE} 篇，当前 {len(req.documents)} 篇",
        )

    uploaded: list[dict] = []
    failed: list[dict] = []
    for idx, item in enumerate(req.documents):
        try:
            result = await _upload_single_document(
                DocumentRequest(name=item.name, text=item.text)
            )
            uploaded.append({"index": idx, **result})
        except HTTPException as e:
            failed.append({"index": idx, "name": item.name, "error": e.detail})

    return {"uploaded": uploaded, "failed": failed}


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


@app.put("/documents/{doc_id}")
async def update_document(doc_id: str, req: DocumentUpdateRequest):
    """更新文档内容或名称（至少传一个）。

    策略：**先删旧片段、再按新内容重新入库**——保证 chunks 与 metadata 一致。
    为什么不 in-place 覆盖：片段数量变化时旧 id 会残留，且 name 改动需刷新
    每个片段 metadata.doc_name，走"删 + 重 add"路径最简单一致。
    """
    meta = DOCUMENTS.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    data = req.model_dump(exclude_unset=True)
    if "text" not in data and "name" not in data:
        raise HTTPException(status_code=400, detail="name 和 text 至少传一个")

    # 确定最终 name 与 text
    new_name = data.get("name", meta["name"])
    new_text = data.get("text")
    if new_text is None:
        # 仅改 name：必须真正刷新历史 chunk 的 metadata.doc_name（不能只改元数据字典，否则溯源仍指向旧名）
        # 实现：从向量库取回旧 chunks 的原文 → 删旧 → 重新嵌入 + 入库（新 doc_name）
        pipeline = get_pipeline()
        old_ids = [f"{doc_id}:{i}" for i in range(meta["chunk_count"])]
        old_items = await pipeline.vectorstore.get(old_ids)
        if len(old_items) != len(old_ids):
            # 防御：向量库与元数据不一致时拒绝部分更新
            raise HTTPException(
                status_code=500,
                detail=f"chunk 数量不一致：元数据={len(old_ids)} 向量库={len(old_items)}，请走删旧文档再重新上传的流程",
            )
        await pipeline.vectorstore.delete(old_ids)
        for i, item in enumerate(sorted(old_items, key=lambda x: int(x["id"].split(":")[-1]))):
            text = item["metadata"].get("text", "")
            vec = await pipeline.embedder.embed(text)
            await pipeline.vectorstore.add(
                id=f"{doc_id}:{i}",
                vector=vec,
                metadata={"doc_name": new_name, "chunk_index": i, "text": text},
            )
        meta["name"] = new_name
        return {"doc_id": doc_id, **meta, "refreshed_chunks": len(old_items)}

    # 传了 text：走"删旧 + 入新"标准流程
    chunks = chunk_text(new_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档内容为空")

    pipeline = get_pipeline()
    old_ids = [f"{doc_id}:{i}" for i in range(meta["chunk_count"])]
    await pipeline.vectorstore.delete(old_ids)

    for i, chunk in enumerate(chunks):
        vec = await pipeline.embedder.embed(chunk)
        await pipeline.vectorstore.add(
            id=f"{doc_id}:{i}",
            vector=vec,
            metadata={"doc_name": new_name, "chunk_index": i, "text": chunk},
        )

    meta["name"] = new_name
    meta["chunk_count"] = len(chunks)
    meta.pop("name_stale_in_chunks", None)
    return {"doc_id": doc_id, **meta}


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
