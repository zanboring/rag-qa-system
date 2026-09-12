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

import asyncio
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


async def _close_backends(pipeline: RAGPipeline) -> None:
    """关闭 pipeline 各后端持有的资源（HTTP 连接池等）。

    单个后端关闭失败不应阻断其它后端，也不应影响服务退出流程，
    因此逐个 try/except 吞掉异常——关闭阶段的报错没有可做的补救动作。
    """
    for target in (pipeline.embedder, pipeline.llm):
        close = getattr(target, "aclose", None)
        if close is None:
            continue
        try:
            await close()
        except Exception:  # noqa: BLE001 — 关闭失败不应中断退出
            pass


async def _restore_documents() -> int:
    """启动时从向量库反向重建文档注册表，返回恢复的文档数。

    为什么需要重建：DOCUMENTS 是进程内存字典，而向量库可以持久化到磁盘
    （如 Chroma 的 PersistentClient）。若只持久化向量，重启后会出现
    「向量还在、检索能命中，但文档列表是空的」这种不一致状态，
    用户会以为数据丢了。

    解决办法是把向量库当作文档元数据的**单一事实来源**，启动时反向聚合重建，
    从根本上消除双写不一致。因此上传时把 created_at 一并写进了每个 chunk 的 metadata。

    重建失败（后端未实现 list_all、或磁盘不可读）不应阻断服务启动：
    服务仍能正常提供检索能力，只是文档列表暂不可用。
    """
    pipeline = get_pipeline()
    try:
        items = await pipeline.vectorstore.list_all()
    except Exception:  # noqa: BLE001 — 重建失败不阻断启动（含未实现 list_all 的后端）
        return 0

    restored: dict[str, dict] = {}
    for item in items:
        meta = item.get("metadata") or {}
        chunk_id = item.get("id") or ""
        # 片段 id 约定为 {doc_id}:{chunk_index}，用 rsplit 取回 doc_id
        doc_id = chunk_id.rsplit(":", 1)[0] if ":" in chunk_id else chunk_id
        if not doc_id:
            continue
        entry = restored.setdefault(
            doc_id,
            {
                "name": meta.get("doc_name", "未知"),
                "chunk_count": 0,
                "created_at": int(meta.get("created_at", 0) or 0),
            },
        )
        entry["chunk_count"] += 1

    DOCUMENTS.clear()
    DOCUMENTS.update(restored)
    return len(restored)


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline = get_pipeline()  # 启动时预热管道
    restored = await _restore_documents()
    if restored:
        # 放到 app.state 上便于观测，也让测试可以断言恢复行为
        app.state.restored_documents = restored
    try:
        yield
    finally:
        # 释放后端连接池。缺少这一步时，Zhipu / Ollama 后端的连接会驻留到进程退出，
        # 在测试反复建管道、或服务频繁重启的场景下会累积占用连接资源。
        await _close_backends(pipeline)


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

# 单篇文档内部并发向量化的上限。
# 为什么要并发：embed 对云端 API 是 IO 密集调用，串行 await 会让上传耗时随片段数
# 线性增长；并发能显著缩短单篇上传的响应时间。
# 为什么要限制：并发过高会触发服务端限流（429），对本地模型则会造成线程数失控。
# 取 8 是吞吐与限流风险之间的折中。
EMBED_CONCURRENCY = 8


async def _embed_chunks(pipeline: RAGPipeline, chunks: list[str]) -> list[list[float]]:
    """受控并发地对一批片段做向量化，返回与输入**等长且同序**的向量列表。

    抽成独立函数的原因：上传与更新两条路径都需要批量向量化，且更新路径必须
    「先算完向量再删旧数据」（见 update_document 的说明）。把这个顺序约束
    集中在一处实现，避免两处逻辑各自演化后走偏。

    asyncio.gather 保证返回顺序与输入顺序一致，因此调用方可以安全地用
    zip(chunks, vectors) 恢复 chunk_index 与原文位置的对应关系。
    """
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)

    async def embed_one(chunk: str) -> list[float]:
        async with semaphore:
            return await pipeline.embedder.embed(chunk)

    return list(await asyncio.gather(*(embed_one(chunk) for chunk in chunks)))


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
    created_at = int(time.time())

    vectors = await _embed_chunks(pipeline, chunks)
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        await pipeline.vectorstore.add(
            id=f"{doc_id}:{i}",
            vector=vec,
            metadata={
                "doc_name": req.name,
                "chunk_index": i,
                "text": chunk,
                # created_at 冗余写进每个 chunk 的 metadata，使向量库成为文档元数据的
                # 单一事实来源——重启后可据此完整重建文档注册表（含入库时间）。
                "created_at": created_at,
            },
        )
    DOCUMENTS[doc_id] = {
        "name": req.name,
        "chunk_count": len(chunks),
        "created_at": created_at,
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

    策略：**先算出新向量 → 再删旧片段 → 最后写入新片段**，保证 chunks 与 metadata 一致。
    为什么不 in-place 覆盖：片段数量变化时旧 id 会残留，且 name 改动需刷新
    每个片段 metadata.doc_name，走"重算 + 删 + 重 add"路径最简单一致。

    为什么把「算向量」放在「删旧」之前：embed 是可能失败的外部调用
    （网络抖动、模型加载异常、配额耗尽）。若先删旧再逐个 embed，
    一旦中途失败，旧数据已被清除而新数据尚未写入，该文档会永久丢失且无法恢复。
    先算后删把失败窗口收窄到"只读"操作，失败时旧数据完好无损。
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
        # 先按原顺序取出全部原文并完成向量化，成功后再删旧片段。
        # 顺序不可颠倒：若先删后 embed，一旦 embed 失败会同时丢掉旧数据和新数据。
        ordered = sorted(old_items, key=lambda x: int(x["id"].split(":")[-1]))
        texts = [item["metadata"].get("text", "") for item in ordered]
        vectors = await _embed_chunks(pipeline, texts)

        await pipeline.vectorstore.delete(old_ids)
        for i, (text, vec) in enumerate(zip(texts, vectors)):
            await pipeline.vectorstore.add(
                id=f"{doc_id}:{i}",
                vector=vec,
                metadata={
                    "doc_name": new_name,
                    "chunk_index": i,
                    "text": text,
                    # 保留原入库时间：改名不应改变文档的创建时间
                    "created_at": meta.get("created_at", 0),
                },
            )
        meta["name"] = new_name
        return {"doc_id": doc_id, **meta, "refreshed_chunks": len(old_items)}

    # 传了 text：走"删旧 + 入新"标准流程
    chunks = chunk_text(new_text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="文档内容为空")

    pipeline = get_pipeline()

    # 顺序很关键：先在内存中完成全部向量化，成功后再删除旧片段。
    # 若反过来（先删旧、再逐个 embed），一旦 embed 抛错（网络抖动、模型异常、配额耗尽），
    # 旧数据已被清除而新数据未写入，该文档会永久丢失且无法从向量库恢复。
    vectors = await _embed_chunks(pipeline, chunks)

    old_ids = [f"{doc_id}:{i}" for i in range(meta["chunk_count"])]
    await pipeline.vectorstore.delete(old_ids)

    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        await pipeline.vectorstore.add(
            id=f"{doc_id}:{i}",
            vector=vec,
            metadata={
                "doc_name": new_name,
                "chunk_index": i,
                "text": chunk,
                # 保留原入库时间：更新内容不应改变文档在列表中的创建时间
                "created_at": meta.get("created_at", 0),
            },
        )

    meta["name"] = new_name
    meta["chunk_count"] = len(chunks)
    meta.pop("name_stale_in_chunks", None)
    return {"doc_id": doc_id, **meta}


@app.get("/health")
async def health():
    """健康检查。

    除了存活状态，还回显当前生效的后端与关键参数。排查"为什么线上行为和预期不符"时，
    第一件事就是确认实例实际加载的是哪套配置——环境变量没生效是最常见的原因。
    """
    return {
        "status": "ok",
        "backends": {
            "embedding": config.EMBEDDING_BACKEND,
            "vector": config.VECTOR_BACKEND,
            "llm": config.LLM_BACKEND,
        },
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        # 向量库是否真正持久化。两个条件必须同时满足：后端选了 chroma 且配置了落盘路径。
        # 只看 CHROMA_PATH 会在 VECTOR_BACKEND=memory 时误报 true，让人误以为数据不会丢。
        "vector_persistent": config.VECTOR_BACKEND == "chroma" and bool(config.CHROMA_PATH),
        "documents": len(DOCUMENTS),
    }
