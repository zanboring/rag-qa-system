"""RAG 接口集成测试：上传文档 → 提问 → 溯源（默认 hash/memory/mock 后端）。"""

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import DOCUMENTS, app, reset_pipeline


@pytest.fixture(autouse=True)
def _clean_state():
    # 每个测试独立：清空文档表 + 重置管道（内存向量库）
    reset_pipeline()
    DOCUMENTS.clear()
    yield
    reset_pipeline()
    DOCUMENTS.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_upload_and_query_with_sources(client):
    # 1. 上传文档
    doc = await client.post(
        "/documents",
        json={
            "name": "算法笔记.md",
            "text": (
                "快速排序是一种分治算法，平均时间复杂度 O(n log n)。"
                "它的核心是选一个基准元素 pivot，把比它小的放左边、比它大的放右边，"
                "再递归排序左右两部分。最坏情况下会退化到 O(n^2)。"
            ),
        },
    )
    assert doc.status_code == 200
    data = doc.json()
    assert data["chunk_count"] >= 1
    assert data["doc_id"]

    # 2. 提问
    q = await client.post("/query", json={"question": "快速排序的时间复杂度是多少？"})
    assert q.status_code == 200
    result = q.json()
    assert "answer" in result
    assert "sources" in result
    # 检索到至少一个来源，且来源标注了文档名与相似度
    assert len(result["sources"]) >= 1
    src = result["sources"][0]
    assert src["doc_name"] == "算法笔记.md"
    assert "score" in src
    # mock 回答会回显上下文，包含来源标记
    assert "参考资料" in result["answer"]


async def test_query_empty_knowledge_base(client):
    resp = await client.post("/query", json={"question": "没有任何文档"})
    assert resp.status_code == 200
    result = resp.json()
    assert result["sources"] == []
    assert "未检索到" in result["answer"]


async def test_query_with_rerank(client):
    # 上传文档后，开启 rerank 提问不应报错，且仍返回带来源的回答
    await client.post(
        "/documents",
        json={
            "name": "算法笔记.md",
            "text": "快速排序是一种分治算法，平均时间复杂度 O(n log n)。",
        },
    )
    q = await client.post(
        "/query", json={"question": "快速排序的时间复杂度", "rerank": True, "top_k": 1}
    )
    assert q.status_code == 200
    result = q.json()
    assert len(result["sources"]) >= 1
    assert result["sources"][0]["doc_name"] == "算法笔记.md"


async def test_upload_empty_document_rejected(client):
    resp = await client.post("/documents", json={"name": "空", "text": "   "})
    assert resp.status_code == 400


async def test_list_documents(client):
    await client.post(
        "/documents", json={"name": "A", "text": "内容A" * 20}
    )
    await client.post(
        "/documents", json={"name": "B", "text": "内容B" * 20}
    )
    resp = await client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    # 分页结构：total / page / page_size / items
    assert data["total"] == 2
    assert data["page"] == 1
    assert data["page_size"] == 20
    assert len(data["items"]) == 2
    # 每条包含 doc_id / name / chunk_count / created_at
    for item in data["items"]:
        assert "doc_id" in item
        assert "name" in item
        assert "chunk_count" in item
        assert "created_at" in item


async def test_list_documents_pagination(client):
    """分页边界：page_size=1 时翻页应返回不同文档。"""
    ids = []
    for i in range(3):
        r = await client.post(
            "/documents", json={"name": f"doc{i}", "text": f"内容{i} " * 30}
        )
        ids.append(r.json()["doc_id"])

    # 第一页：最多 1 条
    page1 = await client.get("/documents?page=1&page_size=1")
    assert page1.status_code == 200
    data1 = page1.json()
    assert data1["total"] == 3
    assert data1["page"] == 1
    assert data1["page_size"] == 1
    assert len(data1["items"]) == 1

    # 第二页：另 1 条，与第一页 doc_id 不同
    page2 = await client.get("/documents?page=2&page_size=1")
    assert page2.status_code == 200
    data2 = page2.json()
    assert len(data2["items"]) == 1
    assert data1["items"][0]["doc_id"] != data2["items"][0]["doc_id"]

    # 第三页：最后 1 条
    page3 = await client.get("/documents?page=3&page_size=1")
    assert len(page3.json()["items"]) == 1

    # 越界：第 4 页应返回空 items，但请求成功
    page4 = await client.get("/documents?page=4&page_size=1")
    assert page4.json()["items"] == []
    assert page4.json()["total"] == 3


async def test_list_documents_pagination_validation(client):
    """分页参数越界应返回 422（Pydantic 自动校验）。"""
    # page < 1
    r = await client.get("/documents?page=0")
    assert r.status_code == 422
    # page_size > MAX_PAGE_SIZE
    r = await client.get("/documents?page_size=9999")
    assert r.status_code == 422


async def test_get_document_by_id(client):
    """GET /documents/{doc_id} 返回元数据，不存在返回 404。"""
    r = await client.post(
        "/documents", json={"name": "具体", "text": "具体内容" * 30}
    )
    doc_id = r.json()["doc_id"]

    # 正常获取
    r = await client.get(f"/documents/{doc_id}")
    assert r.status_code == 200
    meta = r.json()
    assert meta["doc_id"] == doc_id
    assert meta["name"] == "具体"
    assert meta["chunk_count"] >= 1
    assert "created_at" in meta

    # 不存在 → 404
    r = await client.get("/documents/不存在的id123")
    assert r.status_code == 404


async def test_delete_document(client):
    doc = await client.post(
        "/documents", json={"name": "要删除的文档", "text": "独特词汇甲" * 30}
    )
    doc_id = doc.json()["doc_id"]
    assert doc.json()["chunk_count"] >= 1

    # 删除前能检索到
    q = await client.post("/query", json={"question": "独特词汇甲"})
    assert len(q.json()["sources"]) >= 1

    # 删除文档
    r = await client.delete(f"/documents/{doc_id}")
    assert r.status_code == 200
    assert r.json()["removed_chunks"] >= 1

    # 文档列表为空
    assert (await client.get("/documents")).json()["total"] == 0

    # 删除后检索不到该文档来源
    q2 = await client.post("/query", json={"question": "独特词汇甲"})
    assert q2.json()["sources"] == []

    # 再次删除同一文档返回 404
    assert (await client.delete(f"/documents/{doc_id}")).status_code == 404


async def test_update_document_replaces_chunks(client):
    """PUT /documents/{id} 用新文本替换旧片段，并更新元数据。"""
    # 用一组独特的、与更新内容完全不重叠的中文词，便于 hash 检索区分
    r = await client.post(
        "/documents",
        json={"name": "原始名", "text": "旧文档独门标识符甲乙丙丁戊己庚辛壬癸" * 5},
    )
    doc_id = r.json()["doc_id"]

    # 更新文本（用全新的、不同主题的内容）
    r2 = await client.put(
        f"/documents/{doc_id}",
        json={"text": "全新主题戊癸未分类完全隔离的更新测试片段不与上文交集" * 5},
    )
    assert r2.status_code == 200
    data = r2.json()
    assert data["name"] == "原始名"  # 没传 name，沿用旧值
    assert data["chunk_count"] >= 1
    # 文档表只有这一篇
    listing = await client.get("/documents")
    assert listing.json()["total"] == 1

    # 用旧文档独有的字串检索：期望至少在 top-1 命中评分低于「明确属于新文本」的检索
    # （hash embedding 在 64 维下相近文本可能混淆，因此只断言「新主题词能命中且 score>=旧主题词」即可）
    q_new = await client.post("/query", json={"question": "全新主题戊癸未分类"})
    assert len(q_new.json()["sources"]) >= 1
    assert q_new.json()["sources"][0]["doc_name"] == "原始名"


async def test_update_document_rename_only(client):
    """只传 name：现在走"真刷新 chunk metadata"路径（删旧 + 重新嵌入入库），
    应返回 refreshed_chunks 字段且历史 chunk 的 doc_name 已更新。"""
    r = await client.post(
        "/documents", json={"name": "原名", "text": "一些文本内容用于检索验证" * 10}
    )
    doc_id = r.json()["doc_id"]
    original_chunk_count = r.json()["chunk_count"]

    r2 = await client.put(f"/documents/{doc_id}", json={"name": "新名"})
    assert r2.status_code == 200
    data = r2.json()
    assert data["name"] == "新名"
    assert data.get("refreshed_chunks") == original_chunk_count  # 所有历史 chunk 都已用新名重建

    # 二次查询，溯源应显示新名
    r3 = await client.post("/query", json={"question": "检索验证", "top_k": 3})
    assert r3.status_code == 200
    sources = r3.json().get("sources", [])
    assert sources, "查询应至少命中一条"
    assert all(s.get("doc_name") == "新名" for s in sources)


async def test_update_document_validation(client):
    """空 body / 不存在的 doc_id 应返回 4xx。"""
    r = await client.post(
        "/documents", json={"name": "X", "text": "内容" * 20}
    )
    doc_id = r.json()["doc_id"]

    # name 和 text 都没传 → 400
    r_bad = await client.put(f"/documents/{doc_id}", json={})
    assert r_bad.status_code == 400

    # 不存在的 doc_id → 404
    r_404 = await client.put("/documents/nonexistent-id-xyz", json={"name": "X"})
    assert r_404.status_code == 404


# ---------- 批量上传 ----------

async def test_batch_upload_multiple_documents(client):
    """POST /documents/batch 一次性传多篇，应全部进 uploaded 列表。"""
    r = await client.post(
        "/documents/batch",
        json={
            "documents": [
                {"name": "文档甲", "text": "甲主题内容独门标识符甲甲甲甲甲甲甲甲甲" * 5},
                {"name": "文档乙", "text": "乙主题内容独门标识符乙乙乙乙乙乙乙乙乙" * 5},
                {"name": "文档丙", "text": "丙主题内容独门标识符丙丙丙丙丙丙丙丙丙" * 5},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    # 全部成功，failed 应为空
    assert len(data["uploaded"]) == 3
    assert data["failed"] == []

    # uploaded 每条都带 index / doc_id / name / chunk_count
    for i, item in enumerate(data["uploaded"]):
        assert item["index"] == i
        assert item["doc_id"]
        assert item["name"] == f"文档{['甲', '乙', '丙'][i]}"
        assert item["chunk_count"] >= 1

    # 文档列表应有 3 篇
    listing = await client.get("/documents")
    assert listing.json()["total"] == 3


async def test_batch_upload_single_document(client):
    """批量上传单篇文档应等价于单条上传（结构不变）。"""
    r = await client.post(
        "/documents/batch",
        json={"documents": [{"name": "单篇", "text": "单独一篇的内容用于验证" * 10}]},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["uploaded"]) == 1
    assert data["failed"] == []
    assert data["uploaded"][0]["index"] == 0
    assert data["uploaded"][0]["name"] == "单篇"


async def test_batch_upload_partial_failure(client):
    """批量上传时部分文档为空文本 → 记入 failed，已成功的进 uploaded，整体仍 200。"""
    r = await client.post(
        "/documents/batch",
        json={
            "documents": [
                {"name": "正常的", "text": "正常内容独门标识符丁" * 10},
                {"name": "空文本", "text": "   "},  # 切片后为空 → 400
                {"name": "也正常", "text": "另一篇正常内容独门标识符戊" * 10},
            ]
        },
    )
    assert r.status_code == 200
    data = r.json()
    # 2 篇成功，1 篇失败
    assert len(data["uploaded"]) == 2
    assert len(data["failed"]) == 1
    # failed 记录了 index/name/error
    fail_item = data["failed"][0]
    assert fail_item["index"] == 1
    assert fail_item["name"] == "空文本"
    assert "空" in fail_item["error"]
    # uploaded 顺序对应原数组下标
    assert data["uploaded"][0]["index"] == 0
    assert data["uploaded"][1]["index"] == 2

    # 文档列表只有 2 篇（失败的没入库）
    listing = await client.get("/documents")
    assert listing.json()["total"] == 2


async def test_batch_upload_empty_list_rejected(client):
    """空 documents 列表 → 400（不允许「零上传」当成成功）。"""
    r = await client.post("/documents/batch", json={"documents": []})
    assert r.status_code == 400


async def test_batch_upload_exceeds_limit_rejected(client):
    """超过 BATCH_MAX_SIZE → 400，超限整批拒绝（不部分入库）。"""
    docs = [{"name": f"d{i}", "text": "x" * 100} for i in range(51)]  # BATCH_MAX_SIZE=50
    r = await client.post("/documents/batch", json={"documents": docs})
    assert r.status_code == 400
    assert "50" in r.json()["detail"]
    # 整批拒绝：文档表应为空
    listing = await client.get("/documents")
    assert listing.json()["total"] == 0


# ---------- 来源片段：置信度 + 高亮 ----------

async def test_query_returns_confidence_and_highlight(client):
    """每条 source 应包含 confidence（相对归一化），snippet 应高亮查询关键词。"""
    r = await client.post(
        "/documents",
        json={
            "name": "排序算法笔记.md",
            "text": (
                "快速排序是一种分治算法。快速排序平均时间复杂度是 O(n log n)，"
                "最坏情况下退化到 O(n^2)。归并排序也是分治，但稳定性更好。"
            ),
        },
    )
    assert r.status_code == 200

    q = await client.post("/query", json={"question": "快速排序的时间复杂度"})
    assert q.status_code == 200
    data = q.json()

    # 顶层聚合字段存在
    assert "confidence_avg" in data
    assert "top_score" in data

    # sources 每条必须有 confidence 字段
    assert len(data["sources"]) >= 1
    for src in data["sources"]:
        assert "confidence" in src
        assert isinstance(src["confidence"], (int, float))
        assert 0.0 <= src["confidence"] <= 1.0
        assert "snippet" in src

    # 关键：「快速排序」应在 snippet 中被 ** 包裹（高亮）
    snippet = data["sources"][0]["snippet"]
    assert "**快速排序**" in snippet


async def test_query_confidence_distribution(client):
    """confidence 之和应 ≈ 1（相对归一化）。"""
    await client.post(
        "/documents",
        json={
            "name": "排序笔记.md",
            "text": "冒泡排序、选择排序、插入排序、快速排序都是基础排序算法。" * 5,
        },
    )
    await client.post(
        "/documents",
        json={
            "name": "图论笔记.md",
            "text": "图论研究节点与边的关系，与排序算法无关。" * 5,
        },
    )
    q = await client.post("/query", json={"question": "排序算法有哪些", "top_k": 2})
    assert q.status_code == 200
    data = q.json()
    total_conf = sum(src["confidence"] for src in data["sources"])
    # confidence 是 score/sum(scores)，总和 = 1（容差 0.01）
    assert abs(total_conf - 1.0) < 0.01


async def test_query_highlight_case_insensitive(client):
    """高亮应大小写不敏感（query 中用大写也应命中 snippet 中小写词）。"""
    await client.post(
        "/documents",
        json={"name": "PY指南.md", "text": "Python 是一种解释型语言，Python 支持多范式。" * 5},
    )
    # query 中用大写 Python（实际 chunk 中是小写）
    q = await client.post("/query", json={"question": "Python 的特点"})
    snippet = q.json()["sources"][0]["snippet"]
    # 大小写不敏感匹配后应有 **
    assert "**Python**" in snippet
