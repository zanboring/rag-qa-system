"""vectorstore 单元测试：入库、检索排序、Top-K。"""

import asyncio

from app.embedder import HashEmbedder
from app.vectorstore import MemoryVectorStore


def test_add_and_search_topk():
    store = MemoryVectorStore()
    e = HashEmbedder(dim=128)

    async def run():
        # 入库 3 条
        await store.add("a", await e.embed("Python 快速排序算法"), {"text": "排序"})
        await store.add("b", await e.embed("Python 快速排序的实现"), {"text": "排序"})
        await store.add("c", await e.embed("足球比赛结果"), {"text": "体育"})

        hits = await store.search(await e.embed("快速排序算法怎么写"), top_k=2)
        assert len(hits) == 2
        # 最相关的应是 a 或 b，而不是 c
        assert hits[0]["id"] in ("a", "b")
        assert hits[0]["id"] != "c"
        # 分数降序
        assert hits[0]["score"] >= hits[1]["score"]
        assert "metadata" in hits[0]

    asyncio.run(run())


def test_search_empty_store():
    store = MemoryVectorStore()
    e = HashEmbedder(dim=128)

    async def run():
        hits = await store.search(await e.embed("任意问题"), top_k=3)
        assert hits == []

    asyncio.run(run())


def test_delete():
    store = MemoryVectorStore()
    e = HashEmbedder(dim=128)

    async def run():
        await store.add("doc1:0", await e.embed("内容A"), {"text": "A"})
        await store.add("doc1:1", await e.embed("内容B"), {"text": "B"})
        await store.add("doc2:0", await e.embed("内容C"), {"text": "C"})

        removed = await store.delete(["doc1:0", "doc1:1"])
        assert removed == 2
        assert len(store) == 1

        # 删除后检索不再命中 doc1 的内容
        hits = await store.search(await e.embed("内容A"), top_k=3)
        assert all(h["id"] != "doc1:0" for h in hits)
        assert all(h["id"] != "doc1:1" for h in hits)

    asyncio.run(run())
