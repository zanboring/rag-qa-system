"""embedder 单元测试：确定性、维度、相似性方向。"""

import asyncio

from app.embedder import HashEmbedder
from app.vectorstore import _cosine


def test_hash_embedder_deterministic():
    e = HashEmbedder(dim=128)
    v1 = asyncio.run(e.embed("人工智能与机器学习"))
    v2 = asyncio.run(e.embed("人工智能与机器学习"))
    assert v1 == v2


def test_hash_embedder_dim_and_normalized():
    e = HashEmbedder(dim=128)
    v = asyncio.run(e.embed("hello world"))
    assert len(v) == 128
    # L2 归一化后模长应约等于 1
    norm = sum(x * x for x in v)
    assert abs(norm - 1.0) < 1e-6


def test_similar_text_closer_than_different():
    e = HashEmbedder(dim=256)
    a = asyncio.run(e.embed("如何用 Python 写一个快速排序算法"))
    b = asyncio.run(e.embed("Python 快速排序算法的实现方式"))
    c = asyncio.run(e.embed("今天天气很好适合出去散步"))
    assert _cosine(a, b) > _cosine(a, c)
