"""reranker 单元测试：词法重叠计算 + RRF 融合后的精排效果。"""

from app.reranker import LexicalReranker, _lexical_overlap


def test_lexical_overlap_basics():
    # 语义相近文本有重叠
    assert _lexical_overlap("快速排序", "快速排序算法") > 0
    # 完全不相关的文本重叠为 0
    assert _lexical_overlap("快速排序", "足球比赛") == 0.0
    # 空查询 / 空文本安全返回 0
    assert _lexical_overlap("", "内容") == 0.0
    assert _lexical_overlap("内容", "") == 0.0


def test_lexical_overlap_bounded():
    # 重叠系数值域应为 [0, 1]
    score = _lexical_overlap("快速排序算法", "快速排序的 Python 实现")
    assert 0.0 <= score <= 1.0


def test_rerank_prefers_lexical_match():
    """RRF 应把「向量分略低但词法高度相关」的片段提到最前。

    三候选设计：a 向量分最高但词法无关；b 词法完全命中；c 词法部分命中。
    只靠向量分，a 排第一；融合词法后，b 应反超 a。
    """
    reranker = LexicalReranker()
    hits = [
        {"id": "a", "score": 0.99, "metadata": {"text": "足球比赛结果，梅西进球"}},
        {"id": "b", "score": 0.80, "metadata": {"text": "快速排序算法时间复杂度"}},
        {"id": "c", "score": 0.60, "metadata": {"text": "快速排序的 python 实现"}},
    ]
    top = reranker.rerank("快速排序算法", hits, top_k=3)
    assert top[0]["id"] == "b"


def test_rerank_truncates_to_top_k():
    reranker = LexicalReranker()
    hits = [
        {"id": f"c{i}", "score": 1.0 - i * 0.05, "metadata": {"text": f"内容片段 {i}"}}
        for i in range(10)
    ]
    top = reranker.rerank("内容片段 0", hits, top_k=3)
    assert len(top) == 3


def test_rerank_empty_hits():
    reranker = LexicalReranker()
    assert reranker.rerank("任意问题", [], top_k=3) == []
