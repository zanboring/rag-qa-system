"""评测指标单元测试。

为什么指标实现也要测：指标是评测体系的地基。指标算错时，评测报告依然是
"完整、漂亮、有数字"的——错误不会报错，只会给出错误结论，进而把调优方向带偏。
因此指标层必须有测试覆盖，尤其是边界情况（空输入、除零、实际返回少于 K）。
"""

from eval.metrics import (
    aggregate_scores,
    count_relevant_in_corpus,
    dcg_at_k,
    hit_at_k,
    is_relevant,
    keyword_coverage,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_flags,
    relevance_grades,
)


# ---------------------------------------------------------------------------
# 相关性判定
# ---------------------------------------------------------------------------

def test_is_relevant_requires_doc_and_keyword():
    """必须同时满足「文档匹配」与「关键串全部包含」才算相关。"""
    sample = {"relevant_docs": ["a.md"], "must_contain": ["快速排序"]}

    hit_ok = {"metadata": {"doc_name": "a.md", "text": "快速排序是一种分治算法"}}
    hit_wrong_doc = {"metadata": {"doc_name": "b.md", "text": "快速排序是一种分治算法"}}
    hit_missing_keyword = {"metadata": {"doc_name": "a.md", "text": "归并排序是稳定排序"}}

    assert is_relevant(hit_ok, sample) is True
    assert is_relevant(hit_wrong_doc, sample) is False
    assert is_relevant(hit_missing_keyword, sample) is False


def test_is_relevant_all_keywords_required():
    """标注多个关键串时必须全部命中——只中一个不足以证明检索到了正确片段。"""
    sample = {"relevant_docs": ["a.md"], "must_contain": ["快速排序", "O(n log n)"]}

    full = {"metadata": {"doc_name": "a.md", "text": "快速排序平均复杂度 O(n log n)"}}
    partial = {"metadata": {"doc_name": "a.md", "text": "快速排序是分治算法"}}

    assert is_relevant(full, sample) is True
    assert is_relevant(partial, sample) is False


def test_is_relevant_without_keywords_checks_doc_only():
    """未标注关键串时退化为只判文档名。"""
    sample = {"relevant_docs": ["a.md"]}
    hit = {"metadata": {"doc_name": "a.md", "text": "任意内容"}}
    assert is_relevant(hit, sample) is True


def test_is_relevant_handles_dirty_metadata():
    """metadata 缺失或为 None 时不应抛异常（脏数据防御）。"""
    assert is_relevant({}, {"relevant_docs": ["a.md"]}) is False
    assert is_relevant({"metadata": None}, {"relevant_docs": ["a.md"]}) is False


def test_relevance_flags_and_grades():
    sample = {"relevant_docs": ["a.md"], "must_contain": ["x", "y"]}
    hits = [
        {"metadata": {"doc_name": "a.md", "text": "x y"}},   # 命中，等级 2
        {"metadata": {"doc_name": "b.md", "text": "x y"}},   # 文档不匹配
    ]
    assert relevance_flags(hits, sample) == [1, 0]
    assert relevance_grades(hits, sample) == [2.0, 0.0]


def test_count_relevant_in_corpus():
    """Recall 的分母必须由库中真实存在的相关片段数决定。"""
    corpus_chunks = [
        {"metadata": {"doc_name": "a.md", "text": "快速排序"}},
        {"metadata": {"doc_name": "a.md", "text": "归并排序"}},
        {"metadata": {"doc_name": "b.md", "text": "快速排序"}},
    ]
    sample = {"relevant_docs": ["a.md"], "must_contain": ["快速排序"]}
    assert count_relevant_in_corpus(sample, corpus_chunks) == 1


# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------

def test_hit_at_k():
    flags = [0, 0, 1, 0, 0]
    assert hit_at_k(flags, 1) == 0.0
    assert hit_at_k(flags, 2) == 0.0
    assert hit_at_k(flags, 3) == 1.0
    assert hit_at_k(flags, 5) == 1.0
    assert hit_at_k([], 3) == 0.0


def test_recall_at_k():
    flags = [1, 0, 1, 0, 0]
    assert recall_at_k(flags, 3, total_relevant=2) == 1.0
    assert recall_at_k(flags, 1, total_relevant=2) == 0.5
    # K 小于相关片段总数时，Recall@K 上限本身就小于 1
    assert recall_at_k([1, 0, 0], 1, total_relevant=3) == 1 / 3
    # 分母为 0 的边界：不应除零
    assert recall_at_k(flags, 3, total_relevant=0) == 0.0


def test_precision_at_k_uses_actual_returned_count():
    """分母用实际返回条数，而非 K。

    库中片段少于 K 时，用 K 作分母会惩罚一个并不存在的错误，
    人为压低准确率。
    """
    assert precision_at_k([1, 0, 1], 3) == 2 / 3
    assert precision_at_k([1, 0], 5) == 0.5
    assert precision_at_k([], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([1, 0, 0]) == 1.0
    assert reciprocal_rank([0, 1, 0]) == 0.5
    assert reciprocal_rank([0, 0, 1]) == 1 / 3
    assert reciprocal_rank([0, 0, 0]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_dcg_discounts_lower_positions():
    """位置越靠后折损越重：高相关结果排前面应得更高 DCG。"""
    assert dcg_at_k([2.0, 0.0], 2) > dcg_at_k([0.0, 2.0], 2)


def test_ndcg_perfect_and_worst_ordering():
    # 完美排序时 IDCG == DCG，nDCG 为 1
    assert ndcg_at_k([2.0, 1.0], 2) == 1.0
    # 逆序时应小于 1
    assert 0 < ndcg_at_k([1.0, 2.0], 2) < 1.0
    # 全不相关时为 0
    assert ndcg_at_k([0.0, 0.0], 2) == 0.0


def test_ndcg_uses_recalled_relevant_only():
    """IDCG 只用已召回的相关结果做理想排序，不虚构未召回的高等级结果。

    若把未召回的高等级结果算进理想排序，分母会被放大，nDCG 被系统性低估。
    """
    # 只召回了一个等级 1 的结果，且排在第一位 → 已是当前可达的最优排序
    assert ndcg_at_k([1.0, 0.0, 0.0], 3) == 1.0


# ---------------------------------------------------------------------------
# 生成与统计
# ---------------------------------------------------------------------------

def test_keyword_coverage():
    sample = {"answer_keywords": ["O(n log n)", "分治"]}
    assert keyword_coverage("快速排序平均 O(n log n)，是分治算法", sample) == 1.0
    assert keyword_coverage("快速排序是分治算法", sample) == 0.5
    assert keyword_coverage("不知道", sample) == 0.0
    # 无标注关键词时返回 0（而非除零或 1）
    assert keyword_coverage("任意内容", {"answer_keywords": []}) == 0.0


def test_keyword_coverage_is_case_insensitive():
    sample = {"answer_keywords": ["Python"]}
    assert keyword_coverage("python 是解释型语言", sample) == 1.0


def test_percentile():
    values = [1, 2, 3, 4, 5]
    assert percentile(values, 0) == 1
    assert percentile(values, 50) == 3
    assert percentile(values, 100) == 5
    # 线性插值
    assert percentile(values, 25) == 2
    assert percentile([], 50) == 0.0
    assert percentile([7], 50) == 7


def test_aggregate_scores_macro_average():
    """逐样本平均（macro），避免样本量不同的分组互相主导。"""
    rows = [
        {"id": "1", "mrr": 1.0, "ndcg_at_5": 1.0},
        {"id": "2", "mrr": 0.0, "ndcg_at_5": 0.0},
    ]
    result = aggregate_scores(rows)
    assert result["mrr"] == 0.5
    assert result["ndcg_at_5"] == 0.5
    # 空输入返回空字典，而不是抛异常
    assert aggregate_scores([]) == {}
