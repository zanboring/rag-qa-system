"""RAG 评测指标实现（纯函数，零第三方依赖，可独立单测）。

设计原则
--------
1. **不依赖 chunk_id 的标注方式**
   评测集用「文档名 + 必须包含的关键串」标注相关性，而不是硬编码 `doc_id:chunk_index`。
   原因：chunk_id 依赖切片参数（chunk_size / overlap），一旦调参，标注全部失效，
   评测集就废了。用内容包含关系标注，换切片策略后评测集依然有效。

2. **区分「有无答案」两类样本**
   - 有答案样本（answerable）：参与 Hit@K / Recall@K / MRR / nDCG 等检索指标。
   - 无答案样本（unanswerable）：库里确实没有答案，正确行为是「拒答」。这类样本
     的 relevant 集合为空，若混入检索指标会出现「0 分」或「除零」的错误语义——
     它考的是「不该乱答」，属于另一类能力，单独用 abstention 指标统计。

3. **召回与排序分开看**
   Recall@K 只看「有没有捞回来」（检索能力上限），MRR / nDCG 看「捞回来的排得好不好」
   （排序能力）。两者分开，才能定位问题出在召回还是重排。

术语
----
retrieved : 已按相关性从高到低排好序的命中列表，元素为 hit dict
            （约定含 `metadata.doc_name` 与 `metadata.text`，与 vectorstore.search 返回一致）
relevant  : 标注中判定的相关 hit 集合（由 is_relevant 逐条判定得到）
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# 相关性判定：把「检索结果」映射为「是否相关」
# ---------------------------------------------------------------------------

# 检索结果中，必须包含的元数据字段
REQUIRED_META_KEYS = ("doc_name", "text")


def _meta_of(hit: dict) -> dict:
    """从 hit 中取出 metadata；兼容 metadata 缺失或为 None 的脏数据。"""
    meta = hit.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def is_relevant(hit: dict, sample: dict) -> bool:
    """判定单个检索结果是否命中该样本标注的相关内容。

    判定规则（两条同时满足才算相关）：
    1. 片段所属文档在 `sample["relevant_docs"]` 中；
    2. 片段文本包含 `sample["must_contain"]` 中的**全部**关键串（未标注则该条不限制）。

    为什么要求「全部包含」：同一篇文档可能被切成多个片段，主题各不相同。
    若只判文档名，凡是该文档的任意片段都算命中，会把「检索到同文档的错误片段」
    误判为成功，导致指标虚高。要求关键串全部命中，等价于要求「检索到了真正
    含有答案的那一段」，更接近真实语义。
    """
    meta = _meta_of(hit)
    if meta.get("doc_name") not in sample.get("relevant_docs", []):
        return False
    text = meta.get("text") or ""
    for keyword in sample.get("must_contain", []):
        if keyword not in text:
            return False
    return True


def relevance_flags(retrieved: Sequence[dict], sample: dict) -> list[int]:
    """把有序检索结果转为相关性标记序列 [1, 0, 1, ...]，长度等于 retrieved。"""
    return [1 if is_relevant(hit, sample) else 0 for hit in retrieved]


def relevance_grades(retrieved: Sequence[dict], sample: dict) -> list[float]:
    """带分级的相关度序列，用于 nDCG。

    分级约定：must_contain 标注得越多，说明标注越具体，命中它的价值越高。
    实现上取「命中该条所需的关键串数量」作为等级（至少 1），
    未命中为 0。这样当同一文档的多个片段都被召回时，越具体的片段得分越高。
    """
    grades: list[float] = []
    for hit in retrieved:
        if not is_relevant(hit, sample):
            grades.append(0.0)
        else:
            grades.append(max(1.0, float(len(sample.get("must_contain", [])))))
    return grades


def count_relevant_in_corpus(sample: dict, corpus_chunks: Sequence[dict]) -> int:
    """统计整个知识库中「相关片段」的总数，作为 Recall 的分母。

    Recall@K = Top-K 中命中的相关片段数 / 库中全部相关片段数。
    分母必须由库中真实存在的相关片段数决定，不能凭标注拍脑袋给一个数，
    否则 Recall 会失真（例如标注写了 2 个但库里实际只有 1 个，Recall 永远到不了 1）。
    """
    return sum(1 for chunk in corpus_chunks if is_relevant(chunk, sample))


# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------

def hit_at_k(flags: Sequence[int], k: int) -> float:
    """Top-K 内是否存在相关片段（0 或 1）。衡量最朴素的「有没有捞到」。"""
    return 1.0 if any(flags[:k]) else 0.0


def recall_at_k(flags: Sequence[int], k: int, total_relevant: int) -> float:
    """Top-K 召回率 = 命中数 / 库中相关片段总数。

    注意：K 小于 total_relevant 时，Recall@K 上限本身就小于 1（不可能一次全捞回），
    这是 Recall@K 的固有性质，不是 bug。
    """
    if total_relevant <= 0:
        return 0.0
    return min(1.0, sum(flags[:k]) / total_relevant)


def precision_at_k(flags: Sequence[int], k: int) -> float:
    """Top-K 准确率 = 命中数 / 实际返回条数。

    分母用「实际返回条数」而不是 K：当库中片段少于 K 时，用 K 作分母会人为压低
    准确率，惩罚一个并不存在的错误。
    """
    actual = len(flags[:k])
    if actual == 0:
        return 0.0
    return sum(flags[:k]) / actual


def reciprocal_rank(flags: Sequence[int]) -> float:
    """MRR 的单样本贡献：第一个相关片段的排名倒数（未命中为 0）。

    只看第一名相关结果的位置，对「第一个正确答案排在第几」敏感，
    适合评价「用户能否一眼看到答案」的场景。
    """
    for rank, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def dcg_at_k(grades: Sequence[float], k: int) -> float:
    """折损累计增益 DCG@K = Σ (2^rel_i - 1) / log2(i + 1)。

    使用指数增益（2^rel - 1）而非线性增益：相关性越高的结果，位置前移带来的
    收益增长越快，符合「第一个结果最重要」的检索直觉。
    """
    total = 0.0
    for i, grade in enumerate(grades[:k], start=1):
        total += (2.0 ** grade - 1.0) / math.log2(i + 1)
    return total


def ndcg_at_k(grades: Sequence[float], k: int) -> float:
    """归一化折损累计增益 nDCG@K = DCG@K / IDCG@K。

    IDCG 用「把当前所有相关结果按等级降序排列」的理想顺序计算。
    只对已召回的相关结果做理想排序（而非假设库中存在无限多个最高等级结果），
    避免分母被虚构的理想值放大，使 nDCG 在召回不全时依然可比。
    """
    dcg = dcg_at_k(grades, k)
    if dcg == 0.0:
        return 0.0
    ideal = sorted((g for g in grades if g > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def average(values: Iterable[float]) -> float:
    """安全均值：空序列返回 0.0，避免统计空样本时抛异常。"""
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数（p 取 0~100）。

    延迟这类指标看均值没意义（会被极端值带偏），P50 / P95 才能反映真实体感。
    自己实现而不依赖 numpy：评测指标层保持零依赖，便于在任何环境复现结果。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


# ---------------------------------------------------------------------------
# 生成指标（规则版，不依赖 LLM）
# ---------------------------------------------------------------------------

def keyword_coverage(answer: str, sample: dict) -> float:
    """答案对标注关键词的覆盖率，作为「答案正确性」的零依赖代理指标。

    这是在无 API Key 环境下也能跑出数据的兜底指标。它衡量的是「答案里有没有
    出现标准答案的关键要素」，属于必要不充分条件：
    - 覆盖率低 ⇒ 答案几乎肯定不对（可用作强否决信号）；
    - 覆盖率高 ⇏ 语义一定正确（可能关键词堆砌）。
    因此它与 LLM-as-judge 互补：规则指标负责可复现的底线，judge 负责语义判断。
    """
    keywords = sample.get("answer_keywords", [])
    if not keywords:
        return 0.0
    answer_lower = (answer or "").lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


def context_recall(retrieved: Sequence[dict], sample: dict) -> float:
    """检索到的上下文对标准答案要素的覆盖度（检索侧视角的「喂料够不够」）。

    与 answer_keywords 的区别：这里检查的是 **上下文** 是否包含答案要素，
    而 keyword_coverage 检查的是 **最终答案**。两者对比可以定位问题：
    - 上下文有、答案没有 ⇒ 生成环节丢了信息（Prompt 或模型问题）；
    - 上下文本身就没有 ⇒ 检索环节漏了（该调切片 / embedding / top_k）。
    """
    keywords = sample.get("answer_keywords", [])
    if not keywords:
        return 0.0
    joined = "\n".join(_meta_of(hit).get("text", "") for hit in retrieved).lower()
    hits = sum(1 for kw in keywords if kw.lower() in joined)
    return hits / len(keywords)


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """把逐样本的指标行聚合成总体指标（各指标分别取均值）。

    分开聚合而不是先算总再平均：不同样本的相关片段数不同，
    逐样本算完再平均（macro-average）能避免「相关片段多的样本」主导总分。
    """
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "id"]
    return {key: round(average(float(row[key]) for row in rows if key in row), 4) for key in keys}
