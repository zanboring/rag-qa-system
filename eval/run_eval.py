"""RAG 评测主流程。

用法
----
    # 零依赖跑基线（规则裁判，不消耗任何 API 调用），同时对比重排与哈希维度
    python -m eval.run_eval --judge rule --hash-dims 256,4096

    # 换成 BGE embedding（需 pip install sentence-transformers）
    python -m eval.run_eval --embedding hash,bge --judge rule

    # 接真实模型跑生成指标（需 DEEPSEEK_API_KEY；裁判也会自动复用 DeepSeek）
    python -m eval.run_eval --llm deepseek --tag deepseek

    # 调整召回条数与并发
    python -m eval.run_eval --top-k 5 --concurrency 8

产出
----
    eval/results/report[_<tag>].md   汇总报告（指标对比表 + 自动诊断 + 失败案例）
    eval/results/detail_<配置>.json  逐样本明细，便于定位单条失败

    --tag 用于区分不同后端的评测结果（如 report.md 为 mock 基线、
    report_deepseek.md 为真实模型），避免后跑的覆盖先跑的。

评测流程
--------
1. 自检评测集（validate_set）——标注有错就不进入评测，避免指标被静默污染
2. 按配置构建 pipeline，把语料整体索引进向量库
3. 对每条 query 走完整链路（检索 → 可选重排 → 生成），记录耗时
4. 三类指标分别打分：
   - 检索指标（answerable 样本）：Hit / Recall / Precision / MRR / nDCG
   - 生成指标：规则指标（关键词覆盖）+ LLM 裁判指标（忠实度、答案相关性）
   - 拒答指标（unanswerable 样本）：正确拒答率
5. 输出对比报告与明细

为什么把三类指标分开
--------------------
它们考的是**不同的能力**，混在一起算平均会互相掩盖问题：
- 检索指标被压低 → 问题在切片 / embedding / top_k
- 检索正常但忠实度低 → 问题在提示词或生成模型
- 无答案样本上乱答 → 拒答机制缺失
分开统计才能据此定位薄弱环节，这也是这份评测存在的意义。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app import config
from app.chunker import chunk_text
from app.embedder import get_embedder
from app.llm import get_llm
from app.rag import RAGPipeline
from app.vectorstore import get_vectorstore

from eval import metrics as M
from eval.judge import RuleBasedJudge, get_judge
from eval.validate_set import load_corpus, load_golden_set, print_summary, validate

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# 评测使用的召回条数。取 5 是为了能同时报告 @1/@3/@5 三档指标，
# 观察指标随 K 的变化趋势——若 recall@5 明显高于 recall@3，
# 说明正确片段确实被召回了、只是排序不够靠前，问题在精排而非召回。
DEFAULT_TOP_K = 5


@dataclass
class EvalConfig:
    """一组待评测的后端配置。"""

    embedding: str
    vector: str
    llm: str
    rerank: bool
    top_k: int = DEFAULT_TOP_K
    # 仅对 hash embedding 生效：哈希桶数量（即向量维度）。
    # 把它暴露成配置项的原因：hash embedding 的主要误差来源是哈希碰撞——
    # 3-gram 的种类数远多于桶数时，不同 n-gram 会落进同一个桶，向量区分度下降。
    # 对比不同维度可以在不引入任何模型的前提下，验证「表示能力是检索质量瓶颈」。
    hash_dim: int | None = None

    @property
    def label(self) -> str:
        """配置标签，用于文件名与报告表格行名。"""
        head = self.embedding
        if self.embedding == "hash" and self.hash_dim:
            head = f"hash{self.hash_dim}"
        return f"{head}+{self.vector}+{self.llm}+rerank{'on' if self.rerank else 'off'}"


@dataclass
class SampleResult:
    """单条样本的评测明细。"""

    id: str
    question: str
    type: str
    difficulty: str
    # 检索
    retrieved_docs: list[str] = field(default_factory=list)
    relevance: list[int] = field(default_factory=list)
    total_relevant: int = 0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_3: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    # 生成
    answer: str = ""
    keyword_coverage: float = 0.0
    context_recall: float = 0.0
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    judge_reason: str = ""
    judge_error: str | None = None
    # 拒答
    is_unanswerable: bool = False
    abstained: bool = False
    # 性能
    latency_ms: float = 0.0


async def build_index(pipeline: RAGPipeline, corpus: dict[str, str]) -> list[dict]:
    """把语料索引进向量库，返回全部 chunk（用于统计库中相关片段总数）。

    与 app/main.py 的上传逻辑保持一致：同样按 CHUNK_SIZE / CHUNK_OVERLAP 切片，
    片段 id 同样采用 `{doc_id}:{index}` 约定。这里不走 HTTP 接口，
    是为了让评测不受网络栈与序列化开销干扰，测得更接近检索本身的真实耗时。
    """
    chunks: list[dict] = []
    for doc_name, text in corpus.items():
        pieces = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for index, piece in enumerate(pieces):
            vector = await pipeline.embedder.embed(piece)
            await pipeline.vectorstore.add(
                id=f"{doc_name}:{index}",
                vector=vector,
                metadata={"doc_name": doc_name, "chunk_index": index, "text": piece},
            )
            chunks.append(
                {"id": f"{doc_name}:{index}", "metadata": {"doc_name": doc_name, "text": piece}}
            )
    return chunks


async def evaluate_sample(
    pipeline: RAGPipeline,
    sample: dict,
    corpus_chunks: list[dict],
    chunk_text_by_id: dict[str, str],
    judge,
    rerank: bool,
    top_k: int,
) -> SampleResult:
    """跑单条样本：检索 → 生成 → 打分。"""
    result = SampleResult(
        id=sample["id"],
        question=sample["question"],
        type=sample.get("type", ""),
        difficulty=sample.get("difficulty", ""),
        is_unanswerable=sample.get("type") == "unanswerable",
    )

    # 用真实链路作答，同时记录端到端耗时（含生成，这是用户实际等待的时间）
    started = time.perf_counter()
    response = await pipeline.ask(sample["question"], top_k=top_k, rerank=rerank)
    result.latency_ms = round((time.perf_counter() - started) * 1000, 2)

    sources = response.get("sources", [])
    result.retrieved_docs = [s.get("doc_name", "") for s in sources]

    # 还原 hit 结构（含 metadata.text），供指标层做内容级相关性判定。
    # pipeline.ask 返回的 sources 只带 snippet（已截断到 120 字符），不能用于 must_contain
    # 判定——关键串可能落在截断点之后，用 snippet 会让本该命中的样本被判为未命中。
    # 因此按片段 id 从索引里取回完整原文。
    hits = []
    for src in sources:
        doc_name = src.get("doc_name")
        chunk_index = src.get("chunk_index")
        full_text = chunk_text_by_id.get(f"{doc_name}:{chunk_index}", "")
        hits.append({"metadata": {"doc_name": doc_name, "text": full_text}})

    # ---------- 检索指标：仅对有答案样本计算 ----------
    if not result.is_unanswerable:
        flags = M.relevance_flags(hits, sample)
        grades = M.relevance_grades(hits, sample)
        result.relevance = flags
        result.total_relevant = M.count_relevant_in_corpus(sample, corpus_chunks)

        result.hit_at_1 = M.hit_at_k(flags, 1)
        result.hit_at_3 = M.hit_at_k(flags, 3)
        result.hit_at_5 = M.hit_at_k(flags, 5)
        result.recall_at_3 = M.recall_at_k(flags, 3, result.total_relevant)
        result.recall_at_5 = M.recall_at_k(flags, 5, result.total_relevant)
        result.precision_at_3 = M.precision_at_k(flags, 3)
        result.mrr = M.reciprocal_rank(flags)
        result.ndcg_at_5 = M.ndcg_at_k(grades, 5)

        # ---------- 规则生成指标 ----------
        result.answer = response.get("answer", "")
        result.keyword_coverage = M.keyword_coverage(result.answer, sample)
        result.context_recall = M.context_recall(hits, sample)

    # ---------- 裁判打分（两类样本都要，用于拒答判定） ----------
    # 传给裁判的上下文必须是**实际被召回的片段**，且与 hits 严格一一对应。
    # 若按「召回文档」去库里捞该文档的全部片段，会给裁判喂进比实际检索结果更多的内容，
    # 导致忠实度虚高——那测的就不是系统真实看到的信息了。
    context_texts = [h["metadata"]["text"] for h in hits if h["metadata"]["text"]]

    judge_score = await judge.score(sample["question"], response.get("answer", ""), context_texts)
    result.answer = response.get("answer", "")
    result.faithfulness = judge_score.faithfulness
    result.answer_relevancy = judge_score.answer_relevancy
    result.judge_reason = judge_score.reason
    result.judge_error = judge_score.error
    # answered 为 None（裁判未给出）时，退化为"答案非空且未含拒答词"的规则判断
    result.abstained = not (
        judge_score.answered
        if judge_score.answered is not None
        else bool((response.get("answer") or "").strip())
    )

    return result


async def evaluate_config(
    cfg: EvalConfig,
    corpus: dict[str, str],
    samples: list[dict],
    judge,
    concurrency: int,
) -> tuple[list[SampleResult], float]:
    """跑完一组配置的全部样本，返回明细与索引耗时。"""
    # 每组配置用独立的向量库实例，避免不同配置之间的数据互相污染
    os.environ["EMBEDDING_BACKEND"] = cfg.embedding
    os.environ["VECTOR_BACKEND"] = cfg.vector
    os.environ["LLM_BACKEND"] = cfg.llm
    config.EMBEDDING_BACKEND = cfg.embedding
    config.VECTOR_BACKEND = cfg.vector
    config.LLM_BACKEND = cfg.llm
    if cfg.hash_dim:
        os.environ["HASH_EMBED_DIM"] = str(cfg.hash_dim)
        config.HASH_EMBED_DIM = cfg.hash_dim

    pipeline = RAGPipeline(get_embedder(), get_vectorstore(), get_llm())

    index_started = time.perf_counter()
    corpus_chunks = await build_index(pipeline, corpus)
    index_seconds = time.perf_counter() - index_started

    # 片段 id → 原文 的映射，供逐样本按 id O(1) 取回完整片段文本
    chunk_text_by_id = {c["id"]: c["metadata"]["text"] for c in corpus_chunks}

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(sample: dict) -> SampleResult:
        async with semaphore:
            return await evaluate_sample(
                pipeline, sample, corpus_chunks, chunk_text_by_id, judge, cfg.rerank, cfg.top_k
            )

    rows = await asyncio.gather(*(run_one(s) for s in samples))
    return list(rows), index_seconds


# ---------------------------------------------------------------------------
# 聚合与报告
# ---------------------------------------------------------------------------

RETRIEVAL_KEYS = (
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "recall_at_3",
    "recall_at_5",
    "precision_at_3",
    "mrr",
    "ndcg_at_5",
)
RULE_GEN_KEYS = ("keyword_coverage", "context_recall")


def summarize(rows: list[SampleResult], llm_backend: str = "") -> dict:
    """把逐样本明细聚合为总体指标。

    三类指标分别聚合，且如实标注哪些指标在当前环境下没有数据（值为 None）。

    参数 llm_backend 用于判断生成类指标是否成立：mock 后端不产生真实答案，
    它对应的忠实度 / 相关性 / 拒答指标一律标为不可用。
    """
    answerable = [r for r in rows if not r.is_unanswerable]
    unanswerable = [r for r in rows if r.is_unanswerable]

    summary: dict = {
        "sample_total": len(rows),
        "answerable": len(answerable),
        "unanswerable": len(unanswerable),
    }

    for key in RETRIEVAL_KEYS:
        summary[key] = round(M.average(getattr(r, key) for r in answerable), 4)

    for key in RULE_GEN_KEYS:
        summary[key] = round(M.average(getattr(r, key) for r in answerable), 4)

    # 生成类指标是否可用。
    #
    # 判断依据是「配置里的 llm 后端」，而不是让裁判去识别 mock 输出。
    # 原因：接入真实裁判（如 DeepSeek）后，mock 的模板答案会被裁判判为"未作答"，
    # 于是误拒率飙到 90%+，结论会变成"系统拒答能力有问题"——
    # 但真正的问题是"测量对象本身不成立"（mock 压根不产生答案）。
    # 只看输出、不看配置，就会把"没测到"误读成"测出来不好"。
    mock_ratio = M.average(1.0 if r.judge_error == "mock_backend" else 0.0 for r in rows)
    summary["mock_backend_ratio"] = round(mock_ratio, 4)
    generation_ok = llm_backend != "mock" and mock_ratio < 0.5
    summary["generation_metrics_available"] = generation_ok

    # 裁判指标：只在真正产出分数、且生成后端真实时统计，
    # 既不拿 0 冒充"评分为零"，也不拿 mock 的分数冒充生成质量
    for key in ("faithfulness", "answer_relevancy"):
        values = [getattr(r, key) for r in answerable if getattr(r, key) is not None]
        summary[key] = round(M.average(values), 4) if (values and generation_ok) else None

    # 拒答指标：无答案样本上正确拒答的比例
    if unanswerable and generation_ok:
        summary["abstention_accuracy"] = round(
            M.average(1.0 if r.abstained else 0.0 for r in unanswerable), 4
        )
        # 误拒率：有答案样本上错误拒答的比例（与拒答率必须一起看）
        summary["false_abstention_rate"] = round(
            M.average(1.0 if r.abstained else 0.0 for r in answerable), 4
        )
    else:
        summary["abstention_accuracy"] = None
        summary["false_abstention_rate"] = None

    latencies = [r.latency_ms for r in rows]
    summary["latency_p50_ms"] = round(M.percentile(latencies, 50), 2)
    summary["latency_p95_ms"] = round(M.percentile(latencies, 95), 2)

    return summary


def diagnose(summary: dict) -> list[str]:
    """根据指标自动定位瓶颈环节。

    报告只罗列数字是不够的——面试和实际调优真正需要的是"下一步该改哪里"。
    这里用几个指标之间的关系做自动判断：

    - Hit@5 低 → 正确片段压根没被召回，问题在召回侧
    - Hit@5 高但 MRR 低 → 召回了但排序差，问题在精排侧
    - Hit@1 远低于 Hit@5 → 首位不可靠，用户第一眼看不到答案
    """
    notes: list[str] = []
    hit1 = summary.get("hit_at_1") or 0.0
    hit5 = summary.get("hit_at_5") or 0.0
    mrr = summary.get("mrr") or 0.0
    recall5 = summary.get("recall_at_5") or 0.0

    if hit5 < 0.5:
        notes.append(
            f"Top-5 命中率仅 {hit5:.2f}，正确片段大量未被召回 → 瓶颈在**召回侧**"
            "（检查切片粒度、embedding 语义能力、top_k 是否过小）"
        )
    elif hit5 - mrr > 0.25:
        notes.append(
            f"Hit@5 ({hit5:.2f}) 显著高于 MRR ({mrr:.2f})：正确片段已被召回但排名靠后 "
            "→ 瓶颈在**排序侧**，优化重排的收益最大"
        )
    else:
        notes.append(f"Hit@5 ({hit5:.2f}) 与 MRR ({mrr:.2f}) 较为接近，召回与排序表现均衡")

    if hit5 > 0 and hit1 < hit5 / 2:
        notes.append(
            f"Hit@1 ({hit1:.2f}) 不到 Hit@5 的一半，首位答案不可靠 → "
            "建议开启重排，或引入 cross-encoder 做精排"
        )

    if recall5 > hit5 + 0.05:
        notes.append(
            f"Recall@5 ({recall5:.2f}) 高于 Hit@5 ({hit5:.2f})，说明同一问题需要多个片段才能覆盖，"
            "top_k 偏小会限制完整作答"
        )

    if summary.get("generation_metrics_available") is False:
        notes.append(
            "当前为 mock 生成后端，生成质量与拒答指标不可用（见上方说明），"
            "检索指标不受影响、仍然有效"
        )

    return notes


def _fmt(value, digits: int = 4, suffix: str = "") -> str:
    """表格单元格格式化：None 显示为 n/a（而不是 0），并标注无数据。"""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def build_report(results: list[dict], judge_name: str, corpus_size: int) -> str:
    """生成 Markdown 评测报告。"""
    lines: list[str] = []
    lines.append("# RAG 评测报告")
    lines.append("")
    lines.append(f"- 语料文档数：**{corpus_size}**")
    lines.append(f"- 评测样本数：**{results[0]['summary']['sample_total']}**"
                 f"（有答案 {results[0]['summary']['answerable']} / "
                 f"无答案 {results[0]['summary']['unanswerable']}）")
    lines.append(f"- 裁判后端：**{judge_name}**")
    lines.append("")

    if judge_name == "rule-based":
        lines.append("> **注意**：当前未配置裁判模型 API Key，已降级为规则裁判。")
        lines.append("> 因此 `faithfulness`（忠实度）与 `answer_relevancy`（答案相关性）"
                     "两个指标显示为 `n/a`——")
        lines.append("> 这两项必须由模型做语义判断，用关键词规则冒充会得出误导性的数字。")
        lines.append("> 配置 `JUDGE_API_KEY`（或 `ZHIPU_API_KEY`）后重跑即可获得这两项指标。")
        lines.append("")

    # ---------- 检索指标对比 ----------
    lines.append("## 检索指标（有答案样本）")
    lines.append("")
    header = "| 配置 | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | P@3 | MRR | nDCG@5 |"
    lines.append(header)
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for res in results:
        s = res["summary"]
        lines.append(
            f"| `{res['config']}` | {_fmt(s['hit_at_1'])} | {_fmt(s['hit_at_3'])} | "
            f"{_fmt(s['hit_at_5'])} | {_fmt(s['recall_at_3'])} | {_fmt(s['recall_at_5'])} | "
            f"{_fmt(s['precision_at_3'])} | {_fmt(s['mrr'])} | {_fmt(s['ndcg_at_5'])} |"
        )
    lines.append("")

    # ---------- 自动诊断 ----------
    lines.append("## 自动诊断")
    lines.append("")
    lines.append("根据指标之间的关系自动定位瓶颈环节（召回侧 / 排序侧）：")
    lines.append("")
    for res in results:
        lines.append(f"**`{res['config']}`**")
        lines.append("")
        for note in diagnose(res["summary"]):
            lines.append(f"- {note}")
        lines.append("")

    # ---------- 生成指标对比 ----------
    lines.append("## 生成指标（有答案样本）")
    lines.append("")
    lines.append("| 配置 | 关键词覆盖 | 上下文覆盖 | 忠实度 | 答案相关性 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for res in results:
        s = res["summary"]
        lines.append(
            f"| `{res['config']}` | {_fmt(s['keyword_coverage'])} | {_fmt(s['context_recall'])} | "
            f"{_fmt(s['faithfulness'])} | {_fmt(s['answer_relevancy'])} |"
        )
    lines.append("")
    lines.append("说明：**关键词覆盖**与**上下文覆盖**是零依赖规则指标，反映答案／上下文"
                 "对标准答案要素的覆盖程度，")
    lines.append("属于必要不充分条件（覆盖低必然是错的，覆盖高不必然对）。"
                 "**忠实度**与**答案相关性**由裁判模型给出。")
    lines.append("")

    if any(res["summary"].get("generation_metrics_available") is False for res in results):
        lines.append("> **重要**：部分配置使用 mock 生成后端。mock 的输出是固定模板字符串，"
                     "并非真实作答，")
        lines.append("> 因此其中的**关键词覆盖**天然接近于 0——这反映的是"
                     "「mock 不能答题」，**不代表检索或系统质量差**。")
        lines.append("> 同理，这些配置下的拒答指标显示为 `n/a`："
                     "mock 的答案既不构成作答、也不构成拒答，无法用于评判拒答能力。")
        lines.append("> 这部分结论需在接入真实生成后端（`--llm deepseek` / `--llm ollama`）后重新评测。")
        lines.append("")

    # ---------- 拒答指标 ----------
    lines.append("## 拒答能力（无答案样本）")
    lines.append("")
    lines.append("| 配置 | 正确拒答率 | 误拒率 | P50 延迟 | P95 延迟 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for res in results:
        s = res["summary"]
        lines.append(
            f"| `{res['config']}` | {_fmt(s['abstention_accuracy'])} | "
            f"{_fmt(s['false_abstention_rate'])} | {_fmt(s['latency_p50_ms'], 1, ' ms')} | "
            f"{_fmt(s['latency_p95_ms'], 1, ' ms')} |"
        )
    lines.append("")
    lines.append("说明：正确拒答率过低说明系统倾向于在无依据时强行作答（幻觉风险）；"
                 "误拒率过高说明系统过于保守。两个指标必须一起看。")
    lines.append("")

    # ---------- 按难度分层 ----------
    lines.append("## 按难度分层（nDCG@5 / MRR）")
    lines.append("")
    for res in results:
        rows = res["rows"]
        lines.append(f"### 配置 `{res['config']}`")
        lines.append("")
        lines.append("| 难度 | 样本数 | MRR | nDCG@5 |")
        lines.append("| --- | --- | --- | --- |")
        for level in ("easy", "medium", "hard"):
            subset = [r for r in rows if r["difficulty"] == level and not r["is_unanswerable"]]
            if not subset:
                continue
            lines.append(
                f"| {level} | {len(subset)} | "
                f"{_fmt(M.average(r['mrr'] for r in subset))} | "
                f"{_fmt(M.average(r['ndcg_at_5'] for r in subset))} |"
            )
        lines.append("")

    # ---------- 失败案例 ----------
    lines.append("## 失败案例分析")
    lines.append("")
    for res in results:
        rows = [r for r in res["rows"] if not r["is_unanswerable"]]
        # 失败定义：Top-5 内没有任何相关片段（召回彻底失败），按 MRR 升序取最差几条
        failures = sorted(rows, key=lambda r: (r["mrr"], r["ndcg_at_5"]))[:5]
        if not failures:
            continue
        lines.append(f"### 配置 `{res['config']}`")
        lines.append("")
        lines.append("| 样本 | 难度 | MRR | 召回文档 | 说明 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in failures:
            top_docs = ", ".join(dict.fromkeys(r["retrieved_docs"]))[:60] or "（无召回）"
            note = "完全未命中" if r["mrr"] == 0 else "命中但排名靠后"
            lines.append(
                f"| {r['id']} | {r['difficulty']} | {_fmt(r['mrr'])} | {top_docs} | {note} |"
            )
        lines.append("")

    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    # 1. 加载并自检评测集——标注有问题就不进入评测
    corpus = load_corpus()
    samples = load_golden_set()
    print_summary(samples, corpus)
    problems = validate(samples, corpus)
    if problems:
        print(f"\n评测集自检未通过（{len(problems)} 个问题），中止评测：")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("评测集自检通过。\n")

    # 2. 组装配置矩阵
    embeddings = [e.strip() for e in args.embedding.split(",") if e.strip()]
    rerank_options = [False, True] if args.rerank == "both" else [args.rerank == "on"]
    hash_dims: list[int | None] = [int(d) for d in args.hash_dims.split(",") if d.strip()]

    configs: list[EvalConfig] = []
    for emb in embeddings:
        # 哈希维度只对 hash 后端有意义；其它后端忽略该参数，
        # 否则会对 bge 生成多份完全相同的配置，白白重复跑评测。
        dims: list[int | None] = hash_dims if emb == "hash" else [None]
        for dim in dims:
            for rr in rerank_options:
                configs.append(
                    EvalConfig(
                        embedding=emb,
                        vector=args.vector,
                        llm=args.llm,
                        rerank=rr,
                        top_k=args.top_k,
                        hash_dim=dim,
                    )
                )

    # --judge rule 强制使用零依赖规则裁判。
    # 用途：纯检索评测（调切片、调 top_k、对比 embedding）不需要裁判，
    # 强制规则裁判可以在不消耗任何 API 调用的情况下复现检索指标。
    judge = RuleBasedJudge() if args.judge == "rule" else get_judge()
    print(f"裁判后端：{judge.name}")
    print(f"待评测配置：{[c.label for c in configs]}\n")

    results = []
    try:
        for cfg in configs:
            print(f"=== 评测配置：{cfg.label} ===")
            started = time.perf_counter()
            rows, index_seconds = await evaluate_config(
                cfg, corpus, samples, judge, args.concurrency
            )
            elapsed = time.perf_counter() - started

            summary = summarize(rows, cfg.llm)
            results.append(
                {
                    "config": cfg.label,
                    "llm_backend": cfg.llm,
                    "config_detail": asdict(cfg),
                    "summary": summary,
                    "rows": [asdict(r) for r in rows],
                    "index_seconds": round(index_seconds, 3),
                    "eval_seconds": round(elapsed, 3),
                }
            )
            print(
                f"  完成：MRR={summary['mrr']:.4f} nDCG@5={summary['ndcg_at_5']:.4f} "
                f"Hit@5={summary['hit_at_5']:.4f} "
                f"P50={summary['latency_p50_ms']:.1f}ms 索引耗时={index_seconds:.2f}s"
            )
    finally:
        if hasattr(judge, "aclose"):
            await judge.aclose()

    # 3. 落盘
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for res in results:
        detail_path = RESULTS_DIR / f"detail_{res['config']}.json"
        detail_path.write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    suffix = f"_{args.tag}" if args.tag else ""
    report = build_report(results, judge.name, len(corpus))
    report_path = RESULTS_DIR / f"report{suffix}.md"
    report_path.write_text(report, encoding="utf-8")

    summary_path = RESULTS_DIR / f"summary{suffix}.json"
    summary_path.write_text(
        json.dumps(
            [{k: v for k, v in res.items() if k != "rows"} for res in results],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n评测完成。报告：{report_path}")
    print(f"明细：{RESULTS_DIR}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 评测")
    parser.add_argument(
        "--embedding",
        default="hash",
        help="embedding 后端，逗号分隔（hash / bge / zhipu），默认 hash",
    )
    parser.add_argument("--vector", default="memory", help="向量库后端（memory / chroma）")
    parser.add_argument(
        "--llm", default="mock", help="生成后端（mock / ollama / deepseek / glm4）"
    )
    parser.add_argument(
        "--rerank",
        default="both",
        choices=("on", "off", "both"),
        help="是否开启重排；both 表示两种都跑，用于对比重排增益",
    )
    parser.add_argument(
        "--hash-dims",
        default="256",
        dest="hash_dims",
        help="hash embedding 的向量维度，逗号分隔（如 256,4096）。仅对 hash 后端生效，"
             "用于观察哈希碰撞对检索质量的影响",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, dest="top_k")
    parser.add_argument("--concurrency", type=int, default=4, help="样本并发数")
    parser.add_argument(
        "--tag",
        default="",
        help="输出文件名后缀（如 --tag deepseek 会写出 report_deepseek.md）。"
             "用于同时保留多组不同后端的评测结果，避免后跑的覆盖先跑的",
    )
    parser.add_argument(
        "--judge",
        default="auto",
        choices=("auto", "rule"),
        help="裁判后端。auto 按环境变量自动选择；rule 强制使用零依赖规则裁判，"
             "适合只关心检索指标的评测，不消耗 API 调用",
    )
    return parser.parse_args()


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
