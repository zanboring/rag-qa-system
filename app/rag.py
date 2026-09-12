"""RAG 编排：检索 → 组装 Prompt → 生成 → 溯源。

面试可讲点（RAG 全链路）：
1. Retrieval（检索）：query 向量化 → 向量库取 Top-K 相关片段。
2. Augmented（增强）：把片段拼成结构化 Prompt（System + 上下文 + 问题）。
3. Generation（生成）：交给 LLM 生成答案。
4. 溯源：返回每个引用片段的来源文档 + 相似度 + 关键词高亮，保证可解释、防幻觉。

置信度（confidence）：基于「top1 / Σscores」得到的相对置信度 [0, 1]。
- 语义：当第一名片段远高于其他片段时，confidence 接近 1（强证据）；
        当各片段分数接近时，confidence 偏低（无明显证据，需要谨慎）。
- 用途：前端可基于此决定是否展示"低置信度提示"，或要求 LLM 输出"资料不足"。
"""

from __future__ import annotations

import re

from app.reranker import LexicalReranker, Reranker

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。"
    "请仅依据下面提供的【参考资料】回答问题，并尽量标注引用来源编号；"
    "如果资料不足以回答，请明确说明「资料中未找到相关信息」，不要编造。"
)

# 高亮用 Markdown 风格 **加粗**，与常见 LLM 输出格式兼容
_HIGHLIGHT_OPEN = "**"
_HIGHLIGHT_CLOSE = "**"
# 关键词长度限制：太短易误命中；过长则几乎不可能命中
_MIN_KEYWORD_LEN = 2
_MAX_KEYWORDS = 8
# snippet 最长字符数：超过会截断到该长度并在末尾追加省略号
SNIPPET_MAX_LEN = 120


def _extract_keywords(question: str) -> list[str]:
    """从 query 抽取关键词用于高亮。

    策略（零依赖、不依赖分词器）：
    - 英文 / 数字按非字母数字边界切分；
    - 中文：先按「非汉字」切，再按停用词细分，过滤掉长度 < 2 的段；
    - 去重 + 长度限制（_MAX_KEYWORDS）。
    实际生产可接入 jieba / HanLP；这里给零依赖的演示实现。

    示例："快速排序的时间复杂度是多少？" → ["快速排序", "时间复杂度", "多少"]
    """
    if not question:
        return []

    # 1) 英文 / 数字单词（按非字母数字切分）
    en_tokens = re.findall(r"[A-Za-z0-9]+", question)

    # 2) 中文：先把非汉字替换为空格，再按停用词逐字插入空格（让单字停用词也参与切分）
    cn_only = re.sub(r"[^\u4e00-\u9fff]", " ", question)
    stopwords = {"的", "是", "在", "我", "你", "他", "她", "它", "吗", "呢", "啊", "了", "和"}
    for sw in stopwords:
        cn_only = cn_only.replace(sw, " ")
    cn_tokens = [seg for seg in cn_only.split() if len(seg) >= _MIN_KEYWORD_LEN]

    # 3) 长度过滤 + 去重（保留首次出现顺序）
    seen: set[str] = set()
    keywords: list[str] = []
    for token in en_tokens + cn_tokens:
        if len(token) < _MIN_KEYWORD_LEN:
            continue
        key = token.lower() if token.isascii() else token
        if key in seen:
            continue
        seen.add(key)
        keywords.append(token)
        if len(keywords) >= _MAX_KEYWORDS:
            break

    return keywords


def _highlight_text(text: str, keywords: list[str]) -> str:
    """在文本中包裹关键词（大小写不敏感），不修改原文、不破坏原始 token 边界。

    修复：用单次正则 + alternation，长关键词排在前面避免子串被二次包裹
    （旧版依次 re.sub 会出现 "快速排序" 内部的 "排序" 再次被包成 **快速**排序**）。
    """
    if not text or not keywords:
        return text

    # 去空 + 去重 + 按长度降序（关键修复：长关键词优先匹配，子串不会再被二次包裹）
    kws = sorted({k for k in keywords if k}, key=len, reverse=True)
    if not kws:
        return text

    # 合并为单条 alternation 正则，IGNORECASE 保持原行为
    pattern = re.compile("|".join(re.escape(kw) for kw in kws), re.IGNORECASE)
    return pattern.sub(lambda m: f"{_HIGHLIGHT_OPEN}{m.group(0)}{_HIGHLIGHT_CLOSE}", text)


def _confidence_from_scores(scores: list[float]) -> list[float]:
    """把绝对 score（余弦相似度等）转换成「相对置信度」 [0, 1]。

    计算：confidence_i = score_i / Σscores
    - 全部相等 → 每个 1/n（无明显证据）
    - 单点高分其余为 0 → 接近 1（强证据）
    - 注意：当 Σscores 接近 0 时全部置 0（极端边界）
    """
    total = sum(scores)
    if total <= 0:
        return [0.0 for _ in scores]
    return [round(s / total, 4) for s in scores]


class RAGPipeline:
    def __init__(self, embedder, vectorstore, llm, reranker: Reranker | None = None):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        # 默认启用零依赖词法重排器；传 None 可关闭重排
        self.reranker = reranker if reranker is not None else LexicalReranker()

    async def ask(self, question: str, top_k: int = 3, rerank: bool = False) -> dict:
        # 开启重排时，先多召回候选（3 倍），再精排回 top_k
        recall_k = top_k * 3 if rerank else top_k
        q_vec = await self.embedder.embed(question)
        hits = await self.vectorstore.search(q_vec, recall_k)
        if not hits:
            return {
                "answer": "知识库为空或未检索到相关内容。",
                "sources": [],
                "confidence_avg": 0.0,
                "top_score": 0.0,
            }
        if rerank and self.reranker is not None:
            hits = self.reranker.rerank(question, hits, top_k)

        # 置信度：基于所有命中片段的绝对 score 做相对归一化
        raw_scores = [float(h["score"]) for h in hits]
        confidences = _confidence_from_scores(raw_scores)

        keywords = _extract_keywords(question)

        context_parts = []
        sources = []
        for i, (hit, confidence) in enumerate(zip(hits, confidences), 1):
            meta = hit["metadata"]
            full_text = meta.get("text", "") or ""
            # snippet 取前 SNIPPET_MAX_LEN 字符（带高亮）
            snippet_src = full_text[:SNIPPET_MAX_LEN]
            snippet = _highlight_text(snippet_src, keywords)

            context_parts.append(
                f"[{i}] 来源《{meta.get('doc_name', '未知')}》片段{meta.get('chunk_index', 0)}：\n"
                f"{full_text}"
            )
            sources.append(
                {
                    "doc_name": meta.get("doc_name"),
                    "chunk_index": meta.get("chunk_index"),
                    "score": round(float(hit["score"]), 4),
                    "confidence": confidence,
                    "snippet": snippet,
                }
            )

        prompt = (
            "【参考资料】\n"
            + "\n\n".join(context_parts)
            + f"\n\n【用户问题】\n{question}"
        )
        answer = await self.llm.generate(SYSTEM_PROMPT, prompt)

        # 顶层置信度聚合：top_score = max(confidence)；confidence_avg = 平均
        top_conf = max(confidences) if confidences else 0.0
        avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        return {
            "answer": answer,
            "sources": sources,
            "confidence_avg": avg_conf,
            "top_score": round(float(max(raw_scores)), 4) if raw_scores else 0.0,
        }
