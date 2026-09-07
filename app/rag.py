"""RAG 编排：检索 → 组装 Prompt → 生成 → 溯源。

面试可讲点（RAG 全链路）：
1. Retrieval（检索）：query 向量化 → 向量库取 Top-K 相关片段。
2. Augmented（增强）：把片段拼成结构化 Prompt（System + 上下文 + 问题）。
3. Generation（生成）：交给 LLM 生成答案。
4. 溯源：返回每个引用片段的来源文档 + 相似度，保证可解释、防幻觉。
"""

from app.reranker import LexicalReranker, Reranker

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。"
    "请仅依据下面提供的【参考资料】回答问题，并尽量标注引用来源编号；"
    "如果资料不足以回答，请明确说明「资料中未找到相关信息」，不要编造。"
)


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
            return {"answer": "知识库为空或未检索到相关内容。", "sources": []}
        if rerank and self.reranker is not None:
            hits = self.reranker.rerank(question, hits, top_k)

        context_parts = []
        sources = []
        for i, hit in enumerate(hits, 1):
            meta = hit["metadata"]
            context_parts.append(
                f"[{i}] 来源《{meta.get('doc_name', '未知')}》片段{meta.get('chunk_index', 0)}：\n"
                f"{meta.get('text', '')}"
            )
            sources.append(
                {
                    "doc_name": meta.get("doc_name"),
                    "chunk_index": meta.get("chunk_index"),
                    "score": round(hit["score"], 4),
                    "snippet": (meta.get("text") or "")[:80],
                }
            )

        prompt = (
            "【参考资料】\n"
            + "\n\n".join(context_parts)
            + f"\n\n【用户问题】\n{question}"
        )
        answer = await self.llm.generate(SYSTEM_PROMPT, prompt)
        return {"answer": answer, "sources": sources}
