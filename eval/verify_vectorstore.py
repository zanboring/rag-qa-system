"""跨向量库一致性验证：Chroma 与内存库应给出相同的排序与 score。

用法
----
    pip install -r requirements-chroma.txt
    python -m eval.verify_vectorstore

为什么需要这个验证
------------------
Chroma 的默认距离度量是 **L2（欧氏距离）**，而本项目其它后端用的是**余弦相似度**。
如果直接把 `1 - distance` 当成相似度使用，在 L2 下数值是错的：

    L2 距离 = 0.5  ⇒  实际余弦 ≈ 0.875，而 1 - 0.5 = 0.5

更麻烦的是**排序结果依然正确**（向量归一化后 L2 与余弦单调等价），
所以常规功能测试完全发现不了问题——它只会让 `score` 失去绝对量纲，
进而让依赖 score 绝对值的下游计算（置信度归一化、固定阈值判拒答）悄悄失真。

修复方式是创建 collection 时指定 `metadata={"hnsw:space": "cosine"}`。
本脚本用同一批文本分别写入两个后端并检索，逐条比对**排序**与 **score 数值**，
给出修复是否生效的直接证据——这类"能跑通但数值错"的问题，
只能靠跨后端的一致性比对来暴露。
"""

from __future__ import annotations

import asyncio
import sys

from app.embedder import HashEmbedder
from app.vectorstore import MemoryVectorStore

DOCS = [
    ("d:0", "快速排序是一种分治算法，平均时间复杂度 O(n log n)。"),
    ("d:1", "归并排序同样是分治，但需要额外的辅助空间。"),
    ("d:2", "堆排序利用二叉堆结构，原地排序但常数因子较大。"),
    ("d:3", "冒泡排序通过相邻元素交换实现，平均复杂度 O(n^2)。"),
    ("d:4", "向量数据库用于存储和检索高维嵌入向量。"),
]
QUERIES = ["快速排序的时间复杂度", "排序算法有哪些", "向量数据库检索"]

# Chroma 内部以 float32 存储，允许浮点精度差异
SCORE_TOLERANCE = 1e-4


async def main() -> int:
    try:
        from app.vectorstore import ChromaVectorStore
    except ImportError:
        print("未安装 chromadb。请先执行：pip install -r requirements-chroma.txt", file=sys.stderr)
        return 1

    try:
        chroma = ChromaVectorStore("consistency_check")
    except ImportError:
        print("未安装 chromadb。请先执行：pip install -r requirements-chroma.txt", file=sys.stderr)
        return 1

    embedder = HashEmbedder(dim=512)
    memory = MemoryVectorStore()

    # 同一批向量同时写入两个后端
    for doc_id, text in DOCS:
        vector = await embedder.embed(text)
        metadata = {"doc_name": doc_id, "chunk_index": 0, "text": text}
        await memory.add(doc_id, vector, metadata)
        await chroma.add(doc_id, vector, metadata)

    order_mismatch = 0
    score_mismatch = 0

    for query in QUERIES:
        query_vec = await embedder.embed(query)
        m_hits = await memory.search(query_vec, 3)
        c_hits = await chroma.search(query_vec, 3)

        m_ids = [h["id"] for h in m_hits]
        c_ids = [h["id"] for h in c_hits]

        print(f"查询：{query}")
        print(f"  memory  top3={m_ids}  top1_score={m_hits[0]['score']:.6f}")
        print(f"  chroma  top3={c_ids}  top1_score={c_hits[0]['score']:.6f}")

        if m_ids != c_ids:
            order_mismatch += 1
            print("  ⚠ 排序不一致")
        else:
            worst = max(abs(a["score"] - b["score"]) for a, b in zip(m_hits, c_hits))
            ok = worst < SCORE_TOLERANCE
            if not ok:
                score_mismatch += 1
            print(f"  score 最大差值 = {worst:.2e}  {'OK' if ok else 'MISMATCH'}")
        print()

    print("=" * 60)
    if order_mismatch == 0 and score_mismatch == 0:
        print("结论：两个后端的排序与 score 完全一致 ✅")
        print("      cosine space 修复有效，score 量纲已对齐")
        return 0

    print(f"结论：排序不一致 {order_mismatch} 处，score 超差 {score_mismatch} 处 ❌")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
