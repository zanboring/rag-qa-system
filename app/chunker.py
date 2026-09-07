"""文档切片：固定长度 + 重叠。

面试可讲点：
- 为什么切片：大模型上下文窗口有限，且过长的段落会稀释检索精度，
  把长文档切成小块，让检索命中更精准的片段。
- 为什么重叠：固定切分会把一句话从中间切断，丢失跨边界语义；
  相邻块之间留一段重叠，保证边界语义不丢失。
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """把长文本切成若干块，相邻块之间保留 overlap 个字符重叠。"""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
    return chunks
