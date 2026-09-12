"""文档切片：固定长度 + 重叠。

面试可讲点：
- 为什么切片：大模型上下文窗口有限，且过长的段落会稀释检索精度，
  把长文档切成小块，让检索命中更精准的片段。
- 为什么重叠：固定切分会把一句话从中间切断，丢失跨边界语义；
  相邻块之间留一段重叠，保证边界语义不丢失。
- 边界优先（chunk_text_smart）：固定切片的好处是实现简单、可预测，
  坏处是切点可能正落在单词/句子中间；prefer_boundary=True 时优先
  在最近的段落/句子分隔处断开，减少「一个句子被切成两半」的概率。
"""

import re


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


# 边界优先级（从强到弱）：段落 > 句子 > 分句。
# 匹配后切到「匹配结束位置之后」，例如 "Hello world.\nNext" 切到 "." 后，
# 下一块从 "Next" 开头，避免把分隔符本身带进前一块的尾部又带进后一块的头部。
_BOUNDARY_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\s*\n"),          # 段落分隔（空行）
    re.compile(r"\n"),               # 行分隔
    re.compile(r"[。！？]"),          # 中文句末
    re.compile(r"[.!?](?=\s|$)"),    # 英文句末（要求后接空白或串尾，避免误切缩写）
    re.compile(r"[，,；;]"),          # 中英文分句
    re.compile(r"\s"),               # 任意空白（最后兜底）
)


def _find_boundary(text: str, start: int, hard_end: int) -> int:
    """在 [start, hard_end] 区间内找最近的语义边界位置。

    - 找不到任何边界时返回 hard_end（行为与 chunk_text 一致）。
    - 切点位置 >= start + chunk_size * MIN_KEEP_RATIO，避免把块切得太碎
      （极端情况：句号恰好贴近 chunk_size，会让下一块开头只剩 1 个字符）。
    - 切点要求 > start（至少前进 1 个字符），防止原地打转。
    """
    MIN_KEEP_RATIO = 0.5  # 切点至少保留前半段 50%，过近则视为无效

    window_len = hard_end - start
    min_keep = start + max(int(window_len * MIN_KEEP_RATIO), 1)
    if min_keep <= start:
        min_keep = start + 1

    search_start = max(min_keep, start + 1)
    window = text[search_start:hard_end]

    # 取每种分隔符在窗口内的「最后一个匹配」（最靠近 hard_end，切得最大块）
    for pattern in _BOUNDARY_PATTERNS:
        last_end: int | None = None
        for m in pattern.finditer(window):
            last_end = m.end()
        if last_end is not None:
            cut_at = search_start + last_end
            if cut_at > start and cut_at <= hard_end:
                return cut_at

    return hard_end


def chunk_text_smart(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    prefer_boundary: bool = True,
) -> list[str]:
    """边界感知的切片：在 chunk_size 上限内优先找最近的语义分隔处。

    与 chunk_text 的关系：
    - 切片长度同样受 chunk_size 约束（不会切出超长块）。
    - 重叠机制完全相同。
    - 区别仅在切点选择：当窗口内存在段落/句子/分句边界时，
      优先切到边界后；找不到时回退到 chunk_size 硬切，保证行为不退化。
    - prefer_boundary=False 时退化为固定长度切片，与 chunk_text 一致。

    设计取舍：
    - 没把 prefer_boundary 默认改成 True：保持「旧调用方 / 旧测试结果不变」，
      新调用方显式启用，避免隐式改变线上行为。
    - 边界优先级串行查找而非单条大正则：可读性优先，O(N) 文本 + 6 个小正则
      比预编译一条巨型 alternation 更易维护；性能上无可见差距。
    """
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")

    if not prefer_boundary:
        return chunk_text(text, chunk_size, overlap)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        hard_end = min(start + chunk_size, n)
        if hard_end >= n:
            # 最后一段：直接收下剩余内容，无需找边界
            piece = text[start:hard_end].strip()
            if piece:
                chunks.append(piece)
            break

        end = _find_boundary(text, start, hard_end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end <= start:
            # 防止零推进造成的死循环（极端短窗口）
            end = hard_end
        start = max(end - overlap, start + 1)
    return chunks
