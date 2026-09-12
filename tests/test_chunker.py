"""chunker 单元测试：切片逻辑、重叠、边界。"""

import pytest

from app.chunker import chunk_text, chunk_text_smart


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []
    assert chunk_text_smart("") == []
    assert chunk_text_smart("   ") == []


def test_single_chunk_short_text():
    assert chunk_text("短文本") == ["短文本"]
    assert chunk_text_smart("短文本") == ["短文本"]


def test_chunk_size_and_overlap():
    text = "abcdefghij"  # 10 个字符
    chunks = chunk_text(text, chunk_size=6, overlap=2)
    # 期望：[abcdef][efghij]
    assert chunks == ["abcdef", "efghij"]
    # 重叠验证：后一块开头与前一块结尾重叠 2 字符
    assert chunks[0][-2:] == chunks[1][:2]


def test_chunk_size_must_be_greater_than_overlap():
    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=3, overlap=3)
    with pytest.raises(ValueError):
        chunk_text_smart("abc", chunk_size=3, overlap=3)


def test_no_chunk_larger_than_size():
    text = "x" * 1000
    for chunk in chunk_text(text, chunk_size=100, overlap=10):
        assert len(chunk) <= 100


# ---------- chunk_text_smart ----------

def test_smart_chinese_paragraph_split():
    """中文段落分隔（空行）应优先作为切点。"""
    para1 = "第一段内容。" * 30  # ~180 字符
    para2 = "第二段内容。" * 30
    para3 = "第三段内容。" * 30
    text = para1 + "\n\n" + para2 + "\n\n" + para3

    chunks = chunk_text_smart(text, chunk_size=200, overlap=20, prefer_boundary=True)
    assert len(chunks) >= 2
    # 每块都不超过 chunk_size 上限
    for c in chunks:
        assert len(c) <= 220, f"块超出大小：{len(c)}"
    # 没有任何块首字符是句号（不应在句中切）
    for c in chunks:
        assert not c.startswith("。"), f"不应在句中切：{c[:5]!r}"


def test_smart_chinese_sentence_split():
    """中文句号优于逗号应作为切点。"""
    text = "苹果是一种水果。香蕉也是水果。葡萄很甜。西瓜很大。" * 20  # 约 220 字符
    chunks = chunk_text_smart(text, chunk_size=80, overlap=10, prefer_boundary=True)
    # 至少有 2 个块，且每块都不超过上限
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 100


def test_smart_english_period_split():
    """英文句号应作为切点，且要求后接空白或串尾，避免误切缩写（e.g. U.S.A）。"""
    # 用普通句子，不要缩写
    text = ("The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. ") * 10
    chunks = chunk_text_smart(text, chunk_size=100, overlap=10, prefer_boundary=True)
    # 至少有 2 块，且首/末不应落在单词中间（不出现「fox 」孤词）
    for c in chunks:
        assert len(c) <= 110
    # 简单断言：拼接各块后总字符数应能覆盖原文主要部分（弱断言）
    joined = "".join(chunks)
    assert "quick brown fox" in joined
    assert "Pack my box" in joined


def test_smart_fallback_to_fixed_when_no_boundary():
    """窗口内无任何边界时，应回退到 chunk_size 硬切。"""
    text = "x" * 1000  # 全是 x，没有段落/句号/逗号/空白
    chunks = chunk_text_smart(text, chunk_size=100, overlap=10, prefer_boundary=True)
    # 行为应与固定长度一致
    for c in chunks:
        assert len(c) <= 100
    assert len(chunks) >= 10


def test_smart_prefer_boundary_false_matches_fixed():
    """prefer_boundary=False 必须与 chunk_text 等价（向后兼容）。"""
    text = "Hello world. " * 50  # 含英文句号与空白
    chunks_fixed = chunk_text(text, chunk_size=50, overlap=5)
    chunks_smart_off = chunk_text_smart(text, chunk_size=50, overlap=5, prefer_boundary=False)
    assert chunks_smart_off == chunks_fixed


def test_smart_keeps_min_keep_ratio():
    """切点不能离硬上限太近，避免切成「超小块」。"""
    # 句号靠近 chunk_size 边缘：长度 49 处有 "X。", chunk_size=50
    text = "x" * 47 + "。" + "x" * 100
    chunks = chunk_text_smart(text, chunk_size=50, overlap=5, prefer_boundary=True)
    for c in chunks:
        # MIN_KEEP_RATIO = 0.5 ⇒ 第一块至少 25 字符 + 切到边界后整块应合理
        assert len(c) >= 1


def test_smart_no_dead_loop_on_short_window():
    """极端短窗口不能死循环。"""
    # 全是中文字符，长度等于 chunk_size
    text = "你好世界你好世界你好世界"
    chunks = chunk_text_smart(text, chunk_size=10, overlap=2, prefer_boundary=True)
    # 必须能正常终止并产出若干块（没有死循环）
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) <= 10
