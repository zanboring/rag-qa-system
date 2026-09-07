"""chunker 单元测试：切片逻辑、重叠、边界。"""

from app.chunker import chunk_text


def test_empty_text():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_single_chunk_short_text():
    assert chunk_text("短文本") == ["短文本"]


def test_chunk_size_and_overlap():
    text = "abcdefghij"  # 10 个字符
    chunks = chunk_text(text, chunk_size=6, overlap=2)
    # 期望：[abcdef][efghij]
    assert chunks == ["abcdef", "efghij"]
    # 重叠验证：后一块开头与前一块结尾重叠 2 字符
    assert chunks[0][-2:] == chunks[1][:2]


def test_chunk_size_must_be_greater_than_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("abc", chunk_size=3, overlap=3)


def test_no_chunk_larger_than_size():
    text = "x" * 1000
    for chunk in chunk_text(text, chunk_size=100, overlap=10):
        assert len(chunk) <= 100
