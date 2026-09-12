"""评测裁判的单元测试：JSON 解析容错、分数夹取、规则裁判的拒答判定。

只覆盖不依赖网络的部分。LLMJudge 的真实调用属于集成测试（需要 API Key），
不放进单元测试——否则 `pytest` 会在无网络 / 无 Key 的环境下失败，
而一个"本地跑不过的测试套件"等于没有测试。
"""

import pytest

from eval.judge import (
    MOCK_MARKER,
    JudgeScore,
    RuleBasedJudge,
    _clamp,
    _extract_json,
)


# ---------------------------------------------------------------------------
# _extract_json：模型输出格式不稳定，解析必须容错
# ---------------------------------------------------------------------------

def test_extract_json_plain():
    assert _extract_json('{"faithfulness": 1.0}') == {"faithfulness": 1.0}


def test_extract_json_from_markdown_fence():
    """模型常把 JSON 包在 ```json 代码块里。"""
    raw = '```json\n{"faithfulness": 0.8, "answer_relevancy": 0.9}\n```'
    assert _extract_json(raw) == {"faithfulness": 0.8, "answer_relevancy": 0.9}


def test_extract_json_with_surrounding_text():
    """模型常在 JSON 前后附带解释性文字。"""
    raw = '好的，我的评估如下：\n{"faithfulness": 0.5}\n以上是评分依据。'
    assert _extract_json(raw) == {"faithfulness": 0.5}


def test_extract_json_with_nested_object():
    """嵌套结构不能被"第一个 { 到最后一个 }"的朴素截取弄坏。"""
    raw = '{"scores": {"faithfulness": 0.7}, "reason": "ok"}'
    assert _extract_json(raw) == {"scores": {"faithfulness": 0.7}, "reason": "ok"}


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        _extract_json("这里完全没有 JSON")


# ---------------------------------------------------------------------------
# _clamp
# ---------------------------------------------------------------------------

def test_clamp_bounds():
    assert _clamp(1.5) == 1.0
    assert _clamp(-0.3) == 0.0
    assert _clamp(0.42) == 0.42


def test_clamp_invalid_returns_none():
    """非法值必须返回 None（视为无效），而不是当成 0 分。

    这个区别很关键：把"解析失败"当成"模型给了 0 分"会系统性压低指标，
    而且从报告上完全看不出异常——只会表现为"系统效果差"。
    """
    assert _clamp("不是数字") is None
    assert _clamp(None) is None


# ---------------------------------------------------------------------------
# JudgeScore
# ---------------------------------------------------------------------------

def test_judge_score_usable():
    assert JudgeScore(faithfulness=0.8).usable is True
    assert JudgeScore(faithfulness=None).usable is False


# ---------------------------------------------------------------------------
# RuleBasedJudge：零依赖降级路径
# ---------------------------------------------------------------------------

async def test_rule_judge_detects_abstention():
    judge = RuleBasedJudge()
    score = await judge.score("问题", "资料中未找到相关信息，无法回答。", [])
    assert score.answered is False
    # 规则裁判不产出语义指标，必须是 None 而不是 0
    assert score.faithfulness is None
    assert score.answer_relevancy is None


async def test_rule_judge_detects_answered():
    judge = RuleBasedJudge()
    score = await judge.score("问题", "快速排序的平均复杂度是 O(n log n)。", [])
    assert score.answered is True


async def test_rule_judge_flags_mock_output():
    """mock 输出必须被标记，且 answered 为 None。

    若把 mock 判为 answered=False，上层会将其解读为"拒答"，
    于是 mock 模式下所有样本都算拒答，拒答率与误拒率同时为 1.0——
    报告会得出自相矛盾的结论。
    """
    judge = RuleBasedJudge()
    score = await judge.score("问题", f"{MOCK_MARKER} 已依据参考资料生成回答。", [])
    assert score.answered is None
    assert score.error == "mock_backend"


async def test_rule_judge_name_is_stable():
    """裁判标识会写进报告，改名要谨慎（测试作为变更提醒）。"""
    assert RuleBasedJudge().name == "rule-based"
