"""评测集自检：在跑评测之前，先验证标注本身是自洽的。

为什么需要这一步
----------------
评测集的标注错误会**静默污染所有指标**：如果某条样本的 `must_contain` 写错了字
（或指向了不存在的文档），这条样本永远不可能被判定为命中，对应的检索指标被永久
压低，而报告里看不出任何异常——只会表现为"系统效果差"，导致把时间浪费在调优
代码而非修正标注上。

因此把"校验评测集"作为评测流水线的第一道门禁：
    python -m eval.validate_set

校验项：
1. 每条样本的必填字段齐全，id 全局唯一；
2. relevant_docs 引用的文档真实存在；
3. must_contain 的每个关键串确实出现在其所标注的文档中（若不出现，标注必然失效）；
4. answer_keywords 的每个关键词能在标注文档中找到（作为答案要素的合理性检查）；
5. unanswerable 样本不得标注 relevant_docs（否则语义自相矛盾）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
CORPUS_DIR = EVAL_DIR / "corpus"
GOLDEN_SET = EVAL_DIR / "golden_set.jsonl"

REQUIRED_FIELDS = ("id", "question", "type", "difficulty", "relevant_docs", "must_contain")


class ValidationError(ValueError):
    """评测集结构或标注内容不合法。"""


def load_corpus() -> dict[str, str]:
    """加载语料库，返回 {文档名: 全文}。文档名即上传时的 name。"""
    corpus: dict[str, str] = {}
    for path in sorted(CORPUS_DIR.glob("*.md")):
        corpus[path.name] = path.read_text(encoding="utf-8")
    if not corpus:
        raise ValidationError(f"语料目录为空：{CORPUS_DIR}")
    return corpus


def load_golden_set() -> list[dict]:
    """加载评测集（JSON Lines，每行一个样本）。"""
    if not GOLDEN_SET.exists():
        raise ValidationError(f"评测集不存在：{GOLDEN_SET}")
    samples: list[dict] = []
    for lineno, raw in enumerate(GOLDEN_SET.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            samples.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValidationError(f"第 {lineno} 行不是合法 JSON：{exc}") from exc
    if not samples:
        raise ValidationError("评测集为空")
    return samples


def validate(samples: list[dict], corpus: dict[str, str]) -> list[str]:
    """执行全部校验，返回问题描述列表（空列表表示全部通过）。"""
    problems: list[str] = []
    seen_ids: set[str] = set()

    for sample in samples:
        sid = sample.get("id", "<无 id>")

        # 1. 字段完整性
        for field in REQUIRED_FIELDS:
            if field not in sample:
                problems.append(f"[{sid}] 缺少必填字段：{field}")
        if "id" in sample:
            if sid in seen_ids:
                problems.append(f"[{sid}] id 重复")
            seen_ids.add(sid)

        docs = sample.get("relevant_docs", [])
        is_unanswerable = sample.get("type") == "unanswerable"

        # 2. 文档引用有效性
        for doc in docs:
            if doc not in corpus:
                problems.append(f"[{sid}] relevant_docs 引用了不存在的文档：{doc}")

        # 3. 无答案样本不得标注相关文档（语义自相矛盾）
        if is_unanswerable:
            if docs:
                problems.append(f"[{sid}] unanswerable 样本不应标注 relevant_docs")
            if sample.get("must_contain"):
                problems.append(f"[{sid}] unanswerable 样本不应标注 must_contain")
            continue

        # 4. 有答案样本必须至少标注一个文档
        if not docs:
            problems.append(f"[{sid}] 有答案样本必须标注 relevant_docs")
            continue

        # 5. must_contain 必须真实存在于所标注的文档中
        #    这是最关键的一项：关键串写错/过时会让该样本永远判不中
        joined_docs = "\n".join(corpus[doc] for doc in docs if doc in corpus)
        for keyword in sample.get("must_contain", []):
            if keyword not in joined_docs:
                problems.append(
                    f"[{sid}] must_contain 关键串在标注文档中找不到：{keyword!r}"
                )

        # 6. answer_keywords 作为答案要素，应能在标注文档中找到
        #    （允许出现在任意一篇标注文档中，因为多跳问题的要素可能分散在多篇）
        for keyword in sample.get("answer_keywords", []):
            if keyword not in joined_docs:
                problems.append(
                    f"[{sid}] answer_keywords 关键词在标注文档中找不到：{keyword!r}"
                )

    return problems


def print_summary(samples: list[dict], corpus: dict[str, str]) -> None:
    """打印评测集构成概况，便于确认难度与类型分布是否合理。"""
    from collections import Counter

    print(f"语料文档数：{len(corpus)}")
    print(f"评测样本数：{len(samples)}")
    print(f"  类型分布：{dict(Counter(s.get('type') for s in samples))}")
    print(f"  难度分布：{dict(Counter(s.get('difficulty') for s in samples))}")
    unanswerable = sum(1 for s in samples if s.get("type") == "unanswerable")
    print(f"  无答案样本：{unanswerable} 条（考察拒答能力）")

    # 语料是否被覆盖：未出现在任何样本中的文档说明评测覆盖不全
    covered = {doc for s in samples for doc in s.get("relevant_docs", [])}
    uncovered = sorted(set(corpus) - covered)
    if uncovered:
        print(f"  未被任何样本标注的文档：{uncovered}")
    else:
        print("  全部语料文档均被标注覆盖")


def main() -> int:
    try:
        corpus = load_corpus()
        samples = load_golden_set()
    except ValidationError as exc:
        print(f"加载失败：{exc}", file=sys.stderr)
        return 1

    print_summary(samples, corpus)
    print("-" * 60)

    problems = validate(samples, corpus)
    if problems:
        print(f"校验未通过，发现 {len(problems)} 个问题：")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("评测集校验通过：字段完整、文档引用有效、关键串全部可命中。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
