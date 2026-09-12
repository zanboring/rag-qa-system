"""RAG 评测体系。

模块划分：
    metrics.py       指标实现（纯函数、零依赖、可单测）
    validate_set.py  评测集自检门禁（跑评测前必须先过）
    judge.py         LLM-as-judge（云端/v1 兼容接口，可降级为规则打分）
    run_eval.py      评测主流程：加载语料 → 索引 → 检索 → 生成 → 打分 → 报告

目录约定：
    corpus/          评测语料（Markdown，文件名即文档 name）
    golden_set.jsonl 标注评测集（每行一个样本）
    results/         评测输出（JSON 明细 + Markdown 报告）
"""
