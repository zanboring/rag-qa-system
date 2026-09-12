# RAG 评测报告

- 语料文档数：**16**
- 评测样本数：**54**（有答案 50 / 无答案 4）
- 裁判后端：**rule-based**

> **注意**：当前未配置裁判模型 API Key，已降级为规则裁判。
> 因此 `faithfulness`（忠实度）与 `answer_relevancy`（答案相关性）两个指标显示为 `n/a`——
> 这两项必须由模型做语义判断，用关键词规则冒充会得出误导性的数字。
> 配置 `JUDGE_API_KEY`（或 `ZHIPU_API_KEY`）后重跑即可获得这两项指标。

## 检索指标（有答案样本）

| 配置 | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | P@3 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hash+memory+mock+rerankoff` | 0.2800 | 0.5400 | 0.5800 | 0.4900 | 0.5300 | 0.1867 | 0.4090 | 0.4495 |
| `hash+memory+mock+rerankon` | 0.4400 | 0.6200 | 0.7000 | 0.5800 | 0.6500 | 0.2267 | 0.5403 | 0.5798 |

## 自动诊断

根据指标之间的关系自动定位瓶颈环节（召回侧 / 排序侧）：

**`hash+memory+mock+rerankoff`**

- Hit@5 (0.58) 与 MRR (0.41) 较为接近，召回与排序表现均衡
- Hit@1 (0.28) 不到 Hit@5 的一半，首位答案不可靠 → 建议开启重排，或引入 cross-encoder 做精排
- 当前为 mock 生成后端，生成质量与拒答指标不可用（见上方说明），检索指标不受影响、仍然有效

**`hash+memory+mock+rerankon`**

- Hit@5 (0.70) 与 MRR (0.54) 较为接近，召回与排序表现均衡
- 当前为 mock 生成后端，生成质量与拒答指标不可用（见上方说明），检索指标不受影响、仍然有效

## 生成指标（有答案样本）

| 配置 | 关键词覆盖 | 上下文覆盖 | 忠实度 | 答案相关性 |
| --- | --- | --- | --- | --- |
| `hash+memory+mock+rerankoff` | 0.1317 | 0.6733 | n/a | n/a |
| `hash+memory+mock+rerankon` | 0.1450 | 0.7933 | n/a | n/a |

说明：**关键词覆盖**与**上下文覆盖**是零依赖规则指标，反映答案／上下文对标准答案要素的覆盖程度，
属于必要不充分条件（覆盖低必然是错的，覆盖高不必然对）。**忠实度**与**答案相关性**由裁判模型给出。

> **重要**：部分配置使用 mock 生成后端。mock 的输出是固定模板字符串，并非真实作答，
> 因此其中的**关键词覆盖**天然接近于 0——这反映的是「mock 不能答题」，**不代表检索或系统质量差**。
> 同理，这些配置下的拒答指标显示为 `n/a`：mock 的答案既不构成作答、也不构成拒答，无法用于评判拒答能力。
> 这部分结论需在接入真实生成后端（`--llm glm4` 或 `--llm ollama`）后重新评测。

## 拒答能力（无答案样本）

| 配置 | 正确拒答率 | 误拒率 | P50 延迟 | P95 延迟 |
| --- | --- | --- | --- | --- |
| `hash+memory+mock+rerankoff` | n/a | n/a | 2.9 ms | 3.6 ms |
| `hash+memory+mock+rerankon` | n/a | n/a | 4.3 ms | 6.0 ms |

说明：正确拒答率过低说明系统倾向于在无依据时强行作答（幻觉风险）；误拒率过高说明系统过于保守。两个指标必须一起看。

## 按难度分层（nDCG@5 / MRR）

### 配置 `hash+memory+mock+rerankoff`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.6481 | 0.6812 |
| medium | 27 | 0.3623 | 0.4057 |
| hard | 14 | 0.3452 | 0.3852 |

### 配置 `hash+memory+mock+rerankon`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.7778 | 0.7778 |
| medium | 27 | 0.5302 | 0.5819 |
| hard | 14 | 0.4071 | 0.4485 |

## 失败案例分析

### 配置 `hash+memory+mock+rerankoff`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 10-prompt-structure.md, 16-glm4-api.md | 完全未命中 |
| q05 | easy | 0.0000 | 14-ragas-judge.md, 06-vector-db-selection.md, 01-rag-overvie | 完全未命中 |
| q06 | medium | 0.0000 | 04-vector-similarity.md, 02-chunking-strategy.md, 12-abstent | 完全未命中 |
| q09 | medium | 0.0000 | 16-glm4-api.md, 03-embedding-models.md, 02-chunking-strategy | 完全未命中 |
| q16 | medium | 0.0000 | 02-chunking-strategy.md, 14-ragas-judge.md, 08-rerank-rrf.md | 完全未命中 |

### 配置 `hash+memory+mock+rerankon`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 16-glm4-api.md, 15-ollama-local.md, 10-prompt-structure.md,  | 完全未命中 |
| q05 | easy | 0.0000 | 14-ragas-judge.md, 02-chunking-strategy.md, 01-rag-overview. | 完全未命中 |
| q16 | medium | 0.0000 | 08-rerank-rrf.md, 14-ragas-judge.md, 06-vector-db-selection. | 完全未命中 |
| q18 | medium | 0.0000 | 05-ann-index.md, 01-rag-overview.md, 16-glm4-api.md, 06-vect | 完全未命中 |
| q20 | hard | 0.0000 | 06-vector-db-selection.md, 04-vector-similarity.md, 15-ollam | 完全未命中 |
