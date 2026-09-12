# RAG 评测报告

- 语料文档数：**16**
- 评测样本数：**54**（有答案 50 / 无答案 4）
- 裁判后端：**llm:deepseek-chat**

## 检索指标（有答案样本）

| 配置 | Hit@1 | Hit@3 | Hit@5 | Recall@3 | Recall@5 | P@3 | MRR | nDCG@5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hash256+memory+deepseek+rerankoff` | 0.2800 | 0.5400 | 0.5800 | 0.4900 | 0.5300 | 0.1867 | 0.4090 | 0.4495 |
| `hash256+memory+deepseek+rerankon` | 0.4400 | 0.6200 | 0.7000 | 0.5800 | 0.6500 | 0.2267 | 0.5403 | 0.5798 |
| `hash4096+memory+deepseek+rerankoff` | 0.6800 | 0.8800 | 0.9200 | 0.8300 | 0.8900 | 0.3133 | 0.7790 | 0.8092 |
| `hash4096+memory+deepseek+rerankon` | 0.7400 | 0.9400 | 0.9600 | 0.9000 | 0.9300 | 0.3400 | 0.8283 | 0.8576 |

## 自动诊断

根据指标之间的关系自动定位瓶颈环节（召回侧 / 排序侧）：

**`hash256+memory+deepseek+rerankoff`**

- Hit@5 (0.58) 与 MRR (0.41) 较为接近，召回与排序表现均衡
- Hit@1 (0.28) 不到 Hit@5 的一半，首位答案不可靠 → 建议开启重排，或引入 cross-encoder 做精排

**`hash256+memory+deepseek+rerankon`**

- Hit@5 (0.70) 与 MRR (0.54) 较为接近，召回与排序表现均衡

**`hash4096+memory+deepseek+rerankoff`**

- Hit@5 (0.92) 与 MRR (0.78) 较为接近，召回与排序表现均衡

**`hash4096+memory+deepseek+rerankon`**

- Hit@5 (0.96) 与 MRR (0.83) 较为接近，召回与排序表现均衡

## 生成指标（有答案样本）

| 配置 | 关键词覆盖 | 上下文覆盖 | 忠实度 | 答案相关性 |
| --- | --- | --- | --- | --- |
| `hash256+memory+deepseek+rerankoff` | 0.6533 | 0.6733 | 0.9760 | 0.9600 |
| `hash256+memory+deepseek+rerankon` | 0.7800 | 0.7933 | 0.9850 | 0.9790 |
| `hash4096+memory+deepseek+rerankoff` | 0.9183 | 0.9533 | 0.9910 | 0.9890 |
| `hash4096+memory+deepseek+rerankon` | 0.9567 | 0.9933 | 0.9960 | 0.9970 |

说明：**关键词覆盖**与**上下文覆盖**是零依赖规则指标，反映答案／上下文对标准答案要素的覆盖程度，
属于必要不充分条件（覆盖低必然是错的，覆盖高不必然对）。**忠实度**与**答案相关性**由裁判模型给出。

## 拒答能力（无答案样本）

| 配置 | 正确拒答率 | 误拒率 | P50 延迟 | P95 延迟 |
| --- | --- | --- | --- | --- |
| `hash256+memory+deepseek+rerankoff` | 1.0000 | 0.3600 | 1276.8 ms | 2025.6 ms |
| `hash256+memory+deepseek+rerankon` | 1.0000 | 0.2400 | 1267.8 ms | 1937.2 ms |
| `hash4096+memory+deepseek+rerankoff` | 1.0000 | 0.0600 | 1307.9 ms | 2168.2 ms |
| `hash4096+memory+deepseek+rerankon` | 1.0000 | 0.0200 | 1379.3 ms | 2301.2 ms |

说明：正确拒答率过低说明系统倾向于在无依据时强行作答（幻觉风险）；误拒率过高说明系统过于保守。两个指标必须一起看。

## 按难度分层（nDCG@5 / MRR）

### 配置 `hash256+memory+deepseek+rerankoff`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.6481 | 0.6812 |
| medium | 27 | 0.3623 | 0.4057 |
| hard | 14 | 0.3452 | 0.3852 |

### 配置 `hash256+memory+deepseek+rerankon`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.7778 | 0.7778 |
| medium | 27 | 0.5302 | 0.5819 |
| hard | 14 | 0.4071 | 0.4485 |

### 配置 `hash4096+memory+deepseek+rerankoff`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.9444 | 0.9590 |
| medium | 27 | 0.7420 | 0.7692 |
| hard | 14 | 0.7440 | 0.7901 |

### 配置 `hash4096+memory+deepseek+rerankon`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.9444 | 0.9590 |
| medium | 27 | 0.8056 | 0.8378 |
| hard | 14 | 0.7976 | 0.8308 |

## 失败案例分析

### 配置 `hash256+memory+deepseek+rerankoff`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 10-prompt-structure.md, 16-glm4-api.md | 完全未命中 |
| q05 | easy | 0.0000 | 14-ragas-judge.md, 06-vector-db-selection.md, 01-rag-overvie | 完全未命中 |
| q06 | medium | 0.0000 | 04-vector-similarity.md, 02-chunking-strategy.md, 12-abstent | 完全未命中 |
| q09 | medium | 0.0000 | 16-glm4-api.md, 03-embedding-models.md, 02-chunking-strategy | 完全未命中 |
| q16 | medium | 0.0000 | 02-chunking-strategy.md, 14-ragas-judge.md, 08-rerank-rrf.md | 完全未命中 |

### 配置 `hash256+memory+deepseek+rerankon`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 16-glm4-api.md, 15-ollama-local.md, 10-prompt-structure.md,  | 完全未命中 |
| q05 | easy | 0.0000 | 14-ragas-judge.md, 02-chunking-strategy.md, 01-rag-overview. | 完全未命中 |
| q16 | medium | 0.0000 | 08-rerank-rrf.md, 14-ragas-judge.md, 06-vector-db-selection. | 完全未命中 |
| q18 | medium | 0.0000 | 05-ann-index.md, 01-rag-overview.md, 16-glm4-api.md, 06-vect | 完全未命中 |
| q20 | hard | 0.0000 | 06-vector-db-selection.md, 04-vector-similarity.md, 15-ollam | 完全未命中 |

### 配置 `hash4096+memory+deepseek+rerankoff`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 01-rag-overview.md, 11-hallucination.md, | 完全未命中 |
| q28 | hard | 0.0000 | 09-cross-encoder.md, 08-rerank-rrf.md, 05-ann-index.md | 完全未命中 |
| q30 | medium | 0.0000 | 09-cross-encoder.md, 08-rerank-rrf.md | 完全未命中 |
| q49 | medium | 0.0000 | 03-embedding-models.md, 13-ir-metrics.md, 16-glm4-api.md, 14 | 完全未命中 |
| q10 | medium | 0.2000 | 03-embedding-models.md, 16-glm4-api.md, 07-bm25-hybrid.md | 命中但排名靠后 |

### 配置 `hash4096+memory+deepseek+rerankon`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 01-rag-overview.md, 11-hallucination.md, | 完全未命中 |
| q28 | hard | 0.0000 | 08-rerank-rrf.md, 09-cross-encoder.md | 完全未命中 |
| q49 | medium | 0.2500 | 16-glm4-api.md, 03-embedding-models.md, 14-ragas-judge.md, 1 | 命中但排名靠后 |
| q09 | medium | 0.3333 | 03-embedding-models.md, 16-glm4-api.md, 15-ollama-local.md,  | 命中但排名靠后 |
| q10 | medium | 0.3333 | 03-embedding-models.md, 16-glm4-api.md, 15-ollama-local.md | 命中但排名靠后 |
