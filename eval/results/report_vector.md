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
| `hash4096+memory+mock+rerankoff` | 0.6800 | 0.8800 | 0.9200 | 0.8300 | 0.8900 | 0.3133 | 0.7790 | 0.8092 |
| `hash4096+memory+mock+rerankon` | 0.7400 | 0.9400 | 0.9600 | 0.9000 | 0.9300 | 0.3400 | 0.8283 | 0.8576 |
| `hash4096+chroma+mock+rerankoff` | 0.6800 | 0.8800 | 0.9200 | 0.8300 | 0.8900 | 0.3133 | 0.7790 | 0.8092 |
| `hash4096+chroma+mock+rerankon` | 0.7400 | 0.9400 | 0.9600 | 0.9000 | 0.9300 | 0.3400 | 0.8283 | 0.8576 |

## 自动诊断

根据指标之间的关系自动定位瓶颈环节（召回侧 / 排序侧）：

**`hash4096+memory+mock+rerankoff`**

- Hit@5 (0.92) 与 MRR (0.78) 较为接近，召回与排序表现均衡
- 当前为 mock 生成后端，生成质量与拒答指标不可用（见上方说明），检索指标不受影响、仍然有效

**`hash4096+memory+mock+rerankon`**

- Hit@5 (0.96) 与 MRR (0.83) 较为接近，召回与排序表现均衡
- 当前为 mock 生成后端，生成质量与拒答指标不可用（见上方说明），检索指标不受影响、仍然有效

**`hash4096+chroma+mock+rerankoff`**

- Hit@5 (0.92) 与 MRR (0.78) 较为接近，召回与排序表现均衡
- 当前为 mock 生成后端，生成质量与拒答指标不可用（见上方说明），检索指标不受影响、仍然有效

**`hash4096+chroma+mock+rerankon`**

- Hit@5 (0.96) 与 MRR (0.83) 较为接近，召回与排序表现均衡
- 当前为 mock 生成后端，生成质量与拒答指标不可用（见上方说明），检索指标不受影响、仍然有效

## 生成指标（有答案样本）

| 配置 | 关键词覆盖 | 上下文覆盖 | 忠实度 | 答案相关性 |
| --- | --- | --- | --- | --- |
| `hash4096+memory+mock+rerankoff` | 0.1250 | 0.9533 | n/a | n/a |
| `hash4096+memory+mock+rerankon` | 0.1317 | 0.9933 | n/a | n/a |
| `hash4096+chroma+mock+rerankoff` | 0.1250 | 0.9533 | n/a | n/a |
| `hash4096+chroma+mock+rerankon` | 0.1317 | 0.9933 | n/a | n/a |

说明：**关键词覆盖**与**上下文覆盖**是零依赖规则指标，反映答案／上下文对标准答案要素的覆盖程度，
属于必要不充分条件（覆盖低必然是错的，覆盖高不必然对）。**忠实度**与**答案相关性**由裁判模型给出。

> **重要**：部分配置使用 mock 生成后端。mock 的输出是固定模板字符串，并非真实作答，
> 因此其中的**关键词覆盖**天然接近于 0——这反映的是「mock 不能答题」，**不代表检索或系统质量差**。
> 同理，这些配置下的拒答指标显示为 `n/a`：mock 的答案既不构成作答、也不构成拒答，无法用于评判拒答能力。
> 这部分结论需在接入真实生成后端（`--llm deepseek` / `--llm ollama`）后重新评测。

## 拒答能力（无答案样本）

| 配置 | 正确拒答率 | 误拒率 | P50 延迟 | P95 延迟 |
| --- | --- | --- | --- | --- |
| `hash4096+memory+mock+rerankoff` | n/a | n/a | 35.7 ms | 42.4 ms |
| `hash4096+memory+mock+rerankon` | n/a | n/a | 31.8 ms | 33.6 ms |
| `hash4096+chroma+mock+rerankoff` | n/a | n/a | 4.2 ms | 42.6 ms |
| `hash4096+chroma+mock+rerankon` | n/a | n/a | 8.6 ms | 10.2 ms |

说明：正确拒答率过低说明系统倾向于在无依据时强行作答（幻觉风险）；误拒率过高说明系统过于保守。两个指标必须一起看。

## 按难度分层（nDCG@5 / MRR）

### 配置 `hash4096+memory+mock+rerankoff`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.9444 | 0.9590 |
| medium | 27 | 0.7420 | 0.7692 |
| hard | 14 | 0.7440 | 0.7901 |

### 配置 `hash4096+memory+mock+rerankon`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.9444 | 0.9590 |
| medium | 27 | 0.8056 | 0.8378 |
| hard | 14 | 0.7976 | 0.8308 |

### 配置 `hash4096+chroma+mock+rerankoff`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.9444 | 0.9590 |
| medium | 27 | 0.7420 | 0.7692 |
| hard | 14 | 0.7440 | 0.7901 |

### 配置 `hash4096+chroma+mock+rerankon`

| 难度 | 样本数 | MRR | nDCG@5 |
| --- | --- | --- | --- |
| easy | 9 | 0.9444 | 0.9590 |
| medium | 27 | 0.8056 | 0.8378 |
| hard | 14 | 0.7976 | 0.8308 |

## 失败案例分析

### 配置 `hash4096+memory+mock+rerankoff`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 01-rag-overview.md, 11-hallucination.md, | 完全未命中 |
| q28 | hard | 0.0000 | 09-cross-encoder.md, 08-rerank-rrf.md, 05-ann-index.md | 完全未命中 |
| q30 | medium | 0.0000 | 09-cross-encoder.md, 08-rerank-rrf.md | 完全未命中 |
| q49 | medium | 0.0000 | 03-embedding-models.md, 13-ir-metrics.md, 16-glm4-api.md, 14 | 完全未命中 |
| q10 | medium | 0.2000 | 03-embedding-models.md, 16-glm4-api.md, 07-bm25-hybrid.md | 命中但排名靠后 |

### 配置 `hash4096+memory+mock+rerankon`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 01-rag-overview.md, 11-hallucination.md, | 完全未命中 |
| q28 | hard | 0.0000 | 08-rerank-rrf.md, 09-cross-encoder.md | 完全未命中 |
| q49 | medium | 0.2500 | 16-glm4-api.md, 03-embedding-models.md, 14-ragas-judge.md, 1 | 命中但排名靠后 |
| q09 | medium | 0.3333 | 03-embedding-models.md, 16-glm4-api.md, 15-ollama-local.md,  | 命中但排名靠后 |
| q10 | medium | 0.3333 | 03-embedding-models.md, 16-glm4-api.md, 15-ollama-local.md | 命中但排名靠后 |

### 配置 `hash4096+chroma+mock+rerankoff`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 01-rag-overview.md, 11-hallucination.md, | 完全未命中 |
| q28 | hard | 0.0000 | 09-cross-encoder.md, 08-rerank-rrf.md, 05-ann-index.md | 完全未命中 |
| q30 | medium | 0.0000 | 09-cross-encoder.md, 08-rerank-rrf.md | 完全未命中 |
| q49 | medium | 0.0000 | 03-embedding-models.md, 13-ir-metrics.md, 16-glm4-api.md, 14 | 完全未命中 |
| q10 | medium | 0.2000 | 03-embedding-models.md, 16-glm4-api.md, 07-bm25-hybrid.md | 命中但排名靠后 |

### 配置 `hash4096+chroma+mock+rerankon`

| 样本 | 难度 | MRR | 召回文档 | 说明 |
| --- | --- | --- | --- | --- |
| q03 | medium | 0.0000 | 15-ollama-local.md, 01-rag-overview.md, 11-hallucination.md, | 完全未命中 |
| q28 | hard | 0.0000 | 08-rerank-rrf.md, 09-cross-encoder.md | 完全未命中 |
| q49 | medium | 0.2500 | 16-glm4-api.md, 03-embedding-models.md, 14-ragas-judge.md, 1 | 命中但排名靠后 |
| q09 | medium | 0.3333 | 03-embedding-models.md, 16-glm4-api.md, 15-ollama-local.md,  | 命中但排名靠后 |
| q10 | medium | 0.3333 | 03-embedding-models.md, 16-glm4-api.md, 15-ollama-local.md | 命中但排名靠后 |
