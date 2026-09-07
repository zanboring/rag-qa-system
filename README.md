# RAG 知识库问答系统

> 独立于毕设的「干净版」RAG 实现，用于在面试中完整讲清 RAG 全链路：**切片 → 向量化 → 检索 → 组装 Prompt → 生成 → 溯源**。

## 为什么做这个项目

毕设里的 RAG 是和招聘系统耦合在一起的，面试时很难把「检索增强生成」这条主线单独讲清楚。本项目把 RAG 拆成 6 个独立小模块，每个模块对应一个能主动讲的八股点。

## 项目结构

```
02-RAG知识库问答/
├── app/
│   ├── config.py        # 后端切换（环境变量）
│   ├── chunker.py       # 文档切片（固定长度 + 重叠）
│   ├── embedder.py      # 向量化（hash 确定性 / BGE / 智谱）
│   ├── vectorstore.py   # 向量库（内存余弦 / ChromaDB）
│   ├── llm.py           # 大模型（mock / Ollama / GLM-4）
│   ├── reranker.py      # 重排序（RRF 融合语义 + 词法）
│   ├── rag.py           # RAG 编排（检索 + 重排 + Prompt + 生成 + 溯源）
│   └── main.py          # FastAPI 接口
├── tests/               # 26 个测试（chunker/embedder/vectorstore/reranker/api）
├── requirements.txt
└── pytest.ini
```

## 快速开始

```bash
# 1. 建虚拟环境并装依赖
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows
# .venv/bin/pip install -r requirements.txt     # macOS/Linux

# 2. 启动服务（默认 hash/memory/mock 零依赖后端，无需 GPU/网络/Key）
.venv/Scripts/uvicorn app.main:app --reload

# 3. 跑测试
.venv/Scripts/python -m pytest -v
```

## 接口示例

```bash
# 上传文档
curl -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"name":"算法笔记.md","text":"快速排序是一种分治算法，平均时间复杂度 O(n log n)……"}'

# 提问（返回答案 + 引用来源）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"快速排序的时间复杂度是多少？"}'

# 提问（开启重排序：先多召回 3 倍候选，再 RRF 精排回 top_k）
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"快速排序的时间复杂度是多少？","rerank":true}'

# 分页列出已入库文档（按入库时间倒序）
curl "http://localhost:8000/documents?page=1&page_size=20"

# 获取单个文档元数据
curl http://localhost:8000/documents/{doc_id}

# 删除文档（返回 404 表示文档不存在）
curl -X DELETE http://localhost:8000/documents/{doc_id}
```

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | /documents | 上传文档（切片 + 向量化 + 入库） |
| POST | /query | 提问（检索 + 重排 + 生成 + 溯源，`rerank=true` 开启精排） |
| GET | /documents | 分页列出已入库文档（`page`/`page_size`，按入库时间倒序） |
| GET | /documents/{doc_id} | 获取单个文档元数据 |
| PUT | /documents/{doc_id} | 更新文档（删旧片段 + 重新入库，name/text 至少传一个） |
| DELETE | /documents/{doc_id} | 删除文档及其全部向量片段 |
| GET | /health | 健康检查 + 当前后端 |

## 面试可讲的点

| 模块 | 核心概念 | 能主动聊的 trade-off |
|---|---|---|
| chunker | 切片 + 重叠 | 为什么要有重叠：避免一句话被从中间切断，丢失跨边界语义 |
| embedder | 文本→高维向量 | 为什么默认用确定性 hash embedding：零依赖可离线；生产换 BGE/智谱 |
| vectorstore | 余弦相似度 Top-K | 内存 O(n) 暴力 vs ChromaDB/Milvus 的 ANN 近似检索 |
| llm | 大模型生成 | 本地 Ollama（隐私/免费）vs 云端 GLM-4（效果/便捷） |
| reranker | 重排序 | RRF 融合「语义 + 词法」两个排序信号，无需归一化；生产可换 cross-encoder |
| rag | 检索增强 | 为什么给 LLM「参考资料」：减少幻觉，回答可溯源 |
| list_documents | 分页 + 排序 | 倒序按时间展示与「page 越界返回空数组」的体验设计；真实生产用游标分页 |

## 切换真实后端

```bash
# 用智谱云端 Embedding + GLM-4（需 Key）
set ZHIPU_API_KEY=你的Key
set EMBEDDING_BACKEND=zhipu
set LLM_BACKEND=glm4

# 用本地 Ollama（需先 ollama serve + ollama pull qwen2.5:7b）
set LLM_BACKEND=ollama
set OLLAMA_MODEL=qwen2.5:7b

# 用 ChromaDB 向量库（需 pip install chromadb）
set VECTOR_BACKEND=chroma
```

## 生产化演进方向

- 切片改用「按标题/语义分段 + 递归切分」，而非纯固定长度
- 向量库换 ChromaDB / Milvus，支持持久化与近似检索
- 重排序换 cross-encoder（如 bge-reranker）做深度语义精排（当前为零依赖 RRF 词法重排）
- 引用溯源加「片段高亮」与「来源置信度」
- 支持文档批量上传、整库重建
