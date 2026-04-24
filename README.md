# RAG Learning Lab

一个面向学习和调试的本地 RAG 实验室。项目从 Markdown 知识库出发，完整展示文档扫描、结构化 chunk、embedding、SQLite + FAISS 持久化索引、关键词/向量/混合检索，以及带引用来源的答案生成链路。

这个项目的目标不是把 RAG 封装成黑盒，而是把每个关键环节摊开给你看：文档怎么切、向量怎么建、检索命中了什么、上下文怎么拼、回答引用了哪些来源。

## 功能特性

- Markdown 知识库扫描：递归加载本地 `*.md` 文件。
- 结构感知 chunk：按标题、段落和 fenced code block 切分，保留标题路径和行号。
- 持久化索引：使用 SQLite 保存文档/chunk 元数据，使用 FAISS 保存本地向量索引。
- 多种检索模式：支持关键词检索、向量检索和混合检索。
- 检索调试 UI：前端展示召回 chunk、分数、来源文件、标题路径和原文证据。
- 答案生成：支持 OpenAI-compatible `/chat/completions` 接口，基于检索证据生成带引用的回答。
- 在线调试链路：可关闭持久化索引，按请求实时扫描、切分、embedding 和检索，方便观察参数变化。

## 适合谁

- 正在学习 RAG 基础链路的开发者。
- 想理解 chunk、embedding、向量库和 hybrid retrieval 如何协作的人。
- 想搭建个人 Markdown 知识库问答原型的人。
- 想调试“为什么 RAG 没召回正确资料”的人。

## 快速开始

### 1. 安装后端依赖

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

如果要使用本地 `bge-m3` embedding 模型，还需要安装可选依赖：

```bash
pip install -e ".[local-embedding]"
```

只想先跑通链路，可以使用 `hash` embedding provider，不需要额外模型。

### 2. 安装前端依赖

```bash
cd frontend
npm install
cd ..
```

### 3. 配置环境变量

复制示例配置：

```bash
cp .env.example .env
```

最小可用配置：

```env
SELF_RAG_EMBEDDING_PROVIDER=hash
SELF_RAG_DATABASE_PATH=data/sqlite/self_rag.db
SELF_RAG_FAISS_INDEX_PATH=data/faiss/self_rag.index
```

如果要生成答案，需要配置 OpenAI-compatible LLM：

```env
SELF_RAG_LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
SELF_RAG_LLM_MODEL=doubao-seed-2-0-code-preview-260215
SELF_RAG_LLM_API_KEY=your-local-api-key
```

`SELF_RAG_LLM_API_KEY` 只在本地运行时读取；如果不配置，检索和索引功能仍可正常使用，但“生成回答”不可用。

### 4. 启动开发服务

同时启动后端和前端：

```bash
bash scripts/dev.sh
```

也可以分别启动：

```bash
bash scripts/dev-backend.sh
bash scripts/dev-frontend.sh
```

访问地址：

```text
前端：http://<server-ip>:5173/
API 文档：http://<server-ip>:8800/docs
```

前端通过 Vite 代理访问后端，因此浏览器只需要访问 `5173` 端口。

## 推荐使用链路

前端推荐按下面的顺序使用：

1. 预览相关切分
   只扫描 Markdown 并展示相关 chunk，不写入 SQLite 或 FAISS。用于检查 chunk 是否合理，以及当前问题大概能命中哪些文本块。
2. 构建持久化索引
   全量扫描 Markdown 根目录，重新生成 SQLite 文档/chunk 元数据和 FAISS 向量索引。文档新增、修改或删除后需要重新执行。
3. 运行检索
   查看 keyword/vector/hybrid 实际召回了哪些 chunk、分数如何、证据是否可靠。
4. 生成回答
   复用当前检索模式和 top-k 参数，先检索证据，再把证据组装成带 `[来源 N]` 的上下文交给 LLM，最后返回回答、引用来源、检索证据和 LLM 上下文。

日常最短路径：

```text
文档有变化：构建持久化索引 -> 运行检索 -> 生成回答
文档没变化：运行检索 -> 生成回答
调试 chunk：预览相关切分 -> 调整 chunk 参数 -> 构建持久化索引
```

如果勾选了“使用 SQLite + FAISS 持久化索引”，但还没有构建索引，运行检索或生成回答会提示先构建索引。关闭该选项后，系统会回到按请求实时处理全量 Markdown 的在线调试模式。

## RAG 链路说明

### 持久化索引

```text
Markdown 文件
  -> 扫描
  -> 结构化 chunk
  -> chunk embedding
  -> SQLite 保存文档和 chunk 元数据
  -> FAISS 保存向量索引
```

构建完成后，每次检索只需要：

```text
用户问题
  -> query embedding
  -> FAISS 相似度搜索
  -> SQLite 取回 chunk 原文和来源元数据
```

这避免了每次提问都重新扫描、切分和 embedding 全部 Markdown 文件。

### 混合检索

混合检索会同时运行：

- 关键词检索：适合文件名、命令、版本号、专有名词和精确中文短语。
- 向量检索：适合语义相近但措辞不同的问题。

两路结果会归一化分数后加权融合。前端会展示关键词召回、向量召回和最终融合排序，方便判断问题出在 chunk、embedding、关键词匹配还是融合策略。

### 答案生成

答案生成阶段不会直接让 LLM 自由发挥。后端会先检索 top-k chunk，再构造上下文：

```text
[来源 1] docs/example.md:10-18 > 标题路径
chunk 原文

[来源 2] notes/rag.md:33-41 > 标题路径
chunk 原文

问题：...
```

系统提示要求模型只能基于这些资料回答；资料不足时应说明“当前知识库证据不足”。前端会同时展示最终回答、引用来源、检索证据和实际发送给模型的上下文。

## 项目架构

```text
浏览器
  |
  | http://<server-ip>:5173
  v
React + Vite 前端
  |
  | Vite 代理：/api 和 /health -> http://127.0.0.1:8800
  v
FastAPI 后端
  |
  +-- Markdown 扫描器
  +-- 结构感知文本切分器
  +-- SQLite 文档和 chunk 元数据
  +-- FAISS 本地向量索引
  +-- 关键词检索器
  +-- 向量检索器
  +-- 混合检索器
  +-- OpenAI-compatible 答案生成器
```

## 目录结构

```text
app/
  main.py                  FastAPI 应用、请求/响应模型、API 路由
  config.py                从 .env 和 SELF_RAG_* 环境变量加载运行配置
  db.py                    SQLite 连接辅助函数和表结构
  documents/
    loader.py              递归扫描 Markdown 文件
    models.py              MarkdownDocument 和 MarkdownChunk 数据类
  generation/
    llm.py                 OpenAI-compatible LLM provider
    prompt.py              RAG 上下文和引用 prompt 构造
  indexing/
    chunker.py             结构感知 Markdown 切分
    embeddings.py          hash 和 sentence-transformers embedding provider
    vector_store.py        内存向量检索存储
    persistent_index.py    SQLite + FAISS 持久化索引构建和查询
  retrieval/
    keyword.py             关键词检索
    vector.py              向量检索
    hybrid.py              关键词结果和向量结果加权融合
    schemas.py             共享检索响应模型

frontend/
  src/main.tsx             React RAG 调试控制台
  src/styles.css           应用样式
  vite.config.ts           Vite 服务和后端代理配置

scripts/
  dev.sh                   同时启动后端和前端
  dev-backend.sh           启动 FastAPI
  dev-frontend.sh          启动 Vite

tests/                     后端单元测试和 API 测试
```

## API 参考

```text
GET  /health

POST /api/index/preview
POST /api/index/build
GET  /api/index/status

POST /api/retrieval/keyword
POST /api/retrieval/vector
POST /api/retrieval/hybrid

POST /api/answer
```

核心接口说明：

- `/api/index/preview`：预览 Markdown 扫描和 chunk 结果，不写持久化索引。
- `/api/index/build`：全量重建 SQLite + FAISS 持久化索引。
- `/api/index/status`：查看当前持久化索引状态。
- `/api/retrieval/*`：返回带分数和来源元数据的检索结果。
- `/api/answer`：先检索证据，再生成带引用来源的回答。

增量刷新和定时索引已在前端预留按钮，但当前版本尚未实现。

## 配置参考

所有运行配置定义在 `app/config.py`，可通过 `.env` 中的 `SELF_RAG_` 前缀变量覆盖。

```env
SELF_RAG_DATABASE_PATH=data/sqlite/self_rag.db
SELF_RAG_FAISS_INDEX_PATH=data/faiss/self_rag.index
SELF_RAG_DEFAULT_MARKDOWN_ROOT=/path/to/markdown

SELF_RAG_EMBEDDING_PROVIDER=hash
SELF_RAG_EMBEDDING_MODEL_PATH=models/BAAI/bge-m3

SELF_RAG_LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
SELF_RAG_LLM_MODEL=doubao-seed-2-0-code-preview-260215
SELF_RAG_LLM_API_KEY=your-local-api-key
```

Embedding provider：

```text
hash      无依赖的确定性向量，适合快速本地测试
bge-m3    从 SELF_RAG_EMBEDDING_MODEL_PATH 加载的本地 sentence-transformers 模型
```

LLM provider 使用 OpenAI-compatible `/chat/completions` 协议。

## 开发

运行测试：

```bash
.venv/bin/python -m pytest
```

构建前端：

```bash
cd frontend
npm run build
```

提交前建议至少运行：

```bash
.venv/bin/python -m pytest
cd frontend && npm run build
```

## 当前限制

- 持久化索引目前是手动全量重建，不是增量更新。
- 定时索引尚未实现。
- 只支持 Markdown 文件。
- 答案质量依赖检索结果、chunk 质量和所配置的 LLM。
- 默认没有内置 reranker，混合检索只做关键词和向量结果的加权融合。

## 路线图

- 增量索引：只处理新增、修改和删除的 Markdown 文件。
- 定时索引：后台定期刷新知识库。
- Reranker：对初步召回结果做二次排序。
- 评测集：记录问题、期望来源和命中率，避免调参只靠感觉。
- 更多数据源：PDF、网页、代码仓库等。

## 本地数据

- SQLite 数据默认写入 `data/sqlite/self_rag.db`。
- FAISS 索引默认写入 `data/faiss/self_rag.index`。
- 这些文件是本地运行产物，可以删除后重新构建索引。
