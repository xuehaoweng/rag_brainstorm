# Self RAG

个人 Markdown RAG 学习控制台。这个项目是一个本地优先的实验工作台，用来扫描 Markdown 笔记、构建 SQLite + FAISS 持久化索引、预览文本切分结果，对比关键词检索、向量检索和混合检索的召回效果，并基于检索证据生成带引用来源的回答。

## 快速启动

同时启动后端和前端：

```bash
bash scripts/dev.sh
```

也可以分别启动。

后端：

```bash
bash scripts/dev-backend.sh
```

前端：

```bash
bash scripts/dev-frontend.sh
```

然后在你的机器上打开前端：

```text
http://<server-ip>:5173/
```

FastAPI 文档地址：

```text
http://<server-ip>:8800/docs
```

前端通过 Vite 代理调用后端 API，所以浏览器只需要访问 `5173` 端口。服务器本机需要能够访问 `127.0.0.1:8800`。

## 项目架构

Self RAG 分为 FastAPI 后端和 React/Vite 前端。

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
  +-- SQLite 表结构初始化
```

当前后端支持两条链路：一条是按请求实时扫描和切分文件的调试链路，另一条是手动触发的 SQLite + FAISS 持久化索引链路。推荐日常检索先点击“构建持久化索引”，之后每次检索只向量化查询问题并搜索已有 FAISS 索引。

## 前端使用链路

推荐按下面的顺序使用前端：

1. 预览相关切分。
   这个按钮只扫描 Markdown 并展示相关 chunk，不会写入 SQLite 或 FAISS。它用于检查文档是否被切得合理，以及当前问题大概能命中哪些文本块。
2. 构建持久化索引。
   这个按钮会全量扫描 Markdown 根目录，重新生成 SQLite 文档/chunk 元数据和 FAISS 向量索引。文档有新增、修改或删除时需要重新点一次。
3. 运行检索。
   在“使用 SQLite + FAISS 持久化索引”保持勾选时，后端只向量化当前问题并查询已有索引。这个步骤用于检查 keyword/vector/hybrid 实际召回了哪些证据。
4. 生成回答。
   后端会复用当前检索模式和 top-k 参数，先检索证据，再把证据组装成带 `[来源 N]` 的上下文交给 LLM，最后返回回答、引用来源、检索证据和 LLM 上下文。

日常最短路径：

```text
文档有变化：构建持久化索引 -> 运行检索 -> 生成回答
文档没变化：运行检索 -> 生成回答
调试 chunk：预览相关切分 -> 调整 chunk 参数 -> 构建持久化索引
```

注意：如果勾选了“使用 SQLite + FAISS 持久化索引”，但还没有构建索引，运行检索或生成回答会失败并提示先构建索引。关闭该勾选项后，系统会回到按请求实时扫描、切分、embedding 和检索的在线调试模式。

## 目录结构

```text
app/
  main.py                  FastAPI 应用、请求/响应模型、API 路由
  config.py                从 .env 和 SELF_RAG_* 环境变量加载运行配置
  db.py                    SQLite 连接辅助函数和文档/chunk 表结构
  documents/
    loader.py              递归扫描 Markdown 文件
    models.py              MarkdownDocument 和 MarkdownChunk 数据类
  indexing/
    chunker.py             感知标题、段落和代码块的 Markdown 切分
    embeddings.py          hash 和 sentence-transformers embedding provider
    vector_store.py        内存向量检索存储
    persistent_index.py    SQLite + FAISS 持久化索引构建和查询
  retrieval/
    keyword.py             面向精确词和中文短语的词面检索
    vector.py              基于 embedding 的余弦相似度检索
    hybrid.py              关键词结果和向量结果的加权融合
    schemas.py             共享的检索响应模型

frontend/
  src/main.tsx             React 检索控制台
  src/styles.css           应用样式
  vite.config.ts           Vite 服务和后端代理配置

scripts/
  dev.sh                   同时启动后端和前端
  dev-backend.sh           在 8800 端口启动 FastAPI
  dev-frontend.sh          在 5173 端口启动 Vite

tests/                     后端单元测试和 API 测试
```

## 检索流程

持久化索引流程：

1. 前端点击“构建持久化索引”，把 Markdown 根目录、embedding provider 和 chunk 参数发送给后端。
2. 后端递归加载 `*.md` 文件，按标题、段落和 fenced code block 切分。
3. 文档和 chunk 元数据写入 SQLite。
4. 可检索 chunk 做 embedding，向量写入本地 FAISS index。
5. 后续检索请求只对 query 做 embedding，再查 FAISS 并从 SQLite 取回 chunk 文本。

在线调试流程仍然保留：关闭“使用 SQLite + FAISS 持久化索引”后，后端会按请求实时扫描、切分、embedding 和检索，适合调试 chunk 参数。

答案生成流程：

1. 前端点击“生成回答”。
2. 后端按当前检索模式召回 top-k chunk。
3. 后端把 chunk 格式化为带 `[来源 N]` 的上下文。
4. OpenAI-compatible LLM 只基于这些资料生成回答。
5. 前端展示最终回答、引用来源、检索证据和实际发送给模型的上下文。

## API 列表

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

`/api/index/preview` 用于在正式检索前检查 Markdown 扫描和切分结果是否合理。`/api/index/build` 会全量重建当前知识库的 SQLite + FAISS 索引。增量刷新和定时索引在前端已预留按钮，但当前第一版尚未实现。

## 配置

运行配置定义在 `app/config.py`，可以通过 `.env` 文件里的 `SELF_RAG_` 前缀变量覆盖。

常用配置：

```text
SELF_RAG_DATABASE_PATH=data/sqlite/self_rag.db
SELF_RAG_FAISS_INDEX_PATH=data/faiss/self_rag.index
SELF_RAG_DEFAULT_MARKDOWN_ROOT=/path/to/markdown
SELF_RAG_EMBEDDING_PROVIDER=bge-m3
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

如果要使用 `bge-m3`，需要安装可选的本地 embedding 依赖，并确保模型路径在本机存在。只想验证 API 和 UI 链路时，可以使用 `hash`，不需要额外模型依赖。

LLM provider 使用 OpenAI-compatible `/chat/completions` 协议。真实 API key 只应写入本机 `.env`，不要提交到仓库。

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
