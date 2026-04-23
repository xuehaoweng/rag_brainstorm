# Self RAG

个人 Markdown RAG 学习控制台。这个项目是一个本地优先的实验工作台，用来扫描 Markdown 笔记、预览文本切分结果，并对比关键词检索、向量检索和混合检索的召回效果，为后续接入完整的答案生成链路做准备。

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
  +-- 关键词检索器
  +-- 向量检索器
  +-- 混合检索器
  +-- SQLite 表结构初始化
```

当前后端会直接基于请求里传入的 Markdown 目录执行检索。服务启动时会初始化 SQLite 表结构，这些表定义了文档和文本块后续持久化的形态；不过当前 API 主链路仍然是按请求实时扫描和切分文件。

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

1. 前端把 Markdown 根目录、查询问题、检索模式、`top_k` 和 chunk 大小配置发送给后端。
2. 后端从指定根目录递归加载 `*.md` 文件。
3. Markdown 先按标题切分，再按段落类块继续拆分，并保留 fenced code block 的完整性。
4. 关键词检索会根据完整查询命中、重要词、文件名、标题和中文 n-gram 打分。
5. 向量检索会对查询和可检索 chunk 做 embedding，然后在内存中按向量相似度排序。
6. 混合检索会同时运行关键词检索和向量检索，归一化两边分数后做加权融合。
7. 前端展示检索链路、调试元数据、来源文件、行号范围、分数和原始 chunk 文本。

## API 列表

```text
GET  /health
POST /api/index/preview
POST /api/retrieval/keyword
POST /api/retrieval/vector
POST /api/retrieval/hybrid
```

`/api/index/preview` 用于在正式检索前检查 Markdown 扫描和切分结果是否合理。检索接口会返回带来源元数据的排序 chunk，方便在 UI 里直接审计证据。

## 配置

运行配置定义在 `app/config.py`，可以通过 `.env` 文件里的 `SELF_RAG_` 前缀变量覆盖。

常用配置：

```text
SELF_RAG_DATABASE_PATH=data/sqlite/self_rag.db
SELF_RAG_DEFAULT_MARKDOWN_ROOT=/path/to/markdown
SELF_RAG_EMBEDDING_PROVIDER=bge-m3
SELF_RAG_EMBEDDING_MODEL_PATH=models/BAAI/bge-m3
```

Embedding provider：

```text
hash      无依赖的确定性向量，适合快速本地测试
bge-m3    从 SELF_RAG_EMBEDDING_MODEL_PATH 加载的本地 sentence-transformers 模型
```

如果要使用 `bge-m3`，需要安装可选的本地 embedding 依赖，并确保模型路径在本机存在。只想验证 API 和 UI 链路时，可以使用 `hash`，不需要额外模型依赖。

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
