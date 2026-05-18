# RAG 优化实战：Query 改写 + Reranker + 评估体系，从原理到代码

> 本文基于一个真实的 RAG 学习项目（self_rag），从零实现三个核心优化：Multi-Query 改写、Reranker 精排、评估体系。每个环节先讲原理，再给出完整代码和测试。

## 前言：一个 RAG 系统的典型困境

你搭了一个 RAG 系统，能跑了。用户问"怎么部署？"，系统去知识库里搜，搜到了一些东西，拼给大模型，生成了回答。

但你很快发现几个问题：

- **找不全**：文档里明明有部署指南，但用户说"怎么部署"，文档标题写的是"安装与配置"，没搜到
- **排不准**：搜到了 5 个片段，最相关的排在第 4 位，排在前面的是些沾边但不精确的内容
- **没法量化**：改了参数感觉好了一点，但说不清楚到底好了多少

这三个问题对应三个优化方向：**Query 改写**、**Reranker 精排**、**评估体系**。

本文将带你依次实现它们。

---

## 优化前：baseline 是什么样的

先看优化前的检索流程：

```
用户 query（原样）→ 混合检索（向量 + 关键词加权融合）→ 拼 prompt → LLM 生成 → 返回
```

混合检索的核心代码很简单——分别做向量检索和关键词检索，分数归一化后加权求和：

```python
score = vector_weight * vector_score + keyword_weight * keyword_score
```

这个 baseline 能工作，但有明确的天花板。接下来我们一个个突破。

---

## 第一步：Multi-Query 改写——"同一个问题，换几种问法"

### 原理

用户的 query 通常是口语化、简短、单一视角的。比如：

- 用户问"怎么部署？"
- 但文档里写的是"安装与配置步骤"、"生产环境部署指南"、"服务启动方式"

**一个 query 只能从一个角度检索，很容易漏掉措辞不同但内容相关的文档。**

Multi-Query 的思路：让 LLM 把用户问题改写成多个不同角度的检索查询，分别检索后合并结果。

```
原始 query: "怎么部署？"
     ↓ LLM 改写
query_1: "项目安装与配置步骤"
query_2: "生产环境部署指南"
query_3: "服务启动和运行方式"
     ↓ 各自检索
     ↓ 合并去重（相同 chunk 取最高分）
     ↓ 送入后续流程
```

### 实现

核心代码不到 80 行：

```python
def rewrite_query(query: str, llm, *, num_rewrites: int = 3) -> list[str]:
    """返回 [原始query, 改写1, 改写2, ...]"""
    user_prompt = f"请将以下问题改写为 {num_rewrites} 个不同角度的检索查询...\n原始问题：{query}"
    
    try:
        raw = llm.generate(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    except Exception:
        return [query]  # LLM 挂了？退回原始 query，不影响主流程
    
    rewrites = parse_rewrites(raw)  # 解析每行一个 query
    return [query] + rewrites[:num_rewrites]  # 原始 query 永远在第一位
```

多路检索后的合并逻辑：

```python
def multi_query_retrieval(queries, mode, request):
    best = {}  # key: "doc_path:chunk_index"
    for query in queries:
        results = run_retrieval(mode, query)
        for chunk in results:
            key = f"{chunk.document_path}:{chunk.chunk_index}"
            if key not in best or chunk.score > best[key].score:
                best[key] = chunk  # 同一个 chunk，保留最高分
    return sorted(best.values(), key=lambda c: c.score, reverse=True)
```

### 关键设计决策

**1. 始终保留原始 query**

改写可能丢失用户的精确意图。比如用户搜的就是某个专有名词，改写后反而模糊了。原始 query 是安全网。

**2. 解析容错**

LLM 不一定遵循你的格式要求。你说"每行一个，不要编号"，它可能返回 `1. xxx` 或 `- xxx`。所以解析时要用正则清理各种前缀：

```python
line = re.sub(r"^[\d]+[.)\-]\s*", "", line)  # 去掉 "1. " "2) "
line = re.sub(r"^[-*]\s+", "", line)           # 去掉 "- " "* "
```

**做 LLM 应用的铁律：永远不要信任 LLM 的输出格式。**

**3. Fallback 降级**

LLM 调用失败 → 退回 `[原始query]`，主流程完全不受影响。这是生产级代码和 demo 的分水岭。

### Multi-Query 提升了什么？

**召回率（Recall）**——"相关的文档有多少被找回来了"。

但注意：**准确率不一定提升，甚至可能下降**。因为多个改写 query 会引入一些"沾边但不太相关"的结果。

这就引出了下一步。

---

## 第二步：Reranker 精排——"找到了，但排对了吗？"

### 原理

当前混合检索的排序是加权求和：

```python
score = 0.35 * vector_score + 0.65 * keyword_score
```

这个分数**不可靠**。为什么？

- 向量分数衡量的是"语义相似度"
- 关键词分数衡量的是"词汇重叠度"
- 两个完全不同维度的分数，归一化后硬加在一起，本质上是**混合两把不同刻度的尺子**

来看一个具体例子。用户问："RAG 怎么评估效果？"

```
chunk A: "RAG 评估通常使用 faithfulness、relevance 等指标..."
  → 向量分 0.82, 关键词分 0.30 → 混合分 0.48

chunk C: "评估模型效果时需要准备测试集..."
  → 向量分 0.60, 关键词分 0.90 → 混合分 0.80
```

chunk C 排在 A 前面，但 C 讲的是通用模型评估，跟 RAG 没关系。A 才是直接回答问题的。关键词"评估""效果"在 C 里高频出现，BM25 分数虚高。

**Reranker 怎么解决？**

把 query 和 chunk **拼在一起**送进模型，做交叉注意力，直接判断"这段文本是否真的回答了这个问题"。

```
普通检索（Bi-Encoder）：query → 向量    chunk → 向量    算距离
Reranker（Cross-Encoder）：[query + chunk] 一起编码 → 直接出相关性分数
```

### 实现

我们用 LLM 做 reranker——把所有候选 chunk 一次性发给 LLM 打分：

```python
def rerank(query, chunks, llm, *, top_k=None):
    # 拼成一段，一次性打分（而不是 N 次调用）
    chunks_text = "\n\n".join(
        f"[文本 {i}]\n{chunk.text[:800]}"  # 截取前 800 字控制 token
        for i, chunk in enumerate(chunks, start=1)
    )
    
    raw = llm.generate(
        system_prompt="对每段文本输出 0-10 的相关性分数...",
        user_prompt=f"用户问题：{query}\n\n{chunks_text}"
    )
    
    scores = parse_scores(raw)  # 解析 "[1] 8\n[2] 3\n..."
    # 按分数重排
    scored = [(chunk, scores.get(i+1, 0)) for i, chunk in enumerate(chunks)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

### 和 Multi-Query 配合使用

两步配合才是完整优化：

1. Multi-Query **先把网撒大** → 召回率高（"不漏"）
2. Reranker **再把不相关的筛掉** → 准确率也高（"不错"）

在实现上，当两者同时启用时，Multi-Query 会多取候选（over-fetch），给 Reranker 更大的候选池：

```python
if enable_reranker:
    over_fetch_k = top_k * 3  # 多取 3 倍，让 reranker 来筛
```

### 效果对比

```
                        召回率    准确率
baseline (hybrid)       中        中
+ multi_query           高 ✅     中 ⚠️  （撒大网，可能捞到不相关的）
+ reranker              中        高 ✅  （排序更准，但候选池没变大）
+ multi_query + reranker 高 ✅    高 ✅  （最佳组合）
```

---

## 第三步：评估体系——"好了多少？用数据说话"

### 原理

做了两步优化，怎么知道真的有效？

- "感觉好了一点" ← 不可靠
- "某个 case 变好了" ← 可能另一个 case 变差了

需要一套**标准化测试集 + 自动化指标**。

### 三个核心指标

**Recall@K（召回率）**——"该找到的找到了几个"

```
期望来源 = [chunk_A, chunk_B, chunk_C]
实际召回 = [chunk_A, chunk_C, chunk_D, chunk_E]
Recall@5 = 2/3 = 0.67
```

**Precision@K（准确率）**——"找回来的有多少是对的"

```
实际召回 = [chunk_A, chunk_C, chunk_D, chunk_E]  
其中相关的 = [chunk_A, chunk_C]
Precision@4 = 2/4 = 0.5
```

**MRR（Mean Reciprocal Rank）**——"好结果排在第几位"

```
结果 = [chunk_D, chunk_A, chunk_C, ...]
         不相关    相关（位置 2）
MRR = 1/2 = 0.5  （排名越靠前，分数越高）
```

### 实现

**评估数据集格式**（一个 JSON 文件）：

```json
{
  "name": "my-knowledge-base",
  "cases": [
    {
      "query": "怎么部署？",
      "expected_sources": [
        {"document_path": "deploy.md", "heading_path": "部署"}
      ],
      "expected_answer_contains": ["docker", "配置"]
    }
  ]
}
```

**指标计算**：

```python
def compute_retrieval_metrics(case, results):
    hits = 0
    first_hit_rank = 0
    
    for rank, chunk in enumerate(results, start=1):
        if matches_any_expected(chunk, case.expected_sources):
            hits += 1
            if first_hit_rank == 0:
                first_hit_rank = rank
    
    recall = hits / len(case.expected_sources)
    precision = hits / len(results)
    mrr = 1.0 / first_hit_rank if first_hit_rank > 0 else 0.0
    
    return RetrievalMetrics(recall, precision, mrr)
```

**一键评估**：

```bash
curl -X POST /api/eval -d '{
  "dataset_path": "data/eval/my_eval.json",
  "root": "./docs",
  "mode": "hybrid",
  "top_k": 5
}'
```

返回报告：

```json
{
  "mean_recall": 0.667,
  "mean_precision": 0.4,
  "mean_mrr": 0.833,
  "total_cases": 10
}
```

### 用评估体系对比优化效果

这才是评估体系最核心的价值——**量化对比不同配置的效果**：

```
baseline:                       mean_recall=0.50  mean_mrr=0.60
enable_multi_query=true:        mean_recall=0.75  mean_mrr=0.65
enable_reranker=true:           mean_recall=0.50  mean_mrr=0.85
multi_query + reranker:         mean_recall=0.75  mean_mrr=0.92
```

数据一目了然，不用猜。

---

## 优化后的完整流程

```
用户 query: "怎么部署？"
    │
    ▼
① Query 改写
    → ["怎么部署？", "安装配置步骤", "生产环境部署", "服务启动方式"]
    │
    ▼
② 每个 query 做混合检索（向量 + 关键词）
    → 合并去重，保留最高分
    │
    ▼
③ Reranker 精排
    → LLM 对每个 (query, chunk) 打 0-10 分
    → 按分数重排，取 top_k
    │
    ▼
④ 构建 Prompt + LLM 生成 + 引用校验
    │
    ▼
⑤ 返回带来源引用的答案
```

## 工程设计上的几个原则

写完这三个模块，回头看有几个一致的设计原则值得总结：

### 1. 开关式优化，默认关闭

```python
class AnswerRequest(BaseModel):
    enable_multi_query: bool = False  # 默认关
    enable_reranker: bool = False     # 默认关
```

新功能不影响现有行为。可以单独开、组合开、A/B 对比。

### 2. Fallback 优先于报错

三个模块都遵循同一个模式：

```python
try:
    result = llm.generate(...)
except Exception:
    return fallback  # 退回原始行为，不中断服务
```

LLM 不稳定是常态。Fallback 是生产级 LLM 应用的标配。

### 3. 不信任 LLM 的输出格式

Query 改写和 Reranker 都有双层解析——先尝试期望格式，再 fallback 到宽松解析。因为 LLM 就是不会严格遵循你的指令。

### 4. 先有度量，再调参数

没有评估体系的优化就是盲人摸象。先建立 Recall/Precision/MRR 的 baseline，然后每次改动都能量化对比。

---

## 下一步可以做什么

| 方向 | 说明 |
|------|------|
| HyDE | 让 LLM 先生成"假设性回答"，用回答去检索（比 query 更接近文档语义） |
| 本地 Cross-Encoder | 用 bge-reranker-v2-m3 等模型替代 LLM 打分，更快更便宜 |
| Faithfulness 评估 | 检查回答中的每个事实是否都有证据支撑（抓幻觉） |
| 向量数据库 | 从 FAISS 文件迁移到 Milvus/Qdrant，支持增量更新和并发 |

---

## 小结

三步优化，一条线索：

- **Multi-Query**：解决"找不全"（召回率）
- **Reranker**：解决"排不准"（准确率）  
- **评估体系**：解决"说不清"（量化度量）

每一步都不复杂，核心代码加起来不到 300 行。但组合起来，一个 RAG 系统的检索质量可以有质的提升。

关键不是代码量，而是理解**为什么要做这一步，它解决的是什么问题**。

---

*完整代码见 GitHub 仓库，包含所有实现和 46 个通过的测试用例。*
