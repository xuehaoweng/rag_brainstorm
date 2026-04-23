import { StrictMode, useState, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type ChunkPreview = {
  document_path: string;
  chunk_index: number;
  heading_path: string;
  start_line: number;
  end_line: number;
  text: string;
  text_hash: string;
};

type RetrievedChunk = ChunkPreview & {
  rank: number;
  score: number;
};

type IndexPreviewResponse = {
  document_count: number;
  chunk_count: number;
  chunks: ChunkPreview[];
};

type VectorSearchResponse = {
  document_count: number;
  chunk_count: number;
  results: RetrievedChunk[];
  debug?: {
    provider: string;
    indexed_chunk_count: number;
    query_vector_dimensions: number;
  };
  vector_results?: RetrievedChunk[];
  keyword_results?: RetrievedChunk[];
};

type RetrievalMode = "hybrid" | "vector" | "keyword";
type RetrievalRun = {
  mode: RetrievalMode;
  query: string;
  root: string;
};

const MIN_DISPLAY_SCORE = 0.5;

type RunState =
  | { status: "idle" }
  | { status: "loading"; label: string }
  | { status: "error"; message: string }
  | { status: "preview"; data: IndexPreviewResponse }
  | { status: "vector"; data: VectorSearchResponse; run: RetrievalRun };

function App() {
  const [root, setRoot] = useState("/root/self_rag");
  const [query, setQuery] = useState("RAG 混合检索");
  const [provider, setProvider] = useState("bge-m3");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [maxChars, setMaxChars] = useState(1800);
  const [state, setState] = useState<RunState>({ status: "idle" });
  const isLoading = state.status === "loading";
  const statusLabel = state.status === "idle" ? "待运行" : state.status === "loading" ? "检索中" : state.status === "error" ? "需处理" : "已返回";

  async function previewIndex() {
    setState({ status: "loading", label: "正在扫描 Markdown 并预览切分结果..." });
    const result = await postJson<IndexPreviewResponse>("/api/index/preview", {
      root,
      max_chars: maxChars,
      overlap_chars: 160
    });
    setState(result.ok ? { status: "preview", data: result.data } : { status: "error", message: result.error });
  }

  async function runRetrieval() {
    const needsEmbedding = mode !== "keyword";
    setState({
      status: "loading",
      label: needsEmbedding && provider === "bge-m3" ? "正在加载 bge-m3 并执行混合检索..." : "正在检索知识库..."
    });
    const result = await postJson<VectorSearchResponse>(`/api/retrieval/${mode}`, {
      root,
      query,
      top_k: topK,
      max_chars: maxChars,
      overlap_chars: 160,
      embedding_provider: provider
    });
    setState(result.ok ? { status: "vector", data: result.data, run: { mode, query, root } } : { status: "error", message: result.error });
  }

  return (
    <>
      <a className="skip-link" href="#results">跳到检索结果</a>
      <main className="shell">
        <header className="topbar" aria-label="工作台状态">
          <div>
            <p className="eyebrow">Personal RAG console</p>
            <span>Markdown evidence lab</span>
          </div>
          <div className="run-signal" data-state={state.status}>
            <span aria-hidden="true" />
            {statusLabel}
          </div>
        </header>

        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">个人 RAG 调试台</p>
            <h1>先审计证据，再相信回答。</h1>
            <p className="lede">
              一个面向学习和排错的知识库工作台：扫描 Markdown、观察 chunk、运行关键词/向量/混合检索，并直接核对后端返回的原文证据。
            </p>
            <div className="hero-metrics" aria-label="当前检索参数">
              <Metric label="模式" value={modeLabel(mode)} />
              <Metric label="Top K" value={topK.toString()} />
              <Metric label="Chunk 上限" value={`${maxChars} 字`} />
            </div>
          </div>
          <aside className="status-card" aria-label="当前运行配置">
            <span>当前配置</span>
            <strong>{modeLabel(mode)}</strong>
            <p>{provider === "bge-m3" ? "本地 bge-m3 向量模型，适合真实语义检索。" : "hash 测试向量，无模型依赖，适合快速联调。"}</p>
            <div className="status-path">{root}</div>
          </aside>
        </section>

        <section className="control-grid" aria-label="检索控制台">
          <div className="control-intro">
            <p className="eyebrow">Run recipe</p>
            <h2>配置一次可复现的检索实验</h2>
            <p>优先预览切分，再运行检索。这样可以区分“没召回”和“文档切分不合理”。</p>
          </div>
          <div className="field-group">
            <label className="field wide">
              <span>Markdown 根目录</span>
              <input value={root} onChange={(event) => setRoot(event.target.value)} />
            </label>
            <label className="field query">
              <span>检索问题</span>
              <textarea value={query} onChange={(event) => setQuery(event.target.value)} />
            </label>
          </div>
          <div className="field-group compact-fields">
            <label className="field">
              <span>检索模式</span>
              <select value={mode} onChange={(event) => setMode(event.target.value as RetrievalMode)}>
                <option value="hybrid">混合检索</option>
                <option value="keyword">关键词检索</option>
                <option value="vector">向量检索</option>
              </select>
            </label>
            <label className="field">
              <span>向量模型</span>
              <select value={provider} onChange={(event) => setProvider(event.target.value)}>
                <option value="hash">hash</option>
                <option value="bge-m3">bge-m3</option>
              </select>
            </label>
            <label className="field">
              <span>返回条数 Top K</span>
              <input type="number" min="1" max="50" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
            </label>
            <label className="field">
              <span>文本块最大字符数</span>
              <input type="number" min="300" max="8000" value={maxChars} onChange={(event) => setMaxChars(Number(event.target.value))} />
            </label>
          </div>
          <div className="actions">
            <button type="button" className="secondary" onClick={previewIndex} disabled={isLoading}>预览文本切分</button>
            <button type="button" onClick={runRetrieval} disabled={isLoading}>运行检索</button>
          </div>
        </section>

        <div id="results">
          <ResultPanel state={state} />
        </div>
      </main>
    </>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ResultPanel({ state }: { state: RunState }) {
  if (state.status === "idle") {
    return <EmptyPanel />;
  }
  if (state.status === "loading") {
    return <section className="panel"><div className="loader" /><p>{state.label}</p></section>;
  }
  if (state.status === "error") {
    return <section className="panel error"><h2>请求失败</h2><pre>{state.message}</pre></section>;
  }
  if (state.status === "preview") {
    return (
      <section className="panel">
        <PanelHeader title="文本切分预览" documentCount={state.data.document_count} chunkCount={state.data.chunk_count} />
        <ChunkList chunks={state.data.chunks.slice(0, 20)} />
      </section>
    );
  }
  return (
    <section className="panel">
      <PanelHeader title="检索链路" documentCount={state.data.document_count} chunkCount={state.data.chunk_count} />
      <RetrievalTrace data={state.data} run={state.run} />
    </section>
  );
}

function EmptyPanel() {
  return (
    <section className="panel empty">
      <p className="eyebrow">Ready</p>
      <h2>先扫描知识库，或直接运行一次检索。</h2>
      <p>用 <b>预览文本切分</b> 检查 Markdown 如何被拆成 chunk，再用 <b>运行检索</b> 查看排序后的证据。</p>
    </section>
  );
}

function modeLabel(mode: RetrievalMode) {
  if (mode === "hybrid") {
    return "混合检索";
  }
  if (mode === "keyword") {
    return "关键词检索";
  }
  return "向量检索";
}

function PanelHeader({ title, documentCount, chunkCount }: { title: string; documentCount: number; chunkCount: number }) {
  return (
    <header className="panel-header">
      <h2>{title}</h2>
      <div>
        <span>{documentCount} 个文档</span>
        <span>{chunkCount} 个文本块</span>
      </div>
    </header>
  );
}

function RetrievalTrace({ data, run }: { data: VectorSearchResponse; run: RetrievalRun }) {
  return (
    <div className="trace">
      <TraceStep
        index={1}
        title="输入与切分"
        description="后端先扫描 Markdown 根目录，按标题、段落和代码块切分为可检索文本块。"
      >
        <div className="debug-strip">
          <span>检索模式：<b>{modeLabel(run.mode)}</b></span>
          <span>问题：<b>{run.query}</b></span>
          <span>目录：<b>{run.root}</b></span>
        </div>
      </TraceStep>

      {data.keyword_results ? (
        <TraceStep
          index={2}
          title="关键词召回"
          description="用词面匹配捕捉文件名、标题、专有名词、命令、版本号和中文关键短语。"
        >
          <ChunkList chunks={filterDisplayScores(data.keyword_results)} showScores compact />
        </TraceStep>
      ) : null}

      {data.vector_results || data.debug ? (
        <TraceStep
          index={data.keyword_results ? 3 : 2}
          title="向量召回"
          description="把问题和文本块转成向量，用余弦相似度查找语义接近的内容。当前实现会把文件名、标题路径和正文一起向量化。"
        >
          {data.debug ? (
            <div className="debug-strip">
              <span>向量模型：<b>{data.debug.provider}</b></span>
              <span>参与检索的文本块：<b>{data.debug.indexed_chunk_count}</b></span>
              <span>问题向量维度：<b>{data.debug.query_vector_dimensions}</b></span>
            </div>
          ) : null}
          {data.vector_results ? <ChunkList chunks={filterDisplayScores(data.vector_results)} showScores compact /> : null}
        </TraceStep>
      ) : null}

      <TraceStep
        index={run.mode === "hybrid" ? 4 : data.keyword_results ? 3 : 2}
        title={run.mode === "hybrid" ? "融合排序结果" : "最终排序结果"}
        description={run.mode === "hybrid" ? "归一化关键词分数和向量分数后加权合并，当前更偏向关键词信号，适合你的技术文档检索。" : "当前模式直接使用该检索器的排序结果。"}
      >
        <ChunkList chunks={filterDisplayScores(data.results)} showScores />
      </TraceStep>
    </div>
  );
}

function filterDisplayScores(chunks: RetrievedChunk[]) {
  return chunks.filter((chunk) => chunk.score >= MIN_DISPLAY_SCORE);
}

function TraceStep({ index, title, description, children }: { index: number; title: string; description: string; children: ReactNode }) {
  return (
    <section className="trace-step">
      <header>
        <span>{index}</span>
        <div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </header>
      {children}
    </section>
  );
}

function ChunkList({ chunks, showScores = false, compact = false }: { chunks: Array<ChunkPreview | RetrievedChunk>; showScores?: boolean; compact?: boolean }) {
  if (chunks.length === 0) {
    return <p className="muted">没有 score ≥ {MIN_DISPLAY_SCORE} 的文本块。</p>;
  }
  return (
    <div className={compact ? "chunk-list compact" : "chunk-list"}>
      {chunks.map((chunk) => (
        <article className="chunk" key={`${chunk.document_path}-${chunk.chunk_index}`}>
          <header>
            <div>
              <span className="rank">{"rank" in chunk ? `第 ${chunk.rank} 名` : `文本块 ${chunk.chunk_index}`}</span>
              <h3>{chunk.heading_path || "文档"}</h3>
            </div>
            {showScores && "score" in chunk ? (
              <div className="score" aria-label={`相似度分数 ${chunk.score.toFixed(4)}`}>
                <strong>{chunk.score.toFixed(4)}</strong>
                <span style={{ inlineSize: `${Math.max(4, Math.min(100, chunk.score * 100))}%` }} />
              </div>
            ) : null}
          </header>
          <p className="source">{chunk.document_path}:{chunk.start_line}-{chunk.end_line}</p>
          <pre>{chunk.text}</pre>
        </article>
      ))}
    </div>
  );
}

async function postJson<T>(url: string, body: unknown): Promise<{ ok: true; data: T } | { ok: false; error: string }> {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    if (!response.ok) {
      return { ok: false, error: data.detail ?? response.statusText };
    }
    return { ok: true, data };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
