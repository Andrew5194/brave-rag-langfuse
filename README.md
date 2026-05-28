# Brave Search API — value comparison demo

A LangFuse-instrumented comparison of three pipelines answering the same finance questions:

| Run | Pipeline | What it shows |
| --- | --- | --- |
| `baseline-v1` | LLM only, no retrieval | Hallucination baseline |
| `grounded-v1` | Brave **LLM Context** endpoint → LLM with citation enforcement | Quality lift from grounding |
| `diy-rag-v1` | Manual RAG (Search + fetch + extract + chunk + embed + index + retrieve) → LLM | Infrastructure overhead of doing it yourself |

Three evaluators score every output:
- `citation_rate` (0–1) — LLM-as-judge
- `factuality` (1–5) — LLM-as-judge
- `latency_ms` — measured during execution

## 2-minute live demo

The primary demo surface is the Jupyter notebook **`walkthrough.ipynb`** — it interleaves markdown narration, code execution, and outputs, so you can run cells and talk over them as you walk through.

**Before the demo (one-time):**
```bash
set -a; source .env; set +a   # load env vars into current shell
jupyter lab walkthrough.ipynb # or jupyter notebook
```

**During the demo:**

```
T+0:00–0:30   Cells 1–7: intro + setup + show pipeline files
              → audience sees the file sizes contrast immediately

T+0:30–1:00   Cells 8–13: single question, three pipelines in parallel
              → ~25s wall-clock, visible latency gap
              → audience reads the actual answers side-by-side

T+1:00–1:45   Cells 14–25: dataset upload, experiments, LangFuse link
              → switch to LangFuse comparison view for the scores

T+1:45–2:00   Cells 26–30: code contrast + takeaways
              → grounded vs DIY source side-by-side via inspect.getsource()
```

The notebook also works as a self-paced exploration — collaborators can clone the repo, fill in `.env`, and walk through it themselves.

## The two comparisons this demo demonstrates

**1. Quality** — `baseline` vs `grounded`. Does retrieval reduce hallucinations on questions about recent events?

**2. Infrastructure** — `grounded` vs `diy-rag`. If you build RAG from scratch instead of using Brave's LLM Context endpoint, how much extra code, latency, and dependency surface do you take on — and do you get better answers for the trouble?

## Infrastructure overhead (the `grounded` vs `diy-rag` story)

Both pipelines start from the same place (Brave's search index) and end at the same place (Claude with a citation-forcing prompt). The only thing that changes between them is **what handles the middle**:

| Step | `grounded` (LLM Context) | `diy-rag` (from scratch) |
| --- | --- | --- |
| 1. Search | ✓ included | Brave Web Search API |
| 2. Fetch HTML | ✓ included | `httpx` with timeouts, redirects, UA |
| 3. Extract main content | ✓ included | `trafilatura` |
| 4. Chunk text | ✓ included | Custom paragraph-aware splitter |
| 5. Embed chunks | ✓ included | `sentence-transformers` (local model) |
| 6. Vector index | ✓ included | `faiss-cpu` |
| 7. Top-k retrieval | ✓ included | Inner-product search |
| 8. Generate with citations | Same | Same |

Concretely:

|  | `grounded` | `diy-rag` |
| --- | --- | --- |
| Pipeline file size | ~30 lines | ~200 lines |
| External services | 1 (Brave LLM Context) | 1 (Brave Web Search) + local embedder + local vector store |
| Python deps | `httpx` | `httpx`, `trafilatura`, `sentence-transformers`, `faiss-cpu`, `numpy` |
| First-run setup | API key | API key + ~80MB model download |
| Expected p50 latency | ~1–2s | ~5–10s |
| Failure modes | Brave is down | Brave is down, page fetches fail, extraction fails, embedding fails, FAISS errors, chunking edge cases |

A production DIY setup would add managed vector DB (Pinecone/Weaviate), hosted embeddings (OpenAI/Voyage), and a reranker (Cohere). Each of those is more code, more services, more cost, more failure modes.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) installed (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- A Brave Search API key on a plan that includes the **LLM Context** endpoint (verify at https://api-dashboard.search.brave.com/). Web Search is on most plans.
- An Anthropic API key
- A LangFuse account — Cloud free tier is fine (https://cloud.langfuse.com)
- ~500MB disk for the sentence-transformers model + faiss

Python version is pinned in `.python-version`; uv installs the right interpreter automatically if you don't have it.

## Setup

```bash
git clone <this repo> && cd brave-langfuse-demo
uv sync                              # creates .venv/, installs all deps, writes uv.lock

cp .env.example .env
# Fill in BRAVE_API_KEY, ANTHROPIC_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
set -a; source .env; set +a
```

## Run

```bash
uv run jupyter lab walkthrough.ipynb
```

(`uv run` executes inside the project's venv without needing an explicit `source .venv/bin/activate`.)

Then run cells top-to-bottom. First run downloads the sentence-transformers model (~80MB). After that, the full eval (5 questions × 3 pipelines × judges) completes in ~45–60 seconds with concurrent execution. Small enough to live-demo.

## What to look at in LangFuse

After running the experiment cells, the notebook prints a LangFuse URL. Open it and:

1. **Datasets → brave-finance-eval → Runs** — select all three (`baseline-v1`, `grounded-v1`, `diy-rag-v1`) and click "Compare." Three columns, three scores per row.
2. **Trace view** — click any item to see the full chain. For `diy-rag` traces, check the metadata for the `pipeline` and `latency_ms` fields.
3. **Aggregate scores per run** — these are your headline numbers for the write-up.

## Tuning knobs

While iterating, change `RUN_TAG = "v2"` in the notebook so each variation shows up as a separate column in the LangFuse compare view.

In `pipelines/grounded.py`:
- `GROUNDED_SYSTEM` — tighten or loosen citation rules
- `fetch_brave_context` params — try `count=20`, `context_threshold_mode="lenient"`, or `freshness="pw"`

In `pipelines/diy_rag.py`:
- `chunk_text` size/overlap — affects retrieval granularity
- `retrieve_top_k` value of `k` — affects how much context the LLM sees
- `EMBEDDING_MODEL_NAME` — swap to `BAAI/bge-small-en-v1.5` or others
- `fetch_all` `max_workers` — concurrency for the fetch stage

In the notebook itself:
- `QUESTIONS` list (cell 15) — swap the slant (crypto, macro only, regulatory only, etc.)
- Judge prompts (`evaluation.py`'s `JUDGE_CITATION` / `JUDGE_FACTUALITY`) — change what "good" means

## File layout

```
walkthrough.ipynb     # primary demo surface — narration + code + outputs interleaved
evaluation.py         # library: judges + task wrappers imported by the notebook

pipelines/
    __init__.py       # exposes the three pipeline functions
    baseline.py       #  32 lines — LLM only, no retrieval
    grounded.py       # 110 lines — Brave LLM Context endpoint -> LLM with citations
    diy_rag.py        # 280 lines — manual RAG (search + fetch + extract + chunk + embed + index + retrieve)

pyproject.toml        # project metadata + dependencies (grouped by pipeline)
.python-version       # Python version pin for uv
uv.lock               # generated by `uv sync` — pinned dependency tree (commit this)
.env.example
```

The three pipelines are siblings in `pipelines/`. The size difference between `grounded.py` and `diy_rag.py` *is the point* — both produce cited answers under the same contract; only the retrieval implementation differs.

## Cost notes

For one full sweep (5 questions × 3 runs):
- Brave: ~5 LLM Context calls + ~5 Web Search calls + ~40 page fetches
- Anthropic: ~15 generation calls + ~30 judge calls
- Sentence-transformers: free (runs locally on CPU)

In dollar terms this is well under $1 per sweep. Run as many variations as you want while iterating.

## Caveats

- The refusal-test question in the dataset (forward-looking) is intentionally unanswerable. Both grounded pipelines should refuse cleanly; baseline often won't. That contrast is part of the demo.
- The DIY pipeline uses a small local embedder (`all-MiniLM-L6-v2`) and in-memory FAISS for portability. A production setup with hosted embeddings + managed vector DB would have *more* dependencies and infrastructure, not fewer.
- Judge scoring is non-deterministic. For tighter eval, set `JUDGE_MODEL` to your strongest available model and run each experiment 2–3 times to estimate variance.
