"""
DIY RAG pipeline — what you build when you DON'T use Brave's LLM Context endpoint.

The whole point of this file is to show what's involved.

Steps the Brave LLM Context endpoint handles in a single API call,
done manually here:

  1. Search           -> Brave Web Search API (just for URL discovery)
  2. Fetch            -> httpx with timeouts, redirects, UA header
  3. Extract          -> trafilatura (main-content extraction)
  4. Chunk            -> simple paragraph-aware chunker
  5. Embed            -> sentence-transformers (all-MiniLM-L6-v2, local)
  6. Index            -> FAISS in-memory IndexFlatIP
  7. Retrieve         -> top-k cosine similarity
  8. Generate         -> Claude with citation-forcing prompt (same as grounded)

This is the *minimum* viable DIY pipeline. A production setup would add:
managed vector DB (Pinecone/Weaviate), hosted embeddings (OpenAI/Voyage),
reranker (Cohere Rerank), scraper rotation, retry queues, content
deduplication, embedding cache, etc. Every one of those is more code,
more services, more failure modes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import faiss
import httpx
import numpy as np
import trafilatura
from sentence_transformers import SentenceTransformer

# Same citation contract as the Brave-grounded pipeline. The whole point of
# this file is to swap out everything BEFORE the LLM call — the answer
# format stays identical so eval is apples-to-apples.
from .grounded import GROUNDED_SYSTEM

# ---------------------------------------------------------------------------
# Config (mirrors compare.py so this file is self-contained)
# ---------------------------------------------------------------------------

BRAVE_API_KEY = os.environ["BRAVE_API_KEY"]
BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "claude-opus-4-5")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_claude = anthropic.Anthropic()

# Lazy-load the embedder (~80MB download on first run, a few seconds to load)
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


# ---------------------------------------------------------------------------
# Step 1: Search
# ---------------------------------------------------------------------------

def brave_web_search(query: str, count: int = 10, freshness: str = "pm") -> list[dict]:
    """Brave Web Search — used here ONLY for URL discovery (no content extraction).
    This keeps the comparison apples-to-apples with LLM Context, which starts
    from the same search index."""
    r = httpx.get(
        BRAVE_WEB_SEARCH_URL,
        headers={
            "X-Subscription-Token": BRAVE_API_KEY,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        },
        params={"q": query, "count": count, "freshness": freshness},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return [
        {"url": item["url"], "title": item.get("title", "")}
        for item in data.get("web", {}).get("results", [])
    ]


# ---------------------------------------------------------------------------
# Step 2 + 3: Fetch + Extract
# ---------------------------------------------------------------------------

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; brave-langfuse-demo/0.1; "
        "+https://github.com/your-handle/brave-langfuse-demo)"
    )
}


def fetch_and_extract(url: str, timeout: int = 10) -> str | None:
    """Fetch a URL and extract main content. Returns None on any failure
    (so one bad page doesn't kill the pipeline)."""
    try:
        r = httpx.get(url, timeout=timeout, follow_redirects=True, headers=_FETCH_HEADERS)
        r.raise_for_status()
        if "text/html" not in r.headers.get("content-type", ""):
            return None
        return trafilatura.extract(
            r.text,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:
        return None


def fetch_all(urls: list[str], max_workers: int = 8) -> dict[str, str]:
    """Concurrent fetch+extract. Returns {url: content} for successful fetches."""
    out: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_and_extract, url): url for url in urls}
        for fut in as_completed(futures):
            url = futures[fut]
            content = fut.result()
            if content:
                out[url] = content
    return out


# ---------------------------------------------------------------------------
# Step 4: Chunk
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Paragraph-aware chunker. Splits on blank lines, packs paragraphs
    into chunks up to chunk_size chars. Long paragraphs are character-split
    with overlap."""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(para), step):
                chunks.append(para[i : i + chunk_size])
        elif len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Step 5 + 6: Embed + Index
# ---------------------------------------------------------------------------

def embed(texts: list[str]) -> np.ndarray:
    embedder = _get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    # Inner product on normalized vectors == cosine similarity
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ---------------------------------------------------------------------------
# Step 7: Retrieve
# ---------------------------------------------------------------------------

def retrieve_top_k(index: faiss.Index, query_vector: np.ndarray, k: int = 8) -> list[int]:
    _, indices = index.search(query_vector.reshape(1, -1), k)
    return [int(i) for i in indices[0] if i >= 0]


# ---------------------------------------------------------------------------
# Step 8: Generate (citation contract imported from grounded.py — see top of file)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def diy_rag_pipeline(question: str) -> dict:
    """End-to-end manual RAG. The thing Brave LLM Context replaces with one call."""

    # 1. Search
    search_results = brave_web_search(question, count=10)
    if not search_results:
        return _empty("No search results returned.")

    # 2 + 3. Fetch + extract (concurrent)
    urls = [r["url"] for r in search_results[:8]]
    url_to_title = {r["url"]: r["title"] for r in search_results[:8]}
    extracted = fetch_all(urls)
    if not extracted:
        return _empty("All fetches failed or returned no extractable content.")

    # 4. Chunk every page
    chunks: list[dict] = []
    for url, content in extracted.items():
        for piece in chunk_text(content):
            chunks.append({"text": piece, "url": url, "title": url_to_title.get(url, "")})
    if not chunks:
        return _empty("No chunks produced from extracted content.")

    # 5 + 6. Embed + index
    chunk_vectors = embed([c["text"] for c in chunks])
    index = build_index(chunk_vectors)

    # 7. Retrieve
    query_vector = embed([question])[0]
    top_idx = retrieve_top_k(index, query_vector, k=8)
    top_chunks = [chunks[i] for i in top_idx]

    # Number sources by first appearance in top_chunks (so [1] = most relevant page)
    url_to_n: dict[str, int] = {}
    for c in top_chunks:
        if c["url"] not in url_to_n:
            url_to_n[c["url"]] = len(url_to_n) + 1

    numbered = [
        {"n": url_to_n[c["url"]], "url": c["url"], "title": c["title"], "snippet": c["text"]}
        for c in top_chunks
    ]
    sources_text = "\n\n".join(
        f"[{c['n']}] {c['title']} ({c['url']})\n{c['snippet']}" for c in numbered
    )

    # 8. Generate
    msg = _claude.messages.create(
        model=ANSWER_MODEL,
        max_tokens=600,
        system=GROUNDED_SYSTEM.format(sources=sources_text),
        messages=[{"role": "user", "content": question}],
    )

    unique_sources = [
        {"n": n, "url": url, "title": url_to_title.get(url, "")}
        for url, n in sorted(url_to_n.items(), key=lambda kv: kv[1])
    ]

    return {
        "answer": msg.content[0].text,
        "sources": unique_sources,
        "pipeline_steps": {
            "search_results": len(search_results),
            "pages_fetched": len(extracted),
            "chunks_produced": len(chunks),
            "chunks_used": len(top_chunks),
            "sources_cited": len(unique_sources),
        },
    }


def _empty(reason: str) -> dict:
    return {
        "answer": "I cannot verify this from the provided sources.",
        "sources": [],
        "pipeline_steps": {"failed_reason": reason},
    }
