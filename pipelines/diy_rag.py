"""
DIY RAG pipeline — what you build when you DON'T use Brave's LLM Context endpoint.

The whole point of this file is to show what's involved.

Steps the Brave LLM Context endpoint handles in a single API call,
done manually here:

  1. Search           -> Brave Web Search API (just for URL discovery)
  2. Fetch            -> requests with timeouts, redirects, UA header
  3. Extract          -> trafilatura (main-content extraction)
  4. Chunk            -> simple paragraph-aware chunker
  5. Embed            -> sentence-transformers (all-MiniLM-L6-v2, local)
  6. Index            -> FAISS in-memory IndexFlatIP
  7. Retrieve         -> top-k cosine similarity
  8. Generate         -> Claude with citation-forcing prompt (same as brave_search_api)

This is the *minimum* viable DIY pipeline. A production setup would add:
managed vector DB (Pinecone/Weaviate), hosted embeddings (OpenAI/Voyage),
reranker (Cohere Rerank), scraper rotation, retry queues, content
deduplication, embedding cache, etc. Every one of those is more code,
more services, more failure modes.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import requests
import trafilatura
from sentence_transformers import SentenceTransformer

# Same citation contract both retrieval pipelines share (defined in _llm.py):
# this file swaps out everything BEFORE the LLM call, but the answer format
# stays identical so the eval is apples-to-apples.
from ._llm import CITATION_SYSTEM, format_sources, generate

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BRAVE_API_KEY = os.environ["BRAVE_API_KEY"]
BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lazy-load the embedder (~80MB download on first run, a few seconds to load)
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    """Return the shared sentence-transformer embedder, loading it on first call."""
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
    r = requests.get(
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
        r = requests.get(url, timeout=timeout, headers=_FETCH_HEADERS, allow_redirects=True)
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

PARA_SEP = "\n\n"  # paragraphs are split on, and re-joined by, a blank line


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Paragraph-aware chunker. Splits on blank lines, packs paragraphs
    into chunks up to chunk_size chars. Long paragraphs are character-split
    with overlap."""
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split(PARA_SEP) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Paragraph alone exceeds the budget — flush what we have, then
            # hard-split the long paragraph into overlapping windows.
            if current:
                chunks.append(current)
                current = ""
            step = max(chunk_size - overlap, 1)
            for i in range(0, len(para), step):
                chunks.append(para[i : i + chunk_size])
        elif len(current) + len(para) + len(PARA_SEP) <= chunk_size:
            # Fits alongside what's buffered — append it to the current chunk.
            current = f"{current}{PARA_SEP}{para}" if current else para
        else:
            # Doesn't fit — close the current chunk and start a fresh one.
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Step 5 + 6: Embed + Index
# ---------------------------------------------------------------------------


def embed(texts: list[str]) -> np.ndarray:
    """Encode texts to L2-normalised float32 vectors using the local embedder."""
    embedder = _get_embedder()
    vectors = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype="float32")


def build_index(vectors: np.ndarray) -> faiss.Index:
    """Build a FAISS IndexFlatIP over pre-normalised vectors (inner product == cosine)."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


# ---------------------------------------------------------------------------
# Step 7: Retrieve
# ---------------------------------------------------------------------------


def retrieve_top_k(index: faiss.Index, query_vector: np.ndarray, k: int = 8) -> list[int]:
    """Return the indices of the top-k most similar chunks for query_vector."""
    _, indices = index.search(query_vector.reshape(1, -1), k)
    # FAISS pads the result row with -1 when fewer than k chunks exist; drop those.
    return [int(i) for i in indices[0] if i >= 0]


# ---------------------------------------------------------------------------
# Step 8: Generate (same citation contract as brave_search_api — see _llm.py)
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
    query_vector = embed([question])[0]  # embed() returns a batch; take the one query row
    top_idx = retrieve_top_k(index, query_vector, k=8)
    top_chunks = [chunks[i] for i in top_idx]

    # Assign each page a citation number on first appearance: [1] = most relevant.
    # Chunks from the same page reuse that page's number.
    citation_number: dict[str, int] = {}
    for chunk in top_chunks:
        if chunk["url"] not in citation_number:
            citation_number[chunk["url"]] = len(citation_number) + 1

    # The numbered source block the LLM sees — one entry per retrieved chunk.
    numbered_chunks = [
        {
            "n": citation_number[chunk["url"]],
            "url": chunk["url"],
            "title": chunk["title"],
            "snippet": chunk["text"],
        }
        for chunk in top_chunks
    ]
    sources_text = format_sources(numbered_chunks)

    # 8. Generate
    answer = generate(CITATION_SYSTEM.format(sources=sources_text), question)

    # One entry per unique page, already in citation order
    # (dicts preserve insertion order, so no sort is needed).
    unique_sources = [
        {"n": n, "url": url, "title": url_to_title.get(url, "")}
        for url, n in citation_number.items()
    ]

    return {
        "answer": answer,
        "sources": unique_sources,
        "sources_text": sources_text,
        "pipeline_steps": {
            "search_results": len(search_results),
            "pages_fetched": len(extracted),
            "chunks_produced": len(chunks),
            "chunks_used": len(top_chunks),
            "sources_cited": len(unique_sources),
        },
    }


def _empty(reason: str) -> dict:
    """Return the failure shape for early-exit conditions."""
    return {
        "answer": "I cannot verify this from the provided sources.",
        "sources": [],
        "pipeline_steps": {"failed_reason": reason},
    }
