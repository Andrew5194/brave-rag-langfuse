"""
Shared LLM primitives for all three pipelines.

Houses the Anthropic client singleton, default model constants, the
citation system prompt shared by the retrieval pipelines, and a thin
``generate`` helper that standardises the messages.create call site so
each pipeline expresses only its unique retrieval logic.
"""

from __future__ import annotations

import os

import anthropic

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

ANSWER_MODEL: str = os.environ.get("ANSWER_MODEL", "claude-opus-4-7")

# Shared across all pipeline modules and evaluation.py — one place to bump.
_claude = anthropic.Anthropic(max_retries=6)  # ride out transient 529 overloaded_error with backoff

# ---------------------------------------------------------------------------
# Citation system prompt
# ---------------------------------------------------------------------------

# Both retrieval pipelines (diy_rag and brave_search_api) produce answers
# under the same citation rules so evaluation is apples-to-apples.  The only
# thing that varies between them is HOW the sources were retrieved.
CITATION_SYSTEM = """You are a financial research assistant. Answer the user's question using ONLY the sources provided below.

STRICT RULES:
- Every factual claim (numbers, dates, names, events, decisions) MUST be followed by a citation marker like [1], [2], etc., matching the source numbers below.
- If the sources do not contain enough information to answer, respond exactly: "I cannot verify this from the provided sources."
- Do not use prior knowledge that is not supported by the sources.
- Keep the answer to 3-6 sentences.

SOURCES:
{sources}
"""

# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate(system: str, question: str, max_tokens: int = 600) -> str:
    """Single Claude generation call. Returns the response text.

    Args:
        system: The fully-formatted system prompt (including injected sources).
        question: The user question passed as the human turn.
        max_tokens: Upper token budget for the response (default 600).

    Returns:
        The model's text response as a plain string.
    """
    msg = _claude.messages.create(
        model=ANSWER_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return msg.content[0].text


def format_sources(chunks: list[dict]) -> str:
    """Render a list of chunk dicts into the numbered source block expected by CITATION_SYSTEM.

    Each chunk must have keys ``n``, ``title``, ``url``, and ``snippet``.
    """
    return "\n\n".join(f"[{c['n']}] {c['title']} ({c['url']})\n{c['snippet']}" for c in chunks)
