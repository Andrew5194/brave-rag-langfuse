"""
Baseline pipeline — LLM only, no retrieval.

This is the hallucination baseline. No external knowledge,
no citations, no grounding. Whatever the model knows from training.
"""

from __future__ import annotations

from ._llm import generate

BASELINE_SYSTEM = (
    "You are a financial research assistant. Answer the user's question "
    "concisely and factually. Do not invent figures or dates you are not sure about."
)


def baseline_pipeline(question: str) -> dict:
    """One LLM call. No tools, no retrieval, no citations."""
    return {"answer": generate(BASELINE_SYSTEM, question), "sources": []}
