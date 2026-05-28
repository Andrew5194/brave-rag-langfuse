"""
The three pipelines being compared.

Each implements the same interface: a function taking a question string
and returning a dict with at least `answer` and `sources` keys. The
contract is identical so they can be A/B/C tested on the same dataset.

  baseline    →  LLM only, no retrieval
  grounded    →  Brave LLM Context endpoint -> LLM with citations
  diy_rag     →  Manual RAG (search + fetch + extract + chunk + embed + index + retrieve)
"""

from .baseline import baseline_pipeline
from .grounded import grounded_pipeline
from .diy_rag import diy_rag_pipeline

__all__ = ["baseline_pipeline", "grounded_pipeline", "diy_rag_pipeline"]
