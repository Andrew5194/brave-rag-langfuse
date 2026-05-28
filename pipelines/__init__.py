"""
The three pipelines being compared.

Each implements the same interface: a function taking a question string
and returning a dict with at least `answer` and `sources` keys. The
contract is identical so they can be A/B/C tested on the same dataset.

  baseline         →  LLM only, no retrieval
  diy_rag          →  Manual RAG (search + fetch + extract + chunk + embed + index + retrieve)
  brave_search_api →  Brave LLM Context endpoint -> LLM with citations
"""

from .baseline import baseline_pipeline
from .diy_rag import diy_rag_pipeline
from .brave_search_api import brave_search_api_pipeline

__all__ = ["baseline_pipeline", "diy_rag_pipeline", "brave_search_api_pipeline"]
