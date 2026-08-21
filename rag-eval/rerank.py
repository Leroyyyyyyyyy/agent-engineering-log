"""
Cross-encoder reranking. Lever three of the four RAG levers.

The first stage (bi-encoder) embeds query and document *separately*, so the two
vectors never see each other - similarity is computed after the fact. A
cross-encoder feeds (query, document) through the model *together*, so every
token of the query can attend to every token of the document. That is far more
accurate and far more expensive, which is exactly why it runs second, on a
short candidate list.

The consequence worth knowing before using it: a reranker can only reorder what
the first stage already retrieved. It fixes ranking, never recall. We measured
the ceiling first - English cross_file has 89% of its expected chunks inside
top-50, Chinese has half of them below rank 100. So reranking can help English
a lot and Chinese almost not at all, and that was known before writing this.

Lives in rag-eval/, outside the indexed corpus, so it cannot perturb what it
measures.
"""

import os

MODEL_NAME = os.environ.get("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_model = None


def get_model():
    """Load the cross-encoder once, on first use."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Reorder candidates by cross-encoder relevance and return the best top_k.

    candidates are result dicts from the first stage; each needs "content".
    """
    if not candidates:
        return []

    pairs = []
    for candidate in candidates:
        pairs.append((query, candidate["content"]))

    scores = get_model().predict(pairs)

    scored = []
    for candidate, score in zip(candidates, scores):
        scored.append((float(score), candidate))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    ordered = []
    for _score, candidate in scored[:top_k]:
        ordered.append(candidate)
    return ordered
