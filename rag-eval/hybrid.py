"""
Hybrid retrieval: vector search fused with BM25 keyword search.

Lever two of the four RAG levers. Vector search matches meaning but is blind to
rare literal tokens; BM25 matches tokens but is blind to paraphrase. Code search
needs both - `def execute_grep(` is a token problem, "run a command without a
shell" is a meaning problem.

The BM25 index is built from the documents Chroma actually holds, not from a
fresh chunk_repository() call. That removes a whole class of skew: if the
chunker changed since the last reindex, the two retrievers would otherwise be
searching different corpora.

This module lives in rag-eval/, outside the indexed corpus, so unlike the
chunker experiments it cannot perturb the thing it measures.
"""

import math
import re
from collections import Counter

# BM25 constants. Standard defaults - not tuned, so they are not a hidden variable.
K1 = 1.5
B = 0.75

# Reciprocal Rank Fusion damping. 60 is the value from the original RRF paper.
RRF_K = 60

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """
    Split text into search tokens, keeping identifiers at two granularities.

    `FILE_TOOLS` yields "file_tools", "file" and "tools"; `VectorStore` yields
    "vectorstore", "vector" and "store". Keeping only the whole identifier means
    a query saying "tool groups" never matches FILE_TOOLS; keeping only the
    pieces loses the precision that makes BM25 worth having on code.

    Non-ASCII text (e.g. Chinese queries) produces no tokens at all, so BM25
    contributes nothing for them. That is a real property of this design, not
    an oversight - see NOTES on the cross-lingual gap.
    """
    tokens = []
    for match in _TOKEN.findall(text):
        whole = match.lower()
        tokens.append(whole)

        # Split on the original casing - lowercasing first would erase the
        # camelCase boundaries that VectorStore/MemoryStore depend on.
        parts = []
        for piece in match.split("_"):
            if piece:
                parts.extend(_CAMEL.split(piece))

        if len(parts) > 1:
            for part in parts:
                lowered = part.lower()
                if lowered and lowered != whole:
                    tokens.append(lowered)
    return tokens


class BM25Index:
    """Plain BM25 over a fixed set of documents."""

    def __init__(self, ids: list[str], documents: list[str]) -> None:
        self.ids = ids
        self.doc_tokens = []
        self.doc_len = []
        for text in documents:
            tokens = tokenize(text)
            self.doc_tokens.append(Counter(tokens))
            self.doc_len.append(len(tokens))

        total_len = 0
        for length in self.doc_len:
            total_len += length
        self.avg_len = total_len / len(self.doc_len) if self.doc_len else 0.0

        # Document frequency per term
        self.doc_freq: Counter[str] = Counter()
        for counts in self.doc_tokens:
            for term in counts:
                self.doc_freq[term] += 1

        self.n_docs = len(documents)

    def idf(self, term: str) -> float:
        """Inverse document frequency, floored at zero for very common terms."""
        seen = self.doc_freq.get(term, 0)
        if seen == 0:
            return 0.0
        value = math.log(1 + (self.n_docs - seen + 0.5) / (seen + 0.5))
        return max(value, 0.0)

    def search(self, query: str, n_results: int) -> list[tuple[str, float]]:
        """Return (doc_id, score) for the top n_results documents."""
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored = []
        for i, counts in enumerate(self.doc_tokens):
            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if freq == 0:
                    continue
                norm = 1 - B + B * (self.doc_len[i] / self.avg_len)
                score += self.idf(term) * (freq * (K1 + 1)) / (freq + K1 * norm)
            if score > 0.0:
                scored.append((self.ids[i], score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:n_results]


def build_bm25(store, collection_name: str) -> tuple[BM25Index, dict]:
    """
    Build a BM25 index from what Chroma actually stores.

    Returns the index plus an id -> result-dict map so fused results look
    exactly like VectorStore.search() output to the caller.
    """
    collection = store.client.get_collection(collection_name)
    got = collection.get(include=["documents", "metadatas"])

    by_id = {}
    for i, doc_id in enumerate(got["ids"]):
        by_id[doc_id] = {
            "content": got["documents"][i],
            "metadata": got["metadatas"][i],
        }

    return BM25Index(got["ids"], got["documents"]), by_id


def fuse(vector_ids: list[str], bm25_ids: list[str], w_vector: float, w_bm25: float) -> list[str]:
    """
    Reciprocal Rank Fusion of two ranked id lists.

    RRF combines *ranks*, not scores, which is the point: cosine distance and
    BM25 scores live on incompatible scales, and normalising them introduces a
    tuning parameter that quietly does the real work.
    """
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(vector_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_vector / (RRF_K + rank)

    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 / (RRF_K + rank)

    ordered = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)

    fused = []
    for doc_id, _ in ordered:
        fused.append(doc_id)
    return fused
