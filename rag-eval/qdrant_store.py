"""
A Qdrant-backed store with the same surface run_eval.py uses.

Why this file lives here and not in the upstream tutorial repo: that repo is a
separate git checkout used for following along, and this swap is our experiment,
not theirs.

Only four entry points are actually exercised by the eval harness, so only
those are implemented:
    search()                        - run_eval.retrieve()
    add_chunks()                    - reindex, via indexer.embedder.index_chunks
    collection_exists()             - reindex
    client.delete_collection()      - reindex

QDRANT_URL=http://localhost:6333  talks to a container.
Unset, it falls back to an embedded on-disk store, so the swap can be verified
without Docker running.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

# Same dimensionality as all-MiniLM-L6-v2, which produced the vectors we index.
VECTOR_SIZE = 384

# Qdrant point ids must be an unsigned int or a UUID - it rejects the
# "<collection>:<filepath>:<start_line>" strings Chroma accepts. uuid5 keeps the
# mapping deterministic (same chunk id -> same point id on every reindex, so an
# upsert overwrites rather than duplicates), and the original string is kept in
# the payload so callers still see the id they wrote.
ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

DEFAULT_LOCAL_PATH = str(Path(__file__).parent / "data" / "qdrant")


def point_id(chunk_id: str) -> str:
    """Deterministic UUID for a chunk id string."""
    return str(uuid.uuid5(ID_NAMESPACE, chunk_id))


class QdrantStore:
    """Drop-in replacement for VectorStore, backed by Qdrant."""

    def __init__(self, url: str | None = None, path: str | None = None) -> None:
        url = url or os.environ.get("QDRANT_URL")
        if url:
            self.client = QdrantClient(url=url)
            self.backend = url
        else:
            # Embedded mode: no server, no Docker. Same query code path.
            self.client = QdrantClient(path=path or DEFAULT_LOCAL_PATH)
            self.backend = f"embedded:{path or DEFAULT_LOCAL_PATH}"

    def get_or_create_collection(self, name: str) -> str:
        """Create the collection if missing. Returns the name, not an object."""
        if not self.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
        return name

    def collection_exists(self, name: str) -> bool:
        """Check if a collection already exists."""
        return self.client.collection_exists(name)

    def add_chunks(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Upsert chunks. Batched for the same reason Chroma's wrapper batches."""
        self.get_or_create_collection(collection_name)

        points = []
        for i in range(len(ids)):
            payload = dict(metadatas[i])
            payload["chunk_id"] = ids[i]
            payload["document"] = documents[i]
            points.append(
                models.PointStruct(
                    id=point_id(ids[i]),
                    vector=embeddings[i],
                    payload=payload,
                )
            )

        batch_size = 500
        for start in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection_name,
                points=points[start : start + batch_size],
                wait=True,
            )

    def search(
        self,
        query_embedding: list[float],
        collection_name: str | None = None,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the same dict shape VectorStore.search() returns.

        Chroma reports a DISTANCE (lower is better, 1 - cosine); Qdrant reports a
        SCORE (higher is better, the cosine itself). Converting here keeps every
        caller - including run_eval's sort and hybrid's by_id map - unchanged.
        """
        if collection_name is None:
            raise ValueError("QdrantStore.search 需要显式的 collection_name")

        hits = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=n_results,
            with_payload=True,
        ).points

        results = []
        for hit in hits:
            payload = dict(hit.payload or {})
            content = payload.pop("document", "")
            chunk_id = payload.pop("chunk_id", str(hit.id))
            results.append(
                {
                    "id": chunk_id,
                    "content": content,
                    "metadata": payload,
                    "distance": 1.0 - hit.score,
                    "collection": collection_name,
                }
            )
        return results
