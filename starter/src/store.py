"""Qdrant local-mode vector store — persistent, no server needed.

Replaces the pgvector starter since Docker is not available on this machine.
Qdrant local mode (QdrantClient(path=...)) stores vectors on disk and supports
cosine similarity search + metadata filtering. GUIDE.md explicitly allows Qdrant.
"""
from __future__ import annotations
from typing import Any
from .models import Chunk


class QdrantVectorStore:
    """Persistent Qdrant store; one collection per strategy."""

    def __init__(self, persist_directory: str, dimensions: int) -> None:
        from qdrant_client import QdrantClient  # type: ignore
        self._dims = dimensions
        self._persist_dir = persist_directory
        self._client = QdrantClient(path=persist_directory)
        print(f"[store] Qdrant local @ {persist_directory}")

    # ------------------------------------------------------------------
    def create_schema(self, strategy: str) -> None:
        """Create (or recreate) a cosine-distance collection for strategy."""
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, VectorParams  # type: ignore

        name = f"strategy_{strategy}"
        # Delete if exists (fresh ingest each run)
        cols = [c.name for c in self._client.get_collections().collections]
        if name in cols:
            self._client.delete_collection(name)
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=self._dims, distance=Distance.COSINE),
        )
        print(f"[store] Collection '{name}' created.")

    # ------------------------------------------------------------------
    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Write chunks with sequential int IDs; payload carries text + metadata."""
        from qdrant_client.models import PointStruct  # type: ignore

        if len(chunks) != len(vectors):
            raise ValueError(
                f"Length mismatch: {len(chunks)} chunks vs {len(vectors)} vectors"
            )

        name = f"strategy_{strategy}"
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source,
                "page": str(chunk.page),
                "text": chunk.text,
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            points.append(PointStruct(id=i, vector=vec, payload=payload))

        # Upsert in batches of 256
        batch = 256
        for i in range(0, len(points), batch):
            self._client.upsert(
                collection_name=name,
                points=points[i : i + batch],
            )
        print(f"[store] Upserted {len(chunks)} points -> {name}.")

    # ------------------------------------------------------------------
    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Rank by cosine similarity; apply metadata filter when provided."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue  # type: ignore

        name = f"strategy_{strategy}"

        qdrant_filter = None
        if filters:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=str(v)))
                for k, v in filters.items()
            ]
            qdrant_filter = Filter(must=conditions)

        results = self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

        hits = []
        for point in results:
            payload = point.payload or {}
            hits.append({
                "text": payload.get("text", ""),
                "score": round(point.score, 6),
                "metadata": {
                    k: v for k, v in payload.items() if k != "text"
                },
            })
        return hits
