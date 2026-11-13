"""Local embedder using fastembed (ONNX runtime, no PyTorch needed).

Model: BAAI/bge-small-en-v1.5 — 384 dims, optimised for English semantic search.
Provider: fastembed (Qdrant), runs fully local, no API key required.
"""
from __future__ import annotations


class FastEmbedEmbedder:
    """Fastembed wrapper satisfying the Embedder protocol."""

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DIMENSIONS = 384

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from fastembed import TextEmbedding  # type: ignore
        self.model_name = model_name
        self.dimensions = self.DIMENSIONS
        print(f"[embedder] Loading fastembed model: {model_name} ...")
        self._model = TextEmbedding(model_name)
        # Warm-up: forces model download on first call
        list(self._model.embed(["warmup"]))
        print(f"[embedder] Ready. dims={self.dimensions}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one normalised vector per text, in input order."""
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        # BGE models use a query prefix for asymmetric retrieval
        prefixed = f"Represent this sentence for searching relevant passages: {question}"
        return list(self._model.embed([prefixed]))[0].tolist()
