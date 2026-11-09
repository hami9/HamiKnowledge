"""Embedding models and adapters for HamiKnowledge.

Provides:
- Pinned configuration candidate: intfloat/multilingual-e5-small (revision 614241f)
- DeterministicEmbeddingAdapter: Fast, reproducible, zero-network adapter for offline tests and CI
- SentenceTransformerEmbedder: PyTorch/SentenceTransformers implementation with pinned revisions
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Protocol, runtime_checkable

_WORD_RE = re.compile(r"[\w-]+", re.UNICODE)
DEFAULT_EMBEDDING_DIM = 384
PINNED_E5_SMALL_ID = "intfloat/multilingual-e5-small"
PINNED_E5_SMALL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for dense embedding models."""

    @property
    def model_id(self) -> str:
        ...

    @property
    def revision(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_query(self, query: str) -> list[float]:
        ...

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        ...


class DeterministicEmbeddingAdapter:
    """Fast, deterministic, zero-network embedding adapter for testing and CI.

    Maps text tokens and character n-grams into a fixed 384-dimensional unit vector
    using reproducible hashing. Texts with overlapping vocabulary produce high cosine
    similarity, accurately simulating dense vector behavior without heavy dependencies.
    """

    def __init__(
        self,
        dimension: int = DEFAULT_EMBEDDING_DIM,
        model_id: str = PINNED_E5_SMALL_ID,
        revision: str = PINNED_E5_SMALL_REVISION,
    ) -> None:
        self._dimension = dimension
        self._model_id = model_id
        self._revision = revision

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    def _hash_token(self, token: str) -> list[float]:
        """Project a token string to a pseudo-random deterministic vector."""
        vec = [0.0] * self._dimension
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        # Derive coordinates deterministically from digest bytes
        for i in range(self._dimension):
            byte_val = digest[i % len(digest)]
            sign = 1.0 if (byte_val & (1 << (i % 8))) else -1.0
            vec[i] = sign * ((byte_val % 127) / 127.0)
        return vec

    def _embed_text(self, text: str) -> list[float]:
        tokens = [t.lower() for t in _WORD_RE.findall(text) if len(t) > 1]
        if not tokens:
            # Fallback for empty or symbol-only text: uniform normalized vector
            val = 1.0 / math.sqrt(self._dimension)
            return [val] * self._dimension

        accum = [0.0] * self._dimension
        for token in tokens:
            t_vec = self._hash_token(token)
            for i in range(self._dimension):
                accum[i] += t_vec[i]

        # Add 3-gram features for subword/morphological similarity
        for token in tokens:
            if len(token) >= 3:
                for j in range(len(token) - 2):
                    ngram = token[j : j + 3]
                    n_vec = self._hash_token(ngram)
                    for i in range(self._dimension):
                        accum[i] += 0.5 * n_vec[i]

        norm = math.sqrt(sum(x * x for x in accum))
        if norm == 0.0:
            val = 1.0 / math.sqrt(self._dimension)
            return [val] * self._dimension

        return [round(x / norm, 6) for x in accum]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_text(query)

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        return [self._embed_text(p) for p in passages]


class SentenceTransformerEmbedder:
    """Production embedder using sentence-transformers with pinned models."""

    def __init__(
        self,
        model_id: str = PINNED_E5_SMALL_ID,
        revision: str = PINNED_E5_SMALL_REVISION,
        dimension: int = DEFAULT_EMBEDDING_DIM,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._dimension = dimension
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._model: Any = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_id,
                revision=self._revision,
            )
        return self._model

    def embed_query(self, query: str) -> list[float]:
        model = self._get_model()
        formatted = f"{self._query_prefix}{query}"
        emb = model.encode(formatted, normalize_embeddings=True)
        return [float(x) for x in emb]

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        if not passages:
            return []
        model = self._get_model()
        formatted = [f"{self._passage_prefix}{p}" for p in passages]
        embeddings = model.encode(formatted, normalize_embeddings=True)
        return [[float(x) for x in emb] for emb in embeddings]
