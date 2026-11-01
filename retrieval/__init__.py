"""HamiKnowledge retrieval package: protocols, search baselines, fusion, and reranking."""

from retrieval.embeddings import (
    DeterministicEmbeddingAdapter,
    EmbeddingModel,
    SentenceTransformerEmbedder,
)
from retrieval.fusion import ReciprocalRankFuser
from retrieval.lexical import InMemoryLexicalRetriever, build_postgres_fts_sql
from retrieval.protocols import (
    Chunk,
    CrossEncoderReranker,
    LexicalRetriever,
    RankFuser,
    RetrievalFilter,
    ScoredCandidate,
    SourceLocation,
    TextChunker,
    TextNormalizer,
    VectorRetriever,
)
from retrieval.reranker import (
    DeterministicRerankerAdapter,
    SentenceTransformerReranker,
)
from retrieval.vector import InMemoryVectorRetriever, build_pgvector_sql

__all__ = [
    "Chunk",
    "CrossEncoderReranker",
    "DeterministicEmbeddingAdapter",
    "DeterministicRerankerAdapter",
    "EmbeddingModel",
    "InMemoryLexicalRetriever",
    "InMemoryVectorRetriever",
    "LexicalRetriever",
    "RankFuser",
    "ReciprocalRankFuser",
    "RetrievalFilter",
    "ScoredCandidate",
    "SentenceTransformerEmbedder",
    "SentenceTransformerReranker",
    "SourceLocation",
    "TextChunker",
    "TextNormalizer",
    "VectorRetriever",
    "build_pgvector_sql",
    "build_postgres_fts_sql",
]
