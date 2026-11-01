"""Unit tests for the four retrieval variants (B0, B1, B2, B3) in RetrievalPipeline."""

from __future__ import annotations

from uuid import UUID, uuid4

from retrieval.embeddings import DeterministicEmbeddingAdapter
from retrieval.fusion import ReciprocalRankFuser
from retrieval.lexical import InMemoryLexicalRetriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.protocols import Chunk, RetrievalFilter, SourceLocation
from retrieval.reranker import DeterministicRerankerAdapter
from retrieval.vector import InMemoryVectorRetriever


def _create_chunk(content: str, ws_id: UUID, cid: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id=uuid4(),
        workspace_id=ws_id,
        source_version="v1",
        content=content,
        search_text=content.lower(),
        location=SourceLocation(start_char=0, end_char=len(content)),
        language="en",
    )


def test_pipeline_b0_b1_b2_b3_execution() -> None:
    ws_id = uuid4()
    embedder = DeterministicEmbeddingAdapter()

    chunk1 = _create_chunk(
        "Configure Pod Security Standards to restrict privileges.", ws_id, "chk_1"
    )
    chunk2 = _create_chunk("Storage volume provisioner and persistent volumes.", ws_id, "chk_2")

    chunks = [chunk1, chunk2]
    embeddings = {
        c.chunk_id: embedder.embed_query(c.search_text)
        for c in chunks
    }

    pipeline = RetrievalPipeline(
        lexical_retriever=InMemoryLexicalRetriever(chunks),
        vector_retriever=InMemoryVectorRetriever(chunks, embeddings),
        embedder=embedder,
        fuser=ReciprocalRankFuser(),
        reranker=DeterministicRerankerAdapter(),
    )

    filt = RetrievalFilter(workspace_id=ws_id)
    query = "Pod Security Standards"

    # Variant B0: Lexical only
    res_b0 = pipeline.execute(query=query, retrieval_filter=filt, variant="B0")
    assert res_b0.variant == "B0"
    assert len(res_b0.candidates) >= 1
    assert res_b0.candidates[0].branch == "lexical"
    assert res_b0.candidates[0].chunk_id == "chk_1"

    # Variant B1: Vector only
    res_b1 = pipeline.execute(query=query, retrieval_filter=filt, variant="B1")
    assert res_b1.variant == "B1"
    assert len(res_b1.candidates) >= 1
    assert res_b1.candidates[0].branch == "vector"

    # Variant B2: Hybrid RRF
    res_b2 = pipeline.execute(query=query, retrieval_filter=filt, variant="B2")
    assert res_b2.variant == "B2"
    assert len(res_b2.candidates) >= 1
    assert res_b2.candidates[0].branch == "fused"

    # Variant B3: Hybrid RRF + Cross-Encoder Reranker
    res_b3 = pipeline.execute(query=query, retrieval_filter=filt, variant="B3")
    assert res_b3.variant == "B3"
    assert len(res_b3.candidates) >= 1
    assert res_b3.candidates[0].branch == "reranked"
    assert res_b3.candidates[0].chunk_id == "chk_1"
    assert res_b3.timing.total_ms >= 0.0


def test_pipeline_zero_cross_tenant_leakage() -> None:
    ws_alpha = uuid4()
    ws_beta = uuid4()
    embedder = DeterministicEmbeddingAdapter()

    chunk_a = _create_chunk("Alpha confidential internal guide.", ws_alpha, "chk_a")
    chunk_b = _create_chunk("Beta private network security topology.", ws_beta, "chk_b")

    chunks = [chunk_a, chunk_b]
    embeddings = {c.chunk_id: embedder.embed_query(c.search_text) for c in chunks}

    pipeline = RetrievalPipeline(
        lexical_retriever=InMemoryLexicalRetriever(chunks),
        vector_retriever=InMemoryVectorRetriever(chunks, embeddings),
        embedder=embedder,
        fuser=ReciprocalRankFuser(),
        reranker=DeterministicRerankerAdapter(),
    )

    # Workspace Alpha searches for Beta terms
    res_leak_check = pipeline.execute(
        query="Beta private network security topology",
        retrieval_filter=RetrievalFilter(workspace_id=ws_alpha),
        variant="B3",
    )
    # ZERO RESULTS LEAKED
    assert len(res_leak_check.candidates) == 0
