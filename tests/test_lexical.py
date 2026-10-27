"""Tests for lexical retrieval and PostgreSQL FTS query generator."""

from __future__ import annotations

import uuid

from retrieval.lexical import InMemoryLexicalRetriever, build_postgres_fts_sql
from retrieval.protocols import Chunk, RetrievalFilter, SourceLocation


def _make_chunk(ws_id: uuid.UUID, doc_id: uuid.UUID, text: str, idx: int = 1) -> Chunk:
    return Chunk(
        chunk_id=f"chk_{idx}",
        document_id=doc_id,
        workspace_id=ws_id,
        source_version="v1",
        content=text,
        search_text=text,
        location=SourceLocation(start_char=0, end_char=len(text)),
        language="en",
    )


def test_build_postgres_fts_sql_enforces_workspace_predicate() -> None:
    ws_id = uuid.uuid4()
    doc_id = uuid.uuid4()
    flt = RetrievalFilter(workspace_id=ws_id, allowed_document_ids=frozenset([doc_id]))

    sql, params = build_postgres_fts_sql("crashloopbackoff", flt, limit=15)

    assert "c.workspace_id = :workspace_id" in sql
    assert "ts_rank_cd" in sql
    assert params["workspace_id"] == str(ws_id)
    assert params["allowed_doc_ids"] == [str(doc_id)]
    assert params["limit"] == 15


def test_in_memory_lexical_retriever_isolates_workspaces() -> None:
    ws_a = uuid.uuid4()
    ws_b = uuid.uuid4()
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    chunk_a = _make_chunk(ws_a, doc_a, "Pod security standards enforce baseline profiles", 1)
    chunk_b = _make_chunk(ws_b, doc_b, "Secret passwords in workspace B should never leak", 2)

    retriever = InMemoryLexicalRetriever([chunk_a, chunk_b])

    # User in workspace A searches for "passwords"
    results_a = retriever.retrieve_lexical(
        query="passwords",
        retrieval_filter=RetrievalFilter(workspace_id=ws_a),
    )
    # Must NOT find chunk_b which belongs to workspace B
    assert len(results_a) == 0

    # User in workspace A searches for "baseline profiles"
    results_found = retriever.retrieve_lexical(
        query="baseline profiles",
        retrieval_filter=RetrievalFilter(workspace_id=ws_a),
    )
    assert len(results_found) == 1
    assert results_found[0].chunk_id == "chk_1"
    assert results_found[0].branch == "lexical"


def test_in_memory_lexical_respects_document_acls() -> None:
    ws_id = uuid.uuid4()
    doc_allowed = uuid.uuid4()
    doc_restricted = uuid.uuid4()

    c1 = _make_chunk(ws_id, doc_allowed, "Public runbook for restartPolicy", 1)
    c2 = _make_chunk(ws_id, doc_restricted, "Internal runbook for restartPolicy", 2)

    retriever = InMemoryLexicalRetriever([c1, c2])

    filter_allowed = RetrievalFilter(
        workspace_id=ws_id,
        allowed_document_ids=frozenset([doc_allowed]),
    )
    results = retriever.retrieve_lexical("restartPolicy", filter_allowed)
    assert len(results) == 1
    assert results[0].document_id == doc_allowed


def test_persian_lexical_retrieval_with_character_variants() -> None:
    ws_id = uuid.uuid4()
    doc_id = uuid.uuid4()

    # Document contains Persian letters (ک and ی)
    chunk = _make_chunk(ws_id, doc_id, "کنترلرهای کوبرنتیز وضعیت واقعی را بررسی می\u200cکنند", 1)
    retriever = InMemoryLexicalRetriever([chunk])

    # Query uses Arabic letters (ك and ي)
    arabic_query = "كنترلرهاي كوبرنتيز"
    results = retriever.retrieve_lexical(
        query=arabic_query,
        retrieval_filter=RetrievalFilter(workspace_id=ws_id),
    )
    assert len(results) == 1
    assert results[0].chunk_id == "chk_1"
