"""Tests for heading-aware chunker with offset preservation."""

from __future__ import annotations

import uuid

from ingestion.chunker import HeadingAwareChunker


def test_chunker_preserves_exact_source_character_offsets() -> None:
    doc_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    version = "v1.0"

    markdown_text = (
        "# Deployments in Kubernetes\n\n"
        "A Deployment provides declarative updates for Pods and ReplicaSets.\n"
        "You describe a desired state in a Deployment.\n\n"
        "## Creating a Deployment\n\n"
        "Here is an example of a Deployment. It creates a ReplicaSet to bring up three pods.\n"
        "The following manifest runs nginx.\n\n"
        "### Deployment Status\n\n"
        "Kubernetes automatically reconciles current state with desired state.\n"
    )

    chunker = HeadingAwareChunker(target_chunk_chars=120, overlap_chars=30)
    chunks = chunker.chunk(
        text=markdown_text,
        document_id=doc_id,
        workspace_id=ws_id,
        source_version=version,
        language="en",
    )

    assert len(chunks) >= 3

    # CRITICAL: Every chunk content must exactly match the raw slice in the source text
    for chunk in chunks:
        loc = chunk.location
        sliced_text = markdown_text[loc.start_char : loc.end_char]
        assert sliced_text == chunk.content, (
            f"Offset mismatch: expected {sliced_text!r}, got {chunk.content!r}"
        )


def test_heading_hierarchy_path_tracking() -> None:
    doc_id = uuid.uuid4()
    ws_id = uuid.uuid4()

    text = (
        "# Storage\n\n"
        "Storage overview text.\n\n"
        "## Persistent Volumes\n\n"
        "A PersistentVolume (PV) is a piece of storage in the cluster.\n\n"
        "### Reclaim Policy\n\n"
        "Current reclaim policies include Retain and Delete.\n"
    )

    chunker = HeadingAwareChunker(target_chunk_chars=300, overlap_chars=40)
    chunks = chunker.chunk(
        text=text,
        document_id=doc_id,
        workspace_id=ws_id,
        source_version="v1.0",
    )

    # Find the chunk about reclaim policy
    reclaim_chunks = [c for c in chunks if "Reclaim Policy" in c.content]
    assert len(reclaim_chunks) >= 1
    assert reclaim_chunks[0].location.heading_path == (
        "Storage",
        "Persistent Volumes",
        "Reclaim Policy",
    )


def test_deterministic_chunk_ids() -> None:
    doc_id = uuid.uuid4()
    ws_id = uuid.uuid4()
    text = "Short text for chunking."

    chunker = HeadingAwareChunker()
    chunks_1 = chunker.chunk(text, doc_id, ws_id, "v1")
    chunks_2 = chunker.chunk(text, doc_id, ws_id, "v1")

    assert len(chunks_1) == len(chunks_2) == 1
    assert chunks_1[0].chunk_id == chunks_2[0].chunk_id
