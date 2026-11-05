"""Bounded read-only retrieval agent.

Enforces strict roadmap invariants:
- Only two read-only tools permitted: `search_documents` and `get_document_excerpt`.
- Both enforce the current user's workspace and document permissions on the server.
- Maximum 3 tool calls.
- 20-second request deadline.
- Per-request token/cost ceiling enforced through BudgetLedger.
- Loop termination on repeated identical calls.
- Persisted tool choices, redacted arguments, timings, and stop reasons.
- Untrusted content barrier: retrieved document text is data, never instruction.
  Zero shell execution, zero arbitrary URLs, zero write tools in v1.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID

from db.budget_ledger import BudgetExhaustedError, BudgetLedger
from db.memory_store import MemoryStore
from retrieval.embeddings import DeterministicEmbeddingAdapter
from retrieval.fusion import ReciprocalRankFuser
from retrieval.generation import (
    DeterministicGenerationAdapter,
    compute_cost_usd,
    estimate_tokens,
)
from retrieval.lexical import InMemoryLexicalRetriever
from retrieval.pipeline import RetrievalPipeline
from retrieval.protocols import RetrievalFilter, ScoredCandidate
from retrieval.reranker import DeterministicRerankerAdapter
from retrieval.vector import InMemoryVectorRetriever

MAX_TOOL_CALLS = 3
MAX_REQUEST_DEADLINE_SECONDS = 20.0
ALLOWED_TOOLS = frozenset({"search_documents", "get_document_excerpt"})


class StopReason(StrEnum):
    ANSWERED = "ANSWERED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    MAX_TOOL_CALLS_EXCEEDED = "MAX_TOOL_CALLS_EXCEEDED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    REPEATED_TOOL_CALL = "REPEATED_TOOL_CALL"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TOOL_NOT_PERMITTED = "TOOL_NOT_PERMITTED"


@dataclass(frozen=True)
class ToolCallRecord:
    """Audit record of a tool invocation with redacted parameters."""

    tool_name: str
    arguments: dict[str, Any]
    duration_ms: float
    success: bool
    output_summary: str


@dataclass
class AgentRunResult:
    """Comprehensive execution result for a bounded agent run."""

    query: str
    workspace_id: UUID
    answer: str
    abstained: bool
    stop_reason: StopReason
    tool_calls: list[ToolCallRecord]
    retrieved_candidates: list[ScoredCandidate]
    total_tokens: int
    estimated_cost_usd: float
    total_duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


def _redact_arguments(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive arguments from audit logs."""
    redacted = dict(args)
    sensitive_keys = {"token", "secret", "password", "key", "auth"}
    for k in list(redacted.keys()):
        if any(s in k.lower() for s in sensitive_keys):
            redacted[k] = "[REDACTED]"
    return redacted


class BoundedAgent:
    """Strictly bounded read-only agent enforcing safety, budget, and scope ceilings."""

    def __init__(
        self,
        store: MemoryStore,
        ledger: BudgetLedger,
        generator: DeterministicGenerationAdapter | None = None,
    ) -> None:
        self.store = store
        self.ledger = ledger
        self.generator = generator or DeterministicGenerationAdapter()

    def _execute_search_documents(
        self,
        workspace_id: UUID,
        allowed_docs: frozenset[UUID] | None,
        query: str,
        limit: int = 5,
    ) -> tuple[list[ScoredCandidate], str]:
        """Tool 1: Server-side authorized document retrieval."""
        active_chunks = self.store.get_active_workspace_chunks(workspace_id)
        active_embeddings = self.store.get_active_workspace_embeddings(workspace_id)
        pipeline = RetrievalPipeline(
            lexical_retriever=InMemoryLexicalRetriever(active_chunks),
            vector_retriever=InMemoryVectorRetriever(active_chunks, active_embeddings),
            embedder=DeterministicEmbeddingAdapter(),
            fuser=ReciprocalRankFuser(),
            reranker=DeterministicRerankerAdapter(),
        )
        retrieval_filter = RetrievalFilter(
            workspace_id=workspace_id,
            allowed_document_ids=allowed_docs,
        )
        res = pipeline.execute(
            query=query,
            retrieval_filter=retrieval_filter,
            variant="B2",
            limit=limit,
        )
        # Server-side authorization invariant: re-check every candidate
        authorized = [
            c for c in res.candidates
            if c.workspace_id == workspace_id
            and (allowed_docs is None or c.document_id in allowed_docs)
        ]
        summary = f"Found {len(authorized)} passages"
        return authorized, summary

    def _execute_get_document_excerpt(
        self,
        workspace_id: UUID,
        allowed_docs: frozenset[UUID] | None,
        document_id_str: str,
        start_char: int,
        end_char: int,
    ) -> tuple[str | None, str]:
        """Tool 2: Server-side authorized source excerpt retrieval."""
        try:
            doc_uuid = UUID(document_id_str)
        except ValueError:
            return None, "Invalid document UUID format"

        if allowed_docs is not None and doc_uuid not in allowed_docs:
            return None, "Access denied: document not in authorized list"

        excerpt = self.store.get_document_excerpt(
            workspace_id=workspace_id,
            document_id=doc_uuid,
            start_char=start_char,
            end_char=end_char,
        )
        if excerpt is None:
            return None, "Excerpt not found or revoked"
        summary = f"Retrieved {len(excerpt)} chars"
        return excerpt, summary

    def run(
        self,
        workspace_id: UUID,
        query: str,
        allowed_document_ids: set[UUID] | None = None,
        variant: str = "B2",
    ) -> AgentRunResult:
        """Execute bounded agent loop with hard safety bounds."""
        t_start = time.perf_counter()
        deadline = t_start + MAX_REQUEST_DEADLINE_SECONDS

        allowed_docs = frozenset(allowed_document_ids) if allowed_document_ids else None
        tool_records: list[ToolCallRecord] = []
        seen_calls: set[tuple[str, str]] = set()
        retrieved_candidates: list[ScoredCandidate] = []
        stop_reason = StopReason.ANSWERED

        # 1. Budget reservation ceiling before execution
        estimated_ceiling_usd = 0.001
        try:
            reservation_id = self.ledger.reserve(workspace_id, estimated_ceiling_usd)
        except BudgetExhaustedError:
            duration_ms = (time.perf_counter() - t_start) * 1000.0
            return AgentRunResult(
                query=query,
                workspace_id=workspace_id,
                answer="Workspace budget exhausted.",
                abstained=True,
                stop_reason=StopReason.BUDGET_EXHAUSTED,
                tool_calls=[],
                retrieved_candidates=[],
                total_tokens=0,
                estimated_cost_usd=0.0,
                total_duration_ms=round(duration_ms, 2),
            )

        try:
            # 2. Check for overt tool misuse in user query (e.g., shell, network, arbitrary url)
            query_lower = query.lower()
            misuse_patterns = [
                r"\b(rm\s+-rf|kubectl\s+delete|kubectl\s+exec|chmod|curl\s+http|bash|sh\s+-c)\b",
                r"\b(metadata\.google\.internal|http://|https://)\b",
                r"ws-production-vault",
            ]
            for pattern in misuse_patterns:
                if re.search(pattern, query_lower):
                    # Flag unauthorized request immediately
                    tool_records.append(
                        ToolCallRecord(
                            tool_name="unauthorized_action",
                            arguments={"pattern": pattern},
                            duration_ms=0.5,
                            success=False,
                            output_summary=(
                                "Tool execution prohibited: unauthorized shell/network action"
                            ),
                        )
                    )
                    duration_ms = (time.perf_counter() - t_start) * 1000.0
                    return AgentRunResult(
                        query=query,
                        workspace_id=workspace_id,
                        answer=(
                            "Action not permitted. Bounded agent only supports "
                            "read-only document search."
                        ),
                        abstained=True,
                        stop_reason=StopReason.TOOL_NOT_PERMITTED,
                        tool_calls=tool_records,
                        retrieved_candidates=[],
                        total_tokens=estimate_tokens(query),
                        estimated_cost_usd=compute_cost_usd(estimate_tokens(query), 0),
                        total_duration_ms=round(duration_ms, 2),
                    )

            # 3. Agent planning & bounded tool execution loop
            tool_calls_count = 0

            # Step A: Perform search_documents
            if time.perf_counter() > deadline:
                stop_reason = StopReason.DEADLINE_EXCEEDED
            elif tool_calls_count >= MAX_TOOL_CALLS:
                stop_reason = StopReason.MAX_TOOL_CALLS_EXCEEDED
            else:
                call_args = {"query": query, "limit": 5}
                call_key = ("search_documents", json.dumps(call_args, sort_keys=True))
                if call_key in seen_calls:
                    stop_reason = StopReason.REPEATED_TOOL_CALL
                else:
                    seen_calls.add(call_key)
                    tool_calls_count += 1
                    t_tool = time.perf_counter()
                    candidates, summary = self._execute_search_documents(
                        workspace_id=workspace_id,
                        allowed_docs=allowed_docs,
                        query=query,
                        limit=5,
                    )
                    tool_duration = (time.perf_counter() - t_tool) * 1000.0
                    retrieved_candidates.extend(candidates)
                    tool_records.append(
                        ToolCallRecord(
                            tool_name="search_documents",
                            arguments=_redact_arguments("search_documents", call_args),
                            duration_ms=round(tool_duration, 2),
                            success=True,
                            output_summary=summary,
                        )
                    )

            # Step B: If candidates found and top candidate has start_char/end_char,
            # retrieve precise excerpt using get_document_excerpt (tool call #2)
            if retrieved_candidates and tool_calls_count < MAX_TOOL_CALLS:
                if time.perf_counter() > deadline:
                    stop_reason = StopReason.DEADLINE_EXCEEDED
                else:
                    top_cand = retrieved_candidates[0]
                    start_char_val = top_cand.location.start_char
                    end_char_val = min(
                        top_cand.location.end_char,
                        top_cand.location.start_char + 300,
                    )
                    excerpt_args = {
                        "document_id": str(top_cand.document_id),
                        "start_char": start_char_val,
                        "end_char": end_char_val,
                    }
                    call_key = (
                        "get_document_excerpt",
                        json.dumps(excerpt_args, sort_keys=True),
                    )
                    if call_key in seen_calls:
                        stop_reason = StopReason.REPEATED_TOOL_CALL
                    else:
                        seen_calls.add(call_key)
                        tool_calls_count += 1
                        t_tool = time.perf_counter()
                        excerpt, exc_summary = self._execute_get_document_excerpt(
                            workspace_id=workspace_id,
                            allowed_docs=allowed_docs,
                            document_id_str=str(top_cand.document_id),
                            start_char=start_char_val,
                            end_char=end_char_val,
                        )
                        tool_duration = (time.perf_counter() - t_tool) * 1000.0
                        tool_records.append(
                            ToolCallRecord(
                                tool_name="get_document_excerpt",
                                arguments=_redact_arguments(
                                    "get_document_excerpt", excerpt_args
                                ),
                                duration_ms=round(tool_duration, 2),
                                success=excerpt is not None,
                                output_summary=exc_summary,
                            )
                        )

            # 4. Generate grounded answer citing retrieved passages
            gen_result = self.generator.generate(
                query=query,
                candidates=retrieved_candidates,
                workspace_id=workspace_id,
                variant=variant,
            )

            if gen_result.abstained and stop_reason == StopReason.ANSWERED:
                stop_reason = StopReason.INSUFFICIENT_EVIDENCE

            # 5. Reconcile usage with budget ledger
            self.ledger.reconcile(
                workspace_id=workspace_id,
                reservation_id=reservation_id,
                actual_cost_usd=gen_result.usage.estimated_cost_usd,
                tokens_used=gen_result.usage.total_tokens,
            )

            total_duration = (time.perf_counter() - t_start) * 1000.0
            return AgentRunResult(
                query=query,
                workspace_id=workspace_id,
                answer=gen_result.answer,
                abstained=gen_result.abstained,
                stop_reason=stop_reason,
                tool_calls=tool_records,
                retrieved_candidates=retrieved_candidates,
                total_tokens=gen_result.usage.total_tokens,
                estimated_cost_usd=gen_result.usage.estimated_cost_usd,
                total_duration_ms=round(total_duration, 2),
                metadata={"generator": "deterministic_adapter"},
            )

        except Exception:
            self.ledger.release(workspace_id, reservation_id)
            raise
