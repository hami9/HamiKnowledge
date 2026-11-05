"""Multi-tenant query and retrieval cache with strict isolation boundaries.

Invariants:
- Cache keys MUST include workspace_id, permission_version, corpus_version,
  model, and retrieval_settings.
- Cross-boundary cache hits are strictly impossible by construction.
- Document deletion, reindexing, or ACL changes immediately invalidate scoped caches.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class CacheKey:
    """Strictly isolated cache key incorporating all roadmap-mandated dimensions."""

    workspace_id: UUID
    permission_version: str
    corpus_version: str
    model: str
    retrieval_settings: str
    query_hash: str

    @classmethod
    def create(
        cls,
        workspace_id: UUID,
        permission_version: str,
        corpus_version: str,
        model: str,
        retrieval_settings: dict[str, Any],
        query: str,
    ) -> CacheKey:
        settings_str = json.dumps(retrieval_settings, sort_keys=True)
        query_hash = hashlib.sha256(query.strip().encode("utf-8")).hexdigest()
        return cls(
            workspace_id=workspace_id,
            permission_version=permission_version,
            corpus_version=corpus_version,
            model=model,
            retrieval_settings=settings_str,
            query_hash=query_hash,
        )


class QueryCache:
    """Thread-safe in-memory cache supporting tenant isolation and invalidation."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[CacheKey, Any] = {}
        # Track active permission versions per workspace
        self._workspace_permission_versions: dict[UUID, int] = {}

    def get_permission_version(self, workspace_id: UUID | str) -> str:
        ws_uuid = UUID(str(workspace_id))
        with self._lock:
            ver = self._workspace_permission_versions.get(ws_uuid, 1)
            return f"v{ver}"

    def bump_permission_version(self, workspace_id: UUID | str) -> str:
        """Bump permission version on document deletion, reindex, or ACL revocation."""
        ws_uuid = UUID(str(workspace_id))
        with self._lock:
            ver = self._workspace_permission_versions.get(ws_uuid, 1) + 1
            self._workspace_permission_versions[ws_uuid] = ver
            self.invalidate_workspace(ws_uuid)
            return f"v{ver}"

    def get(self, key: CacheKey) -> Any | None:
        with self._lock:
            return self._entries.get(key)

    def set(self, key: CacheKey, value: Any) -> None:
        with self._lock:
            self._entries[key] = value

    def invalidate_workspace(self, workspace_id: UUID | str) -> int:
        """Invalidate all cached responses for a given workspace."""
        ws_uuid = UUID(str(workspace_id))
        with self._lock:
            to_remove = [k for k in self._entries if k.workspace_id == ws_uuid]
            for k in to_remove:
                del self._entries[k]
            return len(to_remove)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._workspace_permission_versions.clear()


# Global query cache instance
global_query_cache = QueryCache()
