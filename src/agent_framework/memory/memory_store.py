"""Persistent memory store for agent learning across tasks.

Memories are stored as JSON files keyed by (repo, agent_type).
Each memory entry has a category, content, and metadata for
relevance scoring and recency decay.

Storage layout:
  .agent-communication/memory/{repo_slug}/{agent_type}.json
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Guard against unbounded growth
MAX_MEMORIES_PER_STORE = 200
MAX_CONTENT_LENGTH = 2000


@dataclass
class MemoryEntry:
    """Single memory item stored by an agent."""
    category: str          # e.g. "repo_structure", "test_commands", "conventions", "tool_patterns"
    content: str           # The actual learned information
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source_task_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def touch(self) -> None:
        """Update access metadata."""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryStore:
    """Read/write persistent memories per (repo, agent_type) pair.

    Thread-safe via atomic writes (temp file + rename).
    """

    def __init__(self, workspace: Path, enabled: bool = True):
        self._workspace = Path(workspace)
        self._enabled = enabled
        self._base_dir = self._workspace / ".agent-communication" / "memory"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _store_path(self, repo_slug: str, agent_type: str) -> Path:
        safe_repo = repo_slug.replace("/", "__")
        return self._base_dir / safe_repo / f"{agent_type}.json"

    def _load_entries(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load memory store {path}: {e}")
        return []

    def _save_entries(self, path: Path, entries: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}")
        try:
            tmp_path.write_text(json.dumps(entries, indent=2))
            tmp_path.replace(path)
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def remember(
        self,
        repo_slug: str,
        agent_type: str,
        category: str,
        content: str,
        source_task_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """Store a new memory. Returns True on success."""
        if not self._enabled:
            return False

        content = content[:MAX_CONTENT_LENGTH]
        path = self._store_path(repo_slug, agent_type)
        entries = self._load_entries(path)

        # Deduplicate: if same category+content exists, just touch it
        for entry in entries:
            if entry.get("category") == category and entry.get("content") == content:
                entry["last_accessed"] = time.time()
                entry["access_count"] = entry.get("access_count", 0) + 1
                self._save_entries(path, entries)
                return True

        new_entry = MemoryEntry(
            category=category,
            content=content,
            source_task_id=source_task_id,
            tags=tags or [],
        )
        entries.append(asdict(new_entry))

        # Evict oldest entries if over limit
        if len(entries) > MAX_MEMORIES_PER_STORE:
            entries.sort(key=lambda e: e.get("last_accessed", 0))
            entries = entries[-MAX_MEMORIES_PER_STORE:]

        self._save_entries(path, entries)
        return True

    def recall(
        self,
        repo_slug: str,
        agent_type: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[MemoryEntry]:
        """Retrieve memories, optionally filtered by category/tags."""
        if not self._enabled:
            return []

        path = self._store_path(repo_slug, agent_type)
        entries = self._load_entries(path)

        results = []
        for raw in entries:
            if category and raw.get("category") != category:
                continue
            if tags:
                entry_tags = set(raw.get("tags", []))
                if not entry_tags.intersection(tags):
                    continue
            try:
                results.append(MemoryEntry(**{
                    k: v for k, v in raw.items()
                    if k in MemoryEntry.__dataclass_fields__
                }))
            except (TypeError, KeyError):
                continue

        # Sort by recency, return top N
        results.sort(key=lambda m: m.last_accessed, reverse=True)
        return results[:limit]

    def recall_all(self, repo_slug: str, agent_type: str) -> List[MemoryEntry]:
        """Retrieve all memories for a repo/agent pair."""
        return self.recall(repo_slug, agent_type, limit=MAX_MEMORIES_PER_STORE)

    def forget(self, repo_slug: str, agent_type: str, category: str, content: str) -> bool:
        """Remove a specific memory. Returns True if found and removed."""
        if not self._enabled:
            return False

        path = self._store_path(repo_slug, agent_type)
        entries = self._load_entries(path)

        original_count = len(entries)
        entries = [
            e for e in entries
            if not (e.get("category") == category and e.get("content") == content)
        ]

        if len(entries) < original_count:
            self._save_entries(path, entries)
            return True
        return False
