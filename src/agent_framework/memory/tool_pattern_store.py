"""Persistent storage for tool usage pattern recommendations.

Stores per-repo pattern recommendations with dedup, scoring, and eviction.
Mirrors the atomic write pattern from MemoryStore.

Storage layout:
  .agent-communication/memory/{repo_slug}/tool_patterns.json
"""

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .tool_pattern_analyzer import ToolPatternRecommendation

logger = logging.getLogger(__name__)

MAX_PATTERNS = 50
# Patterns older than 14 days lose half their score
RECENCY_HALF_LIFE = 14 * 86400


class ToolPatternStore:
    """Read/write persistent tool pattern recommendations per repo.

    Thread-safe via atomic writes (temp file + rename).
    """

    def __init__(self, workspace: Path, enabled: bool = True):
        self._workspace = Path(workspace)
        self._enabled = enabled
        self._base_dir = self._workspace / ".agent-communication" / "memory"

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _store_path(self, repo_slug: str) -> Path:
        safe_repo = repo_slug.replace("/", "__")
        return self._base_dir / safe_repo / "tool_patterns.json"

    def _load_entries(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load tool patterns {path}: {e}")
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

    def _score(self, entry: Dict[str, Any]) -> float:
        """Score = hit_count * recency_factor (exponential decay)."""
        age = time.time() - entry.get("last_seen", 0)
        recency_factor = 2 ** (-age / RECENCY_HALF_LIFE)
        return entry.get("hit_count", 1) * recency_factor

    def store_patterns(
        self, repo_slug: str, recommendations: List[ToolPatternRecommendation]
    ) -> int:
        """Merge new recommendations into the store. Returns count stored."""
        if not self._enabled or not recommendations:
            return 0

        path = self._store_path(repo_slug)
        entries = self._load_entries(path)

        # Index existing entries by pattern_id
        by_id: Dict[str, Dict] = {e["pattern_id"]: e for e in entries if "pattern_id" in e}

        stored = 0
        for rec in recommendations:
            existing = by_id.get(rec.pattern_id)
            if existing:
                existing["hit_count"] = existing.get("hit_count", 1) + 1
                existing["last_seen"] = time.time()
            else:
                by_id[rec.pattern_id] = asdict(rec)
                by_id[rec.pattern_id]["last_seen"] = time.time()
            stored += 1

        all_entries = list(by_id.values())

        # Evict beyond top MAX_PATTERNS by score
        if len(all_entries) > MAX_PATTERNS:
            all_entries.sort(key=self._score, reverse=True)
            all_entries = all_entries[:MAX_PATTERNS]

        self._save_entries(path, all_entries)
        return stored

    def get_top_patterns(
        self, repo_slug: str, limit: int = 5, max_chars: int = 1500
    ) -> List[ToolPatternRecommendation]:
        """Return top-N patterns within char budget, scored by hit_count * recency."""
        if not self._enabled:
            return []

        path = self._store_path(repo_slug)
        entries = self._load_entries(path)
        if not entries:
            return []

        # Score and sort descending
        entries.sort(key=self._score, reverse=True)

        results = []
        total_chars = 0
        for entry in entries[:limit]:
            tip = entry.get("tip", "")
            if total_chars + len(tip) > max_chars:
                break
            try:
                results.append(ToolPatternRecommendation(
                    pattern_id=entry["pattern_id"],
                    tip=tip,
                    hit_count=entry.get("hit_count", 1),
                    last_seen=entry.get("last_seen", 0),
                ))
                total_chars += len(tip)
            except (KeyError, TypeError):
                continue

        return results
