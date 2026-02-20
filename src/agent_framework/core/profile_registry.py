"""File-based cache of LLM-generated specialization profiles.

Stores generated profiles so identical or similar tasks can reuse them without
re-running LLM generation. Scoring uses file pattern overlap, extension overlap,
and tag-based keyword matching.
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .engineer_specialization import SpecializationProfile, match_patterns

logger = logging.getLogger(__name__)


@dataclass
class GeneratedProfileEntry:
    """A generated profile plus metadata for cache matching and eviction."""

    profile: SpecializationProfile
    created_at: float
    last_matched_at: float
    match_count: int
    source_task_id: str
    tags: List[str]
    file_extensions: List[str]


def _entry_to_dict(entry: GeneratedProfileEntry) -> Dict:
    """Serialize entry to a JSON-compatible dict."""
    d = {
        "profile": {
            "id": entry.profile.id,
            "name": entry.profile.name,
            "description": entry.profile.description,
            "file_patterns": entry.profile.file_patterns,
            "prompt_suffix": entry.profile.prompt_suffix,
            "teammates": entry.profile.teammates,
            "tool_guidance": entry.profile.tool_guidance,
        },
        "created_at": entry.created_at,
        "last_matched_at": entry.last_matched_at,
        "match_count": entry.match_count,
        "source_task_id": entry.source_task_id,
        "tags": entry.tags,
        "file_extensions": entry.file_extensions,
    }
    return d


def _dict_to_entry(d: Dict) -> GeneratedProfileEntry:
    """Deserialize a dict back to a GeneratedProfileEntry."""
    p = d["profile"]
    profile = SpecializationProfile(
        id=p["id"],
        name=p["name"],
        description=p["description"],
        file_patterns=p["file_patterns"],
        prompt_suffix=p["prompt_suffix"],
        teammates=p.get("teammates", {}),
        tool_guidance=p.get("tool_guidance", ""),
    )
    return GeneratedProfileEntry(
        profile=profile,
        created_at=d["created_at"],
        last_matched_at=d["last_matched_at"],
        match_count=d["match_count"],
        source_task_id=d["source_task_id"],
        tags=d.get("tags", []),
        file_extensions=d.get("file_extensions", []),
    )


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity: |intersection| / |union|. Returns 0.0 for empty sets."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _extract_extensions(files: List[str]) -> set:
    """Extract unique lowercase extensions from file paths."""
    exts = set()
    for f in files:
        _, ext = os.path.splitext(f)
        if ext:
            exts.add(ext.lower())
    return exts


def _tokenize(text: str) -> set:
    """Split text into lowercase tokens for tag matching. Keeps alphanumeric terms like k8s, gRPC."""
    return {w.lower() for w in text.split() if len(w) > 3 and w.isalnum()}


class ProfileRegistry:
    """File-based cache of generated specialization profiles."""

    DEFAULT_MAX_PROFILES = 50
    DEFAULT_STALENESS_DAYS = 90

    def __init__(
        self,
        workspace: Path,
        max_profiles: int = DEFAULT_MAX_PROFILES,
        staleness_days: int = DEFAULT_STALENESS_DAYS,
    ):
        self._store_path = (
            workspace / ".agent-communication" / "profile-registry" / "profiles.json"
        )
        self._max_profiles = max_profiles
        self._staleness_days = staleness_days

    def find_matching_profile(
        self,
        files: List[str],
        task_description: str,
        min_score: float = 0.4,
    ) -> Optional[SpecializationProfile]:
        """Score cached profiles and return the best match above threshold.

        Scoring weights:
        - File pattern overlap (0.6): uses match_patterns() from engineer_specialization
        - Extension overlap (0.25): Jaccard similarity of file extensions
        - Tag overlap (0.15): Jaccard similarity of description tokens vs profile tags
        """
        entries = self._load_entries()
        if not entries:
            return None

        task_extensions = _extract_extensions(files)
        task_tokens = _tokenize(task_description)
        total_files = len(files) if files else 1

        best_score = 0.0
        best_entry: Optional[GeneratedProfileEntry] = None

        for entry in entries:
            # File pattern match ratio
            matched = match_patterns(files, entry.profile.file_patterns)
            pattern_score = matched / total_files

            # Extension overlap
            entry_exts = set(entry.file_extensions)
            ext_score = _jaccard(task_extensions, entry_exts)

            # Tag overlap
            entry_tags = {t.lower() for t in entry.tags}
            tag_score = _jaccard(task_tokens, entry_tags)

            score = 0.6 * pattern_score + 0.25 * ext_score + 0.15 * tag_score

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= min_score:
            # Update match metadata
            best_entry.last_matched_at = time.time()
            best_entry.match_count += 1
            self._save_entries(entries)
            logger.debug(
                "Profile '%s' matched with score %.3f (threshold %.2f)",
                best_entry.profile.id,
                best_score,
                min_score,
            )
            return best_entry.profile

        return None

    def store_profile(self, entry: GeneratedProfileEntry) -> None:
        """Persist a new profile entry, evicting stale and duplicate entries."""
        entries = self._load_entries()

        # Remove any existing entry with the same profile ID (update in place)
        new_id = entry.profile.id
        entries = [e for e in entries if e.profile.id != new_id]

        # Evict stale entries
        cutoff = time.time() - (self._staleness_days * 86400)
        entries = [e for e in entries if e.last_matched_at >= cutoff]

        # Evict least-recently-matched if still at capacity
        if len(entries) >= self._max_profiles:
            entries.sort(key=lambda e: e.last_matched_at)
            entries = entries[-(self._max_profiles - 1):]

        entries.append(entry)
        self._save_entries(entries)
        logger.info("Stored generated profile '%s' (%d total)", entry.profile.id, len(entries))

    def record_domain_feedback(
        self,
        profile_id: str,
        domain_tags: List[str],
        mismatch_signal: bool,
    ) -> None:
        """Record domain feedback from debate outcomes.

        When a debate reveals the current specialization was a poor fit,
        this stores a mismatch signal so future profile matching can
        penalize profiles that don't align with the task's domain.

        Args:
            profile_id: ID of the profile that was a poor fit.
            domain_tags: Domain keywords detected in the debate synthesis.
            mismatch_signal: True if this is a negative signal (mismatch).
        """
        feedback = self._load_domain_feedback()

        entry = feedback.get(profile_id, {"mismatches": [], "total_signals": 0})
        entry["total_signals"] = entry.get("total_signals", 0) + 1

        if mismatch_signal:
            mismatches = entry.get("mismatches", [])
            mismatches.append({
                "domain_tags": domain_tags,
                "timestamp": time.time(),
            })
            # Cap stored mismatches to avoid unbounded growth
            if len(mismatches) > 50:
                mismatches = mismatches[-50:]
            entry["mismatches"] = mismatches

        feedback[profile_id] = entry
        self._save_domain_feedback(feedback)

        logger.debug(
            "Recorded domain feedback for profile '%s': mismatch=%s, domains=%s",
            profile_id, mismatch_signal, domain_tags,
        )

    def get_domain_corrections(self, profile_id: str) -> Dict:
        """Get accumulated domain feedback for a profile.

        Returns dict with 'mismatches' list and 'total_signals' count.
        Returns empty dict if no feedback exists.
        """
        feedback = self._load_domain_feedback()
        return feedback.get(profile_id, {})

    def _load_domain_feedback(self) -> Dict:
        """Load domain feedback from disk."""
        feedback_path = self._store_path.parent / "domain_feedback.json"
        try:
            raw = feedback_path.read_text()
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass
        return {}

    def _save_domain_feedback(self, feedback: Dict) -> None:
        """Atomically persist domain feedback to disk."""
        from ..utils.atomic_io import atomic_write_json

        feedback_path = self._store_path.parent / "domain_feedback.json"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(feedback, indent=2)
        atomic_write_json(feedback_path, content)

    def _load_entries(self) -> List[GeneratedProfileEntry]:
        """Load entries from disk. Returns empty list on any I/O or parse error."""
        try:
            raw = self._store_path.read_text()
            data = json.loads(raw)
            return [_dict_to_entry(d) for d in data]
        except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.debug("Could not load profile registry: %s", exc)
            return []

    def _save_entries(self, entries: List[GeneratedProfileEntry]) -> None:
        """Atomically persist entries to disk."""
        from ..utils.atomic_io import atomic_write_json

        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps([_entry_to_dict(e) for e in entries], indent=2)
        atomic_write_json(self._store_path, content)
