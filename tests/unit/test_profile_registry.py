"""Tests for the profile registry (generated profile cache)."""

import json
import time

import pytest
from pathlib import Path

from agent_framework.core.profile_registry import (
    ProfileRegistry,
    GeneratedProfileEntry,
    _entry_to_dict,
    _dict_to_entry,
    _jaccard,
    _extract_extensions,
    _tokenize,
)
from agent_framework.core.engineer_specialization import SpecializationProfile


def _make_profile(
    profile_id="data-pipeline",
    name="Data Pipeline Engineer",
    file_patterns=None,
) -> SpecializationProfile:
    return SpecializationProfile(
        id=profile_id,
        name=name,
        description="Test profile",
        file_patterns=file_patterns or ["**/*.py", "**/pipeline/**"],
        prompt_suffix="X" * 60,
        teammates={},
        tool_guidance="Use pytest",
    )


def _make_entry(
    profile_id="data-pipeline",
    tags=None,
    file_extensions=None,
    last_matched_at=None,
    match_count=1,
    file_patterns=None,
) -> GeneratedProfileEntry:
    return GeneratedProfileEntry(
        profile=_make_profile(profile_id, file_patterns=file_patterns),
        created_at=time.time(),
        last_matched_at=last_matched_at or time.time(),
        match_count=match_count,
        source_task_id="task-1",
        tags=tags or ["data", "pipeline", "etl"],
        file_extensions=file_extensions or [".py"],
    )


class TestHelpers:
    """Tests for utility functions."""

    def test_jaccard_identical(self):
        assert _jaccard({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        assert _jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)

    def test_jaccard_empty(self):
        assert _jaccard(set(), set()) == 0.0

    def test_extract_extensions(self):
        exts = _extract_extensions(["src/main.py", "lib/util.go", "README.md"])
        assert exts == {".py", ".go", ".md"}

    def test_extract_extensions_empty(self):
        assert _extract_extensions([]) == set()

    def test_tokenize(self):
        tokens = _tokenize("Add server for data pipeline processing")
        assert "server" in tokens
        assert "data" in tokens
        assert "pipeline" in tokens
        assert "processing" in tokens
        # Short words (<=3 chars) filtered out
        assert "for" not in tokens
        assert "add" not in tokens

    def test_tokenize_keeps_alphanumeric_domain_terms(self):
        """Domain terms like k8s should be kept (isalnum, not isalpha)."""
        tokens = _tokenize("Deploy k8s cluster with grpc services")
        assert "cluster" in tokens
        assert "grpc" in tokens
        assert "services" in tokens

    def test_tokenize_empty(self):
        assert _tokenize("") == set()


class TestSerialization:
    """Tests for entry serialization round-trip."""

    def test_roundtrip(self):
        entry = _make_entry()
        d = _entry_to_dict(entry)
        restored = _dict_to_entry(d)

        assert restored.profile.id == entry.profile.id
        assert restored.profile.name == entry.profile.name
        assert restored.profile.file_patterns == entry.profile.file_patterns
        assert restored.tags == entry.tags
        assert restored.file_extensions == entry.file_extensions
        assert restored.match_count == entry.match_count

    def test_roundtrip_json(self):
        """Full JSON serialization round-trip."""
        entry = _make_entry()
        json_str = json.dumps(_entry_to_dict(entry))
        restored = _dict_to_entry(json.loads(json_str))
        assert restored.profile.id == entry.profile.id


class TestProfileRegistry:
    """Tests for ProfileRegistry find/store operations."""

    def test_store_and_find(self, tmp_path):
        registry = ProfileRegistry(tmp_path)
        entry = _make_entry(
            file_patterns=["**/*.py", "**/pipeline/**"],
            tags=["data", "pipeline"],
            file_extensions=[".py"],
        )
        registry.store_profile(entry)

        # Files that match the stored profile's patterns
        result = registry.find_matching_profile(
            files=["src/pipeline/etl.py", "src/pipeline/loader.py"],
            task_description="Build data pipeline",
            min_score=0.2,
        )
        assert result is not None
        assert result.id == "data-pipeline"

    def test_find_returns_none_when_empty(self, tmp_path):
        registry = ProfileRegistry(tmp_path)
        result = registry.find_matching_profile(
            files=["src/main.py"],
            task_description="test",
        )
        assert result is None

    def test_find_returns_none_below_threshold(self, tmp_path):
        registry = ProfileRegistry(tmp_path)
        entry = _make_entry(
            file_patterns=["**/*.proto"],
            tags=["grpc", "protobuf"],
            file_extensions=[".proto"],
        )
        registry.store_profile(entry)

        # Completely unrelated files
        result = registry.find_matching_profile(
            files=["src/components/Button.tsx"],
            task_description="Build React component",
            min_score=0.4,
        )
        assert result is None

    def test_stale_eviction(self, tmp_path):
        registry = ProfileRegistry(tmp_path, staleness_days=1)

        # Store an entry that was last matched 2 days ago
        stale_entry = _make_entry(
            profile_id="stale-profile",
            last_matched_at=time.time() - 2 * 86400,
        )
        registry.store_profile(stale_entry)

        # Store a fresh entry
        fresh_entry = _make_entry(profile_id="fresh-profile")
        registry.store_profile(fresh_entry)

        # Reload and check — stale should be evicted
        entries = registry._load_entries()
        ids = [e.profile.id for e in entries]
        assert "fresh-profile" in ids
        assert "stale-profile" not in ids

    def test_capacity_eviction(self, tmp_path):
        registry = ProfileRegistry(tmp_path, max_profiles=3)

        # Store 3 entries
        for i in range(3):
            entry = _make_entry(
                profile_id=f"profile-{i}",
                last_matched_at=time.time() - (3 - i) * 100,
            )
            registry.store_profile(entry)

        # Store a 4th — oldest should be evicted
        new_entry = _make_entry(profile_id="profile-new")
        registry.store_profile(new_entry)

        entries = registry._load_entries()
        ids = [e.profile.id for e in entries]
        assert len(ids) <= 3
        assert "profile-new" in ids
        # profile-0 was oldest, should be evicted
        assert "profile-0" not in ids

    def test_graceful_on_corrupt_file(self, tmp_path):
        registry = ProfileRegistry(tmp_path)
        # Write garbage to the store path
        registry._store_path.parent.mkdir(parents=True, exist_ok=True)
        registry._store_path.write_text("NOT VALID JSON {{{")

        entries = registry._load_entries()
        assert entries == []

    def test_graceful_on_missing_file(self, tmp_path):
        registry = ProfileRegistry(tmp_path)
        entries = registry._load_entries()
        assert entries == []

    def test_find_updates_match_metadata(self, tmp_path):
        registry = ProfileRegistry(tmp_path)
        original_time = time.time() - 1000
        entry = _make_entry(
            file_patterns=["**/*.py", "**/pipeline/**"],
            tags=["data", "pipeline"],
            file_extensions=[".py"],
            last_matched_at=original_time,
            match_count=5,
        )
        registry.store_profile(entry)

        result = registry.find_matching_profile(
            files=["src/pipeline/etl.py", "src/pipeline/loader.py"],
            task_description="Build data pipeline",
            min_score=0.2,
        )
        assert result is not None

        # Reload and verify metadata was updated
        entries = registry._load_entries()
        assert entries[0].match_count == 6
        assert entries[0].last_matched_at > original_time

    def test_duplicate_id_replaces_existing(self, tmp_path):
        """Storing a profile with an existing ID replaces the old entry."""
        registry = ProfileRegistry(tmp_path)
        entry1 = _make_entry(profile_id="my-profile", match_count=5)
        registry.store_profile(entry1)

        entry2 = _make_entry(profile_id="my-profile", match_count=1)
        registry.store_profile(entry2)

        entries = registry._load_entries()
        matching = [e for e in entries if e.profile.id == "my-profile"]
        assert len(matching) == 1
        assert matching[0].match_count == 1

    def test_best_profile_selected(self, tmp_path):
        """When multiple profiles match, the highest-scoring one wins."""
        registry = ProfileRegistry(tmp_path)

        # Profile A: matches .py and pipeline
        entry_a = _make_entry(
            profile_id="pipeline",
            file_patterns=["**/*.py", "**/pipeline/**"],
            tags=["pipeline"],
            file_extensions=[".py"],
        )
        registry.store_profile(entry_a)

        # Profile B: matches .proto only
        entry_b = _make_entry(
            profile_id="grpc",
            file_patterns=["**/*.proto"],
            tags=["grpc"],
            file_extensions=[".proto"],
        )
        registry.store_profile(entry_b)

        result = registry.find_matching_profile(
            files=["src/pipeline/etl.py", "src/pipeline/loader.py"],
            task_description="Build data pipeline",
            min_score=0.2,
        )
        assert result is not None
        assert result.id == "pipeline"
