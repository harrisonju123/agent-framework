"""Read cache manager — cross-step file read deduplication.

Tracks which files were read during each workflow step so downstream agents
can skip re-reading them. Maintains both task-scoped and repo-scoped caches
for persistence across retries and independent tasks.

Cache path: `.agent-communication/read-cache/{root_task_id}.json`
Repo cache:  `.agent-communication/read-cache/_repo-{slug}.json`
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .session_logger import SessionLogger
    from .prompt_builder import PromptBuilder

from .task import Task

_MAX_REPO_CACHE_ENTRIES = 200


def _repo_cache_slug(github_repo: str) -> str:
    """Convert 'owner/repo' to 'owner-repo' for cache file naming."""
    return github_repo.replace("/", "-")


def _to_relative_path(file_path: str, working_dir: Optional[Path]) -> str:
    """Strip worktree prefix for cache portability across chain steps."""
    if not working_dir or not file_path.startswith("/"):
        return file_path
    prefix = str(working_dir).rstrip("/") + "/"
    if file_path.startswith(prefix):
        return file_path[len(prefix):]
    return file_path


class ReadCacheManager:
    """Manages the shared read cache for cross-step file read dedup."""

    def __init__(
        self,
        workspace: Path,
        session_logger: "SessionLogger",
        logger: logging.Logger,
        config_base_id: str,
        prompt_builder: "PromptBuilder",
    ):
        self.workspace = workspace
        self._session_logger = session_logger
        self.logger = logger
        self._config_base_id = config_base_id
        self._prompt_builder = prompt_builder

    def set_session_logger(self, session_logger: "SessionLogger") -> None:
        """Update the session logger (called per-task)."""
        self._session_logger = session_logger

    def populate_read_cache(self, task: Task, working_dir: Optional[Path] = None) -> list[str]:
        """Populate shared read cache with files read during this session.

        Appends to .agent-communication/read-cache/{root_task_id}.json so
        downstream chain steps can skip re-reading the same files.
        Cache keys are repo-relative so they match across worktrees.
        Non-fatal — workflow continues even if this fails.

        Returns the raw file_reads list so callers can reuse it without
        re-parsing the session log.
        """
        try:
            from ..utils.atomic_io import atomic_write_text
            from ..utils.file_summarizer import summarize_file

            file_reads = self._session_logger.extract_file_reads()
            if not file_reads:
                return file_reads

            root_task_id = task.root_id
            cache_dir = self.workspace / ".agent-communication" / "read-cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"{root_task_id}.json"

            # Load existing cache (may have entries from MCP tool calls with summaries)
            existing: dict = {}
            try:
                existing = json.loads(cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}

            entries = existing.get("entries", {})
            pre_existing_keys = set(entries.keys())
            step = task.context.get("workflow_step", "unknown")
            now_iso = datetime.now(timezone.utc).isoformat()

            # Detect files modified in this step so stale summaries get refreshed
            from .chain_state import _collect_files_modified
            modified_in_step = set(_collect_files_modified(working_dir))

            added = 0
            refreshed = 0
            read_keys: set[str] = set()
            for file_path in file_reads:
                # Store repo-relative key for cross-worktree portability
                cache_key = _to_relative_path(file_path, working_dir)
                read_keys.add(cache_key)
                is_modified = cache_key in modified_in_step
                if cache_key not in entries or is_modified:
                    # Preserve original reader lineage when refreshing modified entries
                    old_entry = entries.get(cache_key, {})
                    entry = {
                        "summary": summarize_file(file_path),
                        "read_by": old_entry.get("read_by", self._config_base_id),
                        "read_at": old_entry.get("read_at", now_iso),
                        "workflow_step": old_entry.get("workflow_step", step),
                    }
                    if is_modified:
                        entry["modified_by"] = self._config_base_id
                        entry["modified_at"] = now_iso
                        refreshed += 1
                    else:
                        # New entry -- attribute to current agent
                        entry["read_by"] = self._config_base_id
                        entry["read_at"] = now_iso
                        entry["workflow_step"] = step
                        added += 1
                    entries[cache_key] = entry

            # Measure cache bypass rate at the storage layer (complements
            # measure_cache_effectiveness which measures from prompt-injected paths)
            if pre_existing_keys:
                re_read = read_keys & pre_existing_keys
                # Partition: re-reads of modified files are justified, others are wasteful
                modified_keys = {k for k, v in entries.items() if v.get("modified_by")}
                justified = sorted(re_read & modified_keys)
                wasteful = sorted(re_read - modified_keys)
                wasteful_rate = len(wasteful) / len(pre_existing_keys)
                self._session_logger.log(
                    "read_cache_bypass",
                    cached_files=len(pre_existing_keys),
                    re_read_count=len(re_read),
                    bypass_rate=round(len(re_read) / len(pre_existing_keys), 3),
                    justified_rereads=len(justified),
                    wasteful_rereads=len(wasteful),
                    wasteful_rate=round(wasteful_rate, 3),
                    new_reads=added,
                    total_reads=len(file_reads),
                    re_read_files=sorted(re_read)[:20],
                )

            if added + refreshed == 0:
                return file_reads

            cache_data = {
                "root_task_id": root_task_id,
                "entries": entries,
            }
            atomic_write_text(cache_file, json.dumps(cache_data))
            self.logger.debug(
                f"Read cache: added {added}, refreshed {refreshed} "
                f"({len(entries)} total) for {root_task_id}"
            )

            # Merge into repo-scoped cache for cross-attempt persistence
            github_repo = task.context.get("github_repo")
            if github_repo:
                self._update_repo_cache(cache_dir, github_repo, entries)
            return file_reads
        except Exception as e:
            self.logger.warning(f"Failed to populate read cache for {task.id}: {e}")
            return []

    def measure_cache_effectiveness(
        self, task: Task, file_reads: list[str], *, working_dir: Optional[Path] = None,
    ) -> None:
        """Log how well the read cache prevented redundant file reads.

        Compares the set of paths injected into the prompt (from previous steps)
        against the files actually read during this session. Non-fatal.
        """
        try:
            injected = self._prompt_builder.injected_cache_paths
            if not injected:
                return

            session_paths = {_to_relative_path(p, working_dir) for p in file_reads}

            cache_hits = injected - session_paths   # cached and NOT re-read
            cache_misses = injected & session_paths  # cached but re-read anyway
            new_reads = session_paths - injected     # not cached, first time

            # Load cache to identify modified files for justified/wasteful split
            modified_keys: set[str] = set()
            root_task_id = task.root_id
            cache_file = self.workspace / ".agent-communication" / "read-cache" / f"{root_task_id}.json"
            try:
                cache_data = json.loads(cache_file.read_text(encoding="utf-8"))
                modified_keys = {
                    k for k, v in cache_data.get("entries", {}).items()
                    if v.get("modified_by")
                }
            except (json.JSONDecodeError, OSError):
                pass

            justified = cache_misses & modified_keys   # re-read because file changed
            wasteful = cache_misses - modified_keys    # avoidable re-reads

            total_cached = len(injected)
            hit_rate = len(cache_hits) / total_cached if total_cached else 0.0
            wasteful_rate = len(wasteful) / total_cached if total_cached else 0.0

            self._session_logger.log(
                "read_cache_effectiveness",
                total_cached=total_cached,
                cache_hits=len(cache_hits),
                cache_misses=len(cache_misses),
                justified_rereads=len(justified),
                wasteful_rereads=len(wasteful),
                wasteful_rate=round(wasteful_rate, 3),
                new_reads=len(new_reads),
                hit_rate=round(hit_rate, 3),
                missed_files=sorted(cache_misses)[:20],
                new_files=sorted(new_reads)[:20],
            )
            self.logger.debug(
                f"Read cache effectiveness: {len(cache_hits)}/{total_cached} hits "
                f"({hit_rate:.0%}), {len(cache_misses)} re-reads "
                f"({len(justified)} justified, {len(wasteful)} wasteful), "
                f"{len(new_reads)} new"
            )
        except Exception as e:
            self.logger.debug(f"Cache effectiveness measurement failed (non-fatal): {e}")

    def _update_repo_cache(self, cache_dir: Path, github_repo: str, entries: dict) -> None:
        """Merge entries into repo-scoped cache for cross-attempt persistence.

        Accumulates file-read knowledge across independent task attempts on the
        same repo. New tasks can seed from this when no task-specific cache exists.
        Non-fatal — exceptions are logged and swallowed.
        """
        try:
            from ..utils.atomic_io import atomic_write_text

            slug = _repo_cache_slug(github_repo)
            repo_cache_file = cache_dir / f"_repo-{slug}.json"

            existing: dict = {}
            try:
                existing = json.loads(repo_cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}

            repo_entries = existing.get("entries", {})

            # Merge: newer wins by read_at timestamp
            for path, entry in entries.items():
                if path not in repo_entries:
                    repo_entries[path] = entry
                else:
                    existing_at = repo_entries[path].get("read_at", "")
                    new_at = entry.get("read_at", "")
                    if new_at > existing_at:
                        repo_entries[path] = entry

            # Evict oldest entries if over limit
            if len(repo_entries) > _MAX_REPO_CACHE_ENTRIES:
                sorted_paths = sorted(
                    repo_entries.keys(),
                    key=lambda p: repo_entries[p].get("read_at", ""),
                )
                for path in sorted_paths[:len(repo_entries) - _MAX_REPO_CACHE_ENTRIES]:
                    del repo_entries[path]

            repo_data = {"github_repo": github_repo, "entries": repo_entries}
            atomic_write_text(repo_cache_file, json.dumps(repo_data))
            self.logger.debug(f"Repo cache: {len(repo_entries)} entries for {github_repo}")
        except Exception as e:
            self.logger.warning(f"Failed to update repo cache for {github_repo}: {e}")
