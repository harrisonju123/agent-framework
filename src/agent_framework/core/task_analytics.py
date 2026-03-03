"""Task analytics manager.

Handles post-task analytics: memory extraction, tool pattern analysis,
summary extraction, and optimization metrics recording.
Extracted from agent.py to reduce its size.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .config import AgentConfig
    from .session_logger import SessionLogger
    from ..llm.base import LLMBackend
    from ..utils.rich_logging import ContextLogger
    from ..memory.memory_retriever import MemoryRetriever
    from ..memory.tool_pattern_store import ToolPatternStore
    from .feedback_bus import FeedbackBus

from .task import Task
from ..llm.base import LLMRequest
from ..utils.type_helpers import get_type_str


# Constants for optimization strategies
SUMMARY_CONTEXT_MAX_CHARS = 2000
SUMMARY_MAX_LENGTH = 500
_CONVERSATIONAL_PREFIXES = ("i ", "i'", "thank", "sure", "certainly", "of course", "let me")


class TaskAnalyticsManager:
    """Manages post-task analytics: memories, tool patterns, summaries, metrics."""

    def __init__(
        self,
        config: "AgentConfig",
        logger: "ContextLogger",
        session_logger: "SessionLogger",
        llm: "LLMBackend",
        memory_retriever: "MemoryRetriever",
        tool_pattern_store: "ToolPatternStore",
        optimization_config: dict,
        memory_enabled: bool,
        tool_tips_enabled: bool,
        session_logging_enabled: bool,
        session_logs_dir: Path,
        workspace: Path,
        feedback_bus: Optional["FeedbackBus"] = None,
        code_index_query=None,
    ):
        self.config = config
        self.logger = logger
        self.session_logger = session_logger
        self.llm = llm
        self.memory_retriever = memory_retriever
        self.tool_pattern_store = tool_pattern_store
        self.optimization_config = optimization_config
        self.memory_enabled = memory_enabled
        self.tool_tips_enabled = tool_tips_enabled
        self.session_logging_enabled = session_logging_enabled
        self.session_logs_dir = session_logs_dir
        self.workspace = workspace
        self.feedback_bus = feedback_bus
        self.code_index_query = code_index_query

    def set_session_logger(self, session_logger: "SessionLogger") -> None:
        """Update session logger for new task."""
        self.session_logger = session_logger

    @staticmethod
    def get_repo_slug(task: Task) -> Optional[str]:
        """Extract repo slug from task context."""
        return task.context.get("github_repo")

    def extract_and_store_memories(self, task: Task, response) -> None:
        """Extract learnings from successful response and store as memories."""
        if not self.memory_enabled:
            return

        repo_slug = self.get_repo_slug(task)
        if not repo_slug:
            return

        count = self.memory_retriever.extract_memories_from_response(
            response_content=response.content,
            repo_slug=repo_slug,
            agent_type=self.config.base_id,
            task_id=task.id,
        )
        if count > 0:
            self.logger.info(f"Extracted {count} memories from task {task.id}")
            self.session_logger.log(
                "memory_store",
                repo=repo_slug,
                count=count,
            )

        # Cross-feature learning: route all post-task learnings through FeedbackBus
        if self.feedback_bus:
            try:
                self.feedback_bus.process(task, repo_slug)
            except Exception as e:
                self.logger.debug(f"Feedback bus error (non-fatal): {e}")

    def analyze_tool_patterns(self, task: Task) -> Optional[int]:
        """Run post-task analysis on session log to detect inefficient tool usage.

        Returns total tool call count (or None if analysis was skipped).
        """
        if not self.session_logging_enabled:
            return None

        session_path = self.session_logs_dir / "sessions" / f"{task.id}.jsonl"
        if not session_path.exists():
            return None

        tool_call_count = None
        tool_calls = None
        try:
            from ..memory.tool_pattern_analyzer import ToolPatternAnalyzer, compute_tool_usage_stats

            analyzer = ToolPatternAnalyzer()

            # Reuse stats from compute_tool_stats_for_chain if available
            cached = task.context.pop("_tool_stats_cache", None)
            if cached:
                tool_call_count = cached.get("total_calls")
                self.session_logger.log("tool_usage_stats", agent_id=self.config.id, **cached)
            else:
                tool_calls = analyzer.extract_tool_calls(session_path)
                if tool_calls:
                    stats = compute_tool_usage_stats(tool_calls)
                    tool_call_count = stats.total_calls
                    self.session_logger.log(
                        "tool_usage_stats",
                        agent_id=self.config.id,
                        total_calls=stats.total_calls,
                        tool_distribution=stats.tool_distribution,
                        duplicate_reads=stats.duplicate_reads,
                        read_before_write_ratio=stats.read_before_write_ratio,
                        edit_write_count=stats.edit_write_count,
                        exploration_count=stats.exploration_count,
                        edit_density=stats.edit_density,
                        files_read=stats.files_read,
                        files_written=stats.files_written,
                    )

            # Language mismatch detection
            repo_slug = self.get_repo_slug(task)
            if repo_slug and self.code_index_query:
                try:
                    from ..memory.tool_pattern_analyzer import detect_language_mismatches
                    from dataclasses import asdict
                    project_language = self.code_index_query.get_project_language(repo_slug)
                    if project_language:
                        if tool_calls is None:
                            tool_calls = analyzer.extract_tool_calls(session_path)
                        mismatches = detect_language_mismatches(tool_calls, project_language)
                        if mismatches:
                            self.session_logger.log(
                                "language_mismatch",
                                agent_id=self.config.id,
                                project_language=project_language,
                                repo=repo_slug,
                                mismatch_count=len(mismatches),
                                mismatches=[asdict(m) for m in mismatches],
                            )
                except Exception as e:
                    self.logger.debug(f"Language mismatch detection failed (non-fatal): {e}")

            # Qualitative anti-pattern detection
            if self.tool_tips_enabled:
                if not repo_slug:
                    repo_slug = self.get_repo_slug(task)
                if repo_slug:
                    recommendations = analyzer.analyze_session(session_path)

                    # Check session log for cross-step re-read waste
                    reread_rec = self._check_reread_waste(session_path, analyzer)
                    if reread_rec:
                        recommendations.append(reread_rec)

                    if recommendations:
                        count = self.tool_pattern_store.store_patterns(repo_slug, recommendations)
                        self.logger.debug(f"Stored {count} tool pattern recommendations")
                        self.session_logger.log(
                            "tool_patterns_analyzed",
                            repo=repo_slug,
                            patterns_found=len(recommendations),
                            patterns_stored=count,
                        )
        except Exception as e:
            self.logger.debug(f"Tool pattern analysis failed (non-fatal): {e}")

        return tool_call_count

    @staticmethod
    def _check_reread_waste(session_path, analyzer):
        """Scan session log for read_cache_bypass events with high wasteful rate.

        Returns a ToolPatternRecommendation or None.
        """
        try:
            text = session_path.read_text()
        except OSError:
            return None

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if entry.get("event") != "read_cache_bypass":
                continue
            wasteful = entry.get("wasteful_rereads", 0)
            cached = entry.get("cached_files", 0)
            rec = analyzer.compute_reread_recommendation(wasteful, cached)
            if rec:
                return rec
        return None

    async def extract_summary(self, response_content: str, task: Task, _recursion_depth: int = 0) -> str:
        """Extract key outcomes from agent response.

        Uses two-tier extraction:
        1. Regex patterns (free, fast) - extracts JIRA keys, PR URLs, file paths
        2. Haiku fallback (cheap) - only if regex insufficient
        """
        if not response_content or not response_content.strip():
            return f"Task {get_type_str(task.type)} completed (no output)"

        if _recursion_depth > 0:
            self.logger.debug("Recursion depth exceeded in summary extraction, using fallback")
            return f"Task {get_type_str(task.type)} completed"

        # Try regex extraction first
        extracted = []

        jira_keys = [
            m for m in re.findall(r'\b([A-Z]{2,5}-\d{1,6})\b', response_content)
            if not re.match(r'^(?:HTTP|UTF|ISO|RFC|TCP|UDP|SSH|SSL|TLS|DNS|API|URL|URI|XML|CSV|PDF)-', m)
        ]
        if jira_keys:
            jira_keys = list(set(jira_keys))[:10]
            extracted.append(f"Created/Updated: {', '.join(jira_keys)}")

        pr_urls = re.findall(r'github\.com/[^/]+/[^/]+/pull/(\d+)', response_content)
        if pr_urls:
            pr_urls = list(set(pr_urls))[:5]
            extracted.append(f"PRs: {', '.join(pr_urls)}")

        file_paths = re.findall(
            r'\b(?:src|lib|app|tests?|pkg|internal|cmd)/[^\s:;,]+\.(?:py|ts|tsx|js|jsx|go|rb|java|kt)',
            response_content
        )
        if file_paths:
            file_paths = list(set(file_paths))[:5]
            extracted.append(f"Modified: {', '.join(file_paths)}")

        if len(extracted) >= 2:
            return " | ".join(extracted)

        # Fall back to Haiku for summarization
        if self.optimization_config.get("enable_result_summarization", False):
            summary_prompt = (
                "Extract the key outcomes from the following agent output. "
                "Return exactly 3 bullet points, each on its own line starting with '- '. "
                "Focus on: what was done, what files/PRs were created, and the final status.\n\n"
                "<agent_output>\n"
                f"{response_content[:SUMMARY_CONTEXT_MAX_CHARS]}\n"
                "</agent_output>"
            )
            try:
                summary_response = await self.llm.complete(LLMRequest(
                    prompt=summary_prompt,
                    system_prompt="You are a summarization tool. Output only bullet points. Never converse, ask questions, or add commentary.",
                    model="haiku",
                    temperature=0.0,
                    max_tokens=512,
                ))

                if summary_response.success and summary_response.content:
                    content = summary_response.content.strip()
                    if content.lower().startswith(_CONVERSATIONAL_PREFIXES):
                        self.logger.warning("Haiku returned conversational response, using fallback")
                    else:
                        return content[:SUMMARY_MAX_LENGTH]
                else:
                    self.logger.warning(f"Haiku summary failed: {summary_response.error}")
            except Exception as e:
                self.logger.warning(f"Failed to extract summary with Haiku: {e}")

        return extracted[0] if extracted else f"Task {get_type_str(task.type)} completed"

    def record_optimization_metrics(
        self,
        task: Task,
        legacy_prompt_length: int,
        optimized_prompt_length: int,
        *,
        should_use_optimization_cb=None,
        get_active_optimizations_cb=None,
    ) -> None:
        """Record optimization metrics for post-deployment analysis."""
        try:
            canary_active = should_use_optimization_cb(task) if should_use_optimization_cb else False
            optimizations = get_active_optimizations_cb() if get_active_optimizations_cb else {}

            metrics = {
                "task_id": task.id,
                "task_type": get_type_str(task.type),
                "agent_id": self.config.id,
                "legacy_prompt_chars": legacy_prompt_length,
                "optimized_prompt_chars": optimized_prompt_length,
                "savings_chars": legacy_prompt_length - optimized_prompt_length,
                "savings_percent": (
                    (legacy_prompt_length - optimized_prompt_length) / legacy_prompt_length * 100
                    if legacy_prompt_length > 0 else 0
                ),
                "canary_active": canary_active,
                "optimizations_enabled": optimizations,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            metrics_dir = self.workspace / ".agent-communication" / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            metrics_file = metrics_dir / "optimization.jsonl"

            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
        except PermissionError as e:
            self.logger.warning(f"Permission denied recording optimization metrics: {e}")
        except OSError as e:
            self.logger.warning(f"Failed to record optimization metrics (disk full?): {e}")
        except Exception as e:
            self.logger.debug(f"Unexpected error recording optimization metrics: {e}")
