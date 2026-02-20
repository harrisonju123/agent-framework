"""JSONL-backed store for per-model success/failure outcomes.

Tracks (repo_slug, model_tier, task_type) tuples so the intelligent
router can use historical success rates as a routing signal.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Key type: (repo_slug, model_tier, task_type)
_StoreKey = Tuple[str, str, str]


@dataclass
class _Outcome:
    """Aggregated outcome for a single (repo, model, task_type) combination."""
    successes: int = 0
    failures: int = 0
    total_cost: float = 0.0

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successes / self.total

    @property
    def avg_cost(self) -> float:
        if self.total == 0:
            return 0.0
        return self.total_cost / self.total


class ModelSuccessStore:
    """JSONL-backed store keyed by (repo_slug, model_tier, task_type).

    Loaded into memory at construction and updated incrementally.
    New outcomes are appended atomically to the JSONL file.
    """

    def __init__(self, workspace: Path, enabled: bool = True):
        self._enabled = enabled
        self._path = workspace / ".agent-communication" / "metrics" / "model_success.jsonl"
        self._data: Dict[_StoreKey, _Outcome] = {}
        if enabled:
            self._load()

    def _load(self) -> None:
        """Load existing JSONL records into memory."""
        if not self._path.exists():
            return
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        key = (record["repo"], record["model"], record["task_type"])
                        outcome = self._data.setdefault(key, _Outcome())
                        if record.get("success"):
                            outcome.successes += 1
                        else:
                            outcome.failures += 1
                        outcome.total_cost += record.get("cost", 0.0)
                    except (json.JSONDecodeError, KeyError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to load model success store: {e}")

    def record_outcome(
        self,
        repo_slug: str,
        model_tier: str,
        task_type: str,
        success: bool,
        cost: float = 0.0,
    ) -> None:
        """Record a task outcome and append to JSONL file."""
        if not self._enabled:
            return

        # Normalize model tier to family name
        model_tier = normalize_model_tier(model_tier)
        task_type = task_type.lower().replace("-", "_")

        key = (repo_slug, model_tier, task_type)
        outcome = self._data.setdefault(key, _Outcome())
        if success:
            outcome.successes += 1
        else:
            outcome.failures += 1
        outcome.total_cost += cost

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "repo": repo_slug,
            "model": model_tier,
            "task_type": task_type,
            "success": success,
            "cost": round(cost, 6),
        }
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
        except Exception as e:
            logger.warning(f"Failed to append to model success store: {e}")

    def get_success_rate(
        self,
        repo_slug: str,
        model_tier: str,
        task_type: str,
    ) -> Optional[float]:
        """Return success rate for (repo, model, task_type), or None if no data."""
        model_tier = normalize_model_tier(model_tier)
        task_type = task_type.lower().replace("-", "_")

        key = (repo_slug, model_tier, task_type)
        outcome = self._data.get(key)
        if outcome is None or outcome.total == 0:
            return None
        return outcome.success_rate

    def get_sample_count(
        self,
        repo_slug: str,
        model_tier: str,
        task_type: str,
    ) -> int:
        """Return total outcome count for (repo, model, task_type)."""
        model_tier = normalize_model_tier(model_tier)
        task_type = task_type.lower().replace("-", "_")

        key = (repo_slug, model_tier, task_type)
        outcome = self._data.get(key)
        return outcome.total if outcome else 0

    @property
    def enabled(self) -> bool:
        return self._enabled


def normalize_model_tier(model: str) -> str:
    """Normalize model identifier to tier name (haiku/sonnet/opus)."""
    lower = model.lower()
    if "haiku" in lower:
        return "haiku"
    if "opus" in lower:
        return "opus"
    if "sonnet" in lower:
        return "sonnet"
    return lower


# Backward-compat alias for external importers
_normalize_model_tier = normalize_model_tier
