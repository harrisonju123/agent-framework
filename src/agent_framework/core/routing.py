"""Routing signal handling for tool-based agent handoffs.

Claude subprocesses write routing signals via MCP transfer tools.
The framework reads these signals after Claude exits and validates
them against safety rules before routing.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROUTING_SIGNALS_DIR = ".agent-communication/routing-signals"
ROUTING_METRICS_FILE = "metrics/routing.jsonl"

WORKFLOW_COMPLETE = "__complete__"


@dataclass(frozen=True)
class RoutingSignal:
    target_agent: str
    reason: str
    timestamp: str
    source_agent: str


def read_routing_signal(workspace: Path, task_id: str) -> Optional[RoutingSignal]:
    """Read and delete the routing signal file for a task. Returns None if no signal exists."""
    signal_path = workspace / ROUTING_SIGNALS_DIR / f"{task_id}.json"
    if not signal_path.exists():
        return None

    try:
        data = json.loads(signal_path.read_text())
        signal = RoutingSignal(
            target_agent=data["target_agent"],
            reason=data["reason"],
            timestamp=data["timestamp"],
            source_agent=data["source_agent"],
        )
        signal_path.unlink()
        return signal
    except (json.JSONDecodeError, KeyError, OSError) as e:
        logger.warning(f"Failed to read routing signal for {task_id}: {e}")
        try:
            signal_path.unlink()
        except OSError:
            pass
        return None


def validate_routing_signal(
    signal: RoutingSignal,
    current_agent: str,
    task_type: str,
    agents_config: list,
) -> Optional[str]:
    """Returns validated target agent name, or None to fall back to default chain."""
    target = signal.target_agent

    if target == current_agent:
        logger.warning(f"Routing signal rejected: {current_agent} tried to route to itself")
        return None

    if target == WORKFLOW_COMPLETE:
        return WORKFLOW_COMPLETE

    known_agents = {a.id for a in agents_config} if agents_config else set()
    if known_agents and target not in known_agents:
        logger.warning(f"Routing signal rejected: unknown target '{target}'")
        return None

    if task_type == "escalation":
        logger.warning(f"Routing signal rejected: escalation tasks cannot be re-routed")
        return None

    return target


def log_routing_decision(
    workspace: Path,
    task_id: str,
    agent_id: str,
    signal: Optional[RoutingSignal],
    validated_target: Optional[str],
    used_fallback: bool,
) -> None:
    """Append routing decision to metrics/routing.jsonl."""
    metrics_path = workspace / ROUTING_METRICS_FILE
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": task_id,
        "agent_id": agent_id,
        "signal_target": signal.target_agent if signal else None,
        "signal_reason": signal.reason if signal else None,
        "validated_target": validated_target,
        "used_fallback": used_fallback,
    }

    try:
        with open(metrics_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        logger.debug(f"Failed to write routing metrics: {e}")
