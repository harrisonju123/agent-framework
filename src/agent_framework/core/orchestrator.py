"""Orchestrator for spawning and managing multiple agents."""

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_agents, AgentDefinition


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates multiple agent processes.

    Ported from scripts/start-async-agents.sh and scripts/stop-async-agents.sh
    """

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.comm_dir = self.workspace / ".agent-communication"
        self.pid_file = self.comm_dir / "pids.txt"
        self.lock_dir = self.comm_dir / "locks"
        self.logs_dir = self.workspace / "logs"

        self.processes: Dict[str, subprocess.Popen] = {}
        self._running = False

        # Ensure directories exist
        self.comm_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def spawn_agent(self, agent_id: str) -> subprocess.Popen:
        """
        Spawn a single agent as a subprocess.

        Args:
            agent_id: Agent identifier (e.g., "engineer", "qa")

        Returns:
            subprocess.Popen object
        """
        log_file_path = self.logs_dir / f"{agent_id}.log"
        log_file = open(log_file_path, "w")

        # Spawn agent process
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent_framework.run_agent", agent_id],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=self.workspace,
        )

        logger.info(f"Spawned agent {agent_id} with PID {proc.pid}")
        self.processes[agent_id] = proc
        return proc

    def spawn_all_agents(self, config_path: Optional[Path] = None) -> Dict[str, subprocess.Popen]:
        """
        Spawn all enabled agents from config.

        Args:
            config_path: Path to agents.yaml (default: config/agents.yaml)

        Returns:
            Dict mapping agent_id to Popen object
        """
        if config_path is None:
            config_path = self.workspace / "config" / "agents.yaml"

        # Load agent configs
        agents = load_agents(config_path)
        enabled_agents = [a for a in agents if a.enabled]

        logger.info(f"Starting {len(enabled_agents)} agents")

        # Clean up stale PIDs
        self._clean_stale_pids()

        # Spawn each agent with staggered startup
        for agent_def in enabled_agents:
            self.spawn_agent(agent_def.id)
            time.sleep(0.5)  # 0.5s delay between spawns

        # Verify agents started
        time.sleep(2)
        for agent_id, proc in self.processes.items():
            if proc.poll() is not None:
                logger.error(f"Agent {agent_id} failed to start (exit code: {proc.poll()})")
            else:
                logger.info(f"Agent {agent_id} running (PID: {proc.pid})")

        # Save PIDs
        self._save_pids()

        self._running = True
        return self.processes

    def spawn_watchdog(self) -> subprocess.Popen:
        """Spawn watchdog process."""
        log_file_path = self.logs_dir / "watchdog.log"
        log_file = open(log_file_path, "w")

        proc = subprocess.Popen(
            [sys.executable, "-m", "agent_framework.run_watchdog"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=self.workspace,
        )

        logger.info(f"Spawned watchdog with PID {proc.pid}")
        self.processes["watchdog"] = proc

        # Update PID file
        self._save_pids()

        return proc

    def stop_agent(self, agent_id: str, graceful: bool = True, timeout: int = 5) -> None:
        """
        Stop a single agent.

        Args:
            agent_id: Agent identifier
            graceful: Use SIGTERM first (True) or SIGKILL immediately (False)
            timeout: Seconds to wait after SIGTERM before SIGKILL
        """
        proc = self.processes.get(agent_id)
        if not proc:
            logger.warning(f"Agent {agent_id} not found in process list")
            return

        pid = proc.pid

        if graceful:
            # Try SIGTERM first
            logger.info(f"Sending SIGTERM to {agent_id} (PID {pid})")
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                logger.warning(f"Process {pid} not found")
                return

            # Wait for process to exit
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not self._is_running(pid):
                    logger.info(f"Agent {agent_id} stopped gracefully")
                    del self.processes[agent_id]
                    return
                time.sleep(0.5)

            # Still running, escalate to SIGKILL
            logger.warning(f"Agent {agent_id} didn't stop after {timeout}s, sending SIGKILL")

        # Force kill
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info(f"Sent SIGKILL to {agent_id} (PID {pid})")
        except ProcessLookupError:
            pass

        del self.processes[agent_id]

    def stop_all_agents(self, graceful: bool = True) -> None:
        """
        Stop all running agents.

        Args:
            graceful: Use graceful shutdown (SIGTERM -> SIGKILL)
        """
        logger.info("Stopping all agents")

        # Send SIGTERM to all agents
        if graceful:
            for agent_id, proc in list(self.processes.items()):
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                    logger.info(f"Sent SIGTERM to {agent_id} (PID {proc.pid})")
                except ProcessLookupError:
                    pass

            # Wait 2 seconds
            time.sleep(2)

            # Check which are still running
            still_running = []
            for agent_id, proc in list(self.processes.items()):
                if self._is_running(proc.pid):
                    still_running.append((agent_id, proc.pid))
                else:
                    logger.info(f"Agent {agent_id} stopped")
                    del self.processes[agent_id]

            # Force kill remaining
            if still_running:
                time.sleep(3)
                for agent_id, pid in still_running:
                    if self._is_running(pid):
                        logger.warning(f"Force killing {agent_id} (PID {pid})")
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        if agent_id in self.processes:
                            del self.processes[agent_id]
        else:
            # Kill immediately
            for agent_id, proc in list(self.processes.items()):
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                del self.processes[agent_id]

        # Clean up
        self._cleanup()

    def get_running_agents(self) -> List[str]:
        """Get list of currently running agent IDs."""
        running = []
        for agent_id, proc in self.processes.items():
            if self._is_running(proc.pid):
                running.append(agent_id)
        return running

    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down")
            self.stop_all_agents(graceful=True)
            sys.exit(0)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)

    def _is_running(self, pid: int) -> bool:
        """Check if process is running."""
        try:
            os.kill(pid, 0)  # Signal 0 = check existence
            return True
        except ProcessLookupError:
            return False

    def _save_pids(self) -> None:
        """Save process IDs to file."""
        lines = []
        for agent_id, proc in self.processes.items():
            lines.append(f"{agent_id}:{proc.pid}\n")

        self.pid_file.write_text("".join(lines))
        logger.debug(f"Saved PIDs to {self.pid_file}")

    def _load_pids(self) -> Dict[str, int]:
        """Load PIDs from file."""
        if not self.pid_file.exists():
            return {}

        pids = {}
        for line in self.pid_file.read_text().strip().split("\n"):
            if not line:
                continue
            agent_id, pid_str = line.split(":", 1)
            pids[agent_id] = int(pid_str)

        return pids

    def _clean_stale_pids(self) -> None:
        """Remove stale PIDs from file."""
        pids = self._load_pids()
        if not pids:
            return

        valid_pids = {}
        for agent_id, pid in pids.items():
            if self._is_running(pid):
                logger.warning(
                    f"Agent {agent_id} already running with PID {pid}, not starting"
                )
                valid_pids[agent_id] = pid
            else:
                logger.info(f"Removed stale PID for {agent_id}: {pid}")

        # Write back only valid PIDs
        if valid_pids:
            lines = [f"{aid}:{pid}\n" for aid, pid in valid_pids.items()]
            self.pid_file.write_text("".join(lines))
        else:
            if self.pid_file.exists():
                self.pid_file.unlink()

    def _cleanup(self) -> None:
        """Clean up after shutdown."""
        # Remove lock files
        if self.lock_dir.exists():
            for lock_dir in self.lock_dir.glob("*.lock"):
                if lock_dir.is_dir():
                    for f in lock_dir.iterdir():
                        f.unlink()
                    lock_dir.rmdir()

        # Remove PID file
        if self.pid_file.exists():
            self.pid_file.unlink()

        # Reset in_progress tasks to pending
        self._reset_in_progress_tasks()

        logger.info("Cleanup complete")

    def _reset_in_progress_tasks(self) -> None:
        """Reset all in_progress tasks to pending."""
        import json
        from ..core.task import Task, TaskStatus

        queue_dir = self.comm_dir / "queues"
        if not queue_dir.exists():
            return

        reset_count = 0
        for agent_dir in queue_dir.iterdir():
            if not agent_dir.is_dir():
                continue

            for task_file in agent_dir.glob("*.json"):
                try:
                    task_data = json.loads(task_file.read_text())
                    if task_data.get("status") == "in_progress":
                        task = Task(**task_data)
                        task.reset_to_pending()
                        task_file.write_text(task.model_dump_json(indent=2))
                        reset_count += 1
                except Exception as e:
                    logger.error(f"Error resetting task {task_file}: {e}")

        if reset_count > 0:
            logger.info(f"Reset {reset_count} in_progress tasks to pending")
