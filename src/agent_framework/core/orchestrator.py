"""Orchestrator for spawning and managing multiple agents."""

import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import load_agents, AgentDefinition
from ..safeguards.circuit_breaker import CircuitBreaker
from ..utils.process_utils import kill_process_tree


logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates multiple agent processes.

    Ported from scripts/start-async-agents.sh and scripts/stop-async-agents.sh
    """

    def __init__(
        self,
        workspace: Path,
        enable_circuit_breaker: bool = True,
        check_health_interval: int = 300,  # Check every 5 minutes
    ):
        self.workspace = Path(workspace)
        self.comm_dir = self.workspace / ".agent-communication"
        self.pid_file = self.comm_dir / "pids.txt"
        self.dashboard_pid_file = self.comm_dir / "dashboard.pid"
        self.lock_dir = self.comm_dir / "locks"
        self.logs_dir = self.workspace / "logs"

        self.processes: Dict[str, subprocess.Popen] = {}
        self._running = False

        # Circuit breaker integration
        self.enable_circuit_breaker = enable_circuit_breaker
        self.check_health_interval = check_health_interval
        self.last_health_check = 0.0
        self.circuit_breaker = CircuitBreaker(workspace) if enable_circuit_breaker else None
        self.health_degraded = False
        self.original_replica_count = 1

        # Ensure directories exist
        self.comm_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def spawn_agent(self, agent_id: str, env_vars: dict = None) -> subprocess.Popen:
        """
        Spawn a single agent as a subprocess.

        Args:
            agent_id: Agent identifier (e.g., "engineer", "qa")
            env_vars: Optional environment variables to pass to agent

        Returns:
            subprocess.Popen object
        """
        log_file_path = self.logs_dir / f"{agent_id}.log"
        log_file = open(log_file_path, "w")

        # Merge environment variables
        env = os.environ.copy()
        # Strip CLAUDECODE so agent subprocesses (and their claude CLI children)
        # don't trigger the "nested session" guard
        env.pop("CLAUDECODE", None)
        if env_vars:
            env.update(env_vars)

        try:
            # Spawn agent process in its own session so all child processes
            # (e.g. claude CLI) inherit the group and can be killed together
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent_framework.run_agent", agent_id],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=self.workspace,
                env=env,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise

        # Close file handle after subprocess inherits it (prevents FD exhaustion)
        log_file.close()

        logger.info(f"Spawned agent {agent_id} with PID {proc.pid}")
        self.processes[agent_id] = proc
        return proc

    def spawn_all_agents(
        self,
        config_path: Optional[Path] = None,
        replicas: int = 1,
        log_level: str = "INFO"
    ) -> Dict[str, subprocess.Popen]:
        """
        Spawn all enabled agents from config.

        Args:
            config_path: Path to agents.yaml (default: config/agents.yaml)
            replicas: Number of replicas per agent (default: 1)
            log_level: Logging level for agents (DEBUG, INFO, WARNING, ERROR)

        Returns:
            Dict mapping agent_id to Popen object
        """
        # Clean up stale locks and orphaned tasks from crashed agents
        self._cleanup(remove_pid_file=False)

        if config_path is None:
            config_path = self.workspace / "config" / "agents.yaml"

        # Store original replica count for health management
        self.original_replica_count = replicas

        # Check system health before spawning agents
        if self.enable_circuit_breaker:
            health_report = self.check_system_health()
            if not health_report.passed:
                logger.warning("System health checks failed, spawning with reduced capacity")
                logger.warning(str(health_report))
                # Reduce replicas if health is degraded
                replicas = max(1, replicas // 2)
                self.health_degraded = True

        # Load agent configs
        agents = load_agents(config_path)
        enabled_agents = [a for a in agents if a.enabled]

        total_to_spawn = len(enabled_agents) * replicas
        logger.info(f"Starting {total_to_spawn} agents ({len(enabled_agents)} types × {replicas} replicas)")

        # Clean up stale PIDs
        self._clean_stale_pids()

        # MCP-related environment variables to pass to agents
        mcp_env_vars = [
            "GITHUB_TOKEN",
            "JIRA_URL",
            "JIRA_EMAIL",
            "JIRA_API_TOKEN",
            "JIRA_SERVER",
        ]

        # Environment variables to pass to agents
        env_vars = {"AGENT_LOG_LEVEL": log_level}

        # Add MCP-related env vars if they exist
        for var in mcp_env_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]

        # Spawn each agent with replicas
        for agent_def in enabled_agents:
            for replica_num in range(1, replicas + 1):
                if replicas == 1:
                    agent_id = agent_def.id
                else:
                    agent_id = f"{agent_def.id}-{replica_num}"

                self.spawn_agent(agent_id, env_vars=env_vars)
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

        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent_framework.run_watchdog"],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=self.workspace,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise

        # Close file handle after subprocess inherits it (prevents FD exhaustion)
        log_file.close()

        logger.info(f"Spawned watchdog with PID {proc.pid}")
        self.processes["watchdog"] = proc

        # Update PID file
        self._save_pids()

        return proc

    def spawn_dashboard(self, port: int = 8080) -> Optional[subprocess.Popen]:
        """Spawn dashboard as a background subprocess.

        Stores PID separately from agent PIDs so dashboard shutdown
        doesn't trigger agent task cleanup (_reset_in_progress_tasks).
        """
        if self.is_port_in_use(port):
            existing = self._load_dashboard_pid()
            if existing and self._is_running(existing["pid"]):
                logger.info(f"Dashboard already running on port {port} (PID {existing['pid']})")
                return None
            logger.warning(f"Port {port} in use but no dashboard PID found")
            return None

        log_file_path = self.logs_dir / "dashboard.log"
        log_file = open(log_file_path, "w")

        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent_framework.run_dashboard", str(port)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=self.workspace,
                start_new_session=True,
            )
        except Exception:
            log_file.close()
            raise

        log_file.close()

        logger.info(f"Spawned dashboard with PID {proc.pid} on port {port}")
        self._save_dashboard_pid(proc.pid, port)

        return proc

    def stop_dashboard(self, timeout: int = 5) -> None:
        """Stop the dashboard subprocess."""
        info = self._load_dashboard_pid()
        if not info:
            logger.debug("No dashboard PID file found")
            return

        pid = info["pid"]
        if not self._is_running(pid):
            logger.info("Dashboard process already stopped")
            self._remove_dashboard_pid()
            return

        if not self._is_dashboard_process(pid):
            logger.warning(f"PID {pid} is not a dashboard process (stale or recycled)")
            self._remove_dashboard_pid()
            return

        logger.info(f"Stopping dashboard (PID {pid})")
        kill_process_tree(pid, signal.SIGTERM)

        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._is_running(pid):
                logger.info("Dashboard stopped gracefully")
                self._remove_dashboard_pid()
                return
            time.sleep(0.5)

        logger.warning(f"Dashboard didn't stop after {timeout}s, sending SIGKILL")
        kill_process_tree(pid, signal.SIGKILL)
        self._remove_dashboard_pid()

    def get_dashboard_info(self) -> Optional[Dict]:
        """Get info about a running dashboard.

        Returns dict with 'pid' and 'port' keys, or None if not running.
        """
        info = self._load_dashboard_pid()
        if not info:
            return None

        if not self._is_running(info["pid"]):
            self._remove_dashboard_pid()
            return None

        if not self._is_dashboard_process(info["pid"]):
            self._remove_dashboard_pid()
            return None

        return info

    def _is_dashboard_process(self, pid: int) -> bool:
        """Check if PID belongs to a dashboard process."""
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True, text=True, timeout=5,
            )
            return "agent_framework.run_dashboard" in result.stdout
        except Exception:
            return False

    def _save_dashboard_pid(self, pid: int, port: int) -> None:
        """Save dashboard PID and port to file."""
        self.dashboard_pid_file.write_text(f"{pid}:{port}\n")

    def _load_dashboard_pid(self) -> Optional[Dict]:
        """Load dashboard PID and port from file."""
        if not self.dashboard_pid_file.exists():
            return None
        try:
            content = self.dashboard_pid_file.read_text().strip()
            pid_str, port_str = content.split(":", 1)
            return {"pid": int(pid_str), "port": int(port_str)}
        except (ValueError, TypeError):
            logger.warning(f"Malformed dashboard PID file: {self.dashboard_pid_file}")
            return None

    def _remove_dashboard_pid(self) -> None:
        """Remove dashboard PID file."""
        if self.dashboard_pid_file.exists():
            self.dashboard_pid_file.unlink()

    def is_port_in_use(self, port: int) -> bool:
        """Check if a TCP port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", port)) == 0

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
            # Send SIGTERM to entire process group
            logger.info(f"Sending SIGTERM to {agent_id} (PID {pid})")
            kill_process_tree(pid, signal.SIGTERM)

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

        # Force kill entire process group
        kill_process_tree(pid, signal.SIGKILL)
        logger.info(f"Sent SIGKILL to {agent_id} (PID {pid})")

        del self.processes[agent_id]

    def stop_all_agents(self, graceful: bool = True, timeout: int = 5) -> None:
        """
        Stop all running agents.

        Args:
            graceful: Use graceful shutdown (SIGTERM -> SIGKILL)
            timeout: Seconds to wait after SIGTERM before escalating to SIGKILL
        """
        logger.info("Stopping all agents")

        # If called from a separate process (e.g. `agent stop`),
        # self.processes is empty — recover PIDs from the pid file.
        pids_to_kill: Dict[str, int] = {}
        for agent_id, proc in self.processes.items():
            pids_to_kill[agent_id] = proc.pid

        if not pids_to_kill:
            loaded = self._load_pids()
            # Guard against recycled PIDs when loading from file
            for agent_id, pid in loaded.items():
                if self._is_agent_process(pid):
                    pids_to_kill[agent_id] = pid
                else:
                    logger.warning(f"Skipping {agent_id} PID {pid} — not an agent process (stale or recycled)")
            if pids_to_kill:
                logger.info(f"Loaded {len(pids_to_kill)} PIDs from pid file")

        if not pids_to_kill:
            logger.info("No agent processes to stop")
            self._cleanup()
            return

        if graceful:
            # Send SIGTERM to all agent process groups
            for agent_id, pid in pids_to_kill.items():
                kill_process_tree(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to {agent_id} (PID {pid})")

            # Wait for graceful shutdown
            time.sleep(min(timeout, 2))

            # Force kill any still running
            still_running = {}
            for aid, pid in pids_to_kill.items():
                if self._is_running(pid):
                    still_running[aid] = pid
                else:
                    logger.info(f"Agent {aid} stopped gracefully")
            if still_running:
                remaining = max(0, timeout - 2)
                if remaining:
                    time.sleep(remaining)
                for agent_id, pid in still_running.items():
                    if self._is_running(pid):
                        logger.warning(f"Force killing {agent_id} (PID {pid})")
                        kill_process_tree(pid, signal.SIGKILL)
        else:
            for agent_id, pid in pids_to_kill.items():
                kill_process_tree(pid, signal.SIGKILL)

        self.processes.clear()

        # Clean up
        self._cleanup()

    def get_running_agents(self) -> List[str]:
        """Get list of currently running agent IDs.

        Falls back to the PID file when self.processes is empty,
        which happens when called from a separate CLI invocation.
        """
        running = []
        for agent_id, proc in self.processes.items():
            if self._is_running(proc.pid):
                running.append(agent_id)

        if not running:
            pids = self._load_pids()
            for agent_id, pid in pids.items():
                if self._is_running(pid) and self._is_agent_process(pid):
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
        except (ProcessLookupError, PermissionError):
            return False

    def _is_agent_process(self, pid: int) -> bool:
        """Check if PID belongs to an agent framework process (not a recycled PID)."""
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "args="],
                capture_output=True, text=True, timeout=5,
            )
            return "agent_framework." in result.stdout
        except Exception:
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
            try:
                agent_id, pid_str = line.split(":", 1)
                pids[agent_id] = int(pid_str)
            except (ValueError, TypeError):
                logger.warning(f"Skipping malformed PID line: {line!r}")

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

    def _cleanup(self, remove_pid_file: bool = True) -> None:
        """Clean up after shutdown.

        Args:
            remove_pid_file: Whether to remove the PID file (skip during pre-start cleanup)
        """
        # Remove lock files (each lock is a directory with a pid file inside)
        if self.lock_dir.exists():
            for lock_dir in self.lock_dir.glob("*.lock"):
                try:
                    if lock_dir.is_dir():
                        for f in lock_dir.iterdir():
                            f.unlink()
                        lock_dir.rmdir()
                except OSError as e:
                    logger.warning(f"Failed to remove lock {lock_dir.name}: {e}")

        # Remove PID file
        if remove_pid_file and self.pid_file.exists():
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

    def check_system_health(self):
        """Check system health using circuit breaker.

        Returns:
            CircuitBreakerReport with health status
        """
        if not self.circuit_breaker:
            # Circuit breaker disabled, return passing report
            from ..safeguards.circuit_breaker import CircuitBreakerReport
            report = CircuitBreakerReport()
            report.add_check("circuit_breaker", True, "Circuit breaker disabled")
            return report

        # Run all circuit breaker checks
        report = self.circuit_breaker.run_all_checks()

        # Update last health check time
        self.last_health_check = time.time()

        # Log health status
        if not report.passed:
            logger.warning("System health degraded:")
            logger.warning(str(report))
        else:
            logger.info("System health check passed")

        return report

    def should_check_health(self) -> bool:
        """Check if it's time to run health checks.

        Returns:
            True if health check interval has elapsed
        """
        if not self.enable_circuit_breaker:
            return False

        elapsed = time.time() - self.last_health_check
        return elapsed >= self.check_health_interval

    def handle_health_degradation(self, config_path: Optional[Path] = None):
        """Handle system health degradation by reducing agent replicas.

        Args:
            config_path: Path to agents.yaml
        """
        if self.health_degraded:
            # Already in degraded mode
            return

        logger.warning("System health degraded, reducing agent replicas")

        # Stop half of the agents to reduce load
        agents_to_stop = []
        for agent_id in list(self.processes.keys()):
            # Keep at least one replica of each agent type
            if "-" in agent_id:  # Replica (e.g., "engineer-2")
                base_id, replica_num = agent_id.rsplit("-", 1)
                if int(replica_num) > 1:
                    agents_to_stop.append(agent_id)

        for agent_id in agents_to_stop[:len(agents_to_stop) // 2]:
            logger.info(f"Stopping {agent_id} due to health degradation")
            self.stop_agent(agent_id, graceful=True)

        self.health_degraded = True

    def handle_critical_health(self):
        """Handle critical health issues by pausing task intake.

        This creates a marker file that agents can check to pause polling.
        """
        logger.error("System health CRITICAL - pausing task intake")

        # Create pause marker file
        pause_marker = self.comm_dir / "PAUSE_INTAKE"
        pause_marker.write_text(f"System paused due to critical health issues at {time.time()}")

        self.health_degraded = True

    def resume_from_health_degradation(self, config_path: Optional[Path] = None, replicas: int = None):
        """Resume normal operations after health recovers.

        Args:
            config_path: Path to agents.yaml
            replicas: Number of replicas to restore (default: original count)
        """
        if not self.health_degraded:
            return

        logger.info("System health recovered, resuming normal operations")

        # Remove pause marker if present
        pause_marker = self.comm_dir / "PAUSE_INTAKE"
        if pause_marker.exists():
            pause_marker.unlink()

        # Restore original replica count if we reduced it
        if replicas is None:
            replicas = self.original_replica_count

        current_count = len(self.processes)
        if config_path is None:
            config_path = self.workspace / "config" / "agents.yaml"

        agents = load_agents(config_path)
        enabled_agents = [a for a in agents if a.enabled]
        target_count = len(enabled_agents) * replicas

        if current_count < target_count:
            logger.info(f"Spawning additional agents to reach target: {current_count} -> {target_count}")
            # Spawn additional replicas
            for agent_def in enabled_agents:
                current_replicas = len([p for p in self.processes.keys() if p.startswith(agent_def.id)])
                needed_replicas = replicas - current_replicas

                for i in range(needed_replicas):
                    replica_num = current_replicas + i + 1
                    agent_id = f"{agent_def.id}-{replica_num}"
                    self.spawn_agent(agent_id)
                    time.sleep(0.5)

        self.health_degraded = False
