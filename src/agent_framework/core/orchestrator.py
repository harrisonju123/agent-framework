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
from ..safeguards.circuit_breaker import CircuitBreaker


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
        if env_vars:
            env.update(env_vars)

        try:
            # Spawn agent process
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent_framework.run_agent", agent_id],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=self.workspace,
                env=env,
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

        # Environment variables to pass to agents
        env_vars = {"AGENT_LOG_LEVEL": log_level}

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
