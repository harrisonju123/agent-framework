"""Watchdog for monitoring agent heartbeats and restarting dead agents."""

import asyncio
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from ..core.activity import ActivityManager, AgentActivity, AgentStatus
from ..utils.process_utils import kill_process_tree


logger = logging.getLogger(__name__)


class Watchdog:
    """
    Monitors agent heartbeats and restarts dead agents.

    Ported from scripts/watchdog.sh
    """

    def __init__(
        self,
        workspace: Path,
        heartbeat_timeout: int = 90,
        check_interval: int = 60,
    ):
        self.workspace = Path(workspace)
        self.comm_dir = self.workspace / ".agent-communication"
        self.heartbeat_dir = self.comm_dir / "heartbeats"
        self.pid_file = self.comm_dir / "pids.txt"
        self.queue_dir = self.comm_dir / "queues"
        self.lock_dir = self.comm_dir / "locks"

        self.heartbeat_timeout = heartbeat_timeout
        self.check_interval = check_interval
        self._running = False

        # Activity tracking
        self.activity_manager = ActivityManager(workspace)

    async def run(self) -> None:
        """Main watchdog monitoring loop."""
        logger.info("Watchdog starting")
        self._running = True

        while self._running:
            try:
                # Check all agents
                agent_status = await self.check_all_agents()

                for agent_id, is_healthy in agent_status.items():
                    if not is_healthy:
                        logger.warning(f"Agent {agent_id} is dead or unresponsive")
                        await self.handle_dead_agent(agent_id)

                # Sleep until next check
                await asyncio.sleep(self.check_interval)

            except Exception as e:
                logger.exception(f"Error in watchdog loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def stop(self) -> None:
        """Stop the watchdog loop."""
        logger.info("Watchdog stopping")
        self._running = False

    async def check_heartbeat(self, agent_id: str) -> bool:
        """
        Check if an agent's heartbeat is fresh.

        Args:
            agent_id: Agent identifier

        Returns:
            True if agent is healthy, False if dead
        """
        heartbeat_file = self.heartbeat_dir / agent_id

        if not heartbeat_file.exists():
            logger.warning(f"No heartbeat file for {agent_id}")
            return False

        try:
            # Read heartbeat timestamp
            last_heartbeat = int(heartbeat_file.read_text().strip())
            current_time = int(time.time())
            age = current_time - last_heartbeat

            if age > self.heartbeat_timeout:
                logger.warning(
                    f"Agent {agent_id} heartbeat is stale: {age}s old "
                    f"(timeout: {self.heartbeat_timeout}s)"
                )
                return False

            return True

        except (ValueError, OSError) as e:
            logger.error(f"Error reading heartbeat for {agent_id}: {e}")
            return False

    async def check_all_agents(self) -> Dict[str, bool]:
        """
        Check heartbeats for all agents.

        Returns:
            Dict mapping agent_id to health status (True=healthy, False=dead)
        """
        pids = self._load_pids()
        agent_status = {}

        for agent_id in pids.keys():
            if agent_id == "watchdog":
                continue  # Don't monitor watchdog itself
            agent_status[agent_id] = await self.check_heartbeat(agent_id)

        return agent_status

    async def restart_agent(self, agent_id: str) -> None:
        """
        Restart a dead agent.

        Args:
            agent_id: Agent identifier
        """
        logger.info(f"Restarting agent {agent_id}")

        # Kill process if it exists, using graceful shutdown first
        pids = self._load_pids()
        if agent_id in pids:
            pid = pids[agent_id]
            if self._is_running(pid):
                logger.warning(f"Stopping process for {agent_id} (PID {pid})")
                try:
                    # Send SIGTERM to entire process group for graceful shutdown
                    kill_process_tree(pid, signal.SIGTERM)
                    # Wait up to 5 seconds for graceful exit
                    for _ in range(10):
                        await asyncio.sleep(0.5)
                        if not self._is_running(pid):
                            break
                    else:
                        # Force kill entire group if still running
                        logger.warning(f"Force killing process for {agent_id} (PID {pid})")
                        kill_process_tree(pid, signal.SIGKILL)
                        await asyncio.sleep(0.5)
                except ProcessLookupError:
                    pass

        # Reset in-progress tasks AFTER process is dead to prevent corruption
        await self.reset_in_progress_tasks(agent_id)

        # Spawn new agent process in its own session for clean group kills
        log_file_path = self.workspace / "logs" / f"{agent_id}.log"
        log_file = open(log_file_path, "a")  # Append to existing log

        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "agent_framework.run_agent", agent_id],
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

        logger.info(f"Restarted agent {agent_id} with new PID {proc.pid}")

        # Update PID file
        self._update_pid(agent_id, proc.pid)

    async def reset_in_progress_tasks(self, agent_id: str) -> None:
        """Recover orphaned in_progress tasks for a dead agent using smart 3-tier detection.

        Args:
            agent_id: Agent identifier
        """
        from ..queue.file_queue import FileQueue

        queue = FileQueue(self.workspace)
        result = queue.recover_orphaned_tasks(queue_ids=[agent_id])

        auto_completed = len(result["auto_completed"])
        reset = len(result["reset_to_pending"])
        total = auto_completed + reset
        if total > 0:
            logger.info(
                f"Recovery for {agent_id}: {auto_completed} auto-completed, "
                f"{reset} reset to pending"
            )

    async def handle_dead_agent(self, agent_id: str) -> None:
        """
        Handle a dead or unresponsive agent.

        Args:
            agent_id: Agent identifier
        """
        logger.warning(f"Handling dead agent: {agent_id}")

        # Mark agent as dead in activity
        self.activity_manager.update_activity(AgentActivity(
            agent_id=agent_id,
            status=AgentStatus.DEAD,
            last_updated=datetime.now(timezone.utc)
        ))

        # Reset tasks and restart
        await self.restart_agent(agent_id)

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
            except ValueError:
                continue

        return pids

    def _update_pid(self, agent_id: str, new_pid: int) -> None:
        """Update PID for an agent in the PID file."""
        pids = self._load_pids()
        pids[agent_id] = new_pid

        # Write back
        lines = [f"{aid}:{pid}\n" for aid, pid in pids.items()]
        self.pid_file.write_text("".join(lines))

    def _is_running(self, pid: int) -> bool:
        """Check if process is running."""
        try:
            os.kill(pid, 0)  # Signal 0 = check existence
            return True
        except ProcessLookupError:
            return False
