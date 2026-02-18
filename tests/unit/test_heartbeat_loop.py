"""Tests for background heartbeat loop in Agent."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from agent_framework.core.agent import Agent


@pytest.fixture
def agent(tmp_path):
    """Minimal Agent mock with real heartbeat methods bound."""
    a = MagicMock()
    a._running = True
    a._heartbeat_interval = 0.05  # 50ms for fast tests
    a._heartbeat_task = None
    a.heartbeat_file = tmp_path / "heartbeat"
    a.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

    # Bind real methods
    a._write_heartbeat = Agent._write_heartbeat.__get__(a)
    a._heartbeat_loop = Agent._heartbeat_loop.__get__(a)
    return a


class TestHeartbeatLoop:
    """Tests for Agent._heartbeat_loop()."""

    @pytest.mark.asyncio
    async def test_writes_heartbeat_periodically(self, agent):
        """Background loop writes heartbeat file multiple times."""
        task = asyncio.create_task(agent._heartbeat_loop())
        await asyncio.sleep(0.15)  # ~3 intervals
        agent._running = False
        await asyncio.sleep(0.1)  # Let loop exit
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert agent.heartbeat_file.exists()
        ts = int(agent.heartbeat_file.read_text())
        assert abs(ts - int(time.time())) < 2

    @pytest.mark.asyncio
    async def test_stops_when_running_false(self, agent):
        """Loop exits immediately when _running is already False."""
        agent._running = False
        # Should return without blocking since _running is False from the start
        await asyncio.wait_for(agent._heartbeat_loop(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_cancel_stops_loop(self, agent):
        """Cancelling the task stops the loop."""
        task = asyncio.create_task(agent._heartbeat_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_survives_write_oserror(self, agent):
        """Transient OSError from _write_heartbeat doesn't kill the loop."""
        call_count = 0
        original_write = agent._write_heartbeat

        def flaky_write():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise OSError("disk full")
            original_write()

        agent._write_heartbeat = flaky_write

        task = asyncio.create_task(agent._heartbeat_loop())
        await asyncio.sleep(0.2)  # ~4 intervals — first 2 fail, rest succeed
        agent._running = False
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert call_count >= 3
        assert agent.heartbeat_file.exists()


class TestStopCancelsHeartbeat:
    """Tests that Agent.stop() cancels the background heartbeat task."""

    @pytest.mark.asyncio
    async def test_stop_cancels_heartbeat_task(self, tmp_path):
        """stop() cancels the heartbeat task and awaits it."""
        a = MagicMock()
        a._running = True
        a._heartbeat_interval = 0.05
        a.heartbeat_file = tmp_path / "heartbeat"
        a.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        a.config = MagicMock(id="test-agent")
        a.logger = MagicMock()
        a.llm = MagicMock()
        a._current_task_id = None

        # Bind real methods
        a._write_heartbeat = Agent._write_heartbeat.__get__(a)
        a._heartbeat_loop = Agent._heartbeat_loop.__get__(a)
        a.stop = Agent.stop.__get__(a)

        # Start loop
        a._heartbeat_task = asyncio.create_task(a._heartbeat_loop())
        await asyncio.sleep(0.05)

        # Stop should cancel it
        await a.stop()

        assert a._heartbeat_task.done()
        assert a._running is False
