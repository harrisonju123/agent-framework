"""Entry point for running a single agent as a subprocess."""

import asyncio
import logging
import sys
from pathlib import Path

from .core.agent import Agent, AgentConfig
from .core.config import load_agents, load_config
from .llm.claude_cli_backend import ClaudeCLIBackend
from .queue.file_queue import FileQueue


def setup_logging(agent_id: str, workspace: Path):
    """Setup logging for the agent."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s - {agent_id} - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point for agent subprocess."""
    if len(sys.argv) < 2:
        print("Usage: python -m agent_framework.run_agent <agent_id>")
        sys.exit(1)

    agent_id = sys.argv[1]
    workspace = Path.cwd()

    # Setup logging
    setup_logging(agent_id, workspace)
    logger = logging.getLogger(__name__)

    try:
        # Load framework config
        framework_config = load_config(workspace / "agent-framework.yaml")

        # Load agent definitions
        agents_config_path = workspace / "config" / "agents.yaml"
        agents = load_agents(agents_config_path)

        # Find this agent's config
        agent_def = next((a for a in agents if a.id == agent_id), None)
        if not agent_def:
            logger.error(f"Agent {agent_id} not found in config")
            sys.exit(1)

        # Create agent config
        agent_config = AgentConfig(
            id=agent_def.id,
            name=agent_def.name,
            queue=agent_def.queue,
            prompt=agent_def.prompt,
            poll_interval=framework_config.task.poll_interval,
            max_retries=framework_config.task.max_retries,
            timeout=framework_config.task.timeout,
        )

        # Create LLM backend (Claude CLI mode only for now)
        llm = ClaudeCLIBackend(
            executable=framework_config.llm.claude_cli_executable,
            max_turns=framework_config.llm.claude_cli_max_turns,
            cheap_model=framework_config.llm.claude_cli_cheap_model,
            default_model=framework_config.llm.claude_cli_default_model,
            premium_model=framework_config.llm.claude_cli_premium_model,
        )

        # Create queue
        queue = FileQueue(
            workspace=framework_config.workspace,
            backoff_initial=framework_config.task.backoff_initial,
            backoff_max=framework_config.task.backoff_max,
            backoff_multiplier=framework_config.task.backoff_multiplier,
        )

        # Create and run agent
        agent = Agent(agent_config, llm, queue, framework_config.workspace)

        logger.info(f"Starting agent {agent_id}")
        asyncio.run(agent.run())

    except KeyboardInterrupt:
        logger.info("Agent interrupted, shutting down")
    except Exception as e:
        logger.exception(f"Agent crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
