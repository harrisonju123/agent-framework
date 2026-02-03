"""Entry point for running a single agent as a subprocess."""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Load .env file into environment before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on shell environment

from .core.agent import Agent, AgentConfig
from .core.config import load_agents, load_config, load_jira_config, load_github_config
from .llm.claude_cli_backend import ClaudeCLIBackend
from .queue.file_queue import FileQueue
from .workspace.multi_repo_manager import MultiRepoManager
from .workspace.worktree_manager import WorktreeManager


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
        framework_config = load_config(workspace / "config" / "agent-framework.yaml")

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
        mcp_config_path = None
        if framework_config.llm.use_mcp and framework_config.llm.mcp_config_path:
            mcp_path = Path(framework_config.llm.mcp_config_path)
            if not mcp_path.exists():
                logger.error(f"MCP config file not found: {mcp_path}")
                raise FileNotFoundError(f"MCP config file not found: {mcp_path}")
            mcp_config_path = str(mcp_path)
            logger.info(f"MCP enabled, config path: {mcp_config_path}")

        llm = ClaudeCLIBackend(
            executable=framework_config.llm.claude_cli_executable,
            max_turns=framework_config.llm.claude_cli_max_turns,
            cheap_model=framework_config.llm.claude_cli_cheap_model,
            default_model=framework_config.llm.claude_cli_default_model,
            premium_model=framework_config.llm.claude_cli_premium_model,
            mcp_config_path=mcp_config_path,
        )

        # Create queue
        queue = FileQueue(
            workspace=framework_config.workspace,
            backoff_initial=framework_config.task.backoff_initial,
            backoff_max=framework_config.task.backoff_max,
            backoff_multiplier=framework_config.task.backoff_multiplier,
        )

        # Create MultiRepoManager if GITHUB_TOKEN is available
        github_token = os.environ.get("GITHUB_TOKEN")
        multi_repo_manager = None
        worktree_manager = None

        if github_token:
            multi_repo_manager = MultiRepoManager(
                workspace_root=framework_config.multi_repo.workspace_root,
                github_token=github_token
            )
            logger.info("MultiRepoManager initialized")

            # Create WorktreeManager if worktree mode is enabled
            worktree_config = framework_config.multi_repo.worktree
            if worktree_config.enabled:
                wt_config = worktree_config.to_manager_config()
                worktree_manager = WorktreeManager(
                    config=wt_config,
                    github_token=github_token,
                )
                logger.info(f"WorktreeManager initialized (root: {wt_config.root})")

                # Run startup cleanup of orphaned worktrees
                cleanup_result = worktree_manager.cleanup_orphaned_worktrees()
                if cleanup_result["total"]:
                    logger.info(
                        f"Cleaned up {cleanup_result['total']} worktrees on startup "
                        f"(registered: {cleanup_result['registered']}, unregistered: {cleanup_result['unregistered']})"
                    )
        else:
            logger.warning("No GITHUB_TOKEN, multi-repo features disabled")

        # Load JIRA and GitHub configs for MCP prompt guidance
        jira_config = load_jira_config(workspace / "config" / "jira.yaml")
        github_config = load_github_config(workspace / "config" / "github.yaml")

        # Extract optimization config from framework config
        optimization_config = framework_config.optimization.model_dump() if framework_config.optimization else {}

        # Create and run agent
        agent = Agent(
            agent_config,
            llm,
            queue,
            framework_config.workspace,
            multi_repo_manager=multi_repo_manager,
            jira_config=jira_config,
            github_config=github_config,
            mcp_enabled=framework_config.llm.use_mcp,
            optimization_config=optimization_config,
            worktree_manager=worktree_manager,
        )

        logger.info(f"Starting agent {agent_id}")
        asyncio.run(agent.run())

    except KeyboardInterrupt:
        logger.info("Agent interrupted, shutting down")
    except Exception as e:
        logger.exception(f"Agent crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
