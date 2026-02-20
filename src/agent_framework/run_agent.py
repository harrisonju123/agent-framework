"""Entry point for running a single agent as a subprocess."""

import asyncio
import logging
import os
import re
import signal
import sys
from pathlib import Path

# Load .env file into environment before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on shell environment

# Strip CLAUDECODE early so no subprocess (LLM backend, profile generator, etc.)
# inherits it and triggers the "nested session" guard in Claude CLI.
os.environ.pop("CLAUDECODE", None)

from .core.agent import Agent, AgentConfig
from .core.config import load_agents, load_config, load_jira_config, load_github_config
from .llm.claude_cli_backend import ClaudeCLIBackend
from .queue.file_queue import FileQueue
from .workspace.multi_repo_manager import MultiRepoManager
from .workspace.worktree_manager import WorktreeManager
from .utils.rich_logging import setup_rich_logging


def _get_active_agent_ids(workspace: Path) -> set:
    """Read activity files and return agent IDs that are actively working.

    Used to protect worktrees from cleanup when their owning agent is still alive.
    """
    from .core.activity import ActivityManager, AgentStatus
    try:
        mgr = ActivityManager(workspace)
        return {
            a.agent_id
            for a in mgr.get_all_activities()
            if a.status not in (AgentStatus.IDLE, AgentStatus.DEAD)
        }
    except Exception:
        return set()


def setup_logging(agent_id: str, workspace: Path):
    """Setup logging for the agent."""
    # Use rich logging with better formatting
    return setup_rich_logging(
        agent_id=agent_id,
        workspace=workspace,
        log_level="INFO",
        use_file=True,
        use_json=False,
    )


def main():
    """Main entry point for agent subprocess."""
    if len(sys.argv) < 2:
        print("Usage: python -m agent_framework.run_agent <agent_id>")
        sys.exit(1)

    agent_id = sys.argv[1]
    workspace = Path.cwd()

    # Expose agent ID to child processes (MCP servers use this for task attribution)
    os.environ["AGENT_ID"] = agent_id

    # Setup logging
    setup_logging(agent_id, workspace)
    logger = logging.getLogger(__name__)

    try:
        # Load framework config
        framework_config = load_config(workspace / "config" / "agent-framework.yaml")

        # Load agent definitions
        agents_config_path = workspace / "config" / "agents.yaml"
        agents = load_agents(agents_config_path)

        # Support replica IDs like "engineer-2", "engineer-3"
        # Extract base agent ID by removing trailing -N suffix
        if '-' in agent_id:
            parts = agent_id.split('-')
            # Check if last part is a number (replica ID)
            if parts[-1].isdigit():
                base_agent_id = '-'.join(parts[:-1])
                replica_num = parts[-1]
            else:
                base_agent_id = agent_id
                replica_num = None
        else:
            base_agent_id = agent_id
            replica_num = None

        # Find this agent's config using base ID
        agent_def = next((a for a in agents if a.id == base_agent_id), None)
        if not agent_def:
            logger.error(f"Agent {base_agent_id} not found in config (requested: {agent_id})")
            sys.exit(1)

        # Create agent config (use actual agent_id for replicas)
        agent_name = f"{agent_def.name} #{replica_num}" if replica_num else agent_def.name
        agent_config = AgentConfig(
            id=agent_id,  # Use full replica ID (e.g., engineer-2)
            name=agent_name,
            queue=agent_def.queue,  # All replicas share same queue
            prompt=agent_def.prompt,
            poll_interval=framework_config.task.poll_interval,
            max_retries=framework_config.task.max_retries,
            timeout=framework_config.task.timeout,
            validate_tasks=framework_config.task.validate_tasks,
            validation_mode=framework_config.task.validation_mode,
        )

        # Create LLM backend based on configured mode
        if framework_config.llm.mode == "litellm":
            from .llm.litellm_backend import LiteLLMBackend
            llm = LiteLLMBackend(
                api_key=framework_config.llm.litellm_api_key,
                api_base=framework_config.llm.litellm_api_base,
                cheap_model=framework_config.llm.litellm_cheap_model,
                default_model=framework_config.llm.litellm_default_model,
                premium_model=framework_config.llm.litellm_premium_model,
                logs_dir=Path(framework_config.workspace) / "logs",
            )
            logger.info("Using LiteLLM direct backend")
        else:
            mcp_config_path = None
            if framework_config.llm.use_mcp and framework_config.llm.mcp_config_path:
                mcp_path = Path(framework_config.llm.mcp_config_path)
                # Resolve relative paths relative to workspace
                if not mcp_path.is_absolute():
                    mcp_path = workspace / mcp_path
                if not mcp_path.exists():
                    logger.error(f"MCP config file not found: {mcp_path}")
                    raise FileNotFoundError(f"MCP config file not found: {mcp_path}")
                mcp_config_path = str(mcp_path)
                logger.info(f"MCP enabled, config path: {mcp_config_path}")

            proxy_env = framework_config.llm.get_proxy_env()
            if proxy_env:
                logger.info(f"LLM proxy enabled: {framework_config.llm.proxy_url}")

            llm = ClaudeCLIBackend(
                executable=framework_config.llm.claude_cli_executable,
                max_turns=framework_config.llm.claude_cli_max_turns,
                timeout=framework_config.llm.claude_cli_timeout,
                timeout_large=framework_config.llm.claude_cli_timeout_large,
                timeout_bounded=framework_config.llm.claude_cli_timeout_bounded,
                timeout_simple=framework_config.llm.claude_cli_timeout_simple,
                cheap_model=framework_config.llm.claude_cli_cheap_model,
                default_model=framework_config.llm.claude_cli_default_model,
                premium_model=framework_config.llm.claude_cli_premium_model,
                mcp_config_path=mcp_config_path,
                logs_dir=Path(framework_config.workspace) / "logs",
                proxy_env=proxy_env,
                use_max_account=framework_config.llm.use_max_account,
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

                # Run startup cleanup of orphaned worktrees, protecting agents that are actively working
                protected_ids = _get_active_agent_ids(Path(framework_config.workspace))
                cleanup_result = worktree_manager.cleanup_orphaned_worktrees(
                    protected_agent_ids=protected_ids,
                )
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

        # Extract agentic flow configs (Sprint 1: memory, self-eval, replanning)
        memory_config = getattr(framework_config, "memory", None)
        if isinstance(memory_config, dict):
            pass  # already a dict from extra="allow"
        elif memory_config is not None:
            memory_config = memory_config if isinstance(memory_config, dict) else {}
        else:
            memory_config = {}

        self_eval_config = getattr(framework_config, "self_evaluation", None)
        if not isinstance(self_eval_config, dict):
            self_eval_config = {}

        replan_config = getattr(framework_config, "replanning", None)
        if not isinstance(replan_config, dict):
            replan_config = {}

        session_logging_config = getattr(framework_config, "session_logging", None)
        if not isinstance(session_logging_config, dict):
            session_logging_config = {}

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
            agents_config=agents,
            team_mode_enabled=framework_config.team_mode.enabled,
            team_mode_default_model=framework_config.llm.claude_cli_default_model,
            agent_definition=agent_def,
            workflows_config=framework_config.workflows,
            memory_config=memory_config,
            self_eval_config=self_eval_config,
            replan_config=replan_config,
            session_logging_config=session_logging_config,
            repositories_config=framework_config.repositories,
            pr_lifecycle_config=framework_config.pr_lifecycle.model_dump(),
            code_indexing_config=framework_config.indexing.model_dump(),
            heartbeat_interval=framework_config.safeguards.heartbeat_interval,
            max_consecutive_tool_calls=framework_config.safeguards.max_consecutive_tool_calls,
            max_consecutive_diagnostic_calls=framework_config.safeguards.max_consecutive_diagnostic_calls,
            exploration_alert_threshold=framework_config.optimization.exploration_alert_threshold,
            exploration_alert_thresholds=framework_config.optimization.exploration_alert_thresholds,
        )

        # Let SIGTERM trigger a clean exit through the polling loop
        # so the agent can release locks and reset in-progress tasks
        def _handle_sigterm(signum, frame):
            agent._running = False

        signal.signal(signal.SIGTERM, _handle_sigterm)

        logger.info(f"Starting agent {agent_id}")
        asyncio.run(agent.run())

    except KeyboardInterrupt:
        logger.info("Agent interrupted, shutting down")
    except Exception as e:
        logger.exception(f"Agent crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
