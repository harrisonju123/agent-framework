"""FastAPI web server for agent dashboard."""

import asyncio
import logging
import re
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from ..core.orchestrator import Orchestrator
from ..queue.file_queue import FileQueue
from .data_provider import DashboardDataProvider
from .models import (
    ActiveTaskData,
    AgentData,
    QueueStats,
    EventData,
    FailedTaskData,
    HealthReport,
    DashboardState,
    SuccessResponse,
    AgentActionResponse,
    TaskActionResponse,
    CreateTaskRequest,
    WorkRequest,
    AnalyzeRequest,
    RunTicketRequest,
    OperationResponse,
    LogEntry,
    JIRAValidationRequest,
    JIRAValidationResponse,
    GitHubValidationRequest,
    GitHubValidationResponse,
    SetupConfiguration,
    SetupStatusResponse,
    TeamSessionData,
    AgenticInsightsData,
)

logger = logging.getLogger(__name__)


def create_app(workspace: Path) -> FastAPI:
    """Create FastAPI application with all routes."""
    app = FastAPI(
        title="Agent Dashboard",
        description="Web dashboard for agent framework",
        version="1.0.0",
    )

    # CORS for development (Vite runs on different port)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize data provider and orchestrator
    data_provider = DashboardDataProvider(workspace)
    orchestrator = Orchestrator(workspace)
    start_time = datetime.now(timezone.utc)

    # Load and cache framework config
    from ..core.config import load_config
    framework_config = load_config(workspace / "config" / "agent-framework.yaml")

    # Store in app state
    app.state.data_provider = data_provider
    app.state.orchestrator = orchestrator
    app.state.start_time = start_time
    app.state.workspace = workspace
    app.state.framework_config = framework_config

    # Register routes
    register_routes(app)

    return app


def _validate_agent_id(agent_id: str) -> None:
    """Validate agent_id contains only safe characters."""
    if not re.match(r'^[a-zA-Z0-9_-]+$', agent_id):
        raise HTTPException(status_code=400, detail=f"Invalid agent_id: {agent_id}")


def register_routes(app: FastAPI):
    """Register all API routes."""

    # ============== REST API Endpoints ==============

    @app.get("/api/agents", response_model=list[AgentData])
    async def get_agents():
        """Get all agents with status."""
        return app.state.data_provider.get_agents_data()

    # Bulk agent actions - must be registered BEFORE parameterized routes
    @app.post("/api/agents/start-all", response_model=SuccessResponse)
    async def start_all_agents():
        """Start all agents (like CLI `agent start`)."""
        try:
            running = app.state.orchestrator.get_running_agents()
            if running:
                return SuccessResponse(
                    message=f"Agents already running: {', '.join(running)}"
                )

            app.state.orchestrator.spawn_all_agents()
            return SuccessResponse(message="All agents started")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/stop-all", response_model=SuccessResponse)
    async def stop_all_agents():
        """Stop all agents (like CLI `agent stop`)."""
        try:
            app.state.orchestrator.stop_all_agents(graceful=True)
            return SuccessResponse(message="All agents stopped")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Individual agent actions - parameterized routes after specific ones
    @app.post("/api/agents/{agent_id}/start", response_model=AgentActionResponse)
    async def start_agent(agent_id: str):
        """Start a stopped agent."""
        _validate_agent_id(agent_id)
        try:
            app.state.orchestrator.spawn_agent(agent_id)
            return AgentActionResponse(
                success=True,
                agent_id=agent_id,
                action="start",
                message=f"Agent {agent_id} started",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/{agent_id}/stop", response_model=AgentActionResponse)
    async def stop_agent(agent_id: str):
        """Stop an agent."""
        _validate_agent_id(agent_id)
        try:
            app.state.orchestrator.stop_agent(agent_id, graceful=True)
            return AgentActionResponse(
                success=True,
                agent_id=agent_id,
                action="stop",
                message=f"Agent {agent_id} stopped",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/{agent_id}/restart", response_model=AgentActionResponse)
    async def restart_agent(agent_id: str):
        """Restart a dead agent."""
        _validate_agent_id(agent_id)
        try:
            # Stop then start
            app.state.orchestrator.stop_agent(agent_id, graceful=False)
            await asyncio.sleep(0.5)
            app.state.orchestrator.spawn_agent(agent_id)
            return AgentActionResponse(
                success=True,
                agent_id=agent_id,
                action="restart",
                message=f"Agent {agent_id} restarted",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/queues", response_model=list[QueueStats])
    async def get_queues():
        """Get queue statistics."""
        return app.state.data_provider.get_queue_stats()

    @app.get("/api/tasks/failed", response_model=list[FailedTaskData])
    async def get_failed_tasks(limit: int = Query(default=10, ge=1, le=100)):
        """Get failed tasks."""
        return app.state.data_provider.get_failed_tasks(limit=limit)

    @app.get("/api/tasks/active", response_model=list[ActiveTaskData])
    async def get_active_tasks(limit: int = Query(default=50, ge=1, le=200)):
        """Get all pending and in-progress tasks."""
        return app.state.data_provider.get_active_tasks(limit=limit)

    @app.post("/api/tasks/{task_id}/retry", response_model=TaskActionResponse)
    async def retry_task(task_id: str):
        """Retry a failed task."""
        success = app.state.data_provider.retry_task(task_id)
        if success:
            return TaskActionResponse(
                success=True,
                task_id=task_id,
                action="retry",
                message=f"Task {task_id} queued for retry",
            )
        else:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    @app.post("/api/tasks/{task_id}/cancel", response_model=TaskActionResponse)
    async def cancel_task(task_id: str, request: Request):
        """Cancel a pending or in-progress task."""
        if not re.match(r'^[a-zA-Z0-9_.-]+$', task_id):
            raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")

        body = {}
        try:
            body = await request.json()
        except Exception:
            pass

        reason = body.get("reason") if body else None
        success = app.state.data_provider.cancel_task(task_id, reason)
        if success:
            return TaskActionResponse(
                success=True,
                task_id=task_id,
                action="cancel",
                message=f"Task {task_id} cancelled",
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found or not cancellable",
            )

    @app.delete("/api/tasks/{task_id}", response_model=TaskActionResponse)
    async def delete_task(task_id: str):
        """Permanently delete a task from disk (only PENDING/FAILED/CANCELLED)."""
        if not re.match(r'^[a-zA-Z0-9_.-]+$', task_id):
            raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")

        error = app.state.data_provider.delete_task(task_id)
        if error is None:
            return TaskActionResponse(
                success=True,
                task_id=task_id,
                action="delete",
                message=f"Task {task_id} deleted",
            )
        elif error == "not_deletable":
            raise HTTPException(
                status_code=409,
                detail=f"Task {task_id} is not deletable (cancel in-progress tasks first)",
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found",
            )

    @app.post("/api/tasks", response_model=OperationResponse)
    async def create_task(request: CreateTaskRequest):
        """Create a task directly in a queue."""
        from ..core.task import TaskType

        # Validate task_type against enum
        try:
            TaskType(request.task_type)
        except ValueError:
            valid = [t.value for t in TaskType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid task_type '{request.task_type}'. Valid: {valid}",
            )

        # Validate assigned_to against configured agent queues
        agents_config = app.state.data_provider._get_agents_config()
        valid_queues = {a.queue for a in agents_config if a.enabled}
        if request.assigned_to not in valid_queues:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid queue '{request.assigned_to}'. Valid: {sorted(valid_queues)}",
            )

        try:
            task = app.state.data_provider.create_task(
                title=request.title,
                description=request.description,
                task_type=request.task_type,
                assigned_to=request.assigned_to,
                repository=request.repository,
                priority=request.priority,
            )
            return OperationResponse(
                success=True,
                task_id=task.id,
                message=f"Task created and queued to {request.assigned_to}",
            )
        except Exception as e:
            logger.exception(f"Error creating task: {e}")
            raise HTTPException(status_code=500, detail="Failed to create task")

    @app.get("/api/events", response_model=list[EventData])
    async def get_events(limit: int = Query(default=20, ge=1, le=100)):
        """Get recent activity events."""
        return app.state.data_provider.get_recent_events(limit=limit)

    @app.get("/api/health", response_model=HealthReport)
    async def get_health():
        """Get system health status."""
        return app.state.data_provider.get_health_status()

    @app.get("/api/analytics/agentic", response_model=AgenticInsightsData)
    async def get_agentic_metrics():
        """Get aggregated agentic feature metrics for the observability panel."""
        from ..analytics.agentic_metrics import AgenticMetricsAggregator

        aggregator = AgenticMetricsAggregator(app.state.workspace)
        metrics = aggregator.get_all_metrics()
        return AgenticInsightsData(
            memory=metrics.memory.__dict__,
            self_eval=metrics.self_eval.__dict__,
            replan=metrics.replan.__dict__,
            specialization_distribution=metrics.specialization_distribution,
            context_budget=metrics.context_budget.__dict__,
        )

    @app.post("/api/system/pause", response_model=SuccessResponse)
    async def pause_system():
        """Pause all agents."""
        paused = app.state.data_provider.pause()
        if paused:
            return SuccessResponse(message="System paused")
        else:
            return SuccessResponse(message="System already paused")

    @app.post("/api/system/resume", response_model=SuccessResponse)
    async def resume_system():
        """Resume agents."""
        resumed = app.state.data_provider.resume()
        if resumed:
            return SuccessResponse(message="System resumed")
        else:
            return SuccessResponse(message="System not paused")

    @app.get("/api/system/status")
    async def get_system_status():
        """Get system status (paused state, uptime)."""
        uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()
        return {
            "is_paused": app.state.data_provider.is_paused(),
            "uptime_seconds": int(uptime),
        }

    # ============== Setup API ==============

    @app.post("/api/setup/validate-jira", response_model=JIRAValidationResponse)
    async def validate_jira_credentials(request: JIRAValidationRequest):
        """Test JIRA connection with provided credentials."""
        try:
            from jira import JIRA
            from jira.exceptions import JIRAError

            # Test connection
            jira = JIRA(
                server=request.server,
                basic_auth=(request.email, request.api_token),
                timeout=10
            )

            # Get user info
            user_info = jira.myself()

            return JIRAValidationResponse(
                valid=True,
                message=f"Connected successfully as {user_info.get('displayName', request.email)}",
                user_info={
                    "display_name": user_info.get("displayName"),
                    "email": user_info.get("emailAddress"),
                }
            )

        except JIRAError as e:
            return JIRAValidationResponse(
                valid=False,
                message="Authentication failed",
                error=str(e)
            )
        except Exception as e:
            return JIRAValidationResponse(
                valid=False,
                message="Connection failed",
                error=str(e)
            )

    @app.post("/api/setup/validate-github", response_model=GitHubValidationResponse)
    async def validate_github_token(request: GitHubValidationRequest):
        """Test GitHub token and return rate limit info."""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {request.token}",
                "Accept": "application/vnd.github.v3+json"
            }

            # Test authentication
            response = requests.get("https://api.github.com/user", headers=headers, timeout=10)

            if response.status_code == 200:
                user_data = response.json()

                # Get rate limit
                rate_response = requests.get("https://api.github.com/rate_limit", headers=headers, timeout=10)
                rate_data = rate_response.json() if rate_response.status_code == 200 else None

                return GitHubValidationResponse(
                    valid=True,
                    user=user_data.get("login"),
                    rate_limit=rate_data.get("rate") if rate_data else None
                )
            elif response.status_code == 401:
                return GitHubValidationResponse(
                    valid=False,
                    error="Invalid token or insufficient permissions"
                )
            else:
                return GitHubValidationResponse(
                    valid=False,
                    error=f"Unexpected response: {response.status_code}"
                )

        except requests.exceptions.Timeout:
            return GitHubValidationResponse(
                valid=False,
                error="Connection timeout"
            )
        except Exception as e:
            return GitHubValidationResponse(
                valid=False,
                error=str(e)
            )

    @app.post("/api/setup/save-config", response_model=SuccessResponse)
    async def save_configuration(config: SetupConfiguration):
        """Generate and save all config files atomically."""
        try:
            from ..config.templates import ConfigGenerator

            generator = ConfigGenerator(app.state.workspace)

            # Convert to dict format
            jira_data = {
                "server": config.jira.server,
                "email": config.jira.email,
                "api_token": config.jira.api_token,
                "project": config.jira.project
            }

            github_data = {
                "token": config.github.token
            }

            repos = [
                {
                    "github_repo": r.github_repo,
                    "jira_project": r.jira_project,
                    "name": r.name
                }
                for r in config.repositories
            ]

            # Write all configs
            created_files = generator.write_all_configs(jira_data, github_data, repos)

            return SuccessResponse(
                message=f"Configuration saved successfully. Created {len(created_files)} file(s)."
            )

        except Exception as e:
            logger.exception(f"Error saving configuration: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save configuration: {str(e)}"
            )

    @app.get("/api/setup/status", response_model=SetupStatusResponse)
    async def get_setup_status():
        """Return current setup completion status."""
        import os
        from pathlib import Path

        config_dir = app.state.workspace / "config"

        # Check config files
        framework_config_exists = (config_dir / "agent-framework.yaml").exists()
        jira_config_exists = (config_dir / "jira.yaml").exists()
        github_config_exists = (config_dir / "github.yaml").exists()
        env_exists = (app.state.workspace / ".env").exists()

        # Check environment variables
        jira_configured = all([
            os.getenv("JIRA_SERVER"),
            os.getenv("JIRA_EMAIL"),
            os.getenv("JIRA_API_TOKEN")
        ]) and jira_config_exists

        github_configured = os.getenv("GITHUB_TOKEN") and github_config_exists

        # Count registered repositories
        repo_count = 0
        if framework_config_exists:
            try:
                from ..core.config import load_config
                fw_config = load_config(config_dir / "agent-framework.yaml")
                repo_count = len(fw_config.repositories)
            except (FileNotFoundError, ValueError) as e:
                logger.debug(f"Could not load framework config for setup status: {e}")

        initialized = framework_config_exists and env_exists
        ready_to_start = initialized and jira_configured and github_configured and repo_count > 0

        return SetupStatusResponse(
            initialized=initialized,
            jira_configured=jira_configured,
            github_configured=github_configured,
            repositories_registered=repo_count,
            mcp_enabled=False,  # Not implemented in MVP
            ready_to_start=ready_to_start
        )

    # ============== Error Translation API ==============

    @app.post("/api/errors/translate")
    async def translate_error(request: Request):
        """Translate technical error to user-friendly format."""
        try:
            from ..errors.translator import ErrorTranslator

            body = await request.json()
            error_message = body.get("error_message", "")

            translator = ErrorTranslator()
            exc = Exception(error_message)
            friendly = translator.translate(exc)

            return {
                "title": friendly.title,
                "explanation": friendly.explanation,
                "actions": friendly.actions,
                "documentation": friendly.documentation
            }
        except Exception as e:
            logger.exception(f"Error translating error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ============== Teams API ==============

    @app.get("/api/teams", response_model=list[TeamSessionData])
    async def get_teams():
        """List active Agent Team sessions."""
        return app.state.data_provider.get_active_teams()

    @app.post("/api/tasks/{task_id}/escalate-to-team")
    async def escalate_to_team(task_id: str):
        """Get CLI command to escalate a failed task to an interactive team.

        Agent Teams require an interactive terminal, so this returns
        the CLI command rather than launching directly.
        """
        if not re.match(r'^[a-zA-Z0-9_.-]+$', task_id):
            raise HTTPException(status_code=400, detail=f"Invalid task_id: {task_id}")

        task_data = app.state.data_provider.get_task(task_id)
        if not task_data:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return {
            "command": f"agent team escalate {task_id}",
            "task_id": task_id,
            "task_title": task_data.title,
        }

    # ============== Operations API ==============

    @app.post("/api/operations/work", response_model=OperationResponse)
    async def create_work(request: WorkRequest):
        """Create new work task (like CLI `agent work`).

        Queues a PLANNING task for the architect agent.
        """
        try:
            from ..core.task_builder import build_planning_task

            # Find repository config (use cached config)
            jira_project = None
            repository_name = request.repository.split("/")[-1]  # fallback
            for repo in app.state.framework_config.repositories:
                if repo.github_repo == request.repository:
                    jira_project = repo.jira_project
                    repository_name = repo.name
                    break

            # Build task using shared utility
            task = build_planning_task(
                goal=request.goal,
                workflow=request.workflow,
                github_repo=request.repository,
                repository_name=repository_name,
                jira_project=jira_project,
                created_by="web-dashboard",
            )

            # Queue the task
            queue = FileQueue(app.state.workspace)
            queue.push(task, "architect")

            return OperationResponse(
                success=True,
                task_id=task.id,
                message=f"Work queued for {request.repository}"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in create_work: {e}")
            raise HTTPException(status_code=500, detail="Failed to create work task")

    @app.post("/api/operations/analyze", response_model=OperationResponse)
    async def analyze_repo(request: AnalyzeRequest):
        """Trigger repository analysis (like CLI `agent analyze`).

        Queues an ANALYSIS task for the architect agent.
        """
        try:
            from ..core.task_builder import build_analysis_task

            # Find repository config (use cached config)
            jira_project = None
            for repo in app.state.framework_config.repositories:
                if repo.github_repo == request.repository:
                    jira_project = repo.jira_project
                    break

            # Build task using shared utility
            task = build_analysis_task(
                repository=request.repository,
                severity=request.severity,
                max_issues=request.max_issues,
                dry_run=request.dry_run,
                focus=request.focus,
                jira_project=jira_project,
                created_by="web-dashboard",
            )

            queue = FileQueue(app.state.workspace)
            queue.push(task, "architect")

            return OperationResponse(
                success=True,
                task_id=task_id,
                message=f"Analysis queued for {request.repository}"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in analyze_repo: {e}")
            raise HTTPException(status_code=500, detail="Failed to start repository analysis")

    @app.post("/api/operations/run-ticket", response_model=OperationResponse)
    async def run_ticket(request: RunTicketRequest):
        """Queue a specific JIRA ticket for processing."""
        try:
            from ..core.config import load_jira_config
            from ..integrations.jira.client import JIRAClient

            # Load JIRA config
            jira_config = load_jira_config(app.state.workspace / "config" / "jira.yaml")
            if not jira_config:
                raise HTTPException(
                    status_code=503,
                    detail="JIRA integration not configured"
                )

            jira_client = JIRAClient(jira_config)

            # Fetch the ticket
            issue = jira_client.jira.issue(request.ticket_id)

            # Determine agent assignment
            from ..core.config import load_agents
            agents_config = load_agents(app.state.workspace / "config" / "agents.yaml")
            assigned_to = request.agent or next(
                (a.queue for a in agents_config if a.enabled and a.id == "architect"),
                next(
                    (a.queue for a in agents_config if a.enabled),
                    "engineer"
                )
            )

            # Create task from issue
            task = jira_client.issue_to_task(issue, assigned_to)

            # Queue the task
            queue = FileQueue(app.state.workspace)
            queue.push(task, assigned_to)

            return OperationResponse(
                success=True,
                task_id=task.id,
                message=f"Ticket {request.ticket_id} queued for {assigned_to}"
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.exception(f"Error in run_ticket: {e}")
            raise HTTPException(status_code=500, detail="Failed to queue ticket")

    # ============== Log Streaming API ==============

    @app.get("/api/logs")
    async def list_log_files():
        """Get list of available log files."""
        return app.state.data_provider.get_available_log_files()

    @app.get("/api/logs/{agent_id}")
    async def get_agent_logs(agent_id: str, lines: int = Query(default=100, ge=1, le=1000)):
        """Get recent log lines for an agent."""
        try:
            log_lines = app.state.data_provider.get_agent_logs(agent_id, lines=lines)
            return {"agent": agent_id, "lines": log_lines}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ============== Claude CLI Log Streaming API ==============

    @app.get("/api/logs/claude-cli")
    async def list_claude_cli_logs():
        """Get list of available Claude CLI log files (task IDs)."""
        return app.state.data_provider.get_available_claude_cli_logs()

    @app.get("/api/logs/claude-cli/{task_id}")
    async def get_claude_cli_logs(task_id: str, lines: int = Query(default=100, ge=1, le=1000)):
        """Get recent log lines for a Claude CLI subprocess by task_id."""
        try:
            log_lines = app.state.data_provider.get_claude_cli_logs(task_id, lines=lines)
            return {"task_id": task_id, "lines": log_lines}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/api/logs/claude-cli-active")
    async def get_active_claude_cli_tasks():
        """Get mapping of agent_id to current task_id for agents with active CLI processes."""
        return app.state.data_provider.get_active_claude_cli_tasks()

    # ============== WebSocket ==============

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time updates."""
        await websocket.accept()
        logger.info("WebSocket client connected")

        try:
            while True:
                # Build full dashboard state
                uptime = (datetime.now(timezone.utc) - app.state.start_time).total_seconds()

                state = DashboardState(
                    agents=app.state.data_provider.get_agents_data(),
                    queues=app.state.data_provider.get_queue_stats(),
                    events=app.state.data_provider.get_recent_events(limit=5),
                    failed_tasks=app.state.data_provider.get_failed_tasks(limit=5),
                    health=app.state.data_provider.get_health_status(),
                    is_paused=app.state.data_provider.is_paused(),
                    uptime_seconds=int(uptime),
                    active_teams=app.state.data_provider.get_active_teams(),
                )

                await websocket.send_json(state.model_dump(mode="json"))
                await asyncio.sleep(0.5)  # 500ms refresh interval

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    @app.websocket("/ws/logs")
    async def log_stream(websocket: WebSocket):
        """WebSocket endpoint for real-time log streaming.

        Streams both agent logs and Claude CLI subprocess logs.
        CLI logs include source='claude-cli' and task_id fields.
        """
        await websocket.accept()
        logger.info("Log stream WebSocket client connected")

        # Track file positions for agent logs
        log_positions: Dict[str, int] = {}
        # Track file positions for Claude CLI logs
        cli_log_positions: Dict[str, int] = {}

        try:
            while True:
                # === Stream agent logs ===
                current_positions = app.state.data_provider.get_all_log_positions()

                for agent_id, current_size in current_positions.items():
                    try:
                        last_position = log_positions.get(agent_id, 0)

                        # If this is a new file or we haven't seen it, start from recent
                        if agent_id not in log_positions:
                            # Start from near the end to avoid sending huge backlog
                            log_positions[agent_id] = max(0, current_size - 4096)
                            last_position = log_positions[agent_id]

                        if current_size > last_position:
                            # Read new lines
                            new_lines, new_position = app.state.data_provider.read_log_from_position(
                                agent_id, last_position
                            )
                            log_positions[agent_id] = new_position

                            # Send each line as a separate message
                            for line in new_lines:
                                level = app.state.data_provider.parse_log_level(line)
                                await websocket.send_json({
                                    "agent": agent_id,
                                    "source": "agent",
                                    "line": line,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "level": level,
                                })
                    except WebSocketDisconnect:
                        raise
                    except Exception as e:
                        logger.warning(f"Error streaming agent log {agent_id}: {e}")

                # === Stream Claude CLI subprocess logs ===
                cli_positions = app.state.data_provider.get_all_claude_cli_log_positions()

                # Map task_id to agent_id for active tasks
                active_tasks = app.state.data_provider.get_active_claude_cli_tasks()
                task_to_agent = {v: k for k, v in active_tasks.items()}

                # Only stream logs for active tasks (skip historical logs with unknown agent)
                for task_id, current_size in cli_positions.items():
                    # Skip logs for tasks that are no longer active
                    if task_id not in task_to_agent:
                        continue

                    try:
                        last_position = cli_log_positions.get(task_id, 0)

                        # If this is a new file, start from near the end
                        if task_id not in cli_log_positions:
                            cli_log_positions[task_id] = max(0, current_size - 4096)
                            last_position = cli_log_positions[task_id]

                        if current_size > last_position:
                            new_lines, new_position = app.state.data_provider.read_claude_cli_log_from_position(
                                task_id, last_position
                            )
                            cli_log_positions[task_id] = new_position

                            agent_id = task_to_agent[task_id]

                            for line in new_lines:
                                level = app.state.data_provider.parse_log_level(line)
                                await websocket.send_json({
                                    "agent": agent_id,
                                    "task_id": task_id,
                                    "source": "claude-cli",
                                    "line": line,
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "level": level,
                                })
                    except WebSocketDisconnect:
                        raise
                    except Exception as e:
                        logger.warning(f"Error streaming CLI log {task_id}: {e}")

                # Cleanup: remove entries for task IDs no longer in active tasks
                stale_task_ids = [tid for tid in cli_log_positions if tid not in task_to_agent]
                for tid in stale_task_ids:
                    del cli_log_positions[tid]

                await asyncio.sleep(0.5)  # Poll every 500ms

        except WebSocketDisconnect:
            logger.info("Log stream WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Log stream WebSocket error: {e}")


def run_dashboard_server(
    workspace: Path,
    port: int = 8080,
    open_browser: bool = True,
    dev_mode: bool = False,
):
    """Run the dashboard server.

    Args:
        workspace: Path to workspace directory
        port: Server port (default: 8080)
        open_browser: Auto-open browser (default: True)
        dev_mode: Run in dev mode with Vite (default: False)
    """
    import uvicorn

    app = create_app(workspace)

    # In production mode, serve static files from dist/
    if not dev_mode:
        static_dir = Path(__file__).parent / "frontend" / "dist"
        if static_dir.exists():
            # Serve static assets
            app.mount(
                "/assets",
                StaticFiles(directory=static_dir / "assets"),
                name="assets",
            )

            # Serve index.html for root
            @app.get("/")
            async def serve_index():
                return FileResponse(static_dir / "index.html")

            # SPA fallback via 404 handler - won't interfere with API routes
            @app.exception_handler(404)
            async def spa_fallback(request: Request, exc: Exception):
                path = request.url.path
                # Only serve SPA for non-API/non-WS routes
                if not path.startswith("/api/") and not path.startswith("/ws"):
                    return FileResponse(static_dir / "index.html")
                return JSONResponse(status_code=404, content={"detail": "Not found"})

        else:
            logger.warning(
                f"Static files not found at {static_dir}. "
                "Run 'npm run build' in frontend directory or use --dev mode."
            )

            # Provide a simple HTML page with instructions
            @app.get("/")
            async def serve_placeholder():
                return {
                    "message": "Frontend not built",
                    "instructions": [
                        "cd src/agent_framework/web/frontend",
                        "npm install",
                        "npm run build",
                        "Or run: agent dashboard --dev",
                    ],
                }

    url = f"http://localhost:{port}"
    logger.info(f"Starting dashboard server at {url}")

    if open_browser:
        # Open browser after slight delay
        import threading

        def open_browser_delayed():
            import time
            time.sleep(1)
            webbrowser.open(url)

        threading.Thread(target=open_browser_delayed, daemon=True).start()

    # Run uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
