"""Integration tests for the engineer specialization pipeline.

Tests the full flow from task → detection → prompt/team composition,
including YAML config loading, enable/disable, and hint overrides.
"""

import tempfile
from datetime import datetime, UTC
from pathlib import Path
from unittest.mock import patch

import yaml

from agent_framework.core.config import (
    AgentDefinition,
    TeammateDefinition,
    SpecializationConfig,
    SpecializationProfileConfig,
    SpecializationTeammateConfig,
    load_specializations,
    clear_config_cache,
)
from agent_framework.core.engineer_specialization import (
    detect_specialization,
    apply_specialization_to_prompt,
    get_specialization_enabled,
    _load_profiles,
    BACKEND_PROFILE,
    FRONTEND_PROFILE,
    INFRASTRUCTURE_PROFILE,
)
from agent_framework.core.task import Task, TaskType, TaskStatus, PlanDocument
from agent_framework.core.team_composer import compose_default_team


def _make_task(files=None, context=None, description=""):
    """Helper to create a task for integration tests."""
    plan = None
    if files:
        plan = PlanDocument(
            objectives=["Test"],
            approach=["Test"],
            success_criteria=["Test"],
            files_to_modify=files,
        )
    return Task(
        id="integ-test",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Integration test task",
        description=description,
        plan=plan,
        context=context or {},
    )


def _make_agent_def(**overrides):
    """Helper to create a minimal AgentDefinition."""
    defaults = dict(
        id="engineer",
        name="Engineer",
        queue="engineer",
        prompt="You are a software engineer.",
        teammates={
            "code-reviewer": TeammateDefinition(
                description="Reviews code quality",
                prompt="You review code.",
            )
        },
    )
    defaults.update(overrides)
    return AgentDefinition(**defaults)


class TestSpecializationPromptIntegration:
    """Verify that backend/frontend/infra tasks produce specialized prompts."""

    def test_backend_task_produces_backend_prompt(self):
        task = _make_task(files=[
            "cmd/server/main.go",
            "internal/handler/auth.go",
            "internal/service/user.go",
        ])
        profile = detect_specialization(task)
        prompt = apply_specialization_to_prompt("Base engineer prompt.", profile)

        assert "BACKEND SPECIALIZATION" in prompt
        assert "API design" in prompt
        assert "Base engineer prompt." in prompt

    def test_frontend_task_produces_frontend_prompt(self):
        task = _make_task(files=[
            "src/components/Header.tsx",
            "src/pages/Settings.tsx",
            "src/styles/settings.css",
        ])
        profile = detect_specialization(task)
        prompt = apply_specialization_to_prompt("Base engineer prompt.", profile)

        assert "FRONTEND SPECIALIZATION" in prompt
        assert "Component design" in prompt

    def test_infra_task_produces_infra_prompt(self):
        task = _make_task(files=[
            "Dockerfile",
            "docker-compose.yml",
            ".github/workflows/deploy.yml",
        ])
        profile = detect_specialization(task)
        prompt = apply_specialization_to_prompt("Base engineer prompt.", profile)

        assert "INFRASTRUCTURE SPECIALIZATION" in prompt
        assert "Container orchestration" in prompt

    def test_generic_task_returns_base_prompt(self):
        task = _make_task(files=["README.md", "docs/guide.txt"])
        profile = detect_specialization(task)
        prompt = apply_specialization_to_prompt("Base engineer prompt.", profile)

        assert prompt == "Base engineer prompt."


class TestSpecializationTeamIntegration:
    """Verify compose_default_team includes specialized teammates."""

    def test_backend_profile_adds_db_and_api_teammates(self):
        agent_def = _make_agent_def()
        team = compose_default_team(
            agent_def,
            default_model="sonnet",
            specialization_profile=BACKEND_PROFILE,
        )
        assert team is not None
        assert "database-expert" in team
        assert "api-reviewer" in team
        # Base teammate preserved
        assert "code-reviewer" in team

    def test_frontend_profile_adds_ux_and_perf_teammates(self):
        agent_def = _make_agent_def()
        team = compose_default_team(
            agent_def,
            default_model="sonnet",
            specialization_profile=FRONTEND_PROFILE,
        )
        assert "ux-reviewer" in team
        assert "performance-auditor" in team
        assert "code-reviewer" in team

    def test_no_profile_returns_base_teammates_only(self):
        agent_def = _make_agent_def()
        team = compose_default_team(agent_def, default_model="sonnet")
        assert "code-reviewer" in team
        assert "database-expert" not in team


class TestSpecializationDisableIntegration:
    """Verify disabled flag prevents specialization."""

    @patch("agent_framework.core.config.load_specializations")
    def test_global_disable_returns_none(self, mock_load):
        """When enabled=False globally, get_specialization_enabled returns False."""
        mock_load.return_value = SpecializationConfig(enabled=False, profiles=[])
        assert get_specialization_enabled() is False

    @patch("agent_framework.core.config.load_specializations")
    def test_detection_still_works_when_enabled(self, mock_load):
        """When enabled=True, detection proceeds normally."""
        mock_load.return_value = None  # Falls back to defaults
        task = _make_task(files=[
            "cmd/main.go",
            "internal/api/handler.go",
            "pkg/models/user.go",
        ])
        profile = detect_specialization(task)
        assert profile is not None
        assert profile.id == "backend"


class TestSpecializationHintIntegration:
    """Verify hint overrides file-based detection in the full pipeline."""

    def test_hint_overrides_to_backend(self):
        """Frontend files + backend hint → backend profile + backend prompt."""
        task = _make_task(
            files=[
                "src/components/Button.tsx",
                "src/pages/Home.tsx",
                "src/styles/app.scss",
            ],
            context={"specialization_hint": "backend"},
        )
        profile = detect_specialization(task)
        assert profile.id == "backend"

        prompt = apply_specialization_to_prompt("Base.", profile)
        assert "BACKEND SPECIALIZATION" in prompt

    def test_invalid_hint_falls_back_to_file_detection(self):
        task = _make_task(
            files=[
                "Dockerfile",
                "docker-compose.yml",
                "k8s/deployment.yaml",
            ],
            context={"specialization_hint": "does-not-exist"},
        )
        profile = detect_specialization(task)
        assert profile.id == "infrastructure"


class TestSpecializationYAMLLoading:
    """Verify YAML config loads and produces correct profiles."""

    def test_load_from_yaml_file(self):
        """Write a YAML config to a temp file and load it."""
        config_data = {
            "enabled": True,
            "profiles": [
                {
                    "id": "ml-engineer",
                    "name": "ML Engineer",
                    "description": "Machine learning specialist",
                    "file_patterns": ["**/*.ipynb", "**/models/**"],
                    "prompt_suffix": "ML SPECIALIZATION",
                    "tool_guidance": "Use jupyter, pytorch",
                    "teammates": {
                        "data-scientist": {
                            "description": "Reviews data pipelines",
                            "prompt": "You review data pipelines.",
                        }
                    },
                }
            ],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            tmp_path = Path(f.name)

        try:
            # Clear cache to force reload
            clear_config_cache()

            result = load_specializations(tmp_path)
            assert result is not None
            assert result.enabled is True
            assert len(result.profiles) == 1
            assert result.profiles[0].id == "ml-engineer"
            assert result.profiles[0].teammates["data-scientist"].description == "Reviews data pipelines"
        finally:
            tmp_path.unlink()

    def test_fallback_when_no_yaml(self):
        """Non-existent path → returns None."""
        result = load_specializations(Path("/tmp/nonexistent-spec-config.yaml"))
        assert result is None

    def test_disabled_in_yaml(self):
        """YAML with enabled: false."""
        config_data = {"enabled": False, "profiles": []}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            tmp_path = Path(f.name)

        try:
            clear_config_cache()
            result = load_specializations(tmp_path)
            assert result is not None
            assert result.enabled is False
        finally:
            tmp_path.unlink()


class TestSpecializationActivityTracking:
    """Verify that specialization information is tracked in activity logs."""

    def test_activity_has_specialization_field(self):
        """AgentActivity model includes specialization field."""
        from agent_framework.core.activity import AgentActivity, AgentStatus

        activity = AgentActivity(
            agent_id="engineer",
            status=AgentStatus.WORKING,
            specialization="backend",
            last_updated=datetime.now(UTC),
        )

        assert activity.specialization == "backend"

        # Test serialization/deserialization
        data = activity.model_dump()
        assert data["specialization"] == "backend"

        restored = AgentActivity(**data)
        assert restored.specialization == "backend"

    def test_activity_specialization_is_optional(self):
        """Specialization field is optional for backward compatibility."""
        from agent_framework.core.activity import AgentActivity, AgentStatus

        activity = AgentActivity(
            agent_id="architect",
            status=AgentStatus.IDLE,
            last_updated=datetime.now(UTC),
        )

        assert activity.specialization is None

        # Test serialization/deserialization without specialization
        data = activity.model_dump()
        assert "specialization" in data
        assert data["specialization"] is None

        restored = AgentActivity(**data)
        assert restored.specialization is None
