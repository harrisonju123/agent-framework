"""Tests for team template loading and prompt building."""

import pytest
from pathlib import Path

from agent_framework.core.team_templates import (
    TeamTemplate,
    load_team_templates,
    build_spawn_prompt,
)


@pytest.fixture
def sample_template():
    return TeamTemplate(
        name="full",
        team_name_prefix="impl",
        lead_role="Architect",
        spawn_prompt="You are the lead architect.",
        plan_approval=True,
        delegate_mode=False,
    )


def test_load_team_templates(tmp_path):
    """Templates load correctly from YAML."""
    config_dir = tmp_path
    templates_file = config_dir / "team-templates.yaml"
    templates_file.write_text("""
templates:
  full:
    team_name_prefix: impl
    lead_role: Architect
    spawn_prompt: "Lead the team."
    plan_approval: true
    delegate_mode: false
  debug:
    team_name_prefix: debug
    lead_role: Debug Lead
    spawn_prompt: "Debug the issue."
""")

    templates = load_team_templates(config_dir)

    assert len(templates) == 2
    assert "full" in templates
    assert "debug" in templates
    assert templates["full"].lead_role == "Architect"
    assert templates["full"].plan_approval is True
    assert templates["debug"].delegate_mode is False


def test_load_team_templates_missing_file(tmp_path):
    """Returns empty dict when config file doesn't exist."""
    templates = load_team_templates(tmp_path)
    assert templates == {}


def test_load_team_templates_empty_file(tmp_path):
    """Returns empty dict for empty YAML file."""
    (tmp_path / "team-templates.yaml").write_text("")
    templates = load_team_templates(tmp_path)
    assert templates == {}


def test_build_spawn_prompt_basic(sample_template):
    """Basic prompt uses template spawn_prompt."""
    prompt = build_spawn_prompt(sample_template)
    assert "You are the lead architect." in prompt


def test_build_spawn_prompt_with_repo(sample_template):
    """Repo info is included in prompt."""
    prompt = build_spawn_prompt(sample_template, repo_info="Repository: org/app")
    assert "Repository: org/app" in prompt
    assert "Target Repository" in prompt


def test_build_spawn_prompt_with_task_context(sample_template):
    """Task context is included in prompt."""
    prompt = build_spawn_prompt(
        sample_template,
        task_context="Task failed with: connection timeout",
    )
    assert "connection timeout" in prompt
    assert "Task Context" in prompt


def test_build_spawn_prompt_with_team_context(sample_template):
    """Team context doc is included in prompt."""
    prompt = build_spawn_prompt(
        sample_template,
        team_context_doc="Use queue_task_for_agent to hand off work.",
    )
    assert "queue_task_for_agent" in prompt
    assert "Pipeline Integration" in prompt


def test_build_spawn_prompt_all_context(sample_template):
    """All context sections appear when provided."""
    prompt = build_spawn_prompt(
        sample_template,
        task_context="Error details",
        repo_info="org/repo",
        team_context_doc="MCP tools available",
    )
    assert "Error details" in prompt
    assert "org/repo" in prompt
    assert "MCP tools available" in prompt
