"""Tests for engineer specialization system."""

import pytest
from datetime import datetime, UTC

from agent_framework.core.engineer_specialization import (
    detect_file_patterns,
    match_patterns,
    detect_specialization,
    apply_specialization_to_prompt,
    get_specialized_teammates,
    BACKEND_PROFILE,
    FRONTEND_PROFILE,
    INFRASTRUCTURE_PROFILE,
    KNOWN_SOURCE_EXTENSIONS,
)
from agent_framework.core.task import Task, TaskType, TaskStatus, PlanDocument


def create_test_task(
    files_in_plan=None,
    files_in_context=None,
    description_text="",
) -> Task:
    """Create a test task with specified files."""
    plan = None
    if files_in_plan:
        plan = PlanDocument(
            objectives=["Test"],
            approach=["Test"],
            success_criteria=["Test passes"],
            files_to_modify=files_in_plan,
        )

    context = {}
    if files_in_context:
        context["files"] = files_in_context

    return Task(
        id="test-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title="Test task",
        description=description_text,
        plan=plan,
        context=context,
    )


class TestDetectFilePatterns:
    """Tests for file pattern detection."""

    def test_detect_from_plan(self):
        """Should extract files from plan.files_to_modify."""
        task = create_test_task(files_in_plan=["src/api/handlers.go", "src/models/user.go"])
        files = detect_file_patterns(task)
        assert "src/api/handlers.go" in files
        assert "src/models/user.go" in files

    def test_detect_from_context(self):
        """Should extract files from context.files."""
        task = create_test_task(files_in_context=["src/components/Button.tsx"])
        files = detect_file_patterns(task)
        assert "src/components/Button.tsx" in files

    def test_detect_from_description(self):
        """Should parse file paths from description text."""
        task = create_test_task(description_text="Update src/handlers/auth.py and tests/test_auth.py")
        files = detect_file_patterns(task)
        assert "src/handlers/auth.py" in files
        assert "tests/test_auth.py" in files

    def test_detect_from_structured_findings(self):
        """Should extract files from structured_findings."""
        task = Task(
            id="test-task",
            type=TaskType.IMPLEMENTATION,
            status=TaskStatus.PENDING,
            priority=1,
            created_by="test",
            assigned_to="engineer",
            created_at=datetime.now(UTC),
            title="Fix issues",
            description="Fix security issues",
            context={
                "structured_findings": {
                    "findings": [
                        {"file": "src/api/auth.go", "line": 42, "severity": "CRITICAL"},
                        {"file": "src/db/query.go", "line": 89, "severity": "HIGH"},
                    ]
                }
            },
        )
        files = detect_file_patterns(task)
        assert "src/api/auth.go" in files
        assert "src/db/query.go" in files

    def test_deduplicate_files(self):
        """Should deduplicate files from multiple sources."""
        task = create_test_task(
            files_in_plan=["src/api/handler.go"],
            files_in_context=["src/api/handler.go"],
        )
        files = detect_file_patterns(task)
        assert files.count("src/api/handler.go") == 1

    def test_description_regex_ignores_non_source_extensions(self):
        """Should not match version strings or URL-like paths in descriptions."""
        task = create_test_task(
            description_text="Use version/2.0.1 and check http/1.1 and config/data.yaml"
        )
        files = detect_file_patterns(task)
        # version/2.0.1 has no known source extension
        assert not any("version" in f for f in files)
        assert not any("http" in f for f in files)
        # .yaml is not in KNOWN_SOURCE_EXTENSIONS
        assert not any("config/data.yaml" in f for f in files)

    def test_description_regex_matches_known_extensions(self):
        """Should match paths with known source extensions."""
        task = create_test_task(
            description_text="Modify src/utils/helper.go and lib/parser.ts"
        )
        files = detect_file_patterns(task)
        assert "src/utils/helper.go" in files
        assert "lib/parser.ts" in files


class TestMatchPatterns:
    """Tests for pattern matching."""

    def test_match_go_files(self):
        """Should match Go files against backend patterns."""
        files = ["src/api/handler.go", "cmd/server/main.go", "internal/service/user.go"]
        matches = match_patterns(files, BACKEND_PROFILE.file_patterns)
        assert matches == 3

    def test_match_tsx_files(self):
        """Should match TypeScript React files against frontend patterns."""
        files = ["src/components/Button.tsx", "src/pages/Home.tsx", "src/styles/app.css"]
        matches = match_patterns(files, FRONTEND_PROFILE.file_patterns)
        assert matches == 3

    def test_match_dockerfile(self):
        """Should match infrastructure-specific files."""
        files = ["Dockerfile", "docker-compose.yml", "k8s/deployment.yaml"]
        matches = match_patterns(files, INFRASTRUCTURE_PROFILE.file_patterns)
        assert matches == 3

    def test_no_matches(self):
        """Should return 0 for files that don't match any pattern."""
        files = ["README.md", "LICENSE", "notes.txt"]
        matches = match_patterns(files, BACKEND_PROFILE.file_patterns)
        assert matches == 0

    def test_match_each_file_once(self):
        """Should count each file only once even if it matches multiple patterns."""
        # A Go test file matches both **/*.go and **/*_test.go patterns
        files = ["src/handler_test.go"]
        matches = match_patterns(files, BACKEND_PROFILE.file_patterns)
        assert matches == 1

    def test_generic_yaml_not_matched_as_infrastructure(self):
        """Generic YAML files should not match infrastructure patterns."""
        files = ["config/agents.yaml", "data/fixtures.yml", "settings.yaml"]
        matches = match_patterns(files, INFRASTRUCTURE_PROFILE.file_patterns)
        assert matches == 0

    def test_plain_ts_js_not_matched_as_frontend(self):
        """Plain .ts/.js files should not match frontend patterns (could be backend Node.js)."""
        files = ["src/server.ts", "src/routes/api.js", "src/middleware/auth.ts"]
        matches = match_patterns(files, FRONTEND_PROFILE.file_patterns)
        assert matches == 0


class TestDetectSpecialization:
    """Tests for specialization detection."""

    def test_backend_specialization_go(self):
        """Should detect backend specialization for Go files."""
        task = create_test_task(files_in_plan=[
            "cmd/server/main.go",
            "internal/api/handler.go",
            "internal/service/user.go",
        ])
        profile = detect_specialization(task)
        assert profile == BACKEND_PROFILE

    def test_backend_specialization_python(self):
        """Should detect backend specialization for Python files."""
        task = create_test_task(files_in_plan=[
            "src/api/views.py",
            "src/models/user.py",
            "tests/test_api.py",
        ])
        profile = detect_specialization(task)
        assert profile == BACKEND_PROFILE

    def test_frontend_specialization_react(self):
        """Should detect frontend specialization for React files."""
        task = create_test_task(files_in_plan=[
            "src/components/Button.tsx",
            "src/pages/Home.tsx",
            "src/styles/app.scss",
        ])
        profile = detect_specialization(task)
        assert profile == FRONTEND_PROFILE

    def test_frontend_specialization_vue(self):
        """Should detect frontend specialization for Vue files."""
        task = create_test_task(files_in_plan=[
            "src/components/Button.vue",
            "src/views/Home.vue",
        ])
        profile = detect_specialization(task)
        assert profile == FRONTEND_PROFILE

    def test_infrastructure_specialization_docker(self):
        """Should detect infrastructure specialization for Docker files."""
        task = create_test_task(files_in_plan=[
            "Dockerfile",
            "docker-compose.yml",
            ".github/workflows/ci.yml",
        ])
        profile = detect_specialization(task)
        assert profile == INFRASTRUCTURE_PROFILE

    def test_infrastructure_specialization_k8s(self):
        """Should detect infrastructure specialization for Kubernetes files."""
        task = create_test_task(files_in_plan=[
            "k8s/deployment.yaml",
            "k8s/service.yaml",
            "helm/values.yaml",
        ])
        profile = detect_specialization(task)
        assert profile == INFRASTRUCTURE_PROFILE

    def test_infrastructure_not_triggered_by_generic_yaml(self):
        """Generic YAML config files should not trigger infrastructure specialization."""
        task = create_test_task(files_in_plan=[
            "config/agents.yaml",
            "config/settings.yml",
            "data/fixtures.yaml",
        ])
        profile = detect_specialization(task)
        assert profile is None

    def test_mixed_patterns_backend_dominant(self):
        """Should pick backend when it has clear majority."""
        task = create_test_task(files_in_plan=[
            "src/api/handler.go",
            "src/service/user.go",
            "internal/db/query.go",
            "src/components/Button.tsx",  # One frontend file
        ])
        profile = detect_specialization(task)
        assert profile == BACKEND_PROFILE

    def test_mixed_patterns_no_clear_winner(self):
        """Should return None for mixed patterns with no clear winner."""
        task = create_test_task(files_in_plan=[
            "src/api/handler.go",
            "src/components/Button.tsx",
        ])
        profile = detect_specialization(task)
        # 50/50 split, no specialization
        assert profile is None

    def test_no_files_detected(self):
        """Should return None when no files are detected."""
        task = create_test_task()
        profile = detect_specialization(task)
        assert profile is None

    def test_minimum_threshold(self):
        """Should require at least 2 matching files."""
        task = create_test_task(files_in_plan=["src/handler.go"])
        profile = detect_specialization(task)
        # Only 1 file, below threshold
        assert profile is None


class TestApplySpecializationToPrompt:
    """Tests for prompt specialization."""

    def test_apply_backend_specialization(self):
        """Should append backend context to base prompt."""
        base_prompt = "You are a software engineer."
        specialized = apply_specialization_to_prompt(base_prompt, BACKEND_PROFILE)

        assert base_prompt in specialized
        assert "BACKEND SPECIALIZATION" in specialized
        assert "API design and implementation" in specialized
        assert "Database schema design" in specialized

    def test_apply_frontend_specialization(self):
        """Should append frontend context to base prompt."""
        base_prompt = "You are a software engineer."
        specialized = apply_specialization_to_prompt(base_prompt, FRONTEND_PROFILE)

        assert base_prompt in specialized
        assert "FRONTEND SPECIALIZATION" in specialized
        assert "Component design and composition" in specialized
        assert "accessibility" in specialized

    def test_apply_infrastructure_specialization(self):
        """Should append infrastructure context to base prompt."""
        base_prompt = "You are a software engineer."
        specialized = apply_specialization_to_prompt(base_prompt, INFRASTRUCTURE_PROFILE)

        assert base_prompt in specialized
        assert "INFRASTRUCTURE SPECIALIZATION" in specialized
        assert "Container orchestration" in specialized
        assert "Infrastructure as Code" in specialized

    def test_no_specialization(self):
        """Should return base prompt unchanged when no profile."""
        base_prompt = "You are a software engineer."
        specialized = apply_specialization_to_prompt(base_prompt, None)
        assert specialized == base_prompt


class TestGetSpecializedTeammates:
    """Tests for teammate merging."""

    def test_merge_with_base_teammates(self):
        """Should merge specialization teammates into base."""
        base = {"code-reviewer": {"description": "Reviews code", "prompt": "Review code."}}
        merged = get_specialized_teammates(base, BACKEND_PROFILE)

        assert "code-reviewer" in merged
        assert "database-expert" in merged
        assert "api-reviewer" in merged

    def test_no_profile_returns_base(self):
        """Should return base unchanged when no profile."""
        base = {"code-reviewer": {"description": "Reviews code", "prompt": "Review code."}}
        result = get_specialized_teammates(base, None)
        assert result == base

    def test_specialization_overrides_same_key(self):
        """Specialization teammates should override base with same key."""
        base = {"database-expert": {"description": "Old", "prompt": "Old prompt."}}
        merged = get_specialized_teammates(base, BACKEND_PROFILE)

        # Should be overridden by BACKEND_PROFILE's database-expert
        assert "query optimization" in merged["database-expert"]["description"]


class TestEndToEnd:
    """End-to-end tests for the specialization system."""

    def test_backend_workflow(self):
        """Test full workflow for backend specialization."""
        task = create_test_task(
            files_in_plan=[
                "internal/api/handler.go",
                "internal/service/user_service.go",
                "internal/models/user.go",
            ],
            description_text="Add new user registration endpoint",
        )

        profile = detect_specialization(task)
        assert profile == BACKEND_PROFILE

        base_prompt = "You are the Software Engineer responsible for implementing features."
        specialized_prompt = apply_specialization_to_prompt(base_prompt, profile)

        assert "BACKEND SPECIALIZATION" in specialized_prompt
        assert "API design and implementation" in specialized_prompt
        assert "parameterized queries" in specialized_prompt

    def test_frontend_workflow(self):
        """Test full workflow for frontend specialization."""
        task = create_test_task(
            files_in_plan=[
                "src/components/UserProfile.tsx",
                "src/pages/Dashboard.tsx",
                "src/styles/dashboard.scss",
            ],
            description_text="Build user profile dashboard",
        )

        profile = detect_specialization(task)
        assert profile == FRONTEND_PROFILE

        base_prompt = "You are the Software Engineer responsible for implementing features."
        specialized_prompt = apply_specialization_to_prompt(base_prompt, profile)

        assert "FRONTEND SPECIALIZATION" in specialized_prompt
        assert "accessibility" in specialized_prompt
        assert "Component design" in specialized_prompt

    def test_generic_workflow(self):
        """Test workflow when no specialization applies."""
        task = create_test_task(
            files_in_plan=["README.md"],
            description_text="Update documentation",
        )

        profile = detect_specialization(task)
        assert profile is None

        base_prompt = "You are the Software Engineer responsible for implementing features."
        specialized_prompt = apply_specialization_to_prompt(base_prompt, profile)

        assert specialized_prompt == base_prompt
        assert "SPECIALIZATION" not in specialized_prompt
