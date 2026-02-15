"""Tests for the profile generator (LLM-based profile creation)."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent_framework.core.profile_generator import ProfileGenerator, GenerationResult
from agent_framework.core.engineer_specialization import SpecializationProfile
from agent_framework.core.task import Task, TaskType, TaskStatus
from datetime import datetime, UTC


def _make_task(title="Build gRPC server", description="Implement gRPC endpoints for data service"):
    return Task(
        id="test-task",
        type=TaskType.IMPLEMENTATION,
        status=TaskStatus.PENDING,
        priority=1,
        created_by="test",
        assigned_to="engineer",
        created_at=datetime.now(UTC),
        title=title,
        description=description,
    )


VALID_PROFILE_JSON = json.dumps({
    "id": "grpc-services",
    "name": "gRPC Service Engineer",
    "description": "Specializes in gRPC service development",
    "file_patterns": ["**/*.proto", "**/grpc/**", "**/*_grpc.go"],
    "prompt_suffix": "You are a gRPC specialist. Focus on protobuf schema design, service definitions, and efficient serialization.",
    "tool_guidance": "Use protoc, grpcurl, buf for linting and testing.",
    "tags": ["grpc", "protobuf", "services", "rpc", "streaming"],
    "file_extensions": [".proto", ".go"],
})


class TestValidateAndConvert:
    """Tests for _validate_and_convert (no LLM calls)."""

    def test_valid_json(self):
        gen = ProfileGenerator(Path("/tmp"))
        result = gen._validate_and_convert(VALID_PROFILE_JSON, ["backend", "frontend"])

        assert result is not None
        assert isinstance(result, GenerationResult)
        assert result.profile.id == "grpc-services"
        assert result.profile.name == "gRPC Service Engineer"
        assert "**/*.proto" in result.profile.file_patterns
        assert len(result.profile.prompt_suffix) > 50

    def test_valid_json_with_markdown_fences(self):
        """LLM sometimes wraps output in markdown code blocks."""
        gen = ProfileGenerator(Path("/tmp"))
        wrapped = f"```json\n{VALID_PROFILE_JSON}\n```"
        result = gen._validate_and_convert(wrapped, [])
        assert result is not None
        assert result.profile.id == "grpc-services"

    def test_invalid_json(self):
        gen = ProfileGenerator(Path("/tmp"))
        result = gen._validate_and_convert("NOT JSON {{{", [])
        assert result is None

    def test_missing_required_field(self):
        gen = ProfileGenerator(Path("/tmp"))
        incomplete = json.dumps({
            "id": "test",
            "name": "Test",
            # missing description, file_patterns, prompt_suffix
        })
        result = gen._validate_and_convert(incomplete, [])
        assert result is None

    def test_empty_file_patterns(self):
        gen = ProfileGenerator(Path("/tmp"))
        data = json.loads(VALID_PROFILE_JSON)
        data["file_patterns"] = []
        result = gen._validate_and_convert(json.dumps(data), [])
        assert result is None

    def test_id_collision_rejected(self):
        gen = ProfileGenerator(Path("/tmp"))
        data = json.loads(VALID_PROFILE_JSON)
        data["id"] = "backend"
        result = gen._validate_and_convert(json.dumps(data), ["backend", "frontend"])
        assert result is None

    def test_short_prompt_suffix_rejected(self):
        gen = ProfileGenerator(Path("/tmp"))
        data = json.loads(VALID_PROFILE_JSON)
        data["prompt_suffix"] = "Too short"
        result = gen._validate_and_convert(json.dumps(data), [])
        assert result is None

    def test_tags_and_extensions_in_result(self):
        """Tags and extensions are returned in GenerationResult, not monkey-patched."""
        gen = ProfileGenerator(Path("/tmp"))
        result = gen._validate_and_convert(VALID_PROFILE_JSON, [])
        assert result is not None
        assert "grpc" in result.tags
        assert ".proto" in result.file_extensions

    def test_missing_optional_fields_default(self):
        """Missing tool_guidance and tags should default gracefully."""
        gen = ProfileGenerator(Path("/tmp"))
        data = json.loads(VALID_PROFILE_JSON)
        del data["tool_guidance"]
        del data["tags"]
        del data["file_extensions"]
        result = gen._validate_and_convert(json.dumps(data), [])
        assert result is not None
        assert result.profile.tool_guidance == ""
        assert result.tags == []
        assert result.file_extensions == []


class TestGenerateProfile:
    """Tests for generate_profile (with mocked subprocess)."""

    @patch("agent_framework.core.profile_generator.run_command")
    def test_successful_generation(self, mock_run):
        mock_run.return_value = MagicMock(stdout=VALID_PROFILE_JSON)

        gen = ProfileGenerator(Path("/tmp"))
        task = _make_task()
        result = gen.generate_profile(task, ["service.proto", "server.go"], ["backend"])

        assert result is not None
        assert result.profile.id == "grpc-services"
        mock_run.assert_called_once()

        # Verify the command includes the right flags
        cmd = mock_run.call_args[0][0]
        assert "--model" in cmd
        assert "haiku" in cmd
        assert "--max-turns" in cmd
        assert "1" in cmd
        assert "--dangerously-skip-permissions" in cmd

    @patch("agent_framework.core.profile_generator.run_command")
    def test_subprocess_failure_returns_none(self, mock_run):
        from agent_framework.utils.subprocess_utils import SubprocessError
        mock_run.side_effect = SubprocessError(
            cmd="claude", returncode=1, stderr="error"
        )

        gen = ProfileGenerator(Path("/tmp"))
        task = _make_task()
        result = gen.generate_profile(task, ["service.proto"], [])
        assert result is None

    @patch("agent_framework.core.profile_generator.run_command")
    def test_invalid_llm_output_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(stdout="Sorry, I can't generate that.")

        gen = ProfileGenerator(Path("/tmp"))
        task = _make_task()
        result = gen.generate_profile(task, ["service.proto"], [])
        assert result is None

    @patch("agent_framework.core.profile_generator.run_command")
    def test_model_passed_through(self, mock_run):
        mock_run.return_value = MagicMock(stdout=VALID_PROFILE_JSON)

        gen = ProfileGenerator(Path("/tmp"), model="sonnet")
        task = _make_task()
        gen.generate_profile(task, ["service.proto"], [])

        cmd = mock_run.call_args[0][0]
        model_idx = cmd.index("--model")
        assert cmd[model_idx + 1] == "sonnet"
