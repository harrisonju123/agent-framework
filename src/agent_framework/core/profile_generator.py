"""LLM-based specialization profile generation.

When no static profile matches a task's files, this module generates a new
SpecializationProfile via a cheap LLM call (haiku). The generated profile uses
the same dataclass as static profiles, so downstream consumers work unchanged.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .engineer_specialization import SpecializationProfile
from ..utils.subprocess_utils import run_command, SubprocessError

logger = logging.getLogger(__name__)

# Structured output schema the LLM must follow
_OUTPUT_SCHEMA = {
    "id": "string (lowercase, hyphenated, e.g. 'data-pipeline')",
    "name": "string (human-readable, e.g. 'Data Pipeline Engineer')",
    "description": "string (one sentence describing the specialization)",
    "file_patterns": ["list of glob patterns matching relevant files"],
    "prompt_suffix": "string (>50 chars, specialization guidance for the engineer)",
    "tool_guidance": "string (common tools for this domain)",
    "tags": ["list of semantic keyword tags for matching future tasks"],
    "file_extensions": ["list of dominant file extensions, e.g. .proto, .go"],
}

_GENERATION_PROMPT_TEMPLATE = """You are a profile generator for an engineer specialization system.

Given a task and its files, generate a specialization profile that will help an engineer agent
work effectively in this domain.

TASK TITLE: {title}
TASK DESCRIPTION: {description}

FILES:
{file_list}

EXISTING PROFILE IDs (do NOT reuse these): {existing_ids}

Generate a JSON object matching this exact schema:
{schema}

Rules:
- The "id" must be unique, lowercase, hyphenated (not any of the existing IDs)
- "file_patterns" must use glob syntax (e.g. "**/*.proto", "**/pipeline/**")
- "prompt_suffix" must be >50 characters with actionable specialization guidance
- "tags" should be 5-15 semantic keywords for matching similar future tasks
- "file_extensions" should list the dominant extensions you see in the file list
- Output ONLY valid JSON, no markdown fences, no explanation

JSON:"""


@dataclass
class GenerationResult:
    """Bundles the generated profile with metadata for the registry."""

    profile: SpecializationProfile
    tags: List[str]
    file_extensions: List[str]


class ProfileGenerator:
    """Generates specialization profiles via LLM for novel task domains."""

    def __init__(self, workspace: Path, model: str = "haiku"):
        self._workspace = workspace
        self._model = model

    def generate_profile(
        self,
        task,
        files: List[str],
        existing_profile_ids: List[str],
    ) -> Optional[GenerationResult]:
        """Generate a profile via claude CLI (haiku, --max-turns 1).

        Returns None on any failure — generation is best-effort and never
        blocks the main workflow.
        """
        file_list = "\n".join(f"- {f}" for f in files[:50])
        prompt = _GENERATION_PROMPT_TEMPLATE.format(
            title=task.title,
            description=task.description[:500],
            file_list=file_list,
            existing_ids=", ".join(existing_profile_ids) or "(none)",
            schema=json.dumps(_OUTPUT_SCHEMA, indent=2),
        )

        try:
            result = run_command(
                [
                    "claude",
                    "--print",
                    "--model", self._model,
                    "--dangerously-skip-permissions",
                    "--max-turns", "1",
                    "--output-format", "text",
                    "-p", prompt,
                ],
                cwd=self._workspace,
                timeout=30,
                check=True,
            )
            raw_output = result.stdout.strip()
        except (SubprocessError, OSError) as exc:
            logger.warning("Profile generation LLM call failed: %s", exc)
            return None

        return self._validate_and_convert(raw_output, existing_profile_ids)

    def _validate_and_convert(
        self,
        raw_json: str,
        existing_ids: List[str],
    ) -> Optional[GenerationResult]:
        """Validate LLM output and convert to GenerationResult.

        Rejects: invalid JSON, missing fields, empty file_patterns,
        ID collision with static profiles, prompt_suffix <50 chars.
        """
        # Strip markdown fences if LLM wrapped output
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Profile generation returned invalid JSON: %s", exc)
            return None

        # Required fields
        required = ["id", "name", "description", "file_patterns", "prompt_suffix"]
        for field in required:
            if field not in data or not data[field]:
                logger.warning("Generated profile missing required field: %s", field)
                return None

        profile_id = data["id"]

        # ID collision check
        if profile_id in existing_ids:
            logger.warning(
                "Generated profile ID '%s' collides with existing profile", profile_id
            )
            return None

        # File patterns must be non-empty list
        if not isinstance(data["file_patterns"], list) or len(data["file_patterns"]) == 0:
            logger.warning("Generated profile has empty file_patterns")
            return None

        # Prompt suffix quality gate
        prompt_suffix = data["prompt_suffix"]
        if not isinstance(prompt_suffix, str) or len(prompt_suffix) < 50:
            logger.warning(
                "Generated profile prompt_suffix too short (%d chars, need 50+)",
                len(prompt_suffix) if isinstance(prompt_suffix, str) else 0,
            )
            return None

        profile = SpecializationProfile(
            id=profile_id,
            name=data["name"],
            description=data["description"],
            file_patterns=data["file_patterns"],
            prompt_suffix=prompt_suffix,
            teammates={},
            tool_guidance=data.get("tool_guidance", ""),
        )

        logger.info("Generated profile '%s': %s", profile.id, profile.name)
        return GenerationResult(
            profile=profile,
            tags=data.get("tags", []),
            file_extensions=data.get("file_extensions", []),
        )
