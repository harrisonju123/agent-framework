"""Validation utilities for repository names, branch names, and identifiers."""

import re


def validate_branch_name(branch_name: str) -> str:
    """
    Validate and sanitize git branch name.

    Args:
        branch_name: Branch name to validate

    Returns:
        Validated branch name

    Raises:
        ValueError: If branch name is invalid
    """
    if not branch_name:
        raise ValueError("Branch name cannot be empty")

    # Strict whitelist
    if not re.match(r'^[a-zA-Z0-9/_-]+$', branch_name):
        raise ValueError(f"Invalid branch name: {branch_name}")

    if branch_name.startswith('/') or branch_name.endswith('/'):
        raise ValueError("Branch name cannot start or end with /")

    if '..' in branch_name or '@{' in branch_name:
        raise ValueError("Branch name contains invalid sequence")

    if len(branch_name) > 255:
        raise ValueError("Branch name too long")

    return branch_name


def validate_identifier(value: str, name: str = "identifier") -> str:
    """
    Validate agent_id or task_id to prevent path traversal.

    Args:
        value: Identifier value to validate
        name: Name of the identifier (for error messages)

    Returns:
        Validated identifier

    Raises:
        ValueError: If identifier is invalid
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")

    # Only allow alphanumeric, dash, underscore
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise ValueError(f"Invalid {name}: {value}")

    if '..' in value or '/' in value or '\\' in value:
        raise ValueError(f"{name} contains invalid characters: {value}")

    if len(value) > 128:
        raise ValueError(f"{name} too long")

    return value


def validate_owner_repo(owner_repo: str) -> str:
    """
    Validate repository name format (owner/repo).

    Args:
        owner_repo: Repository name in owner/repo format

    Returns:
        Validated repository name

    Raises:
        ValueError: If repository name format is invalid
    """
    if not owner_repo:
        raise ValueError("Repository name cannot be empty")

    # Must be in format "owner/repo"
    if not re.match(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+$', owner_repo):
        raise ValueError(
            f"Invalid repository format: {owner_repo}. Must be 'owner/repo'"
        )

    # Prevent path traversal
    if '..' in owner_repo or owner_repo.startswith('/'):
        raise ValueError(f"Invalid repository name: {owner_repo}")

    # Must contain exactly one slash
    if owner_repo.count('/') != 1:
        raise ValueError(
            f"Repository must be in format 'owner/repo': {owner_repo}"
        )

    return owner_repo
