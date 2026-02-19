"""Per-worktree virtualenv manager for Python project isolation.

Agents work in isolated git worktrees. When the engineer creates new Python
modules, imports fail because the global editable install points elsewhere.
This module creates a .venv/ inside each worktree so `pip install -e .` makes
all new modules importable. The venv's PATH is injected into the Claude CLI
subprocess env — no LLM cooperation needed.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from ..utils.subprocess_utils import run_command

logger = logging.getLogger(__name__)


class VenvManager:
    """Creates and manages per-worktree Python virtualenvs."""

    VENV_DIR = ".venv"

    def setup_venv(self, worktree_path: Path) -> Optional[Dict[str, str]]:
        """Set up a virtualenv in the worktree if it's an installable Python project.

        Returns env vars dict to inject into subprocess, or None if not applicable.
        Idempotent: reuses existing valid venv. Non-fatal: returns None on any error.
        """
        try:
            if not self._is_python_project(worktree_path):
                return None

            venv_path = worktree_path / self.VENV_DIR

            if self._venv_is_valid(venv_path):
                logger.debug(f"Reusing existing venv: {venv_path}")
                return self._build_env_vars(venv_path)

            if not self._create_venv(venv_path):
                return None

            # Install failure is non-fatal — the venv still provides isolated Python
            self._install_project(worktree_path, venv_path)

            return self._build_env_vars(venv_path)

        except Exception as e:
            logger.warning(f"Venv setup failed for {worktree_path}: {e}")
            return None

    def _is_python_project(self, path: Path) -> bool:
        """Detect installable Python projects (not bare requirements.txt)."""
        if (path / "setup.py").exists():
            return True

        if (path / "setup.cfg").exists() and self._setup_cfg_has_metadata(path / "setup.cfg"):
            return True

        pyproject = path / "pyproject.toml"
        if pyproject.exists() and self._pyproject_has_build_system(pyproject):
            return True

        return False

    def _setup_cfg_has_metadata(self, setup_cfg: Path) -> bool:
        """Check if setup.cfg has [metadata] section (installable package)."""
        try:
            import configparser
            config = configparser.ConfigParser()
            config.read(str(setup_cfg))
            return config.has_section("metadata")
        except Exception:
            return False

    def _pyproject_has_build_system(self, pyproject: Path) -> bool:
        """Check if pyproject.toml has [build-system] section."""
        try:
            import tomllib
            with open(pyproject, "rb") as f:
                data = tomllib.load(f)
            return "build-system" in data
        except Exception:
            return False

    def _venv_is_valid(self, venv_path: Path) -> bool:
        """Check if an existing venv has a working Python interpreter."""
        return (venv_path / "bin" / "python").exists()

    def _create_venv(self, venv_path: Path) -> bool:
        """Create a new virtualenv. Returns True on success."""
        try:
            logger.info(f"Creating virtualenv: {venv_path}")
            run_command(
                [sys.executable, "-m", "venv", str(venv_path)],
                check=True,
                timeout=60,
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to create venv: {e}")
            return False

    def _install_project(self, worktree_path: Path, venv_path: Path) -> bool:
        """Run pip install -e . (and requirements.txt if present). Returns True on success."""
        pip = str(venv_path / "bin" / "pip")

        try:
            cmd = [pip, "install", "-e", "."]
            if (worktree_path / "requirements.txt").exists():
                cmd.extend(["-r", "requirements.txt"])

            logger.info(f"Installing project in editable mode: {worktree_path}")
            run_command(cmd, cwd=worktree_path, check=True, timeout=120)
            return True
        except Exception as e:
            logger.warning(f"pip install -e . failed (venv still usable): {e}")
            return False

    def _build_env_vars(self, venv_path: Path) -> Dict[str, str]:
        """Build env vars that activate the venv for child processes."""
        venv_bin = str(venv_path / "bin")
        original_path = os.environ.get("PATH", "")
        return {
            "VIRTUAL_ENV": str(venv_path),
            "PATH": f"{venv_bin}:{original_path}",
        }
