"""Unit tests for VenvManager."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent_framework.workspace.venv_manager import VenvManager


@pytest.fixture
def venv_mgr():
    return VenvManager()


class TestIsPythonProject:
    """Tests for _is_python_project detection."""

    def test_detects_setup_py(self, venv_mgr, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")
        assert venv_mgr._is_python_project(tmp_path) is True

    def test_detects_pyproject_with_build_system(self, venv_mgr, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
        )
        assert venv_mgr._is_python_project(tmp_path) is True

    def test_rejects_pyproject_without_build_system(self, venv_mgr, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'foo'\n")
        assert venv_mgr._is_python_project(tmp_path) is False

    def test_detects_setup_cfg_with_metadata(self, venv_mgr, tmp_path):
        (tmp_path / "setup.cfg").write_text("[metadata]\nname = foo\nversion = 1.0\n")
        assert venv_mgr._is_python_project(tmp_path) is True

    def test_rejects_setup_cfg_without_metadata(self, venv_mgr, tmp_path):
        (tmp_path / "setup.cfg").write_text("[options]\ninstall_requires = requests\n")
        assert venv_mgr._is_python_project(tmp_path) is False

    def test_rejects_bare_requirements_txt(self, venv_mgr, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests\nflask\n")
        assert venv_mgr._is_python_project(tmp_path) is False

    def test_rejects_empty_directory(self, venv_mgr, tmp_path):
        assert venv_mgr._is_python_project(tmp_path) is False

    def test_detects_pyproject_when_setup_cfg_lacks_metadata(self, venv_mgr, tmp_path):
        """setup.cfg without [metadata] doesn't short-circuit past valid pyproject.toml."""
        (tmp_path / "setup.cfg").write_text("[options]\ninstall_requires = requests\n")
        (tmp_path / "pyproject.toml").write_text(
            '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n'
        )
        assert venv_mgr._is_python_project(tmp_path) is True

    def test_rejects_non_python_project(self, venv_mgr, tmp_path):
        (tmp_path / "go.mod").write_text("module github.com/example/foo")
        (tmp_path / "package.json").write_text('{"name": "foo"}')
        assert venv_mgr._is_python_project(tmp_path) is False


class TestVenvIsValid:
    """Tests for _venv_is_valid check."""

    def test_valid_venv(self, venv_mgr, tmp_path):
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")
        assert venv_mgr._venv_is_valid(venv) is True

    def test_missing_venv(self, venv_mgr, tmp_path):
        assert venv_mgr._venv_is_valid(tmp_path / ".venv") is False

    def test_venv_without_python(self, venv_mgr, tmp_path):
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        assert venv_mgr._venv_is_valid(venv) is False


class TestBuildEnvVars:
    """Tests for _build_env_vars output."""

    def test_env_vars_format(self, venv_mgr, tmp_path):
        venv = tmp_path / ".venv"
        result = venv_mgr._build_env_vars(venv)

        assert result["VIRTUAL_ENV"] == str(venv)
        assert result["PATH"].startswith(str(venv / "bin") + ":")
        # Original PATH is appended
        assert os.environ.get("PATH", "") in result["PATH"]

    def test_env_vars_has_exactly_two_keys(self, venv_mgr, tmp_path):
        result = venv_mgr._build_env_vars(tmp_path / ".venv")
        assert set(result.keys()) == {"VIRTUAL_ENV", "PATH"}


class TestSetupVenvIdempotency:
    """Tests for setup_venv idempotency and skip behavior."""

    def test_skips_non_python_project(self, venv_mgr, tmp_path):
        result = venv_mgr.setup_venv(tmp_path)
        assert result is None

    def test_reuses_existing_valid_venv(self, venv_mgr, tmp_path):
        """Existing valid venv is reused without creating a new one."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")
        venv = tmp_path / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\n")

        with patch.object(venv_mgr, "_create_venv") as mock_create:
            result = venv_mgr.setup_venv(tmp_path)

        mock_create.assert_not_called()
        assert result is not None
        assert result["VIRTUAL_ENV"] == str(venv)

    @patch("agent_framework.workspace.venv_manager.run_command")
    def test_creates_and_installs_on_fresh_project(self, mock_run_cmd, venv_mgr, tmp_path):
        """Fresh Python project gets venv created and project installed."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        # After run_command for venv creation, make the venv "exist"
        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "-m" in cmd and "venv" in cmd:
                venv = tmp_path / ".venv"
                (venv / "bin").mkdir(parents=True, exist_ok=True)
                (venv / "bin" / "python").write_text("#!/bin/sh\n")
                (venv / "bin" / "pip").write_text("#!/bin/sh\n")
            return MagicMock(returncode=0)

        mock_run_cmd.side_effect = side_effect
        result = venv_mgr.setup_venv(tmp_path)

        assert result is not None
        assert "VIRTUAL_ENV" in result
        # At least 2 calls: venv creation + pip install
        assert mock_run_cmd.call_count >= 2


class TestSetupVenvErrorHandling:
    """Tests for setup_venv error handling (non-fatal behavior)."""

    @patch("agent_framework.workspace.venv_manager.run_command")
    def test_returns_none_on_venv_creation_failure(self, mock_run_cmd, venv_mgr, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")
        from agent_framework.utils.subprocess_utils import SubprocessError
        mock_run_cmd.side_effect = SubprocessError("venv", 1, "failed")

        result = venv_mgr.setup_venv(tmp_path)
        assert result is None

    @patch("agent_framework.workspace.venv_manager.run_command")
    def test_returns_env_vars_on_install_failure(self, mock_run_cmd, venv_mgr, tmp_path):
        """pip install failure is non-fatal — venv still provides isolated Python."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")
        from agent_framework.utils.subprocess_utils import SubprocessError

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "-m" in cmd and "venv" in cmd:
                # Venv creation succeeds
                venv = tmp_path / ".venv"
                (venv / "bin").mkdir(parents=True, exist_ok=True)
                (venv / "bin" / "python").write_text("#!/bin/sh\n")
                (venv / "bin" / "pip").write_text("#!/bin/sh\n")
                return MagicMock(returncode=0)
            # pip install fails
            raise SubprocessError("pip", 1, "install failed")

        mock_run_cmd.side_effect = side_effect
        result = venv_mgr.setup_venv(tmp_path)

        # Should still return env vars despite install failure
        assert result is not None
        assert "VIRTUAL_ENV" in result

    def test_returns_none_on_unexpected_exception(self, venv_mgr, tmp_path):
        """Unexpected exceptions are caught and return None."""
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")

        with patch.object(venv_mgr, "_create_venv", side_effect=RuntimeError("boom")):
            result = venv_mgr.setup_venv(tmp_path)

        assert result is None


class TestSetupVenvWithRequirementsTxt:
    """Tests that requirements.txt is included in pip install when present."""

    @patch("agent_framework.workspace.venv_manager.run_command")
    def test_includes_requirements_txt(self, mock_run_cmd, venv_mgr, tmp_path):
        (tmp_path / "setup.py").write_text("from setuptools import setup\nsetup()")
        (tmp_path / "requirements.txt").write_text("requests\n")

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if "-m" in cmd and "venv" in cmd:
                venv = tmp_path / ".venv"
                (venv / "bin").mkdir(parents=True, exist_ok=True)
                (venv / "bin" / "python").write_text("#!/bin/sh\n")
                (venv / "bin" / "pip").write_text("#!/bin/sh\n")
            return MagicMock(returncode=0)

        mock_run_cmd.side_effect = side_effect
        venv_mgr.setup_venv(tmp_path)

        # Find the pip install call
        pip_calls = [
            c for c in mock_run_cmd.call_args_list
            if any("pip" in str(a) for a in c[0])
        ]
        assert len(pip_calls) == 1
        pip_cmd = pip_calls[0][0][0]
        assert "-r" in pip_cmd
        assert "requirements.txt" in pip_cmd
