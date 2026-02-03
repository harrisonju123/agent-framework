"""Docker container executor for isolated test runs."""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import docker
from docker.errors import ContainerError, ImageNotFound, APIError

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of a command execution in a container."""
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float


class DockerExecutor:
    """Execute commands in isolated Docker containers.

    Provides a sandboxed environment for running tests without affecting
    the host system. Handles container lifecycle, volume mounting, and
    output capture.
    """

    def __init__(
        self,
        image: str = "golang:1.22",
        timeout: int = 600,  # 10 minute default
        memory_limit: str = "2g",
        cpu_limit: float = 2.0,
    ):
        """Initialize Docker executor.

        Args:
            image: Docker image to use for execution
            timeout: Maximum execution time in seconds
            memory_limit: Container memory limit (e.g., "2g", "512m")
            cpu_limit: Number of CPUs to allocate
        """
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._client: Optional[docker.DockerClient] = None

    @property
    def client(self) -> docker.DockerClient:
        """Lazy-initialize Docker client."""
        if self._client is None:
            try:
                self._client = docker.from_env()
                # Verify connection
                self._client.ping()
            except docker.errors.DockerException as e:
                raise RuntimeError(
                    f"Failed to connect to Docker daemon. Is Docker running? {e}"
                )
        return self._client

    def ensure_image(self) -> None:
        """Pull the Docker image if not present."""
        try:
            self.client.images.get(self.image)
            logger.debug(f"Image {self.image} already present")
        except ImageNotFound:
            logger.info(f"Pulling image: {self.image}")
            self.client.images.pull(self.image)
            logger.info(f"Image {self.image} pulled successfully")

    def run_command(
        self,
        command: str,
        repo_path: Path,
        working_dir: str = "/workspace",
        env: Optional[dict] = None,
    ) -> ExecutionResult:
        """Run a command in a container with the repo mounted.

        Args:
            command: Command to execute (e.g., "go test ./...")
            repo_path: Local path to mount as /workspace
            working_dir: Working directory inside container
            env: Environment variables to set

        Returns:
            ExecutionResult with exit code, stdout, stderr, and duration
        """
        import time

        self.ensure_image()

        # Validate repo path
        repo_path = Path(repo_path).resolve()
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")

        # Build container config
        volumes = {
            str(repo_path): {"bind": working_dir, "mode": "rw"}
        }

        environment = env or {}

        # Go-specific environment
        if "golang" in self.image.lower():
            environment.setdefault("GOPROXY", "https://proxy.golang.org,direct")
            environment.setdefault("GOSUMDB", "sum.golang.org")

        container = None
        start_time = time.time()

        try:
            logger.info(f"Running command in container: {command}")

            container = self.client.containers.run(
                self.image,
                command=f"/bin/sh -c '{command}'",
                volumes=volumes,
                working_dir=working_dir,
                environment=environment,
                mem_limit=self.memory_limit,
                nano_cpus=int(self.cpu_limit * 1e9),
                detach=True,
                remove=False,  # Don't auto-remove, we need to get logs
            )

            # Wait for completion with timeout
            result = container.wait(timeout=self.timeout)
            exit_code = result.get("StatusCode", -1)

            # Get logs
            stdout = container.logs(stdout=True, stderr=False).decode("utf-8", errors="replace")
            stderr = container.logs(stdout=False, stderr=True).decode("utf-8", errors="replace")

            duration = time.time() - start_time

            logger.info(
                f"Command completed: exit_code={exit_code}, duration={duration:.1f}s"
            )

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_seconds=duration,
            )

        except ContainerError as e:
            duration = time.time() - start_time
            logger.error(f"Container error: {e}")
            return ExecutionResult(
                exit_code=e.exit_status,
                stdout="",
                stderr=str(e),
                duration_seconds=duration,
            )

        except APIError as e:
            duration = time.time() - start_time
            logger.error(f"Docker API error: {e}")
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=f"Docker API error: {e}",
                duration_seconds=duration,
            )

        finally:
            # Clean up container
            if container:
                try:
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove container: {e}")

    def run_tests(
        self,
        repo_path: Path,
        test_cmd: str = "go test ./...",
        env: Optional[dict] = None,
    ) -> ExecutionResult:
        """Convenience method for running tests.

        Args:
            repo_path: Path to the repository
            test_cmd: Test command to run
            env: Additional environment variables

        Returns:
            ExecutionResult with test output
        """
        return self.run_command(test_cmd, repo_path, env=env)

    def health_check(self) -> bool:
        """Check if Docker is available and working."""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.warning(f"Docker health check failed: {e}")
            return False
