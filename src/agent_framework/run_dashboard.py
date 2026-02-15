"""Entry point for running the web dashboard as a subprocess."""

import logging
import sys
from pathlib import Path

from .web.server import run_dashboard_server


def setup_logging():
    """Setup logging for the dashboard subprocess."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - dashboard - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """Main entry point for dashboard subprocess.

    Usage: python -m agent_framework.run_dashboard [port]
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    except ValueError:
        logger.error(f"Invalid port: {sys.argv[1]}")
        sys.exit(1)
    workspace = Path.cwd()

    logger.info(f"Starting dashboard on port {port}")

    try:
        run_dashboard_server(
            workspace=workspace,
            port=port,
            open_browser=False,
        )
    except KeyboardInterrupt:
        logger.info("Dashboard interrupted, shutting down")
    except Exception as e:
        logger.exception(f"Dashboard crashed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
