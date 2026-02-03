"""Entry point for running the watchdog as a subprocess."""

import asyncio
import logging
import sys
from pathlib import Path

from .core.config import load_config
from .safeguards.watchdog import Watchdog


def setup_logging(workspace: Path):
    """Setup logging for the watchdog."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - watchdog - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def main_async():
    """Main async entry point."""
    workspace = Path.cwd()
    setup_logging(workspace)
    logger = logging.getLogger(__name__)

    try:
        # Load config
        framework_config = load_config(workspace / "agent-framework.yaml")

        # Create watchdog
        watchdog = Watchdog(
            workspace=framework_config.workspace,
            heartbeat_timeout=framework_config.safeguards.heartbeat_timeout,
            check_interval=framework_config.safeguards.watchdog_interval,
        )

        logger.info("Starting watchdog")
        await watchdog.run()

    except KeyboardInterrupt:
        logger.info("Watchdog interrupted, shutting down")
    except Exception as e:
        logger.exception(f"Watchdog crashed: {e}")
        sys.exit(1)


def main():
    """Main entry point for watchdog subprocess."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
