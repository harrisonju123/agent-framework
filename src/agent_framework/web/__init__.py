"""Web dashboard package for agent framework."""

from .server import run_dashboard_server
from .data_provider import DashboardDataProvider

__all__ = ["run_dashboard_server", "DashboardDataProvider"]
