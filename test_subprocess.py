#!/usr/bin/env python3
"""Test script to verify Claude CLI subprocess fixes."""

import asyncio
import logging
import sys
from pathlib import Path

# Configure logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(name)s - %(message)s')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_framework.llm.claude_cli_backend import ClaudeCLIBackend
from agent_framework.llm.base import LLMRequest
from agent_framework.core.task import TaskType

async def test_subprocess():
    """Test Claude CLI subprocess with MCP."""
    print("=" * 60)
    print("Testing Claude CLI Subprocess")
    print("=" * 60)

    # Create backend with MCP enabled
    backend = ClaudeCLIBackend(
        executable="claude",
        max_turns=3,
        cheap_model="claude-haiku-4-5-20251001",
        default_model="claude-sonnet-4-5-20250929",
        premium_model="claude-sonnet-4-5-20250929",
        mcp_config_path="/Users/hju/PycharmProjects/agent-framework/config/mcp-config.json",
        timeout=60,
        logs_dir=Path("logs"),
    )

    # Create a simple test request
    request = LLMRequest(
        prompt="Please respond with exactly: 'Subprocess test successful!'",
        task_type=TaskType.DOCUMENTATION,
        model=None,  # Use default
        retry_count=0,
    )

    print("\n📝 Sending test request...")
    print(f"   Prompt: {request.prompt}")
    print(f"   Task type: {request.task_type}")
    print()

    # Make the request
    response = await backend.complete(request, task_id="test-subprocess-1")

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"✓ Success: {response.success}")
    print(f"✓ Model: {response.model_used}")
    print(f"✓ Latency: {response.latency_ms:.0f}ms")
    print(f"✓ Finish reason: {response.finish_reason}")

    if response.success:
        print(f"\n✅ Response:")
        print(f"   {response.content[:200]}")
    else:
        print(f"\n❌ Error:")
        print(f"   {response.error}")

    print("\n" + "=" * 60)

    return response.success

if __name__ == "__main__":
    success = asyncio.run(test_subprocess())
    sys.exit(0 if success else 1)
