"""Integration test: MCP stdio server via the official client SDK."""

import asyncio
import json
import sys

import pytest

pytestmark = pytest.mark.smoke


async def _check():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-u", "-m", "oneclick_distill.mcp.server"], cwd=".", env=None)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            assert names == ["hardware", "start_pipeline", "pipeline_status"], names
            res = await session.call_tool("hardware", {})
            hw = json.loads(res.content[0].text)
            assert hw["device"] in ("cpu", "cuda")
            return names


def test_mcp_server():
    names = asyncio.run(_check())
    assert "hardware" in names
