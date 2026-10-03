"""Tiny MCP server used only by tests (not collected: no test_ prefix)."""

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fixture-echo")


@mcp.tool()
def echo(text: str) -> str:
    """Return the text unchanged."""
    return text


@mcp.tool()
def server_pid() -> int:
    """Process ID of this server - lets tests prove a session is reused."""
    return os.getpid()


if __name__ == "__main__":
    mcp.run()
