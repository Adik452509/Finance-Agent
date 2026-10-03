"""Second tiny MCP server for tests: stands in for "a newly added data source"."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("fixture-math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run()
