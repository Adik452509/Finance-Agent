"""An MCP server that crashes on startup, to test that the toolbox survives it."""

raise RuntimeError("simulated startup failure (e.g. missing dependency or bad config)")
