"""Start the configured MCP servers and discover their tools automatically.

    async with MCPToolbox() as toolbox:
        agent = create_react_agent(llm, toolbox.tools)
        ...

Servers are listed in config/mcp_servers.json. Adding a data source means
adding one entry there - no agent code changes. That's the point of MCP.

Design choices:
  - One long-lived session per server for the toolbox's lifetime. The
    adapter's convenience `get_tools()` spawns a fresh server process for
    EVERY tool call (slow, and it throws away each server's cache).
  - A server that fails to start is skipped and reported; the remaining
    tools still work. One broken data source must not take down the agent.
  - Config can only launch Python modules from allowed packages, run with
    this venv's interpreter. It is not a way to execute arbitrary commands.
  - Enter and exit the toolbox in the same asyncio task (anyio requirement).
"""

import json
import logging
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = settings.project_root / "config" / "mcp_servers.json"
ALLOWED_MODULE_PREFIXES: tuple[str, ...] = ("src.mcp_servers.",)
# A tool call that takes longer than this fails instead of hanging the agent.
TOOL_TIMEOUT_SECONDS = 90

SERVER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,30}$")
MODULE_RE = re.compile(r"^[A-Za-z_]\w*(\.[A-Za-z_]\w*)*$")


@dataclass(frozen=True)
class ServerSpec:
    name: str
    module: str
    description: str = ""
    enabled: bool = True


def load_server_specs(
    path: Path | None = None,
    allowed_prefixes: tuple[str, ...] = ALLOWED_MODULE_PREFIXES,
) -> list[ServerSpec]:
    """Read and validate the server config; return the enabled servers."""
    path = path or DEFAULT_CONFIG
    raw = json.loads(path.read_text(encoding="utf-8"))
    servers = raw.get("servers")
    if not isinstance(servers, dict) or not servers:
        raise ValueError(f"{path.name}: expected a non-empty 'servers' object")

    specs: list[ServerSpec] = []
    for name, entry in servers.items():
        if not SERVER_NAME_RE.match(name):
            raise ValueError(f"{path.name}: invalid server name {name!r} (lowercase letters, digits, _)")
        module = entry.get("module", "")
        if not MODULE_RE.match(module):
            raise ValueError(f"{path.name}: server {name!r} has an invalid module {module!r}")
        if not module.startswith(allowed_prefixes):
            raise ValueError(
                f"{path.name}: server {name!r} module {module!r} is outside the allowed packages "
                f"{', '.join(allowed_prefixes)}"
            )
        spec = ServerSpec(name, module, str(entry.get("description", "")), bool(entry.get("enabled", True)))
        if spec.enabled:
            specs.append(spec)
        else:
            logger.info("Server %r is disabled in %s", name, path.name)
    return specs


def build_connection(spec: ServerSpec) -> dict:
    """stdio connection: run the module with this venv's Python from the project root."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", spec.module],
        "cwd": str(settings.project_root),
        "encoding": "utf-8",  # figures include ₹ and ¥
        "session_kwargs": {"read_timeout_seconds": timedelta(seconds=TOOL_TIMEOUT_SECONDS)},
    }


class MCPToolbox:
    """Async context manager holding live sessions and the discovered tools."""

    def __init__(self, specs: list[ServerSpec] | None = None) -> None:
        self.specs = specs if specs is not None else load_server_specs()
        self.tools: list[BaseTool] = []
        self.tools_by_server: dict[str, list[str]] = {}
        self.failed: dict[str, str] = {}
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self) -> "MCPToolbox":
        self._stack = AsyncExitStack()
        client = MultiServerMCPClient({s.name: build_connection(s) for s in self.specs})
        try:
            for spec in self.specs:
                try:
                    session = await self._stack.enter_async_context(client.session(spec.name))
                    tools = await load_mcp_tools(session, server_name=spec.name)
                except Exception as e:  # crashed on start, bad import, protocol error...
                    self.failed[spec.name] = f"{type(e).__name__}: {str(e)[:200]}"
                    logger.warning("MCP server %r unavailable: %s", spec.name, self.failed[spec.name])
                    continue
                self.tools.extend(tools)
                self.tools_by_server[spec.name] = [t.name for t in tools]

            self._check_unique_names()
        except BaseException:
            await self._stack.aclose()
            raise
        return self

    async def __aexit__(self, *exc_info) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    def _check_unique_names(self) -> None:
        """Two servers exposing the same tool name would make the agent's choice ambiguous."""
        owners: dict[str, str] = {}
        for server, names in self.tools_by_server.items():
            for name in names:
                if name in owners:
                    raise ValueError(f"Tool name {name!r} is exposed by both {owners[name]!r} and {server!r}")
                owners[name] = server

    def summary(self) -> str:
        lines = []
        for spec in self.specs:
            if spec.name in self.failed:
                lines.append(f"{spec.name}: UNAVAILABLE ({self.failed[spec.name]})")
            else:
                lines.append(f"{spec.name}: {', '.join(self.tools_by_server.get(spec.name, []))}")
        return "\n".join(lines)
