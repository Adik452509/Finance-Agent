"""The research team as a long-lived service: MCP servers started once, shared by every request.

Scripts open an MCPToolbox per run; a web backend must not - six server processes per
question would take seconds and gigabytes. Team opens them once and keeps them.

One task owns the MCP sessions (Team.run). The MCP client is built on anyio, which requires
a session to be closed by the same task that opened it, so a request never restarts the
servers itself: it calls request_reload() and the owner task does it.

Live tool events (the "→ search_filings(...)" lines) belong to one request, but the graph is
built once, so build_graph gets a single dispatcher that forwards each event to whichever
request is running in the current asyncio context (a ContextVar - each request's task has
its own copy).
"""

import asyncio
import logging
from collections.abc import Callable
from contextvars import ContextVar

logger = logging.getLogger(__name__)

EventSink = Callable[[str, dict], None]
current_sink: ContextVar[EventSink | None] = ContextVar("current_sink", default=None)


def dispatch_event(kind: str, payload: dict) -> None:
    sink = current_sink.get()
    if sink is not None:
        sink(kind, payload)


class Team:
    def __init__(self, toolbox_factory: Callable | None = None, graph_factory: Callable | None = None) -> None:
        self._toolbox_factory = toolbox_factory
        self._graph_factory = graph_factory
        self.graph = None
        self.tools: list = []
        self.failed: dict[str, str] = {}
        self.error: str | None = None   # the team could not start at all
        self.needs_reload = False        # a company was added: the planner's company list is stale
        self._ready = asyncio.Event()
        self._reload = asyncio.Event()
        self._stop = asyncio.Event()

    @property
    def ready(self) -> bool:
        return self.graph is not None

    async def run(self) -> None:
        """Owner task: open the servers, serve until asked to reload or stop, repeat."""
        from src.agents.graph import build_graph
        from src.mcp_client.client import MCPToolbox

        while not self._stop.is_set():
            self._reload.clear()
            try:
                async with (self._toolbox_factory or MCPToolbox)() as toolbox:
                    self.tools, self.failed = toolbox.tools, dict(toolbox.failed)
                    self.graph = (self._graph_factory or build_graph)(self.tools, on_event=dispatch_event)
                    self.error, self.needs_reload = None, False
                    logger.info("Research team ready: %d tools; unavailable: %s", len(self.tools), self.failed or "none")
                    self._ready.set()
                    await self._wait_for_reload_or_stop()
                    self.graph = None
            except Exception as e:  # e.g. a broken config: report it on /status instead of crashing the API
                self.graph, self.error = None, f"{type(e).__name__}: {str(e)[:300]}"
                logger.exception("Research team failed to start")
                self._ready.set()
                await self._wait_for_reload_or_stop()
            self._ready.clear()

    async def _wait_for_reload_or_stop(self) -> None:
        waiters = [asyncio.ensure_future(self._reload.wait()), asyncio.ensure_future(self._stop.wait())]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()

    async def wait_ready(self, timeout: float = 120) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
        except asyncio.TimeoutError:
            return False
        return self.ready

    def request_reload(self) -> None:
        self._ready.clear()
        self._reload.set()

    def request_stop(self) -> None:
        self._stop.set()
