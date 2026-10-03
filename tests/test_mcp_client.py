"""MCP client: config validation, auto-discovery, plug-and-play, resilience.

The integration tests start real server processes (tests/mcp_fixture_*.py)
over stdio, so they take a few seconds - but use no network or API quota.
"""

import asyncio
import json
import sys

import pytest

from src.mcp_client.client import MCPToolbox, ServerSpec, build_connection, load_server_specs

TEST_PREFIXES = ("tests.mcp_fixture_",)
ECHO = ServerSpec("echo", "tests.mcp_fixture_echo")
MATH = ServerSpec("math", "tests.mcp_fixture_math")
BROKEN = ServerSpec("broken", "tests.mcp_fixture_broken")


def _write_config(tmp_path, servers: dict):
    path = tmp_path / "mcp_servers.json"
    path.write_text(json.dumps({"servers": servers}), encoding="utf-8")
    return path


def _discover(specs: list[ServerSpec]) -> MCPToolbox:
    async def run():
        async with MCPToolbox(specs) as toolbox:
            return toolbox
    return asyncio.run(run())


# --- config validation (no processes) -------------------------------------------------------


def test_project_config_lists_the_market_server():
    assert any(s.module == "src.mcp_servers.market_server" for s in load_server_specs())


def test_disabled_servers_are_skipped(tmp_path):
    path = _write_config(tmp_path, {
        "echo": {"module": "tests.mcp_fixture_echo"},
        "math": {"module": "tests.mcp_fixture_math", "enabled": False},
    })
    assert [s.name for s in load_server_specs(path, TEST_PREFIXES)] == ["echo"]


@pytest.mark.parametrize("module", ["os; rm -rf /", "../evil", "tests.mcp_fixture_echo --x", ""])
def test_malformed_module_rejected(tmp_path, module):
    with pytest.raises(ValueError, match="invalid module"):
        load_server_specs(_write_config(tmp_path, {"x1": {"module": module}}), TEST_PREFIXES)


@pytest.mark.parametrize("module", ["subprocess", "os", "src.ingestion.indexer"])
def test_modules_outside_allowed_packages_rejected(tmp_path, module):
    # Config must not become a way to run arbitrary code with -m.
    with pytest.raises(ValueError, match="outside the allowed packages"):
        load_server_specs(_write_config(tmp_path, {"x1": {"module": module}}))


def test_bad_server_name_rejected(tmp_path):
    with pytest.raises(ValueError, match="invalid server name"):
        load_server_specs(_write_config(tmp_path, {"Bad Name!": {"module": "tests.mcp_fixture_echo"}}), TEST_PREFIXES)


def test_connection_uses_this_venv_python_and_project_root():
    conn = build_connection(ECHO)
    assert conn["transport"] == "stdio"
    assert conn["command"] == sys.executable
    assert conn["args"] == ["-m", "tests.mcp_fixture_echo"]
    assert conn["cwd"].endswith("Finance agent")


# --- live discovery (real subprocesses over stdio) ---------------------------------------------


def test_tools_are_discovered_automatically():
    toolbox = _discover([ECHO])
    assert toolbox.tools_by_server == {"echo": ["echo", "server_pid"]}
    assert toolbox.failed == {}


def test_adding_a_server_adds_its_tools_with_no_code_change():
    # Same code path, one more config entry -> new tools appear.
    before = {t.name for t in _discover([ECHO]).tools}
    after = {t.name for t in _discover([ECHO, MATH]).tools}
    assert after - before == {"add"}


def test_tools_are_callable_and_session_is_reused():
    async def run():
        async with MCPToolbox([ECHO, MATH]) as toolbox:
            tools = {t.name: t for t in toolbox.tools}
            echoed = await tools["echo"].ainvoke({"text": "₹9,80,136 crore"})
            total = await tools["add"].ainvoke({"a": 2, "b": 3})
            pid1 = await tools["server_pid"].ainvoke({})
            pid2 = await tools["server_pid"].ainvoke({})
            return echoed, total, pid1, pid2
    echoed, total, pid1, pid2 = asyncio.run(run())
    assert echoed == "₹9,80,136 crore"  # non-ASCII survives the stdio round trip
    assert float(total) == 5.0
    assert pid1 == pid2, "each call must reuse the same server process, not spawn a new one"


def test_broken_server_is_reported_and_others_still_work():
    toolbox = _discover([ECHO, BROKEN, MATH])
    assert "broken" in toolbox.failed
    assert set(toolbox.tools_by_server) == {"echo", "math"}
    assert "UNAVAILABLE" in toolbox.summary()


def test_duplicate_tool_names_across_servers_rejected():
    with pytest.raises(ValueError, match="exposed by both"):
        _discover([ECHO, ServerSpec("echo_copy", "tests.mcp_fixture_echo")])
