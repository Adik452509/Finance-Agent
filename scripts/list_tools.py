"""Start every configured MCP server and show the tools it exposes.

Run from the project root:
    python scripts/list_tools.py

This is exactly what the agent sees: names, descriptions and argument schemas,
discovered at runtime. Add a server to config/mcp_servers.json and it shows up
here with no code changes.
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402

from src.mcp_client.client import MCPToolbox  # noqa: E402

console = Console()


async def main() -> int:
    async with MCPToolbox() as toolbox:
        for spec in toolbox.specs:
            if spec.name in toolbox.failed:
                console.print(f"\n[red]{spec.name}[/red]: UNAVAILABLE - {toolbox.failed[spec.name]}")
                continue
            console.print(f"\n[bold green]{spec.name}[/bold green]  [dim]{spec.module}[/dim]  {spec.description}")
            for tool in (t for t in toolbox.tools if t.name in toolbox.tools_by_server[spec.name]):
                args = tool.args_schema if isinstance(tool.args_schema, dict) else tool.args_schema.model_json_schema()
                params = ", ".join(
                    f"{p}{'' if p in args.get('required', []) else '?'}" for p in args.get("properties", {})
                )
                console.print(f"  [cyan]{tool.name}[/cyan]({params})")
                console.print(f"    [dim]{tool.description.strip().splitlines()[0]}[/dim]")
        console.print(f"\n{len(toolbox.tools)} tools from {len(toolbox.tools_by_server)} server(s)")
        return 1 if toolbox.failed else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(asyncio.run(main()))
