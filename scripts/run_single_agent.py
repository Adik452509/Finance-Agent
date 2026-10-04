"""Ask the Phase 2 single agent a question; watch it choose and call MCP tools live.

Run from the project root:
    python scripts/run_single_agent.py "What is Toyota's current share price?"
    python scripts/run_single_agent.py                    # interactive mode
    python scripts/run_single_agent.py --role reasoning "..."   # stronger model

Starts every server in config/mcp_servers.json, discovers their tools, and gives
them all to one agent. The live trace shows which tool it picked and with what arguments.
"""

import argparse
import asyncio
import logging
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402

from config.llm import get_chat_model  # noqa: E402
from config.settings import settings  # noqa: E402
from src.agents.single_agent import AgentRun, ToolCall, build_single_agent, run_agent  # noqa: E402
from src.mcp_client.client import MCPToolbox  # noqa: E402

warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
console = Console()


def on_event(kind: str, call: ToolCall) -> None:
    if kind == "call":
        args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
        console.print(f"  [cyan]→ {call.name}[/cyan]({args})")
    elif kind == "result":
        status = "[green]ok[/green]" if call.ok else ("[red]error[/red]" if call.ok is False else "done")
        extra = f" - {call.error[:100]}" if call.error else ""
        console.print(f"    [dim]← {status} in {call.seconds:.1f}s{extra}[/dim]")


def explain_error(error: Exception) -> str:
    """Say WHICH limit was hit: per-minute limits clear quickly, daily ones don't."""
    text = str(error)
    if "429" not in text and "rate limit" not in text.lower():
        return f"{type(error).__name__}: {text[:300]}"
    wait = re.search(r"try again in ([\w.]+)", text)
    when = f" Try again in {wait.group(1)}." if wait else ""
    if "per day" in text or "TPD" in text or "RPD" in text:
        return (f"Daily free-tier limit reached for this model.{when} "
                "Each model has its own daily budget: try --role reasoning (or the other way round).")
    return f"Per-minute rate limit hit.{when or ' Wait a minute and retry.'}"


async def ask(agent, question: str) -> None:
    console.print(f"\n[bold]Q:[/bold] {question}")
    try:
        run: AgentRun = await run_agent(agent, question, on_event)
    except Exception as e:
        console.print(f"[red]{explain_error(e)}[/red]")
        return
    console.print(Panel(Markdown(run.answer), title="Answer", border_style="yellow" if run.warnings else "green"))
    console.print(f"[dim]Tools used: {', '.join(run.tools_used) or 'none'}[/dim]")
    for w in run.warnings:
        console.print(f"[yellow]! {w}[/yellow]")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*")
    parser.add_argument("--role", choices=["default", "reasoning"], default="default")
    args = parser.parse_args()

    async with MCPToolbox() as toolbox:
        console.print(f"[dim]{toolbox.summary()}[/dim]")
        if toolbox.failed:
            console.print("[yellow]Some servers are unavailable; their tools are missing.[/yellow]")
        model = settings.reasoning_model if args.role == "reasoning" else settings.default_model
        console.print(f"[dim]model: {model}[/dim]")
        agent = build_single_agent(toolbox.tools, get_chat_model(args.role))

        if args.question:
            await ask(agent, " ".join(args.question))
            return 0
        console.print("[bold]Ask the research agent[/bold] (empty line or 'exit' to quit)")
        while True:
            try:
                question = (await asyncio.to_thread(console.input, "\n[cyan]> [/cyan]")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"", "exit", "quit"}:
                break
            await ask(agent, question)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(asyncio.run(main()))
