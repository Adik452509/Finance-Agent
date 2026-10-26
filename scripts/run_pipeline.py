"""Run the Phase 4 multi-agent team on a question and watch every hand-off live.

Run from the project root:
    python scripts/run_pipeline.py "How much did Reliance's revenue grow from 2023-24 to 2024-25?"
    python scripts/run_pipeline.py                       # interactive mode
    python scripts/run_pipeline.py --all-reasoning "..." # every agent on the stronger model

Shows: the plan, each tool call, the researcher's fact notes, the supervisor's
routing decision, the analysis, the final report, and tokens used per agent.
"""

import argparse
import asyncio
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.markup import escape  # noqa: E402  (agent text contains [brackets] rich would eat)
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from config.llm import get_chat_model  # noqa: E402
from config.settings import settings  # noqa: E402
from run_single_agent import explain_error  # noqa: E402  (scripts/ is on sys.path when run directly)
from src.agents.graph import DEFAULT_ROLES, build_graph  # noqa: E402
from src.agents.state import add_usage  # noqa: E402
from src.mcp_client.client import MCPToolbox  # noqa: E402

warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
console = Console()
TOKEN_LIMIT = 8_000  # Groq free tier, per request


def on_event(kind: str, payload: dict) -> None:
    if kind == "call":
        args = escape(", ".join(f"{k}={v!r}" for k, v in payload["args"].items()))
        console.print(f"    [cyan]→ {payload['name']}[/cyan]({args})")
    elif kind == "result":
        status = "[green]ok[/green]" if payload["ok"] else ("[red]error[/red]" if payload["ok"] is False else "done")
        extra = f" - {escape(payload['error'][:90])}" if payload.get("error") else ""
        console.print(f"    [dim]← {status} in {payload['seconds']}s{extra}[/dim]")


def show_update(node: str, update: dict) -> None:
    if node == "planner":
        plan = update["plan"]
        console.print("[bold magenta]PLANNER[/bold magenta]")
        for s in plan["steps"]:
            ticker = f"  ({s['ticker']})" if s.get("ticker") else ""
            console.print(escape(f"  {s['id']}. [{s['kind']}] {s['goal']}{ticker}"))
        if plan["needs_calculation"]:
            console.print(escape(f"  calculation: {plan['calculation'] or '(yes)'}"))
    elif node == "researcher":
        console.print("[bold magenta]RESEARCHER - fact notes[/bold magenta]")
        for f in update["findings"]:
            console.print(f"  [bold]Step {f['step_id']}[/bold]")
            console.print(escape("\n".join(f"    {line}" for line in f["facts"].splitlines())))
    elif node == "supervisor":
        console.print(f"[bold magenta]SUPERVISOR[/bold magenta] → {update['route']}")
    elif node == "analyst":
        console.print("[bold magenta]ANALYST[/bold magenta]")
        console.print(escape("  " + (update["analysis"] or "(empty)").replace("\n", "\n  ")))
    elif node == "writer":
        revision = update.get("revisions")
        label = f"revision {revision}" if revision else "first draft"
        console.print(f"[bold magenta]WRITER[/bold magenta] → {label} ({len(update.get('draft', ''))} characters)")
    elif node == "critic":
        critique = update.get("critique", {})
        by = "+".join(critique.get("checked_by", []))
        if critique.get("verdict") == "pass":
            console.print(f"[bold magenta]CRITIC[/bold magenta] (round {update.get('critic_rounds')}, {by}) → [green]pass[/green]")
        else:
            console.print(f"[bold magenta]CRITIC[/bold magenta] (round {update.get('critic_rounds')}, {by}) → [red]fail[/red]")
            for issue in critique.get("issues", []):
                console.print(escape(f"    - {issue.get('claim', '')}: {issue.get('problem', '')}"))
    elif node == "finalize":
        console.print(Panel(Markdown(update["report"]), title="REPORT", border_style="green"))
    for w in update.get("warnings", []):
        console.print(f"  [yellow]! {escape(w)}[/yellow]")


def show_usage(usage: dict, roles: dict, seconds: float) -> None:
    table = Table(title=f"Tokens per agent ({seconds:.0f}s)")
    for col in ("agent", "model", "tokens"):
        table.add_column(col)
    for agent in DEFAULT_ROLES:
        if agent in usage:
            model = settings.reasoning_model if roles[agent] == "reasoning" else settings.default_model
            table.add_row(agent, model, f"{usage[agent]:,}")
    table.add_row("[bold]total[/bold]", "", f"[bold]{sum(usage.values()):,}[/bold]")
    console.print(table)


async def ask(graph, roles: dict, question: str) -> None:
    console.rule(f"[bold]{question}")
    usage, started = {}, time.monotonic()
    try:
        async for update in graph.astream({"question": question}, stream_mode="updates"):
            for node, data in update.items():
                show_update(node, data or {})
                usage = add_usage(usage, (data or {}).get("usage", {}))
    except Exception as e:
        console.print(f"[red]{explain_error(e)}[/red]")
        return
    show_usage(usage, roles, time.monotonic() - started)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*")
    parser.add_argument("--all-reasoning", action="store_true", help="put every agent on the stronger model")
    args = parser.parse_args()

    roles = {a: "reasoning" for a in DEFAULT_ROLES} if args.all_reasoning else dict(DEFAULT_ROLES)
    async with MCPToolbox() as toolbox:
        console.print(f"[dim]{toolbox.summary()}[/dim]")
        llms = {agent: get_chat_model(role) for agent, role in roles.items()}
        graph = build_graph(toolbox.tools, llms, on_event=on_event)

        if args.question:
            await ask(graph, roles, " ".join(args.question))
            return 0
        console.print("[bold]Ask the research team[/bold] (empty line or 'exit' to quit)")
        while True:
            try:
                question = (await asyncio.to_thread(console.input, "\n[cyan]> [/cyan]")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"", "exit", "quit"}:
                break
            await ask(graph, roles, question)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(asyncio.run(main()))
