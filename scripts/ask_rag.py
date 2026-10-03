"""Ask questions about the indexed filings from the terminal.

Run from the project root:
    python scripts/ask_rag.py "What were Apple's total net sales in fiscal 2025?"
    python scripts/ask_rag.py                          # interactive: ask several questions
    python scripts/ask_rag.py --show-sources "..."     # also print retrieved passages + scores
    python scripts/ask_rag.py --ticker RELIANCE.NS "What were total borrowings?"

Companies named in a question (Apple, Reliance, Toyota, or their tickers) are
detected automatically; --ticker overrides that.
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402

from config.settings import settings  # noqa: E402
from src.rag.qa import Answer, answer_question  # noqa: E402
from src.rag.retriever import DEFAULT_K, MAX_K, Retriever  # noqa: E402

warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
console = Console()


def show(answer: Answer, show_sources: bool) -> None:
    scope = ", ".join(answer.tickers) if answer.tickers else "all companies"
    console.print(f"[dim]Searched: {scope} | {len(answer.hits)} passages | model: {settings.default_model}[/dim]")
    console.print(Panel(Markdown(answer.text), title="Answer", border_style="green" if not answer.not_found else "yellow"))

    if answer.hits:
        console.print("[bold]Sources[/bold]")
        for i, hit in enumerate(answer.hits, start=1):
            used = "[green]cited[/green]" if i in answer.cited else "[dim]not cited[/dim]"
            console.print(f"  [S{i}] {hit.citation()}  score {hit.score:.3f}  {used}")
            if show_sources:
                console.print(f"[dim]{hit.text[:700]}[/dim]\n")

    for warning in answer.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")


def ask(question: str, args: argparse.Namespace, retriever: Retriever) -> None:
    try:
        answer = answer_question(question, ticker=args.ticker, k=args.k,
                                 min_score=args.min_score, retriever=retriever)
    except ValueError as e:  # bad ticker, empty question, ...
        console.print(f"[red]{e}[/red]")
        return
    except Exception as e:
        text = str(e)
        if "PerDay" in text:
            console.print("[red]Daily Gemini quota reached (each question uses one embedding). "
                          "It resets around midnight Pacific time.[/red]")
        elif "429" in text or "RESOURCE_EXHAUSTED" in text:
            console.print("[red]Rate limited by the API. Wait a minute and try again.[/red]")
        else:
            console.print(f"[red]{type(e).__name__}: {text[:300]}[/red]")
        return
    show(answer, args.show_sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="question to ask (omit for interactive mode)")
    parser.add_argument("--ticker", help="restrict to one company, e.g. AAPL, RELIANCE.NS, 7203.T")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help=f"passages to retrieve (1-{MAX_K})")
    parser.add_argument("--min-score", type=float, default=None, help="drop passages below this similarity")
    parser.add_argument("--show-sources", action="store_true", help="print the retrieved passages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        retriever = Retriever()  # opens the index once for the whole session
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    if args.question:
        ask(" ".join(args.question), args, retriever)
        return 0

    console.print("[bold]Ask about the filings[/bold] (empty line or 'exit' to quit)")
    while True:
        try:
            question = console.input("\n[cyan]> [/cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"", "exit", "quit"}:
            break
        ask(question, args, retriever)
    return 0


if __name__ == "__main__":
    sys.exit(main())
