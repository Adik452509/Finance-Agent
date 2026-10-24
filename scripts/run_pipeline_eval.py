"""Score the Phase 4 multi-agent team against tests/eval/pipeline_questions.json.

Run from the project root:
    python scripts/run_pipeline_eval.py                    # all questions (~90-110k Groq tokens)
    python scripts/run_pipeline_eval.py --only ril_        # just the questions whose id starts with ril_
    python scripts/run_pipeline_eval.py --all-reasoning    # every agent on the stronger model

Each question is checked for: the right figure, no known-wrong figures, the required
tools actually used, the supervisor's route, refusals where the answer doesn't exist,
and news citations. Tokens and warnings are reported per question.

Cost: one full run is roughly half of a model's free daily Groq budget (200k tokens per
model per day), so run it after a fresh budget, not straight after heavy testing.
Exits 0 only if every question passes.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.markup import escape  # noqa: E402
from rich.table import Table  # noqa: E402

from config.llm import get_chat_model  # noqa: E402
from config.settings import settings  # noqa: E402
from src.agents.graph import AGENTS, DEFAULT_ROLES, build_graph  # noqa: E402
from src.evaluation.checks import check_result  # noqa: E402
from src.mcp_client.client import MCPToolbox  # noqa: E402

warnings.filterwarnings("ignore")
console = Console()
QUESTIONS = settings.project_root / "tests" / "eval" / "pipeline_questions.json"
RESULTS = settings.data_dir / "cache" / "pipeline_eval_last.json"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="run only question ids starting with this prefix")
    parser.add_argument("--all-reasoning", action="store_true", help="put every agent on the stronger model")
    parser.add_argument("--delay", type=float, default=20.0, help="seconds between questions (8k tokens/minute limit)")
    args = parser.parse_args()

    questions = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"] if q["id"].startswith(args.only)]
    if not questions:
        console.print(f"[red]No questions match --only {args.only!r}[/red]")
        return 1
    roles = {a: "reasoning" for a in AGENTS} if args.all_reasoning else dict(DEFAULT_ROLES)

    rows, started = [], time.monotonic()
    async with MCPToolbox() as toolbox:
        if toolbox.failed:
            console.print(f"[yellow]Unavailable servers: {toolbox.failed}[/yellow]")
        graph = build_graph(toolbox.tools, {a: get_chat_model(r) for a, r in roles.items()})
        for i, q in enumerate(questions):
            if i:
                await asyncio.sleep(args.delay)
            console.print(f"[dim]{i + 1}/{len(questions)} {q['id']}[/dim]")
            t0 = time.monotonic()
            try:
                final = await graph.ainvoke({"question": q["question"]})
            except Exception as e:
                rows.append({"id": q["id"], "passed": False, "failures": [f"{type(e).__name__}: {str(e)[:200]}"],
                             "notes": [], "tokens": 0, "seconds": round(time.monotonic() - t0)})
                continue
            result = check_result(q["expect"], final)
            rows.append({"id": q["id"], "passed": result.passed, "failures": result.failures, "notes": result.notes,
                         "tokens": sum((final.get("usage") or {}).values()), "seconds": round(time.monotonic() - t0),
                         "route": final.get("route"), "report": final.get("report", "")[:1500]})

    table = Table(title="Phase 4 pipeline eval" + (" (all reasoning)" if args.all_reasoning else ""))
    for col in ("id", "result", "tokens", "time", "detail"):
        table.add_column(col, overflow="fold")
    for r in rows:
        detail = "; ".join(r["failures"]) or ("; ".join(r["notes"]) if r["notes"] else "")
        table.add_row(r["id"], "[green]PASS[/green]" if r["passed"] else "[red]FAIL[/red]",
                      f"{r['tokens']:,}", f"{r['seconds']}s", escape(detail))
    console.print(table)

    passed = sum(r["passed"] for r in rows)
    console.print(f"\nPassed: {passed}/{len(rows)}   Tokens: {sum(r['tokens'] for r in rows):,}   "
                  f"Time: {time.monotonic() - started:.0f}s")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"Full results (including reports): {RESULTS}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(asyncio.run(main()))
