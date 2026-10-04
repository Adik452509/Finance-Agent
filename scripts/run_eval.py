"""Score the RAG pipeline against the verified question set in tests/eval/questions.json.

Run from the project root:
    python scripts/run_eval.py                     # default (cheap) model
    python scripts/run_eval.py --role reasoning    # stronger model, for comparison
    python scripts/run_eval.py --only ril_         # just the questions whose id starts with ril_

Each question is checked on three things:
  - retrieval: did search return a page where the answer actually appears?
  - answer:    does the answer contain the right figure (and none of the known wrong ones)?
               or, for unanswerable questions, did it refuse?
  - citations: did the citation checker raise any warnings?

Question embeddings are cached in data/cache/, so re-running after a prompt or
model change costs no embedding quota - only the first run of a new question does.
Exits 0 only if every answer check passes.
"""

import argparse
import json
import logging
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.embeddings import Embeddings  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from config.llm import get_chat_model, get_embeddings  # noqa: E402
from config.settings import settings  # noqa: E402
from src.rag.qa import answer_question  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402

warnings.filterwarnings("ignore", message=".*Timestamp.utcnow.*")
console = Console()

QUESTIONS = settings.project_root / "tests" / "eval" / "questions.json"
CACHE_DIR = settings.data_dir / "cache"


class CachedQueryEmbeddings(Embeddings):
    """Wraps the real embeddings; remembers each question's vector on disk."""

    def __init__(self, inner: Embeddings, path: Path) -> None:
        self.inner, self.path = inner, path
        self.cache: dict[str, list[float]] = json.loads(path.read_text()) if path.exists() else {}
        self.new = 0

    def embed_query(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.inner.embed_query(text)
            self.new += 1
            self.path.write_text(json.dumps(self.cache))
        return self.cache[text]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)


def answer_with_retry(question: str, attempts: int = 4, **kwargs):
    """Free-tier chat APIs rate-limit bursts; wait it out instead of scoring an error."""
    for attempt in range(1, attempts + 1):
        try:
            return answer_question(question, **kwargs)
        except Exception as e:
            text = str(e)
            if attempt == attempts or not ("429" in text or "rate limit" in text.lower()):
                raise
            hint = re.search(r"try again in ([\d.]+)s", text)
            wait = min(90.0, float(hint.group(1)) + 2) if hint else 30.0
            console.print(f"[yellow]  rate limited; waiting {wait:.0f}s (attempt {attempt}/{attempts})[/yellow]")
            time.sleep(wait)
    raise AssertionError("unreachable")


def digits(text: str) -> str:
    return re.sub(r"[^\d]", "", text)


def contains_figure(answer: str, figure: str) -> bool:
    """Match a figure regardless of grouping style: 9,80,136 == 980,136 == 980136."""
    target = digits(figure)
    for token in re.findall(r"\d[\d,\s ]*\d|\d", answer):
        if digits(token) == target:
            return True
    return False


def check(q: dict, result) -> tuple[bool, str]:
    expect = q["expect"]
    body = result.text.split("\n\nThis is an automated")[0]
    if expect["type"] == "not_found":
        return (True, "refused") if result.not_found else (False, "answered instead of refusing")
    if result.not_found:
        return False, "refused but the answer is in the filings"
    if expect["type"] == "figure":
        wrong = [f for f in expect.get("must_not", []) if contains_figure(body, f)]
        if wrong:
            return False, f"contains wrong figure {', '.join(wrong)}"
        if any(contains_figure(body, f) for f in expect["any_of"]):
            return True, "correct figure"
        return False, "expected figure missing"
    if expect["type"] == "text":
        ok = any(k.lower() in body.lower() for k in expect["any_of"])
        return ok, "keyword found" if ok else "keyword missing"
    return False, f"unknown expect type {expect['type']}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=["default", "reasoning"], default="default")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--only", default="", help="run only question ids starting with this prefix")
    parser.add_argument("--delay", type=float, default=3.0, help="seconds between questions (rate limits)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    questions = [q for q in json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"] if q["id"].startswith(args.only)]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = CachedQueryEmbeddings(get_embeddings(), CACHE_DIR / "eval_query_embeddings.json")
    retriever = Retriever(embeddings=embeddings)
    llm = get_chat_model(args.role)
    model = settings.reasoning_model if args.role == "reasoning" else settings.default_model

    table = Table(title=f"RAG eval - {model} (k={args.k})", show_lines=False)
    for col in ("id", "retrieval", "answer", "top score", "detail"):
        table.add_column(col, overflow="fold")

    rows, started = [], time.monotonic()
    for i, q in enumerate(questions):
        if i:
            time.sleep(args.delay)
        console.print(f"[dim]{i + 1}/{len(questions)} {q['id']}[/dim]")
        try:
            result = answer_with_retry(q["question"], k=args.k, retriever=retriever, llm=llm)
        except Exception as e:
            rows.append({"id": q["id"], "answer_ok": False, "detail": f"{type(e).__name__}: {str(e)[:120]}"})
            table.add_row(q["id"], "-", "[red]ERROR[/red]", "-", rows[-1]["detail"])
            continue

        pages = {h.page for h in result.hits}
        expected_pages = set(q["expect"].get("pages", []))
        retrieval_ok = None if not expected_pages else bool(pages & expected_pages)
        answer_ok, detail = check(q, result)
        top = max((h.score for h in result.hits), default=0.0)
        if result.warnings:
            detail += f" | {len(result.warnings)} citation warning(s)"

        rows.append({"id": q["id"], "type": q["expect"]["type"], "retrieval_ok": retrieval_ok,
                     "answer_ok": answer_ok, "top_score": top, "detail": detail,
                     "answer": result.text.split("\n\nThis is an automated")[0],
                     "pages": sorted(pages)})
        r_cell = "-" if retrieval_ok is None else ("[green]hit[/green]" if retrieval_ok else "[red]miss[/red]")
        a_cell = "[green]PASS[/green]" if answer_ok else "[red]FAIL[/red]"
        table.add_row(q["id"], r_cell, a_cell, f"{top:.3f}", detail)

    console.print(table)

    scored = [r for r in rows if "top_score" in r]
    answer_pass = sum(r["answer_ok"] for r in rows)
    retrieval = [r["retrieval_ok"] for r in scored if r["retrieval_ok"] is not None]
    console.print(f"\nAnswers correct : {answer_pass}/{len(rows)}")
    if retrieval:
        console.print(f"Retrieval hits  : {sum(retrieval)}/{len(retrieval)}")

    answerable = [r["top_score"] for r in scored if r["type"] != "not_found"]
    unanswerable = [r["top_score"] for r in scored if r["type"] == "not_found"]
    if answerable and unanswerable:
        console.print(f"Top score - answerable   : min {min(answerable):.3f}  median {statistics.median(answerable):.3f}")
        console.print(f"Top score - unanswerable : max {max(unanswerable):.3f}  median {statistics.median(unanswerable):.3f}")
    console.print(f"New embeddings used: {embeddings.new}  |  {time.monotonic() - started:.0f}s")

    for r in rows:
        if not r["answer_ok"]:
            console.print(f"\n[red]FAILED {r['id']}[/red]: {r['detail']}\n[dim]{r.get('answer', '')[:500]}[/dim]")

    out = CACHE_DIR / f"eval_last_{args.role}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\nFull results: {out}")
    return 0 if answer_pass == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
