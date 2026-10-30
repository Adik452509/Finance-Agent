"""Start the backend API and the Research Desk together, for local development.

    python scripts/run_dev.py              # backend on :8000, app on http://localhost:8501
    python scripts/run_dev.py --local-db   # force the local SQLite database (e.g. on a network that blocks Postgres)

Ctrl+C stops both.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-db", action="store_true", help="use data/app.db even if DATABASE_URL is set")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if args.local_db:
        env["DATABASE_URL"] = f"sqlite:///{(ROOT / 'data' / 'app.db').as_posix()}"
        env["FILINGS_STORE"] = "local"
        subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, env=env, check=True)

    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1",
                                "--port", "8000"], cwd=ROOT, env=env)
    for _ in range(60):  # wait until the API answers before opening the app
        try:
            httpx.get("http://127.0.0.1:8000/health", timeout=2)
            break
        except httpx.HTTPError:
            time.sleep(1)
    frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "app.py"),
                                 "--server.port", str(args.port)], cwd=ROOT, env=env)
    print(f"\nResearch Desk: http://localhost:{args.port}   (API docs: http://127.0.0.1:8000/docs)\n")
    try:
        frontend.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for process in (frontend, backend):
            process.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
