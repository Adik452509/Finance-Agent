"""Production entrypoint (Docker / Hugging Face Spaces): migrate, then run the backend and the app.

    python scripts/serve.py            # what the container runs

1. Writes Streamlit's login config (.streamlit/secrets.toml) from the environment.
2. Brings the database schema up to date (alembic upgrade head).
3. Starts the backend on 127.0.0.1:8000 - reachable only inside the container.
4. Starts the Research Desk on 0.0.0.0:$PORT (7860 on Hugging Face).
If either process dies, the other is stopped and the container exits non-zero, so the
platform restarts it instead of serving a half-working app.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import settings  # noqa: E402  (validates the configuration: fails fast)
from frontend.auth import write_secrets  # noqa: E402

BACKEND_URL = "http://127.0.0.1:8000"


def log(message: str) -> None:
    print(f"[serve] {message}", flush=True)


def wait_for_backend(backend: subprocess.Popen, seconds: int = 120) -> bool:
    for _ in range(seconds):
        if backend.poll() is not None:
            return False
        try:
            if httpx.get(f"{BACKEND_URL}/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(1)
    return False


def main() -> int:
    port = os.environ.get("PORT", "7860")
    log(f"environment: {settings.app_env} | filings store: {settings.filings_store} | "
        f"database: {settings.masked_database_url()}")
    log("Google login: " + ("on" if write_secrets() else "OFF (dev user)"))

    log("database: alembic upgrade head")
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, check=True)

    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1",
                                "--port", "8000", "--no-access-log"], cwd=ROOT)
    if not wait_for_backend(backend):
        log("backend did not start")
        backend.terminate()
        return 1
    log("backend ready")

    frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "frontend" / "app.py"),
                                 "--server.port", port, "--server.address", "0.0.0.0"], cwd=ROOT)
    log(f"Research Desk on port {port}")

    processes = {"backend": backend, "frontend": frontend}
    stopping = False

    def stop(*_args) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    exit_code = 0
    while not stopping:
        for name, process in processes.items():
            if process.poll() is not None:
                log(f"{name} exited with code {process.returncode}; stopping the container")
                exit_code, stopping = process.returncode or 1, True
                break
        time.sleep(1)
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
