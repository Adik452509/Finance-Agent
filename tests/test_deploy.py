"""Phase 6.7 guard rails for the container: no secrets in the image, not root, the right port."""

from config.settings import settings

ROOT = settings.project_root


def lines(name: str) -> list[str]:
    return [line.strip() for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")]


def test_secrets_and_local_data_never_enter_the_image():
    ignored = lines(".dockerignore")
    # "**/*.db", not "*.db": the bare pattern only covers the top folder (data/app.db leaked once).
    for pattern in (".env", ".streamlit/secrets.toml", "**/*.db", "data/chroma/*", "docs/"):
        assert pattern in ignored, f"{pattern} must be in .dockerignore"


def test_the_container_runs_as_a_normal_user_on_the_hugging_face_port():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "USER user" in dockerfile and "--uid 1000" in dockerfile
    assert "EXPOSE 7860" in dockerfile and "PORT=7860" in dockerfile
    assert "APP_ENV=production" in dockerfile  # production rules apply: login + API token required
    assert dockerfile.strip().endswith('CMD ["python", "scripts/serve.py"]')


def test_the_backend_is_only_reachable_inside_the_container():
    serve = (ROOT / "scripts" / "serve.py").read_text(encoding="utf-8")
    assert '"--host", "127.0.0.1"' in serve        # the API: localhost only
    assert '"--server.address", "0.0.0.0"' in serve  # the app: the only public port


def test_data_servers_inherit_the_environment(monkeypatch):
    # In the container keys exist only as environment variables (no .env); the MCP client's
    # default would start the servers without them (seen in the first Docker run).
    from src.mcp_client.client import ServerSpec, build_connection

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-the-environment")
    env = build_connection(ServerSpec("news", "src.mcp_servers.news_server"))["env"]
    assert env["TAVILY_API_KEY"] == "tvly-from-the-environment" and env["PYTHONIOENCODING"] == "utf-8"


def test_every_imported_package_is_pinned():
    pinned = {line.split("==")[0].split("[")[0].lower() for line in lines("requirements.txt") if "==" in line}
    for package in ("streamlit", "authlib", "fastapi", "uvicorn", "sqlalchemy", "alembic", "psycopg", "pgvector",
                    "httpx", "altair"):
        assert package in pinned, f"{package} must be pinned in requirements.txt"
